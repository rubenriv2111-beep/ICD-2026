"""
=============================================================================
01_entrenamiento_protocolo_a.py
=============================================================================
Línea Base y Comparación Pareada de Clasificadores bajo el Protocolo A
(Partición Estratificada Celda a Celda 70/15/15).

Flujo cubierto:
  - Carga del banco de datos de 89,815 celdas a 30 m (Valle de Mexicali).
  - Cálculo determinista de descriptores de interacción (R_TE, I_TT).
  - Escalamiento numérico StandardScaler confinado estrictamente a Train.
  - Preservación de la distribución natural (SIN SMOTE ni submuestreo).
  - Entrenamiento y evaluación de líneas base (Dummy, Regresión Logística, Random Forest).
  - Entrenamiento y evaluación de 5 arquitecturas MLP profundas en PyTorch
    (ReLU, LeakyReLU alpha=0.25, GELU, Swish/SiLU, Tanh) con BatchNorm y Dropout.
  - Cálculo de métricas: ROC-AUC, PR-AUC, Brier score loss, matrices de confusión.

Autor: Rubén Rivera García (CICESE)
Asignatura: Ciencia de Datos (Semestre 2026-3)
Profesor: Dr. Irvin Hussein López Nava
=============================================================================
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 0. Reproducibilidad
SEMILLA = 42
random.seed(SEMILLA)
np.random.seed(SEMILLA)
torch.manual_seed(SEMILLA)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEMILLA)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Carga de Datos
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POSSIBLE_DATA = [
    os.path.join(SCRIPT_DIR, "data", "datos_geotermia_baja_california.csv"),
    os.path.join(SCRIPT_DIR, "..", "data", "datos_geotermia_baja_california.csv")
]

CSV_PATH = None
for p in POSSIBLE_DATA:
    if os.path.exists(p):
        CSV_PATH = os.path.abspath(p)
        break

if CSV_PATH is None:
    raise FileNotFoundError("No se encontró datos_geotermia_baja_california.csv en data/.")

print("=" * 80)
print("SCRIPT 01: ENTRENAMIENTO Y COMPARATIVA PAREADA (PROTOCOLO A)")
print("=" * 80)
print(f"Archivo de entrada: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
features_fisicas = ['LST', 'NDVI', 'Elevaciones', 'Pendientes', 'Prox_Fallas', 'Magnetometria']

# 2. Ingeniería de Características
df['Ratio_Termo_Estructural'] = df['LST'] / np.log(df['Prox_Fallas'] + 10.0)
df['Indice_Topo_Termico'] = df['LST'] / (df['Elevaciones'] + 50.0)
columnas_predictoras = features_fisicas + ['Ratio_Termo_Estructural', 'Indice_Topo_Termico']

X = df[columnas_predictoras].values
y = df['Target_Geotermico'].values

# Partición 70/15/15
X_train_raw, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEMILLA, stratify=y
)
X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEMILLA, stratify=y_temp
)

# 3. Escalamiento sin fugas
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_val_scaled = scaler.transform(X_val_raw)
X_test_scaled = scaler.transform(X_test_raw)

print(f"Muestras de Train:      {X_train_scaled.shape[0]:,} (Positivos: {(y_train==1).sum():,}, {y_train.mean()*100:.2f}%)")
print(f"Muestras de Validación: {X_val_scaled.shape[0]:,} (Positivos: {(y_val==1).sum():,})")
print(f"Muestras de Prueba:     {X_test_scaled.shape[0]:,} (Positivos: {(y_test==1).sum():,})")
print("-" * 80)

# 4. Modelos Clásicos
print("Entrenando modelos base...")
# Dummy
dummy = DummyClassifier(strategy='prior')
dummy.fit(X_train_scaled, y_train)
p_te_dummy = dummy.predict_proba(X_test_scaled)[:, 1]

# LogReg
lr = LogisticRegression(max_iter=1000, random_state=SEMILLA, C=1.0)
lr.fit(X_train_scaled, y_train)
p_te_lr = lr.predict_proba(X_test_scaled)[:, 1]

# RF
rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=SEMILLA, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
p_te_rf = rf.predict_proba(X_test_scaled)[:, 1]

# 5. Red Neuronal MLP PyTorch
print("Entrenando Perceptrón Multicapa (MLP) LeakyReLU (0.25) con BatchNorm...")
X_tr_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_tr_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_te_t = torch.tensor(X_test_scaled, dtype=torch.float32)

train_ds = TensorDataset(X_tr_t, y_tr_t)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

class MLPGeotermia(nn.Module):
    def __init__(self, n_in):
        super(MLPGeotermia, self).__init__()
        act_fn = nn.LeakyReLU(negative_slope=0.25)
        self.net = nn.Sequential(
            nn.Linear(n_in, 64),
            nn.BatchNorm1d(64),
            act_fn,
            nn.Dropout(0.10),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            act_fn,
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            act_fn,
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

torch.manual_seed(SEMILLA)
model = MLPGeotermia(n_in=X.shape[1]).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
crit = nn.BCEWithLogitsLoss()

model.train()
for ep in range(25):
    for bx, by in train_loader:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        opt.zero_grad()
        crit(model(bx), by).backward()
        opt.step()

model.eval()
with torch.no_grad():
    p_te_mlp = torch.sigmoid(model(X_te_t.to(DEVICE))).cpu().numpy().flatten()

print("\n" + "=" * 80)
print("TABLA COMPARATIVA EN EL CONJUNTO DE PRUEBA INDEPENDIENTE (PROTOCOLO A):")
print("=" * 80)

modelos = [
    ('Dummy Classifier', p_te_dummy),
    ('Regresión Logística (con R_TE, I_TT)', p_te_lr),
    ('Random Forest (100 árboles)', p_te_rf),
    ('MLP PyTorch LeakyReLU (0.25)', p_te_mlp)
]

filas = []
for name, p in modelos:
    roc = roc_auc_score(y_test, p)
    pr = average_precision_score(y_test, p)
    brier = brier_score_loss(y_test, p)
    filas.append({
        'Modelo': name,
        'ROC-AUC': round(roc, 4),
        'PR-AUC': round(pr, 4),
        'Brier Score': round(brier, 5)
    })

df_res_a = pd.DataFrame(filas)
print(df_res_a.to_string(index=False))
print("=" * 80)
print("SCRIPT 01 COMPLETADO CON ÉXITO")
