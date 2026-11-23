"""
================================================================================
SCRIPT 03: INFERENCIA CARTOGRÁFICA REGIONAL Y ÁREA DE APLICABILIDAD (AOA)
================================================================================
Proyecto: Prospección Geotérmica en el Rift de Baja California
Autor: Rubén Rivera García (CICESE) - Ciencia de Datos 2026-3
Profesor: Dr. Irvin Hussein Lopez Nava

Objetivo:
  Genera la cartografía regional comparativa multimodelo y la delimitación
  espacial mediante el Área de Aplicabilidad (AOA Meyer & Pebesma / Índice de Disimilitud).
  Produce la Figura 10 (Gráfico 10) en tira horizontal estándar IEEE de 2 columnas (180 mm):
  - Panel (a): Regresión Logística con descriptores heurísticos (R_TE, I_TT)
  - Panel (b): Random Forest (100 estimadores)
  - Panel (c): MLP Ensamble (3 semillas, LeakyReLU)
  - Panel (d): Área de Aplicabilidad (AOA clasificada mediante DI <= 2.13)
================================================================================
"""

import os
import sys
import time
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

t0 = time.time()
print("=" * 80)
print("SCRIPT 03: INFERENCIA CARTOGRÁFICA REGIONAL Y ÁREA DE APLICABILIDAD (AOA)")
print("================================================================================")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEARCH_DIRS = [
    os.path.join(SCRIPT_DIR, "mapas_qgis"),
    SCRIPT_DIR,
    os.path.join(SCRIPT_DIR, "..", "mapas_qgis")
]

FIG_DIRS = [
    os.path.join(SCRIPT_DIR, "figures")
]
for fd in FIG_DIRS:
    os.makedirs(fd, exist_ok=True)

# 1. LOCALIZAR CAPAS GEOTIFF REGIONALES
def find_tif(name):
    for d in SEARCH_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None

p_lr_path = find_tif("prob_logreg.tif")
p_rf_path = find_tif("prob_rf.tif")
p_mlp_path = find_tif("prob_mlp.tif")
aoa_path = find_tif("aoa_mask.tif")

if not all([p_lr_path, p_rf_path, p_mlp_path, aoa_path]):
    print("ERROR: No se encontraron los 4 GeoTIFFs base. Ejecutar exportar_geotiffs_qgis.py primero.")
    sys.exit(1)

print("[1/3] Cargando capas GeoTIFF de alta resolución (120 m)...")
with rasterio.open(p_lr_path) as s_lr:
    map_lr = s_lr.read(1)
    bounds = s_lr.bounds
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

with rasterio.open(p_rf_path) as s_rf:
    map_rf = s_rf.read(1)

with rasterio.open(p_mlp_path) as s_mlp:
    map_mlp = s_mlp.read(1)

with rasterio.open(aoa_path) as s_aoa:
    raw_aoa = s_aoa.read(1)
    # 0 = NoData, 1 = Dentro, 2 = Fuera
    # Mapear para imshow: 0=Fuera (naranja), 1=Dentro (verde/turquesa), NaN=NoData
    map_aoa = np.full_like(map_lr, np.nan, dtype=np.float32)
    map_aoa[raw_aoa == 1] = 1.0
    map_aoa[raw_aoa == 2] = 0.0

# 2. AUDITORÍA ESTADÍSTICA REGIONAL Y EL CHINERO
print("[2/3] Calculando estadísticas territoriales de AOA y verificación en El Chinero...")
valid_cells = (raw_aoa > 0)
total_valid = int(valid_cells.sum())
dentro_aoa = int((raw_aoa == 1).sum())
fuera_aoa = int((raw_aoa == 2).sum())
pct_dentro = (dentro_aoa / total_valid) * 100.0
pct_fuera = (fuera_aoa / total_valid) * 100.0

print(f"  * Total celdas continentales evaluadas: {total_valid:,}")
print(f"  * Celdas dentro del AOA (DI <= 2.1339): {dentro_aoa:,} ({pct_dentro:.2f}%)")
print(f"  * Celdas fuera del AOA (extrapolación): {fuera_aoa:,} ({pct_fuera:.2f}%)")

# Coordenadas exactas de El Chinero (31.45 N, 115.05 W)
with rasterio.open(aoa_path) as s_aoa:
    r_ch, c_ch = s_aoa.index(685295.32, 3481120.55)
    val_ch = raw_aoa[r_ch, c_ch]
    p_lr_ch = map_lr[r_ch, c_ch]
    p_rf_ch = map_rf[r_ch, c_ch]
    p_mlp_ch = map_mlp[r_ch, c_ch]
    print(f"  * Diagnóstico en El Chinero (685295 m, 3481120 m):")
    print(f"    - Estado AOA: {'Fuera (Extrapolación)' if val_ch == 2 else 'Dentro'} (Valor ráster = {val_ch})")
    print(f"    - Probabilidad Regresión Logística: {p_lr_ch:.4f}")
    print(f"    - Probabilidad Random Forest:       {p_rf_ch:.4f}")
    print(f"    - Probabilidad MLP Ensamble:        {p_mlp_ch:.4f}")

# 3. RENDERIZADO DE TIRA HORIZONTAL IEEE (180 MM = 7.0866 PULGADAS)
print("[3/3] Renderizando tira horizontal de 4 paneles estándar IEEE (180 mm a 300 DPI)...")
fig, axes = plt.subplots(1, 4, figsize=(7.0866, 3.6), dpi=300)
fig.subplots_adjust(left=0.04, right=0.96, top=0.88, bottom=0.22, wspace=0.10)

cmap_prob = plt.cm.viridis
cmap_aoa = ListedColormap(['#d95f02', '#1b9e77']) # 0=Fuera (naranja), 1=Dentro (verde/turquesa)

# Paneles a, b, c
im0 = axes[0].imshow(map_lr, extent=extent, origin='upper', cmap=cmap_prob, vmin=0, vmax=1)
axes[0].set_title("(a) Regresión Logística", fontsize=8, fontweight='bold', pad=4)

im1 = axes[1].imshow(map_rf, extent=extent, origin='upper', cmap=cmap_prob, vmin=0, vmax=1)
axes[1].set_title("(b) Random Forest", fontsize=8, fontweight='bold', pad=4)

im2 = axes[2].imshow(map_mlp, extent=extent, origin='upper', cmap=cmap_prob, vmin=0, vmax=1)
axes[2].set_title("(c) MLP Ensamble", fontsize=8, fontweight='bold', pad=4)

# Panel d
im3 = axes[3].imshow(map_aoa, extent=extent, origin='upper', cmap=cmap_aoa, vmin=0, vmax=1)
axes[3].set_title("(d) Área de Aplicabilidad (AOA)", fontsize=8, fontweight='bold', pad=4)

# Decoraciones en cada panel
for ax in axes:
    ax.tick_params(labelsize=6, length=2, pad=1)
    ax.grid(color='gray', linestyle=':', linewidth=0.3, alpha=0.5)
    # Cerro Prieto polígono de explotación aproximado y El Chinero
    ax.plot([650000, 655000, 655000, 650000, 650000], [3580000, 3580000, 3590000, 3590000, 3580000], 
            color='#900C3F', linewidth=1.0)
    ax.scatter(685295.32, 3481120.55, color='#e7298a', marker='d', s=18, edgecolors='black', linewidth=0.4, zorder=5)

axes[0].set_ylabel("Norte UTM (m)", fontsize=7)
for ax in axes[1:]:
    ax.set_yticklabels([])

# Barra de color continua para (a, b, c)
cbar_ax1 = fig.add_axes([0.06, 0.08, 0.60, 0.035])
cb1 = fig.colorbar(im0, cax=cbar_ax1, orientation='horizontal')
cb1.set_label("Probabilidad / Puntuación Estimada P(x)", fontsize=7, fontweight='bold')
cb1.ax.tick_params(labelsize=6)

# Barra discreta para (d)
cbar_ax2 = fig.add_axes([0.76, 0.08, 0.18, 0.035])
cb2 = fig.colorbar(im3, cax=cbar_ax2, orientation='horizontal', ticks=[0.25, 0.75])
cb2.ax.set_xticklabels(['Fuera', 'Dentro'], fontsize=6)
cb2.set_label("DI <= 2.13", fontsize=7, fontweight='bold')

out_filename = "grafico_10_mapa_susceptibilidad_espacial.png"
for fd in FIG_DIRS:
    out_p = os.path.join(fd, out_filename)
    fig.savefig(out_p, dpi=300, bbox_inches='tight')
    print(f"Figura guardada en: {out_p}")

plt.close(fig)
print(f"\nGeneración cartográfica completada exitosamente en {time.time()-t0:.1f} s")
print("=" * 80)
