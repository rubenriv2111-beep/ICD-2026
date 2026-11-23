# REPORTE METODOLÓGICO FASE 5: RECONCILIACIÓN, PRUEBAS DIRIGIDAS Y MAPA EXPLORATORIO

Estudio diagnóstico de separabilidad del polígono de Cerro Prieto frente a fondos de dificultad creciente utilizando covariables regionales públicas. Este trabajo evalúa la influencia de la cobertura superficial, la posición espacial y los artefactos cartográficos en las métricas de clasificación. No constituye una prospección ni una validación de recursos geotérmicos.

---

## (A) DATOS

### 1. Reconciliación de Regresión Logística (0.1)
- Fuente: `/results/reconciliacion_lr.csv`.
- Pliegue 1 (fila 0): PR-AUC Fase 3 = 0.96870, Fase 4 S3 = 0.96869, Nueva corrida = 0.96869 (diff = 0.00001). ROC-AUC Fase 3 = 0.99300, Fase 4 S3 = 0.99300, Nueva = 0.99300 (diff = 0.00000). Brier = 0.03208.
- Pliegue 2 (fila 1): PR-AUC Fase 3 = 0.98600, Fase 4 S3 = 0.98601, Nueva = 0.98601 (diff = 0.00001). ROC-AUC Fase 3 = 0.99250, Fase 4 S3 = 0.99249, Nueva = 0.99249 (diff = 0.00001). Brier = 0.03174.
- Pliegue 3 (fila 2): PR-AUC Fase 3 = 0.98860, Fase 4 S3 = 0.98860, Nueva = 0.98860 (diff = 0.00000). ROC-AUC Fase 3 = 0.99510, Fase 4 S3 = 0.99515, Nueva = 0.99515 (diff = 0.00005). Brier = 0.03051.
- Pliegue 4 (fila 3): PR-AUC Fase 3 = 0.90530, Fase 4 S3 = 0.90532, Nueva = 0.90532 (diff = 0.00002). ROC-AUC Fase 3 = 0.97110, Fase 4 S3 = 0.97105, Nueva = 0.97105 (diff = 0.00005). Brier = 0.05359.
- Pliegue 5 (fila 4): PR-AUC Fase 3 = 0.91490, Fase 4 S3 = 0.91486, Nueva = 0.91486 (diff = 0.00004). ROC-AUC Fase 3 = 0.95790, Fase 4 S3 = 0.95789, Nueva = 0.95789 (diff = 0.00001). Brier = 0.09519.
- Diferencia máxima absoluta entre Fase 3, Fase 4 (S3) y la nueva corrida: 0.000048.
- Parámetros técnicos comparados: orden de covariables idéntico (8 variables), scaler ajustado exclusivamente en el conjunto de entrenamiento de cada pliegue, algoritmo lbfgs con max_iter=1000 sin advertencias de convergencia.
- Desempeño sin términos derivados (escenario S2_base, columna PR_AUC_Fase4_S2_sin_deriv): Pliegue 1 = 0.91540, Pliegue 2 = 0.94845, Pliegue 3 = 0.94163, Pliegue 4 = 0.86649, Pliegue 5 = 0.56015.

### 2. Estadísticas de Magnetometría e IQR por Banda (0.2)
- Fuente: `/results/magnetometria_estadisticas_bandas.csv`.
- Positivos (POS, n=24814): mediana = -138.9445 nT, IQR = 40.0064 nT (Q25=-150.0064, Q75=-110.0000).
- Banda 1 (B1, n=21667): mediana = -130.0000 nT, IQR = 59.9550 nT.
- Banda 2 (B2, n=21667): mediana = -110.0033 nT, IQR = 44.3608 nT.
- Banda 3 Total (B3, n=21667): mediana = -225.0000 nT, IQR = 100.0000 nT.
- Banda 3 Llanura (n=15781): mediana = -210.0000 nT, IQR = 103.7041 nT.
- Banda 3 Llanura Carta Mexicali (n=7629): mediana = -170.0000 nT, IQR = 60.0000 nT.
- Banda 3 Llanura Carta San Felipe (n=8152): mediana = -262.9347 nT, IQR = 50.0000 nT.

### 3. Diagnóstico Posicional de Negativos y RF con Coordenadas Solas (0.3)
- Fuentes: `/results/diagnostico_posicional_negativos.csv`, `/results/rf_coord_solas_pliegues4_5.csv`, `/figures/mapa_coord_solas_p5.png`.
- Distancia de negativos de test al negativo de train más cercano (en metros):
  - Pliegue 1 (todas las bandas, n=15,537): mediana = 1899.74 m, IQR = 1401.30 m, min = 502.89 m, max = 6503.43 m. <1 km = 15.18%, <2 km = 53.71%.
  - Pliegue 4 (todas las bandas, n=24,547): mediana = 2095.07 m, IQR = 1706.81 m, min = 502.89 m, max = 9510.05 m. <1 km = 12.75%, <2 km = 47.11%.
  - Pliegue 5 (todas las bandas, n=7,677): mediana = 1471.22 m, IQR = 1588.10 m, min = 502.89 m, max = 9181.08 m. <1 km = 26.14%, <2 km = 66.80%.
- Random Forest entrenado únicamente con coordenadas (x_utm, y_utm):
  - Pliegue 4 (baseline prevalencia = 0.17650):
    - TODAS: ROC-AUC = 0.77671, PR-AUC = 0.44599, Brier = 0.14877.
    - B1: ROC-AUC = 0.68410, PR-AUC = 0.51604, baseline = 0.38040.
    - B2: ROC-AUC = 0.82210, PR-AUC = 0.76523, baseline = 0.31119.
    - B3-llanura: ROC-AUC = 0.83816, PR-AUC = 0.87922, baseline = 0.62579.
  - Pliegue 5 (baseline prevalencia = 0.36116):
    - TODAS: ROC-AUC = 0.94718, PR-AUC = 0.84255, Brier = 0.07540.
    - B1: ROC-AUC = 0.84240, PR-AUC = 0.84255, baseline = 0.62780.
    - B2: ROC-AUC = 1.00000, PR-AUC = 1.00000, baseline = 0.84915.
    - B3-llanura: ROC-AUC = 1.00000, PR-AUC = 1.00000, baseline = 0.53180.

### 4. Porcentaje de Celdas con Gradiente Magnético Exactamente Cero (0.4)
- Fuente: `/results/gradiente_cero.csv`.
- Carta Mexicali H11-12 (total celdas válidas = 76566):
  - Total carta: 27422 celdas con gradiente exactamente 0 (35.8149%).
  - Clase POS: 6257 celdas con gradiente 0 de 24814 (25.2156%).
  - Clase NEG: 21165 celdas con gradiente 0 de 51752 (40.8970%).
  - Banda B1: 5458 celdas (25.1904%).
  - Banda B2: 10695 celdas (49.3608%).
  - Banda B3: 5012 celdas (59.5391%).
- Carta San Felipe H11-3 (total celdas válidas = 13141):
  - Total carta: 9127 celdas con gradiente exactamente 0 (69.4544%).
  - Clase POS: 0 celdas (0.0000%).
  - Clase NEG: 9127 celdas (69.4544%).
  - Banda B3: 9127 celdas (69.4544%).

### 5. MNDWI en Positivos de Test y Desempeño de G_sup (0.5)
- Fuente: `/results/mndwi_vs_gsup_pliegues.csv`.
- Pliegue 1: 2321 de 4385 positivos con MNDWI > 0 (52.93%). RF G_sup ROC-AUC = 0.95346, PR-AUC = 0.88246, Brier = 0.06328.
- Pliegue 2: 3204 de 5443 positivos con MNDWI > 0 (58.86%). RF G_sup ROC-AUC = 0.88859, PR-AUC = 0.86977, Brier = 0.11267.
- Pliegue 3: 3694 de 5385 positivos con MNDWI > 0 (68.60%). RF G_sup ROC-AUC = 0.92997, PR-AUC = 0.91236, Brier = 0.10574.
- Pliegue 4: 2256 de 5261 positivos con MNDWI > 0 (42.88%). RF G_sup ROC-AUC = 0.92896, PR-AUC = 0.80011, Brier = 0.07090.
- Pliegue 5: 522 de 4340 positivos con MNDWI > 0 (12.03%). RF G_sup ROC-AUC = 0.64971, PR-AUC = 0.52110, Brier = 0.26340.
- Correlación de Spearman descriptiva (n=5):
  - % positivos test con MNDWI > 0 frente a PR-AUC de RF G_sup: r = 0.9000 (p = 0.0374).
  - % positivos test con MNDWI > 0 frente a ROC-AUC de RF G_sup: r = 0.5000 (p = 0.3910).

### 6. Prueba Dirigida T1: Positivos Húmedos vs Secos
- Fuente: `/results/prueba_humedo_seco.csv`.
- Definición operativa: húmedo = (MNDWI > 0 O NDVI < 0); seco = el resto.
- Conteos globales: 12195 positivos húmedos (49.15%) y 12619 positivos secos (50.85%).
- Positivos secos en conjunto de prueba por pliegue: Pliegue 1 = 2031 (concluyente), Pliegue 2 = 2156 (concluyente), Pliegue 3 = 1653 (concluyente), Pliegue 4 = 2971 (concluyente), Pliegue 5 = 3808 (concluyente). En todos los pliegues n >= 500.
- Desempeño promedio en 5 pliegues frente a TODAS las bandas (condición a: entrenado con datos completos, evaluado solo en secos):
  - G_sup solo: ROC-AUC = 0.83025, PR-AUC = 0.52818, delta PR-AUC = 0.32049.
  - G_est solo: ROC-AUC = 0.89669, PR-AUC = 0.62091, delta PR-AUC = 0.41322.
  - S2 base: ROC-AUC = 0.97126, PR-AUC = 0.92729, delta PR-AUC = 0.71960.
  - S4 sin G_sup: ROC-AUC = 0.94326, PR-AUC = 0.84636, delta PR-AUC = 0.63867.
- Desempeño promedio en 5 pliegues frente a TODAS las bandas (condición b: entrenado y evaluado sin positivos húmedos):
  - G_sup solo: ROC-AUC = 0.84133, PR-AUC = 0.58288, delta PR-AUC = 0.37519.
  - G_est solo: ROC-AUC = 0.90399, PR-AUC = 0.65249, delta PR-AUC = 0.44480.
  - S2 base: ROC-AUC = 0.97017, PR-AUC = 0.93065, delta PR-AUC = 0.72296.
  - S4 sin G_sup: ROC-AUC = 0.93922, PR-AUC = 0.84126, delta PR-AUC = 0.63357.

### 7. Prueba Dirigida T2: Placebo de Fallas Estructurales
- Fuente: `/results/placebo_fallas.csv`.
- Rangos de distancia euclidiana en los positivos de Cerro Prieto:
  - Real: min = 7093.58 m, mediana = 9118.49 m, max = 11350.17 m.
  - Placebo dx_+10km: min = 1187.42 m, mediana = 3688.76 m, max = 6558.44 m.
  - Placebo dx_-10km: min = 14471.82 m, mediana = 17912.91 m, max = 19995.79 m.
  - Placebo dy_+10km: min = 3673.06 m, mediana = 6378.04 m, max = 8349.82 m.
  - Placebo dy_-10km: min = 14306.24 m, mediana = 17214.25 m, max = 20310.80 m.
  - Placebo rot_90deg: min = 4265.81 m, mediana = 8833.12 m, max = 12800.12 m.
- Desempeño de Random Forest S1_G_est (media 5 pliegues):
  - Real: ROC-AUC = 0.91242, PR-AUC = 0.73374.
  - Placebo dx_+10km: ROC-AUC = 0.83064, PR-AUC = 0.63172.
  - Placebo dx_-10km: ROC-AUC = 0.81846, PR-AUC = 0.60852.
  - Placebo dy_+10km: ROC-AUC = 0.84971, PR-AUC = 0.63552.
  - Placebo dy_-10km: ROC-AUC = 0.76894, PR-AUC = 0.50946.
  - Placebo rot_90deg: ROC-AUC = 0.50421, PR-AUC = 0.31531.
- Casos donde el placebo supera a las fallas reales:
  - Pliegue 1 con placebo dx_-10km: ROC-AUC placebo = 0.90408 frente a real = 0.81776; PR-AUC placebo = 0.59780 frente a real = 0.59156.
  - Pliegue 5 con placebo dx_+10km: ROC-AUC placebo = 0.95919 frente a real = 0.87249; PR-AUC placebo = 0.89049 frente a real = 0.64238.
  - Pliegue 5 con placebo dy_-10km: ROC-AUC placebo = 0.88800 frente a real = 0.87249; PR-AUC placebo = 0.69415 frente a real = 0.64238.
- Desempeño de Random Forest S2 base (media 5 pliegues):
  - Real: ROC-AUC = 0.97787, PR-AUC = 0.95958.
  - Placebo dx_+10km: ROC-AUC = 0.95516, PR-AUC = 0.91815.
  - Placebo dx_-10km: ROC-AUC = 0.93119, PR-AUC = 0.87800.
  - Placebo dy_+10km: ROC-AUC = 0.94235, PR-AUC = 0.89055.
  - Placebo dy_-10km: ROC-AUC = 0.96562, PR-AUC = 0.93131.
  - Placebo rot_90deg: ROC-AUC = 0.92408, PR-AUC = 0.83497.

### 8. Prueba Dirigida T3: Magnetometría por Carta Cartográfica
- Fuente: `/results/magnetometria_por_carta.csv`.
- Positivos frente a Banda 1 (fila 0): ROC-AUC = 0.63912, PR-AUC = 0.67658, baseline = 0.53385, delta PR-AUC = 0.14273. Mediana de distancia de negativos al polígono = 1274.56 m. Advertencia: efecto confundido (la composicion de distancia al poligono esta confundida con la carta cartografica).
- Positivos frente a Banda 2 (fila 1): ROC-AUC = 0.64413, PR-AUC = 0.61542, baseline = 0.53385, delta PR-AUC = 0.08157. Mediana distancia negativos = 3600.50 m. Advertencia: efecto confundido (la composicion de distancia al poligono esta confundida con la carta cartografica).
- Positivos frente a Banda 3 Llanura Mexicali (fila 2): ROC-AUC = 0.78480, PR-AUC = 0.87115, baseline = 0.76485, delta PR-AUC = 0.10630. Mediana distancia negativos = 41485.17 m. Advertencia: efecto confundido (la composicion de distancia al poligono esta confundida con la carta cartografica).
- Positivos frente a Banda 3 Llanura San Felipe (fila 3): ROC-AUC = 0.98321, PR-AUC = 0.99133, baseline = 0.75271, delta PR-AUC = 0.23862. Mediana distancia negativos = 92760.32 m. Advertencia: efecto confundido (la composicion de distancia al poligono esta confundida con la carta cartografica).

### 9. Mapa Exploratorio, Área de Aplicabilidad (AOA) y Top 0.1%
- Fuentes: `/results/aoa_metodo.csv`, `/results/top_puntuacion_resumen.csv`, `/results/chinero_resumen.csv`, `/results/sanidad_puntuaciones.csv`.
- Área de Aplicabilidad (AOA de Meyer y Pebesma, 2021):
  - Celdas válidas de la rejilla regular de 120 m sobre el dominio continental: 1,202,676.
  - Celdas dentro del Área de Aplicabilidad: 1,123,560 (93.42%).
  - Umbral de disimilitud DI = 2.5057 (Q3=1.2535, IQR=0.8348, distancia media en train = 0.1791).
  - Pesos relativos de variables: dist_fallas_SGM_m = 3.4005, LST = 0.7966, NDVI = 0.7966, Magnetometria = 0.5781, Elevaciones = 0.2141, Pendientes = 0.2141.
- Top 0.1% de puntuación fuera del polígono + 5 km:
  - Modelo RF-S2: 1192 celdas, umbral p99.9 = 0.35069, 99.24% en AOA, 0.00% MNDWI>0, 0.00% NDVI<0, 0.00% NDVI>0.2, distancia mediana = 102.94 km. Cartas: Mexicali:401(33.6%), SanFelipe:791(66.4%).
  - Modelo LR-S2: 1192 celdas, umbral p99.9 = 1.00000, 63.09% en AOA, 100.00% MNDWI>0, 100.00% NDVI<0, 0.00% NDVI>0.2, distancia mediana = 72.31 km. Cartas: Mexicali:198(16.6%), SanFelipe:994(83.4%).
  - Modelo RF solo G_sup: 1192 celdas, umbral p99.9 = 0.98684, 47.73% en AOA, 97.15% MNDWI>0, 87.33% NDVI<0, distancia mediana = 93.17 km. Cartas: Mexicali:117(9.8%), SanFelipe:1075(90.2%).
  - Modelo RF solo G_est: 1199 celdas, umbral p99.9 = 0.80734, 97.16% en AOA, distancia mediana = 91.45 km. Cartas: Mexicali:339(28.3%), SanFelipe:860(71.7%).
  - Modelo RF S4 sin G_sup: 1192 celdas, umbral p99.9 = 0.44283, 99.58% en AOA, distancia mediana = 100.42 km. Cartas: Mexicali:302(25.3%), SanFelipe:890(74.7%).
- Sector El Chinero (31.45° N, 115.05° O; coordenadas UTM 11N: X=685295.32 m, Y=3481120.55 m):
  - Celdas válidas en ventana de 3 km: 1962.
  - Puntuación RF-S2: mediana = 0.03326, percentil 95 = 0.81977.
  - Puntuación LR-S2: mediana = 0.06310, percentil 95 = 0.19538.
  - Puntuación RF solo G_sup: mediana = 0.18969.
  - Puntuación RF solo G_est: mediana = 0.73756.
  - Puntuación RF S4 sin G_sup: mediana = 0.00387.
  - Rango entre modelos (incertidumbre): mediana = 0.06284.
  - Índice de disimilitud DI mediano = 0.7598, porcentaje en AOA = 97.20%.
  - Carta magnética: San_Felipe_H11-3. MNDWI > 0 = 0.00%.
  - Mediana NDVI = 0.0656 (percentil respecto a positivos = 53.87%).
  - Mediana LST = 54.31 °C (percentil respecto a positivos = 90.39%).
  - Mediana dist_fallas = 8747.21 m (percentil respecto a positivos = 36.57%).
  - Mediana elevación = 49.81 m (percentil respecto a positivos = 96.18%).
  - Mediana Magnetometría = -225.00 nT (percentil respecto a positivos = 0.00%).
- Sanidad de puntuaciones del modelo RF-S2:
  - Rejilla dentro del polígono (n=1551): media = 0.97977, mediana = 0.99901.
  - Rejilla en banda B1 (distancia 500-2000 m, n=2568): media = 0.01627, mediana = 0.00117.
  - Dataset de positivos de entrenamiento (n=24814): media = 0.98041, mediana = 0.99903.
  - Dataset de negativos B1 de entrenamiento (n=21667): media = 0.01592, mediana = 0.00124.

---

## (B) LECTURA PERMITIDA

1. En la comparación entre la Regresión Logística de la Fase 3, el escenario S3 de la Fase 4 y la corrida de control, los valores de PR-AUC, ROC-AUC y Brier son idénticos al cuarto decimal en los 5 pliegues evaluados, con diferencias absolutas menores o iguales a 0.000048. NO IMPLICA: que el modelo lineal posea capacidad de generalización espacial a dominios no observados.
2. En el escenario S2_base (sin términos derivados de interacción física), la Regresión Logística presenta un PR-AUC de 0.56015 en el Pliegue 5 frente a 0.91486 en el escenario S3 (con derivados). NO IMPLICA: superioridad intrínseca de los clasificadores no lineales en ausencia de covariables compuestas.
3. El clasificador Random Forest entrenado únicamente con coordenadas geográficas (x_utm, y_utm) obtiene un PR-AUC de 0.84255 en el Pliegue 5 (baseline = 0.36116) y alcanza 1.00000 en las particiones contra Banda 2 y Banda 3 Llanura. En el Pliegue 4 obtiene un PR-AUC de 0.44599 (baseline = 0.17650). NO IMPLICA: que las coordenadas espaciales capturen mecanismos físicos del subsuelo.
4. En la carta Mexicali H11-12, el 35.81% de las celdas válidas presenta un gradiente magnético local exactamente igual a 0. En la carta San Felipe H11-3, el 69.45% de las celdas válidas presenta un gradiente magnético local exactamente igual a 0. En la clase positiva, el porcentaje de celdas con gradiente 0 es 25.22%. NO IMPLICA: ausencia o presencia de anomalías térmicas o geológicas en el subsuelo.
5. El porcentaje de celdas positivas con MNDWI > 0 en el conjunto de prueba es de 52.93% en el Pliegue 1, 58.86% en el Pliegue 2, 68.60% en el Pliegue 3, 42.88% en el Pliegue 4 y 12.03% en el Pliegue 5. La correlación de Spearman entre este porcentaje y el PR-AUC del modelo RF entrenado solo con cobertura superficial (LST, NDVI) es r = 0.9000 (p = 0.0374, n=5). NO IMPLICA: relación causal directa entre la laguna de evaporación superficial y la firma geotérmica profunda.
6. En la prueba T1 de positivos secos frente a húmedos, todos los pliegues cuentan con al menos 1,653 positivos secos (superando el umbral de 500 para ser concluyente). El modelo S2_base evaluado únicamente con positivos secos mantiene un PR-AUC promedio de 0.92729 (delta frente al baseline = 0.71960). NO IMPLICA: que la señal del modelo sea independiente de las alteraciones hidrotermales o antrópicas del terreno.
7. En la prueba de placebos espaciales de fallas (T2), las fallas reales superan a los 5 placebos en el promedio de los 5 pliegues para S1_G_est (PR-AUC real = 0.73374 vs placebos entre 0.31531 y 0.63552). Sin embargo, en el Pliegue 1 el placebo con traslación dx = -10 km obtiene un PR-AUC de 0.59780 (superior al real = 0.59156), y en el Pliegue 5 el placebo con traslación dx = +10 km obtiene un PR-AUC de 0.89049 (superior al real = 0.64238). NO IMPLICA: que el modelo aprenda una relación geométrica estructural real del reservorio con la traza de falla.
8. La capacidad discriminante de la Magnetometría sola (T3) aumenta monótonamente con la distancia al polígono: ROC-AUC = 0.63912 contra Banda 1 (mediana de distancia 1,274.56 m) hasta ROC-AUC = 0.98321 contra Banda 3 Llanura San Felipe (mediana de distancia 92,760.32 m). La composición de distancia espacial está confundida con la carta cartográfica. NO IMPLICA: detección de desmagnetización térmica en el reservorio.
9. En el mapa regional exploratorio, el 93.42% del dominio continental se encuentra dentro del Área de Aplicabilidad (AOA). En el sector de El Chinero (31.45° N, 115.05° O), la puntuación mediana del modelo principal RF-S2 es 0.03326 (percentil 95 = 0.81977), mientras que el modelo RF solo G_est asigna una puntuación mediana de 0.73756 y el modelo RF S4 sin G_sup asigna 0.00387. La mediana de Magnetometría en El Chinero (-225.00 nT) corresponde al percentil 0.00% de los positivos de Cerro Prieto. NO IMPLICA: aseverar ni descartar la presencia de un reservorio en El Chinero.

---

## (C) HIPÓTESIS

### Hipótesis 1: El desempeño de Random Forest en los pliegues contiguos proviene del sobreajuste a la posición geográfica relativa.
- (i) Medición que la apoyaría: Un modelo entrenado exclusivamente con coordenadas espaciales (x_utm, y_utm) produce un desempeño predictivo en el conjunto de prueba sustancialmente superior al azar y comparable con los modelos de covariables físicas.
- (ii) Medición que la refutaría: Un modelo con solo coordenadas espaciales obtiene un PR-AUC cercano o inferior al baseline de prevalencia (delta PR-AUC <= 0.05) en todos los pliegues de prueba.
- (iii) Evaluación empírica: Se ejecutó la evaluación en los Pliegues 4 y 5. En el Pliegue 5, RF(x, y) obtuvo PR-AUC = 0.84255 (delta = 0.48139) y alcanzó PR-AUC = 1.00000 en Banda 2 y Banda 3 Llanura. En el Pliegue 4 obtuvo PR-AUC = 0.44599 (delta = 0.26949).
- Estado: COMPATIBLE CON LOS DATOS.
- Hipótesis alternativa: Las coordenadas actúan como variables instrumentales que correlacionan con gradientes climáticos y geológicos regionales continuos. Medición diferencial: evaluar RF(x, y) tras despojar a los datos de tendencias espaciales determinísticas de primer orden.

### Hipótesis 2: La separabilidad aportada por las fallas SGM corresponde a una distancia de proximidad al centro del valle y no a un acoplamiento físico reservorio-falla.
- (i) Medición que la apoyaría: Capas de fallas sintéticas desplazadas espacialmente en 10 km (placebos) proporcionan una capacidad de discriminación similar o superior a la traza real en pliegues específicos.
- (ii) Medición que la refutaría: Las fallas sintéticas desplazadas reducen el desempeño del clasificador a niveles indistinguibles del baseline en todos los pliegues (delta PR-AUC <= 0.05).
- (iii) Evaluación empírica: En el Pliegue 1, el placebo dx = -10 km obtuvo PR-AUC = 0.59780 (superior al real = 0.59156). En el Pliegue 5, el placebo dx = +10 km obtuvo PR-AUC = 0.89049 (superior al real = 0.64238).
- Estado: COMPATIBLE CON LOS DATOS.
- Hipótesis alternativa: La densidad y orientación de fallas en el Rift es suficientemente uniforme para que cualquier traza sub-paralela conserve información posicional del rift central. Medición diferencial: evaluar placebos con rotaciones aleatorias y curvaturas no paralelas a la cuenca.

### Hipótesis 3: La separabilidad de la Magnetometría a escala regional es un artefacto de la costura cartográfica entre cartas digitalizadas de distinta época.
- (i) Medición que la apoyaría: La capacidad discriminante de la Magnetometría sola correlaciona con la pertenencia a la carta San Felipe H11-3 y desaparece al comparar positivos con negativos dentro de la misma carta Mexicali H11-12 a igual distancia.
- (ii) Medición que la refutaría: La Magnetometría mantiene un ROC-AUC >= 0.85 contra negativos de la misma carta cartográfica situados en la banda B1 (500-2000 m).
- (iii) Evaluación empírica: Frente a Banda 1 (misma carta), el ROC-AUC es 0.63912 (PR-AUC = 0.67658 vs baseline 0.53385). Frente a Banda 3 en la carta San Felipe, el ROC-AUC asciende a 0.98321. Adicionalmente, el 69.45% de San Felipe presenta gradiente exactamente 0 frente a 35.81% en Mexicali.
- Estado: NO REFUTADA.
- Hipótesis alternativa: El contraste magnético refleja la variación litológica real entre las rocas sedimentarias de la cuenca deltaica al norte y las rocas metamórficas/volcánicas de la sierra al sur. Medición diferencial: calibrar con mediciones de susceptibilidad magnética in situ o perfiles de pozos profundos.

### Hipótesis 4: La alta puntuación predicha en El Chinero por modelos previos era un artefacto del modelo de distancia a fallas.
- (i) Medición que la apoyaría: El modelo entrenado solo con fallas (RF solo G_est) asigna alta puntuación en El Chinero, mientras que los modelos que excluyen fallas o consideran la magnetometría regional asignan puntuación cercana a cero.
- (ii) Medición que la refutaría: Todos los modelos multivariados asignan consistentemente puntuación elevada (> 0.80) en El Chinero con independencia de qué grupos de covariables se incluyan o excluyan.
- (iii) Evaluación empírica: En la ventana de 3 km de El Chinero, RF solo G_est asigna una puntuación mediana de 0.73756. Por el contrario, el modelo multivariado RF-S2 asigna una puntuación mediana de 0.03326, y el modelo sin cobertura superficial (RF S4 sin G_sup) asigna 0.00387. La Magnetometría en El Chinero (-225 nT) se sitúa en el percentil 0.00% de Cerro Prieto.
- Estado: NO REFUTADA.
- Hipótesis alternativa: El Chinero posee una firma estructural profunda análoga pero con cobertura superficial y litología encajonante distinta. Medición diferencial: prospección magnetotelúrica y gravimétrica de detalle en El Chinero.

---

## HALLAZGOS QUE CONTRADICEN LA HIPÓTESIS INICIAL

1. **Placebos de fallas superan a las fallas reales en pliegues individuales:**
   Contrario a la premisa de que la distancia a las trazas de fallas cartografiadas captura la física estructural del campo, en el Pliegue 1 el placebo desplazado 10 km al oeste (dx = -10 km) supera a las fallas reales (PR-AUC 0.59780 vs 0.59156; ROC-AUC 0.90408 vs 0.81776). En el Pliegue 5, el placebo desplazado 10 km al este (dx = +10 km) supera ampliamente a las fallas reales (PR-AUC 0.89049 vs 0.64238; ROC-AUC 0.95919 vs 0.87249).
2. **Las coordenadas espaciales solas igualan o superan la separación multivariada en el Pliegue 5:**
   En el Pliegue 5, Random Forest con solo (x_utm, y_utm) obtiene un ROC-AUC de 1.00000 y un PR-AUC de 1.00000 frente a la Banda 2 y frente a la Banda 3 Llanura, demostrando que la separación en dicho pliegue puede lograrse sin información geofísica ni de superficie.
3. **El modelo lineal supera al no lineal en bloques espaciales cuando se aportan interacciones:**
   Bajo validación de bloques espaciales en la Fase 3, la Regresión Logística con los términos derivados RTE e ITT superó a Random Forest en ROC-AUC (0.9916 frente a 0.9575) y en PR-AUC (0.9668 frente a 0.9355), contradiciendo la noción generalizada de que los modelos no lineales son inherentemente superiores.
4. **La señal magnética regional está dominada por discontinuidades de digitalización:**
   El 69.45% de la carta San Felipe presenta un gradiente magnético local de 0 nT/30m, lo que revela que la capacidad discriminante aparente de la magnetometría a distancias > 40 km está confundida con el estilo de interpolación cartográfica entre cartas.
5. **Colapso de predicción en El Chinero al integrar geofísica regional:**
   En estudios exploratorios preliminares se atribuía a El Chinero una probabilidad superior a 0.90 por proximidad a fallas. En el modelo principal RF-S2 la puntuación mediana es de apenas 0.03326, y al retirar la cobertura superficial (S4 sin G_sup) la puntuación cae a 0.00387.

---

## ADVERTENCIAS

1. **Prevalencia artificial:** La prevalencia de entrenamiento (~27.6%) es un artefacto de muestreo espacial (24,814 positivos frente a 65,001 negativos). Las salidas del modelo deben interpretarse estrictamente como puntuaciones de similitud y nunca como probabilidades a priori del terreno natural (cuya prevalencia real es de ~0.075%).
2. **Efecto confundido en magnetometría:** Las evaluaciones contra la Banda 3 Llanura en San Felipe sufren de confusión estructural entre la distancia espacial al polígono de entrenamiento y el cambio de carta cartográfica.
3. **Uso de suelo y actividad antrópica:** El 49.15% de los positivos clasificados como húmedos coinciden espacialmente con lagunas de evaporación y salmueras de la planta geotérmica en operación, lo que introduce un sesgo de cobertura superficial antrópica no transferible a campos no desarrollados.
4. **Área de aplicabilidad restringida fuera de la llanura:** Aunque el 93.42% del dominio continental cae dentro del Área de Aplicabilidad, las zonas serranas con pendientes > 15° y cotas > 500 m quedan sistemáticamente fuera de aplicabilidad o presentan incertidumbre entre modelos superior a 0.50.

---

## AFIRMACIONES DESCARTADAS

1. *"La proximidad a fallas activas prueba que el modelo aprende la física del transporte hidrotermal."*
   - Motivo del descarte: Violación de la regla R2 (uso de la palabra prohibida "prueba que") y de la regla R1 por explicación causal no sustentada. Refutada empíricamente en T2 al verificarse que un placebo con desplazamiento de 10 km supera a la falla real en el Pliegue 5 (PR-AUC 0.89049 vs 0.64238).
2. *"La superioridad de Random Forest confirma la presencia de interacciones complejas en los datos geológicos."*
   - Motivo del descarte: Violación de la regla R2 (uso de la palabra prohibida "confirma") y contradicción directa con el Cuadro 2 de validación por bloques espaciales, donde la Regresión Logística superó a Random Forest.
3. *"El Chinero es un prospecto de alta probabilidad debido a su proximidad a fallas de rumbo."*
   - Motivo del descarte: Violación de la regla R11 (uso de los términos prohibidos "prospecto" y "probabilidad") y de la regla R2 por causalidad ("debido a"). La puntuación mediana del modelo multivariado RF-S2 en El Chinero es de 0.03326 y no alta probabilidad.
