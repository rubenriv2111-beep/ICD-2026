# Evaluación Diagnóstica de la Separabilidad Espacial en el Rift de Baja California: Alcances y Limitaciones del Aprendizaje Automático Regional

**Autor:** Rubén Rivera García  
**Programa:** Maestría en Ciencias en Ciencias de la Computación  
**Institución:** Centro de Investigación Científica y de Educación Superior de Ensenada (CICESE)  
**Asignatura:** Ciencia de Datos (Semestre 2026-3)  
**Profesor:** Dr. Irvin Hussein López Nava  
**Fecha:** Noviembre de 2026  

---

## 1. Descripción General del Proyecto

Este proyecto presenta una **evaluación diagnóstica, metodológica y reproducible** sobre la capacidad real del aprendizaje automático supervisado para discriminar el campo geotérmico activo de **Cerro Prieto (Baja California, México)** a partir de covariables físicas regionales de acceso público (temperatura superficial satelital LST, vegetación NDVI, cota topográfica, pendiente, magnetometría aerotransportada y distancia euclidiana a fallas tectónicas).

A diferencia de aproximaciones tradicionales de prospección comercial que asumen generalización automática, este estudio formula tres preguntas de investigación formales ($PI_1, PI_2, PI_3$) para responder cuantitativamente:
1. **$PI_1$ (Separabilidad y Dificultad del Fondo):** ¿Qué tan separable es el polígono frente a fondos concéntricos de dificultad creciente (B1: 0.5–2 km, B2: 2–5 km, B3: >5 km) y cuánto de la separabilidad es acoplamiento geofísico frente a posición espacial?
2. **$PI_2$ (Protocolos de Validación):** ¿Cuál es el impacto cuantitativo de sustituir una partición aleatoria convencional (Protocolo A, 70/15/15) por una validación cruzada en 5 bloques espaciales con zona de exclusión perimetral de 500 m (Protocolo B)?
3. **$PI_3$ (Complejidad vs. Calibración):** ¿Superan clasificadores no lineales complejos (Random Forest y Redes Neuronales Perceptrón Multicapa con BatchNorm y LeakyReLU) a una línea base lineal regularizada con descriptores de interacción ($R_{TE}, I_{TT}$) en discriminabilidad espacial (PR-AUC), calibración probabilística (Brier score) y estabilidad de umbrales operativos de decisión?

---

## 2. Origen y Fuentes Geoespaciales de los Datos

El banco analítico integra **89,815 observaciones** discretizadas en celdas regulares de 30 m × 30 m bajo el sistema **UTM Zona 11N (Datum WGS84, EPSG:32611)**:
* **Elevación y Pendiente (CEM 3.0 INEGI):** Continuo de Elevaciones Mexicano a 30 m de resolución (m.s.n.m. y grados de inclinación).
* **LST (Landsat 9 TIRS-2, Banda 10):** Temperatura de superficie terrestre diurna corregida atmosféricamente (°C), captada en condiciones despejadas de verano.
* **NDVI (Landsat 9 OLI-2, Bandas 4 y 5):** Índice de vegetación de diferencia normalizada (vigor fotosintético y humedad).
* **Proximidad a Fallas (SGM 1:250,000):** Distancia euclidiana en metros a trazas tectónicas cartografiadas del sistema de fallas Cerro Prieto - Imperial. Trazas reproyectadas rigurosamente de NAD27 a WGS84 mediante transformación geodésica estándar (corrigiendo un desfase medio de 197.8 m).
* **Magnetometría Residual (SGM):** Anomalía magnética total (nT) obtenida del mosaico continuo regional de las cartas Mexicali (H11-12) y San Felipe (H11-3).
* **Descriptores Heurísticos de Interacción Física:**
  $$R_{TE} = \frac{\text{LST}}{\ln(d_{\text{fallas}} + 10.0)},\qquad I_{TT} = \frac{\text{LST}}{\text{Elevación} + 50.0}$$
* **Etiquetas de Clase:**
  * **Clase Positiva ($y = 1$):** Censo íntegro de 24,814 celdas del polígono concesionado y en operación activa de Cerro Prieto (22.33 km²).
  * **Clase Negativa ($y = 0$):** 65,001 celdas de fondo regional distribuidas equitativamente (21,667 por banda) en estratos concéntricos de dificultad: B1 (0.5–2 km), B2 (2–5 km) y B3 (>5 km). Prevalencia muestral del 27.63% (frente a una prevalencia territorial real de $\approx 0.075\%$).

---

## 3. Requisitos del Sistema e Instalación

Entorno probado y verificado en **Python 3.11**:
```bash
# Clonar el repositorio
git clone <URL_DEL_REPOSITORIO>
cd Proyecto_Ciencia_De_Datos

# Crear y activar entorno virtual
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/macOS:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### Contenido de `requirements.txt`:
* `numpy>=1.26.0`
* `pandas>=2.2.0`
* `scikit-learn>=1.4.0`
* `torch>=2.2.0`
* `matplotlib>=3.8.0`
* `seaborn>=0.13.0`
* `rasterio>=1.3.9`
* `pyproj>=3.6.1`
* `scipy>=1.12.0`
* `shapely>=2.0.0`

---

## 4. Estructura del Repositorio y Arquitectura Modular

```
final_proyect/
│
├── Proyecto_Ciencia_De_Datos_Geotermia.ipynb  # Notebook interactivo principal (Protocolo A, EDA, Modelado, Gráficos 1 al 9)
├── README.md                                  # Guía técnica maestra de reproducibilidad
│
└── documentacion/                             # Repositorio documental, scripts, figuras, datos y cartografía
    ├── Presentacion_FINAL_ICD_RubenRiveraGarcia.pptx # Diapositivas de exposición editables (8 láminas de contenido)
    ├── Presentacion_FINAL_ICD_RubenRiveraGarcia.pdf  # Diapositivas de exposición compiladas en alta definición
    ├── Proyecto_ICD.pdf                       # Manuscrito formal de investigación compilado (5 páginas)
    ├── Proyecto_ICD.tex                       # Manuscrito formal en LaTeX (con figuras y bibliografía)
    ├── reporte_geotermia_final.tex            # Versión preliminar en LaTeX
    ├── requirements.txt                       # Dependencias estrictas del entorno
    ├── .gitignore                             # Exclusión de cachés (__pycache__, checkpoints, temporales)
    ├── 01_entrenamiento_protocolo_a.py        # Protocolo A (70/15/15) y comparación de 5 activaciones
    ├── 02_validacion_espacial_y_diagnosticos.py # Protocolo B (5 bloques, buffer 500 m) y pruebas de control
    ├── 03_mapa_regional_aoa.py                # Inferencia regional 120 m (1.2M px) y delimitación AOA
    ├── construir_dataset.py                   # Pipeline ETL territorial desde rásteres satelitales
    ├── generar_mapa_profesional.py            # Generador cartográfico maestro (300 DPI) con hillshade 3D
    │
    ├── data/
    │   └── datos_geotermia_baja_california.csv # Banco curado de 89,815 observaciones a 30 m (UTM Zona 11N)
    │
    ├── figures/
    │   ├── mapa_5_sectores_profesional.png    # Díptico cartográfico con los 5 bloques y franja buffer (300 DPI)
    │   ├── grafico_caida_pliegue_5.png        # Diagnóstico del colapso de generalización en el Pliegue 5
    │   ├── mapa_regional_profesional.png      # Mapa regional de 4 paneles con AOA y desmitificación de El Chinero
    │   ├── mapa_5_capas_geotermicas_blanco.png # Mosaico temático de las 5 covariables físicas
    │   └── [figuras diagnósticas de curvas ROC, PR, calibración y matrices]
    │
    ├── results/
    │   ├── dataset_final.parquet              # Banco estructurado con covariables físicas y roles
    │   ├── pliegues_roles.parquet             # Partición formal de los 5 pliegues espaciales
    │   ├── metricas_protocolo_A.csv           # Resultados bajo partición aleatoria convencional
    │   ├── metricas_protocolo_B_por_pliegue.csv # Resultados pliegue por pliegue (Protocolo B)
    │   ├── rf_coord_solas_pliegues4_5.csv     # Control espacial: solo coordenadas (x, y)
    │   ├── placebo_fallas.csv                 # Control geológico: fallas desplazadas ±10 km
    │   ├── prueba_humedo_seco.csv             # Control superficial: MNDWI / NDVI
    │   └── chinero_resumen.csv                # Estadísticas y colapso de probabilidad en El Chinero
    │
    └── mapas_qgis/
        ├── dem_120m.tif                       # Ráster continuo de elevación regional a 120 m
        ├── aoa_mask.tif / aoa_mask.qml        # Máscara y simbología QGIS del Área de Aplicabilidad (AOA)
        └── prob_rf.tif / prob_mlp.tif / prob_logreg.tif # Rásteres GeoTIFF de inferencia con estilos QGIS
```

---

## 5. Instrucciones de Ejecución y Reproducibilidad

### Paso 1: Ejecución del Notebook Principal
El notebook contiene todo el desarrollo interactivo con sus contadores de ejecución consecutivos (1 a 18) y salidas completas visibles:
```bash
jupyter notebook Proyecto_Ciencia_De_Datos_Geotermia.ipynb
```
* **Flujo cubierto:** Carga de datos, análisis exploratorio bivariado, preprocesamiento confinado estrictamente a Train (`StandardScaler`), entrenamiento sin SMOTE de líneas base y 5 activaciones profundas (ReLU, LeakyReLU, GELU, Swish, Tanh), curvas de aprendizaje, calibración probabilística, análisis de residuos, variabilidad con 5 semillas y carga de la cartografía regional.

### Paso 2: Reproducción de la Validación Espacial y Pruebas Dirigidas (Cuadros 1 y 2)
```bash
python 02_validacion_espacial_y_diagnosticos.py
```
* **Resultados generados:**
  * Evaluación en los 5 pliegues espaciales con buffer de 500 m.
  * Comprobación de que la Regresión Logística ($PR\text{-}AUC = 0.9527 \pm 0.0398$, $\text{Brier} = 0.04862$) iguala a Random Forest ($0.9555 \pm 0.0686$, $\text{Brier} = 0.06492$) y al MLP Ensamble ($0.9624 \pm 0.0702$, $\text{Brier} = 0.06652$) con menor varianza.
  * Pruebas de coordenadas solas ($PR\text{-}AUC = 0.8425$ en Pliegue 5).
  * Placebos de fallas superando a fallas reales en los Pliegues 1 y 5.
  * Diagnóstico del 49.15% de celdas con humedad antrópica y retención de $PR\text{-}AUC = 0.9273$ en celdas secas.
  * Detección del 69.45% de gradientes nulos en la hoja San Felipe.

### Paso 3: Inferencia Cartográfica Regional y Área de Aplicabilidad (Figura 3 IEEE)
```bash
python 03_mapa_regional_aoa.py
```
* **Resultados generados:**
  * Procesa los 1.2 millones de celdas continentales a resolución de 120 m.
  * Infiere las superficies continuas $P(x) \in [0, 1]$ para Regresión Logística, Random Forest y MLP Ensamble.
  * Calcula el Índice de Disimilitud (DI $\le 2.5057$) delimitando el 62.82% del territorio dentro del AOA.
  * Genera `figures/grafico_10_mapa_susceptibilidad_espacial.png` en tira horizontal de 4 paneles a **180 mm exactos a 300 DPI**.

---

## 6. Principales Hallazgos Científicos

1. **Sesgo por Autocorrelación Espacial:** La partición aleatoria ingenua arroja métricas sobreestimadas ($PR\text{-}AUC > 0.999$, Brier $< 0.004$). Bajo bloques espaciales con zona de exclusión de 500 m, el rendimiento cae a $0.9527 - 0.9624$, emergiendo fuerte varianza inter-pliegues.
2. **Paridad de Modelos Lineales:** La Regresión Logística con los términos de ingeniería ($R_{TE}, I_{TT}$) iguala a Random Forest y a las redes neuronales profundas con mejor calibración probabilística y menor dispersión.
3. **Inestabilidad Extrema del Umbral en Redes:** El umbral óptimo de Youden ($J$) en el MLP varía desde **0.0165** (Pliegue 2) hasta **0.9022** (Pliegue 5), abarcando un rango de **0.8857**, lo que descarta el empleo de umbrales rígidos para prospección regional.
4. **Separabilidad por Confinamiento Espacial:** Un clasificador alimentado únicamente con coordenadas UTM ($x, y$) iguala en el Pliegue 5 a los modelos con 6 variables geofísicas ($PR\text{-}AUC = 0.8425$). Asimismo, placebos sintéticos de fallas desplazadas superan a las fallas verdaderas, demostrando que la proximidad estructural opera como proxy de posición y no como acoplamiento hidrotermal físico.
5. **Desmitificación Concluyente de El Chinero:** En las exploraciones preliminares basadas en fallas univariadas, El Chinero fue catalogado con probabilidades $>0.90$. Sin embargo, la evaluación multivariada integral revela:
   * Puntuación mediana multivariada de apenas **0.03326** (**0.00387** sin variables superficiales).
   * Anomalía aeromagnética de **-225 nT**, ubicada en el percentil **0.00%** de Cerro Prieto (+110 a +350 nT).
   * Localizado fuera del Área de Aplicabilidad (AOA), confirmando que su interés previo se debía a un artefacto cartográfico por la costura del paralelo 32° N.

---

## 7. Declaración sobre el Uso de Inteligencia Artificial Generativa

Se declara el uso de herramientas de inteligencia artificial generativa (*Google DeepMind / Gemini*) con el propósito exclusivo de asistir en la depuración de código computacional en lenguaje Python, la optimización de scripts de análisis numérico y la verificación de estilo tipográfico en código fuente \LaTeX. El planteamiento analítico, procesamiento, interpretación geofísica y conclusiones científicas fueron desarrollados íntegramente por el autor, quien asume la responsabilidad íntegra del manuscrito.
