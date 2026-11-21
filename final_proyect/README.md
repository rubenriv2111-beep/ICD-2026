# PROSPECCIÓN GEOTÉRMICA REGIONAL MEDIANTE APRENDIZAJE AUTOMÁTICO Y REDES NEURONALES PROFUNDAS EN EL RIFT DEL GOLFO DE CALIFORNIA (BAJA CALIFORNIA, MÉXICO)

**Autor:** Rubén Rivera García  
**Programa:** Maestría en Ciencias en Ciencias de la Computación  
**Institución:** Centro de Investigación Científica y de Educación Superior de Ensenada (CICESE)  
**Asignatura:** Ciencia de Datos (Semestre 2026-3)  
**Profesor:** Dr. Jesús Guillermo Falcón-Cardona  
**Fecha de Entrega:** Noviembre de 2026  

---

## 1. Descripción General del Proyecto
Este proyecto aborda la prospección regional y delimitación espacial de zonas con potencial geotérmico en la provincia geológica del Valle de Mexicali y la extensión septentrional del Rift del Golfo de California (Baja California, México). A través de un enfoque riguroso de Ciencia de Datos y Aprendizaje Profundo (*Deep Learning*), se modela la probabilidad de presencia de recursos hidrotermales a partir de variables térmicas satelitales, geomorfológicas y geofísicas.

El proyecto está diseñado bajo estrictos estándares metodológicos de posgrado:
* **Exploración Física Especializada (EDA):** Caracterización estadística, análisis de correlación lineal y no lineal, y dispersión bivariada que demuestra el acoplamiento hidrotermal y el control estructural del sistema de fallas activas.
* **Flujo Libre de Fuga de Datos (*No Data Leakage*):** Partición estratificada *Train/Val/Test* (70% / 15% / 15%) fijada **estrictamente antes** de cualquier ajuste. El escalador (`StandardScaler`) y el balanceo sintético (`SMOTE`) se ajustan **exclusivamente sobre el conjunto de entrenamiento**, evaluando la validación y prueba bajo su prevalencia natural no alterada.
* **Comparativa Justa entre Líneas Base y Redes Profundas:** Comparación sistemática de un Clasificador Dummy, Regresión Logística, Random Forest y **Redes Neuronales Perceptrón Multicapa (MLP) evaluadas bajo 5 funciones de activación (ReLU, LeakyReLU $\alpha=0.25$, GELU, Swish y Tanh)** bajo idéntica arquitectura, optimizador y presupuesto computacional.
* **Evaluación Multidimensional y Análisis de Errores:** Reporte de métricas de ranking independientes del umbral (AUC-ROC, PR-AUC / Average Precision), calibración probabilística (Brier Score), curvas de fiabilidad y diagnóstico geográfico de falsos positivos en zonas desérticas (El Chinero).

---

## 2. Preguntas de Investigación Planteadas y Respondidas
1. **$PI_1$ (Relevancia y Acoplamiento Físico):** ¿Qué covariables geofísicas muestran el mayor poder discriminativo y qué correlaciones físicas describen el acoplamiento hidrotermal en la cuenca?
   * *Hallazgo:* La proximidad inmediata a fallas tectónicas activas ($\text{Prox\_Fallas} < 1,500\text{ m}$) y el ratio termo-estructural son los factores de mayor peso discriminativo, demostrando que la temperatura satelital ($LST$) requiere forzosamente fracturamiento tectónico para indicar potencial geotérmico real.
2. **$PI_2$ (Integridad Metodológica y Prevención de Fuga):** ¿Cómo impacta la partición estricta previa al balanceo sintético en la honestidad de las métricas predictivas?
   * *Hallazgo:* Al restringir `SMOTE` exclusivamente a Train y evaluar Test con su desbalance natural, las métricas reflejan la capacidad de generalización real ($AUC \approx 0.965$, $PR-AUC \approx 0.912$), evitando la sobreestimación optimista artificial ($AUC > 0.999$) causada por contaminación de muestras sintéticas.
3. **$PI_3$ (Efecto de la Función de Activación en Redes Profundas):** ¿Superan las activaciones modernas (LeakyReLU, GELU, Swish) a las formulaciones clásicas (ReLU, Tanh)?
   * *Hallazgo:* Las activaciones modernas continuas superan con claridad a Tanh ($AUC \approx 0.958$). Entre LeakyReLU ($\alpha=0.25$), GELU, Swish y ReLU, las diferencias en Test son marginales ($\approx 0.003$ AUC) y caen dentro del intervalo de variabilidad estocástica de optimización ($\pm 0.004$), destacando LeakyReLU ($\alpha=0.25$) por su convergencia más rápida y menor Brier Score ($0.052$).

---

## 3. Origen y Fuentes de los Datos

| Variable / Capa | Fuente Institucional | Resolución / Escala | Descripción Física |
| :--- | :--- | :--- | :--- |
| **LST (Land Surface Temperature)** | USGS / Landsat 9 TIRS-2 (Banda 10) | 30 m | Temperatura de emisión superficial (°C) adquirida el 25/09/2026 (18:24 UTC / 11:24 PDT). |
| **NDVI** | USGS / Landsat 9 OLI-2 (B4 y B5) | 30 m | Índice normalizado de vegetación para caracterización biológica y discriminación marina. |
| **Elevación y Pendiente** | INEGI (Continuo de Elevaciones CEM 3.0) | 30 m | Modelo digital de terreno (m.s.n.m.) y gradientes de pendiente topográfica (grados). |
| **Proximidad a Fallas** | Servicio Geológico Mexicano (SGM) | Vectorial / 30 m | Distancia euclidiana tridimensional (m) al sistema de fallas activas de la falla San Andrés - Cerro Prieto. |
| **Magnetometría Residual** | Servicio Geológico Mexicano (SGM) | Vuelo aeromagnético | Campo magnético residual de intensidad total (nT), indicador de susceptibilidad y basamento. |
| **Target Geotérmico (Clase 1)** | CFE / Diario Oficial de la Federación | Polígono oficial | Delimitación oficial del Título de Concesión del Campo Geotérmico Cerro Prieto (CFE, 2015). |

### Nota sobre el tamaño de los datos:
La escena completa en formato ráster virtual (`.vrt`) abarca más de **74 millones de celdas** y supera los **2.5 GB**, por lo que no es posible alojarla en un repositorio estándar de GitHub (<50 MB). Para garantizar la **reproducibilidad inmediata**, el repositorio incluye en `data/datos_geotermia_baja_california.csv` una muestra curada georreferenciada de **89,814 registros** (~7.4 MB), conteniendo todos los positivos continentales de Cerro Prieto y una muestra representativa del fondo regional fuera de la zona de amortiguamiento. Aquellos usuarios que deseen regenerar la extracción desde el VRT original pueden ejecutar el script `preparar_datos_proyecto.py`.

---

## 4. Estructura de Archivos del Repositorio

```text
Notebook del proyecto/
│
├── Proyecto_Ciencia_De_Datos_Geotermia.ipynb  # Notebook principal EJECUTADO con todas sus salidas y gráficos visibles
├── README.md                                  # Documentación técnica exhaustiva del proyecto (este archivo)
├── requirements.txt                           # Archivo de dependencias con versiones exactas
│
├── data/                                      # Carpeta de datos del proyecto (<50 MB)
│   └── datos_geotermia_baja_california.csv    # Conjunto de datos curado y georreferenciado (89,814 filas, 7.39 MB)
│
└── figures/                                   # Figuras científicas en alta resolución exportadas por el notebook
    ├── grafico_1_distribuciones_kde.png
    ├── grafico_2_matriz_correlacion.png
    ├── grafico_3_dispersion_bivariada.png
    ├── grafico_4_boxplots_estratificados.png
    ├── grafico_5_curvas_aprendizaje_activaciones.png
    ├── grafico_6_curvas_roc_comparativas.png
    ├── grafico_7_curvas_pr_comparativas.png
    ├── grafico_8_matrices_confusion.png
    ├── grafico_9_calibracion_probabilidades.png
    └── grafico_10_mapa_susceptibilidad_espacial.png
```

---

## 5. Requisitos y Entorno de Ejecución

El proyecto fue desarrollado y probado en **Python 3.11.9 (64-bit)** bajo Windows 11 con aceleración por GPU (NVIDIA GeForce RTX 4060).

### Instalación de Dependencias:
```bash
pip install -r requirements.txt
```

### Contenido de `requirements.txt`:
* `numpy>=1.26.0`
* `pandas>=2.2.0`
* `scikit-learn>=1.4.0`
* `imbalanced-learn>=0.12.0`
* `torch>=2.2.0`
* `matplotlib>=3.8.0`
* `seaborn>=0.13.0`
* `rasterio>=1.3.9`
* `pyproj>=3.6.1`
* `scipy>=1.12.0`

---

## 6. Instrucciones de Reproducción

1. **Clonar el repositorio:**
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd "Proyecto_Ciencia_De_Datos/Notebook del proyecto"
   ```
2. **Abrir y ejecutar el notebook:**
   Inicia Jupyter Lab o Jupyter Notebook:
   ```bash
   jupyter notebook Proyecto_Ciencia_De_Datos_Geotermia.ipynb
   ```
   El notebook utiliza rutas relativas (`os.path.join("data", ...)`), por lo que se ejecuta sin necesidad de modificar directorios locales. Todas las celdas cuentan con contadores consecutivos de ejecución (`[1]`, `[2]`, ..., `[17]`) y las salidas gráficas están completamente integradas.

---

## 7. Declaración Ética de Uso de IA Generativa
En estricto apego a las normas de integridad académica del **CICESE** y la rúbrica de la asignatura de **Ciencia de Datos (2026-3)**:
* Se utilizaron herramientas de inteligencia artificial generativa (*Antigravity / Google Gemini*) exclusivamente para asistencia técnica en la redacción de texto explicativo en Markdown, verificación sintáctica de código bajo el estándar PEP 8 y estilización visual de figuras científicas.
* El diseño conceptual, la delimitación física del área de estudio, la selección de variables geofísicas, la formulación de la metodología sin fuga de información (*No Data Leakage*) y la interpretación de los resultados son de la **autoría y responsabilidad absoluta del alumno: Rubén Rivera García**.
