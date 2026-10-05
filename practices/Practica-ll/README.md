# Práctica 2: Análisis Exploratorio y Técnicas de Preprocesamiento de Datos (Calidad del Agua en México)

## Descripción del Dataset
El conjunto de datos proviene de la **Red Nacional de Medición de Calidad del Agua (RENAMECA)**, administrada por la **Comisión Nacional del Agua (CONAGUA)**.
- **Nombre del archivo:** `TODOS LOS MONITOREOS.xlsb`
- **Hoja analizada:** `RESULTADOS`
- **Periodo temporal:** 2012 – 2025
- **Dimensiones:** 126,070 registros de monitoreo ambiental en cuencas lóticas (ríos), lénticas (lagos y presas), subterráneas (pozos y acuíferos) y costeras de México.
- **Licencia de uso:** Datos Abiertos del Gobierno de México (Libre uso y distribución).

---

## 5 Técnicas de Preprocesamiento Implementadas

1. **Análisis Exploratorio de Datos (EDA):**
   - Resumen estadístico univariado (Tabla 1) con medidas de tendencia central, dispersión y cuantiles.
   - Histogramas de distribución univariada para $pH$ en campo y Oxígeno Disuelto con umbral ecológico de hipoxia ($2.0\text{ mg/L}$) (Figura 1).
   - Matriz de correlación no paramétrica de Spearman para evaluar colinealidades y agrupaciones de contaminantes (Figura 2).

2. **Técnica 1 — Limpieza de Datos (*Data Cleaning*):**
   - Estandarización y conversión de límites analíticos censurados de laboratorio (`< LD` a $\frac{LD}{2}$ y `> LD` al valor nominal) según directrices de US EPA (2000) y APHA (2017).
   - Imputación condicional de valores faltantes basada en la mediana de cada tipología de cuerpo de agua (Lótico, Léntico, Subterráneo, Costero).
   - Filtrado de valores físicamente imposibles.
   - **Comparación Antes vs. Después:** Gráfico de reducción de datos faltantes ($> 200,000 \to 0$) y preservación de la distribución de DBO (Figura 3).

3. **Técnica 2 — Extracción e Ingeniería de Características (*Feature Engineering*):**
   - **Índice de Biodegradabilidad Orgánica ($IBO = \frac{DBO}{DQO}$):** Discrimina efluentes biodegradables de vertidos tóxicos/industriales recalcitrantes.
   - **Índice Potencial de Eutrofización ($IEUT = N_{tot} \times P_{tot}$):** Modela la disponibilidad simultánea de nutrientes según la estequiometría limnológica.
   - **Índice de Carga Orgánica Fecal ($ICOF = DBO \times \log_{10}(COLI\_FEC + 1)$):** Cuantifica el riesgo sanitario conjunto.
   - **Comparación Antes vs. Después:** Gráfico de dispersión cruda ($DBO$ vs $DQO$) frente al espacio de características extraídas ($IBO$ vs $IEUT$) con clara separación de zonas eutrofizadas (Figura 4).

4. **Técnica 3 — Aumento de Datos (*Data Augmentation con SMOTE*):**
   - Tratamiento del desbalance severo en la condición crítica de contaminación / hipoxia ($OD < 2.0\text{ mg/L}$ o $DBO > 30\text{ mg/L}$, ratio $10:1$).
   - Generación de observaciones sintéticas realistas con **SMOTE** ($k=5$).
   - **Comparación Antes vs. Después:** Gráfico de barras de balance de clases ($2,727:273 \to 2,727:2,727$, total $5,454$ muestras) y dispersión espacial de muestras sintéticas en el clúster crítico (Figura 5).

5. **Técnica 4 — Reducción de Dimensionalidad (*PCA*):**
   - Estandarización con `StandardScaler` y proyección ortogonal de 11 variables a 2 Componentes Principales ($PC_1$ y $PC_2$).
   - **Comparación Antes vs. Después:** Scree Plot de varianza acumulada ($42.1\%$ en 2 componentes) y mapa de dispersión 2D estructurado por tipo de cuerpo de agua (Figura 6).

6. **Técnica 5 — Selección de Características (*Feature Selection*):**
   - Diagnóstico y remoción de redundancia colineal extrema ($SDT$ vs $CONDUC\_CAMPO$, $r_s = 0.99$).
   - Evaluación de dependencia no lineal con **Información Mutua** (`mutual_info_classif`) para retener las 5 variables más informativas ($DBO$, $OD$, $DQO$, $P$ y $N$).
   - **Comparación Antes vs. Después:** Matriz de correlación previa vs. ranking de importancia de características (Figura 7).

---

## Requisitos de Ejecución
- Python 3.10 o superior
- Jupyter Notebook, JupyterLab o Visual Studio Code

### Librerías Requeridas
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `imbalanced-learn`
- `python-calamine`

### Instalación de Dependencias
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn python-calamine
```

---

## Instrucciones de Uso
1. Colocar el archivo `TODOS LOS MONITOREOS.xlsb` y `practica-2.ipynb` en la carpeta `C:\Users\ruben\OneDrive\Desktop\Practica-ll\`.
2. Abrir `practica-2.ipynb` en Jupyter Notebook o VS Code.
3. Ejecutar las celdas de forma secuencial de inicio a fin.
4. Todas las figuras comparativas, tablas descriptivas e interpretaciones se despliegan directamente en el cuaderno.

---

## Referencias Bibliográficas (Normas APA 7.ª Edición)
- American Public Health Association, American Water Works Association, & Water Environment Federation. (2017). *Standard methods for the examination of water and wastewater* (23rd ed.). APHA-AWWA-WEF.
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, *16*, 321–357. https://doi.org/10.1613/jair.953
- Comisión Nacional del Agua. (2025a). *Resultados de la Red Nacional de Medición de Calidad del Agua (RENAMECA)*. Gobierno de México.
- Comisión Nacional del Agua. (2025b). *Criterios ecológicos de calidad del agua: Parámetros e indicadores nacionales*. Gobierno de México.
- Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. *Journal of Machine Learning Research*, *3*, 1157–1182.
- Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: A review and recent developments. *Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences*, *374*(2065), Artículo 20150202. https://doi.org/10.1098/rsta.2015.0202
- Metcalf & Eddy, Inc., Burton, F. L., Stensel, H. D., & Tsuchihashi, R. (2014). *Wastewater engineering: Treatment and resource recovery* (5th ed.). McGraw-Hill Education.
- United States Environmental Protection Agency. (2000). *Guidance for data quality assessment: Practical methods for data analysis* (EPA QA/G-9, EPA/600/R-96/084). U.S. EPA.
- Wetzel, R. G. (2001). *Limnology: Lake and river ecosystems* (3rd ed.). Academic Press.
