# Proyecto: Análisis Exploratorio y Preprocesamiento de Datos (Breast Cancer Wisconsin)

## Descripción del Dataset
El archivo de datos contiene registros citológicos obtenidos mediante **Biopsia por Aspiración con Aguja Fina (BAAF / FNA)** de nódulos mamarios, recolectados por el Dr. William H. Wolberg en los Hospitales de la Universidad de Wisconsin, Madison.
- **Nombre del archivo:** `breast-cancer-wisconsin.data`
- **Descripción de atributos:** `breast-cancer-wisconsin.names`
- **Contenido:** 699 registros y 9 variables citológicas cuantificadas en escala de 1 a 10 más el identificador de muestra y la etiqueta diagnóstica (`2 = Benigno`, `4 = Maligno`).
- **Valores ausentes:** 16 registros en el atributo `Bare_Nuclei` indicados con el símbolo `'?'`.

## Técnicas de Preprocesamiento Implementadas
1. **Análisis Exploratorio de Datos (EDA):** Estadísticos descriptivos univariados, histogramas estratificados por diagnóstico y matriz de correlación de Spearman.
2. **Limpieza de Datos:** Identificación y tratamiento de los 16 valores nulos en `Bare_Nuclei` mediante imputación condicional por la mediana según el diagnóstico patológico.
3. **Extracción e Ingeniería de Características (Feature Engineering):** Construcción del *Índice de Atipia Citológica (IAC)* y la *Puntuación de Agresividad y Proliferación (PAP)*.
4. **Aumento de Datos (Data Augmentation):** Balanceo de la clase minoritaria maligna mediante **SMOTE** (*Synthetic Minority Over-sampling Technique*).
5. **Reducción de Dimensionalidad:** Aplicación de **Análisis de Componentes Principales (PCA)** sobre variables estandarizadas con visualización 2D de clusters y análisis de varianza explicada acumulada (Scree plot).
6. **Comparativa Gráfica (Antes vs. Después):** Gráficos dedicados para evaluar el impacto de cada una de las técnicas sobre los datos.

## Requisitos de Ejecución
- Python 3.10 o superior
- Jupyter Notebook o Visual Studio Code

### Librerías Requeridas
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn

### Instalación de Dependencias
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

## Instrucciones de Uso
1. Colocar los archivos `breast-cancer-wisconsin.data`, `breast-cancer-wisconsin.names` y `notebook-preprocesamiento.ipynb` en la misma carpeta.
2. Abrir el archivo `notebook-preprocesamiento.ipynb` en Jupyter Notebook, JupyterLab o VS Code.
3. Ejecutar las celdas de forma secuencial de principio a fin.
4. Todas las figuras, tablas comparativas y análisis interpretativos se visualizan directamente en el documento.

## Referencias
- Wolberg, W. H., & Mangasarian, O. L. (1990). *Multisurface method of pattern separation for medical diagnosis applied to breast cytology.* PNAS, 87(23), 9193–9196.
- Chawla, N. V. et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR, 16, 321–357.
