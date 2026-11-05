# Práctica 3: Predicción de Deserción Estudiantil y Éxito Académico

**Introducción a la Ciencia de Datos 2026**  
Posgrado en Ciencias de la Computación — Centro de Investigación Científica y de Educación Superior de Ensenada (CICESE)  
**Autor:** Ruben Rivera Garcia  

---

## 1. Descripción del Problema y del Dataset

El abandono escolar universitario constituye uno de los mayores desafíos institucionales, pedagógicos y socioeconómicos en los sistemas de educación superior a nivel global. Detectar oportunamente a los estudiantes en situación de vulnerabilidad académica o económica permite desplegar **Sistemas de Alerta Temprana (SAT)**, focalizar programas de tutoría personalizada y asignar apoyos financieros antes de que se produzca la desvinculación formal definitiva.

El conjunto de datos utilizado proviene del repositorio **UCI Machine Learning Repository** (Dataset ID: 697), titulado *Predict Students' Dropout and Academic Success*, donado y publicado por Realinho, Machado, Baptista y Martins (2021, 2022) a partir de registros académicos del **Instituto Politécnico de Portalegre (Portugal)**.

### Características del Conjunto de Datos
- **Instancias totales originales:** 4,424 estudiantes de 17 carreras de licenciatura matriculados entre los ciclos 2008/2009 y 2018/2019.
- **Partición oficial de la práctica:**
  - **Conjunto de entrenamiento (`desercion-entrenamiento.csv`):** 3,539 registros con 38 columnas (30 atributos disponibles al cierre del primer semestre, 6 atributos correspondientes al segundo semestre, identificador único `id` y variable objetivo `Target`).
  - **Conjunto de prueba (`desercion-prueba.csv`):** 885 registros con 31 columnas (los 30 atributos del primer semestre y el identificador `id`). No contiene información del segundo semestre ni la variable objetivo.
- **Variable objetivo (`Target`):** Tres clases cualitativas mutuamente excluyentes:
  1. `Dropout`: Abandono definitivo prematuro de la carrera (1,137 estudiantes en entrenamiento, 32.13%).
  2. `Enrolled`: Estudiante rezagado que continúa matriculado sin haber egresado en el tiempo estándar (635 estudiantes en entrenamiento, 17.94%).
  3. `Graduate`: Egresado que completó satisfactoriamente sus créditos (1,767 estudiantes en entrenamiento, 49.93%).
- **Horizonte temporal de decisión:** Predicción temprana al cierre del **primer semestre escolar**, empleando únicamente los 30 atributos observables en ese instante (demográficos, socioeconómicos, antecedentes escolares, desempeño del primer semestre e indicadores macroeconómicos). Las variables del segundo semestre quedan estrictamente excluidas del modelo predictivo principal.

---

## 2. Los Cinco Clasificadores Implementados

Siguiendo el programa de la unidad, se optimizan y comparan los cinco modelos de clasificación supervisada sin recurrir a ensambles ni redes neuronales:

1. **Árbol de Decisión (`DecisionTreeClassifier`):**
   - *Preprocesamiento:* Invariante al escalamiento numérico. Evaluación de codificación One-Hot para variables nominales con agrupamiento de categorías poco frecuentes (`min_frequency`) frente a codificación ordinal.
   - *Hiperparámetros optimizados:* Criterio de división (`gini`, `entropy`, `log_loss`), profundidad máxima (`max_depth`), tamaño mínimo de hoja (`min_samples_leaf`), poda por complejidad de costo (`ccp_alpha`) y ponderación de clases (`class_weight`).

2. **Naive Bayes:**
   - *Preprocesamiento y Representación:* Evaluación de cinco variantes según la naturaleza estadística de los atributos: Gaussiana (con transformación de potencia Yeo-Johnson), Categórica (con discretización en cuantiles mediante `KBinsDiscretizer` y códigos no negativos), Multinomial y Complementaria (con escalamiento no negativo en $[0, 1]$ mediante `MinMaxScaler`), y Bernoulli (con binarización).
   - *Hiperparámetros optimizados:* Parámetro de suavizado laplaciano (`alpha` o `var_smoothing`), número de intervalos de discretización (`n_bins`) y estimación de probabilidades *a priori*.

3. **k Vecinos Más Cercanos (`KNeighborsClassifier`):**
   - *Preprocesamiento:* Extremada sensibilidad a las distancias métricas. Estandarización de variables continuas y binarias con `StandardScaler` o `MinMaxScaler`, codificación One-Hot de nominales y ponderación de pesos entre bloques transformados (`transformer_weights`).
   - *Hiperparámetros optimizados:* Número de vecinos (`n_neighbors`), función de ponderación de votos (`uniform`, `distance`) y métrica de distancia (`euclidean`, `manhattan`, `cosine`).

4. **Máquinas de Soporte Vectorial (`SVC`):**
   - *Preprocesamiento:* Sensibilidad a productos internos y distancias en el espacio de características. Estandarización obligatoria con `StandardScaler` en continuas/binarias y One-Hot en nominales.
   - *Hiperparámetros optimizados:* Tipo de kernel (`rbf`, `linear`, `poly`, `sigmoid`), parámetro de regularización de margen (`C`), parámetro de curvatura del kernel (`gamma`), grado polinomial (`degree`) y ponderación de clases (`class_weight`).

5. **Regresión Logística Multinomial (`LogisticRegression`):**
   - *Preprocesamiento:* Algoritmos iterativos de optimización (L-BFGS, SAGA) y regularización equitativa de coeficientes. Estandarización obligatoria con `StandardScaler` y One-Hot en nominales.
   - *Hiperparámetros optimizados:* Tipo de penalización (`ninguna`, `l1`, `l2`, `elasticnet`), inverso de regularización (`C`), proporción de mezcla ElasticNet (`l1_ratio`), solucionador compatible (`solver`) y ponderación de clases (`class_weight`).

---

## 3. Protocolo Metodológico y Validación Cruzada

- **Métrica rectora:** $F_1\text{-macro}$. Debido al desbalance de clases (la clase mayoritaria `Graduate` representa casi el 50% de los datos), el *Accuracy* es una métrica engañosa; un clasificador trivial que prediga siempre `Graduate` alcanzaría un 50% de exactitud pero un $F_1\text{-macro}$ de apenas 0.222.
- **Validación cruzada idéntica:** Se utiliza `StratifiedKFold` de 5 pliegues con semilla fija (`SEMILLA = 42`). Todos los clasificadores son evaluados exactamente sobre los mismos pliegues para garantizar comparabilidad estadística absoluta.
- **Prevención de fuga de información (*Data Leakage*):** Todas las transformaciones de preprocesamiento (codificación One-Hot, imputación y estandarización) se encapsulan estrictamente dentro de objetos `Pipeline` y `ColumnTransformer`, ajustándose únicamente con las particiones de entrenamiento de cada pliegue.
- **Búsqueda de hiperparámetros:** Mínimo de 100 configuraciones evaluadas por clasificador mediante búsqueda aleatoria estratificada (`ParameterSampler`), documentando curvas de convergencia y mesetas de optimización.
- **Experimento complementario:** Evaluación en el conjunto de entrenamiento de la ganancia empírica al incorporar los 6 atributos del segundo semestre frente a los costos institucionales de posponer la intervención.

---

## 4. Estructura de Archivos del Proyecto

```text
Practica -lll/
├── desercion-entrenamiento.csv   # Conjunto de entrenamiento (3,539 registros x 38 columnas)
├── desercion-prueba.csv          # Conjunto de prueba oficial (885 registros x 31 columnas)
├── practica-3.ipynb              # Cuaderno Jupyter con el desarrollo íntegro de la práctica
├── predicciones.csv              # Archivo de predicciones finales para evaluación (885 registros)
└── README.md                     # Documentación general y guía metodológica
```

---

## 5. Requisitos de Ejecución

- **Entorno:** Python 3.10 o superior (validado en Python 3.11.9).
- **Herramientas de ejecución:** Jupyter Notebook, JupyterLab o Visual Studio Code.

### Librerías Requeridas
- `pandas` (>= 2.0.0)
- `numpy` (>= 1.24.0)
- `scipy` (>= 1.10.0)
- `matplotlib` (>= 3.7.0)
- `seaborn` (>= 0.12.0)
- `scikit-learn` (>= 1.3.0)

### Instalación de Dependencias
```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn
```

---

## 6. Instrucciones de Uso

1. Colocar los archivos `desercion-entrenamiento.csv`, `desercion-prueba.csv` y el cuaderno `practica-3.ipynb` dentro del mismo directorio de trabajo.
2. Iniciar el entorno interactivo de Jupyter Notebook o VS Code.
3. Abrir `practica-3.ipynb` y ejecutar el cuaderno completo mediante el comando **Restart & Run All**.
4. El cuaderno ejecutará de forma secuencial:
   - El análisis exploratorio univariado y multivariado (Figuras 1, 2 y 3).
   - La selección de atributos y la configuración de los 5 pliegues estratificados.
   - La optimización de los cinco clasificadores y sus respectivas curvas de convergencia (Figuras 4 a 8).
   - La tabla y figuras comparativas de desempeño (Figura 9).
   - El diagnóstico de matriz de confusión y perfiles de deserción (Figuras 10 y 11).
   - El experimento complementario con información del segundo semestre (Figura 12).
   - La exportación y validación automática del archivo `predicciones.csv` para los 885 estudiantes de prueba.

---

## 7. Formato del Archivo de Entrega (`predicciones.csv`)

El archivo de salida cumple estrictamente con las especificaciones del curso:
- **Nombre:** `predicciones.csv`
- **Dimensiones:** Exactamente 885 filas y 2 columnas.
- **Columnas:**
  - `id`: Identificador entero único del estudiante (coincidente 1 a 1 con `desercion-prueba.csv`).
  - `Target`: Predicción en minúsculas restringida exclusivamente a `dropout`, `enrolled` o `graduate`.
- **Integridad:** Sin valores nulos ni duplicados.

---

## 8. Referencias Bibliográficas (Normas APA 7.ª Edición)

- Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. *Journal of Machine Learning Research*, *3*, 1157–1182.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning: Data mining, inference, and prediction* (2nd ed.). Springer. https://doi.org/10.1007/978-0-387-84858-7
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, *12*, 2825–2830.
- Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). Predicting student dropout and academic success. *Data*, *7*(11), 146. https://doi.org/10.3390/data7110146
- Realinho, V., Martins, M. V., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Conjunto de datos]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89. Licencia CC BY 4.0.
- Tinto, V. (1975). Dropout from higher education: A theoretical synthesis of recent research. *Review of Educational Research*, *45*(1), 89–125. https://doi.org/10.3102/00346543045001089
