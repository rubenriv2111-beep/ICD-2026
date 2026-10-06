# Práctica: Diagnóstico Estadístico y Modelado de Regresión Lineal en Vino Tinto

## Descripción del Dataset
El conjunto de datos proviene del repositorio **UCI Machine Learning Repository** (Cortez et al., 2009).
- **Nombre del archivo:** `winequality-red.csv`
- **Contenido original:** 1,599 observaciones de botellas de vino tinto *Vinho Verde* (Portugal) con 11 mediciones fisicoquímicas objetivas y una calificación sensorial de calidad en escala discreta (0 a 10).
- **Contenido depurado:** 1,359 registros únicos independientes tras eliminar 240 filas redundantes idénticas.
- **Licencia:** Dominio público / Uso académico abierto.

---

## Estructura Metodológica y Fases del Proyecto

### 1. Diagnóstico Integral de Calidad de Datos (10 Puntos Metodológicos)
1. **Duplicados:** Identificación y supresión formal de 240 registros idénticos redundantes ($1,599 \to 1,359$ filas), eliminando la fuga de información entre particiones (*data leakage*) y evitando la sobreponderación artificial en la minimización de errores cuadráticos de OLS ($\sum e_i^2$).
2. **Valores Faltantes:** Comprobación de 0 datos nulos en la matriz completa ($100\%$ integridad).
3. **Valores Atípicos (Outliers):** Diagnóstico con diagramas de caja en sulfatos, cloruros y acidez volátil, analizando la sensibilidad cuadrática ($e_i^2$) de Mínimos Cuadrados Ordinarios (OLS).
4. **Errores de Captura:** Verificación de rangos fisicoquímicos enológicos válidos (pH entre 2.74 y 4.01, alcohol entre 8.40% y 14.90%, densidades entre 0.990 y 1.003 g/cm³).
5. **Tipos de Datos:** Estandarización de 11 variables a coma flotante (`float64`) y calidad a entero (`int64`).
6. **Unidades de Medida:** Homogeneización dimensional ($g/\text{dm}^3$, $mg/\text{dm}^3$, $\%$ vol).
7. **Distribución y Dispersión ($X$ vs. $y$):** Análisis de correlación de Spearman, destacando la asociación positiva del alcohol ($r_s = +0.48$) y negativa de la acidez volátil ($r_s = -0.38$). Concentración del $81.8\%$ de los vinos en notas 5 y 6.
8. **Multicolinealidad (VIF):** Cálculo formal del Factor de Inflación de la Varianza con término constante ($VIF < 10$ en todas las variables predictoras).
9. **Independencia Muestral:** Verificación de diseño transversal sin correlación seriada temporal ni espacial.
10. **Transformaciones:** Evaluación de asimetría positiva en cloruros y sulfatos.

---

### 2. Regresión Lineal Simple y Mínimos Cuadrados (Quality vs. Alcohol)
- **Ecuación Ajustada:** $\hat{y} = 1.91 + 0.36 \cdot \text{Alcohol}$
- **Parámetros OLS (Entrenamiento):** Pendiente $\theta_1 = +0.3567$, Intercepto $\theta_0 = 1.9143$.
- **Visualización (Figura 3):**
  - **Figura 3A:** Recta univariada con ecuación explícita y pendiente visible.
  - **Figura 3B:** Representación de distancias verticales residuales ($e_i = y_i - \hat{y}_i$).

---

### 3. Modelado de Regresión Lineal Múltiple (Predecir `quality`)
- **Partición:** 80% Entrenamiento ($1,087$ muestras) / 20% Prueba ($272$ muestras), `random_state=42`.
- **Resultados en Prueba:**
  - **Línea Base (Media de Entrenamiento $\bar{y} = 5.63$):** $RMSE = 0.8443$, $R^2 \approx 0.00$
  - **Regresión Lineal Múltiple:** $RMSE = 0.6565$, $R^2 = 0.3915$ (explica el **$39.15\%$** de la variación sensorial).
- **Visualización:** Gráfico de valores reales vs. predichos e histograma de residuos centrado en cero.

---

### 4. Modelado de Clasificación Binaria (`quality >= 7`)
- **Definición:** `bueno = (quality >= 7).astype(int)`
- **Regla de Decisión:** $\hat{y} \ge 0.5 \to \text{bueno}$, $\hat{y} < 0.5 \to \text{no bueno}$.
- **Resultados en Prueba:**
  - **Exactitud del Modelo Lineal:** $88.24\%$
  - **Exactitud de la Línea Base (Siempre "No Bueno"):** $87.50\%$
  - **Predicciones Continuas fuera de $[0, 1]$:** $60$ de $272$ ($22.06\%$, con valores negativos hasta $-0.17$).
  - **Sensibilidad en clase minoritaria:** Detecta solo $4$ de $34$ vinos buenos ($11.76\%$).
- **Visualización (Figura 4):**
  - **Figura 4A:** Recta ajustada sobre ceros y unos con franjas sombreadas para $\hat{y} < 0$ y $\hat{y} > 1$.
  - **Figura 4B:** Distribución de predicciones continuas del modelo múltiple agrupadas por clase real.

---

### 5. Tabla de Síntesis Metodológica y Hallazgos

| Etapa del Análisis | Metodología Aplicada | Hallazgo Principal Obtenido |
| :--- | :--- | :--- |
| **1. Depuración de Duplicados** | `drop_duplicates()` suprimiendo 240 filas idénticas redundantes ($1,599 \to 1,359$). | Elimina la fuga de datos (*leakage*) y la sobreponderación espuria en OLS ($\sum e_i^2$). |
| **2. Regresión Simple** | Ajuste analítico de mínimos cuadrados univariado ($\text{Quality} \sim \text{Alcohol}$). | El alcohol es el predictor individual dominante ($\theta_1 = +0.3567, \theta_0 = 1.9143$). |
| **3. Regresión Múltiple** | Partición 80/20 con 11 variables fisicoquímicas evaluadas con $RMSE$ y $R^2$. | Reduce el error a $0.6565$ y explica el $39.15\%$ de la variabilidad sensorial del vino. |
| **4. Línea Base (Media)** | Estimación constante con el promedio de entrenamiento ($\bar{y} = 5.63$). | Genera un error típico de $0.8443$ ($R^2 \approx 0.00$), sirviendo de referencia mínima. |
| **5. Binarización de Calidad** | Definición de etiqueta $\text{Bueno} = (\text{Quality} \ge 7)$. | Revela desbalance de clases: solo el $13.57\%$ califica como vino sobresaliente. |
| **6. Clasificador Lineal ($\hat{y} \ge 0.5$)** | Ajuste de la recta sobre ceros y unos con regla de decisión en $0.5$. | Acierta el $88.24\%$, pero solo detecta $4$ de los $34$ vinos buenos ($11.76\%$ de sensibilidad). |
| **7. Línea Base Categórica** | Predicción fija de la clase mayoritaria ("Siempre No Bueno"). | Alcanza $87.50\%$ de exactitud, superando la recta a la línea base en apenas $0.74\%$. |
| **8. Diagnóstico de Rango** | Conteo de predicciones continuas $\hat{y} < 0$ y $\hat{y} > 1$. | Un total de $60$ de $272$ vinos de prueba ($22.06\%$) reciben valores negativos (hasta $-0.17$). |

---

### 6. Discusión Comparativa: ¿Qué le falla a la recta como clasificador?
En regresión, el objetivo es aproximar una magnitud numérica sobre un espacio continuo minimizando la suma de errores cuadráticos ($RMSE$). En clasificación, el propósito consiste en estimar la probabilidad posterior condicional de pertenencia a una categoría discreta ($P(Y=1|X) \in [0, 1]$). La regresión lineal falla como clasificador al proyectar hiperplanos rígidos sin cotas que generan valores negativos e hiper-unitarios carentes de significado probabilístico, carece de la curvatura sigmoidal requerida en zonas de transición y resulta altamente sensible al desbalance de clases, acertando una proporción insignificante por encima de la línea base trivial.

---

## Referencias Bibliográficas (Normas APA 7.ª Edición)
- Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems*, *47*(4), 547–553. https://doi.org/10.1016/j.dss.2009.05.016
- Gómez-Mendoza, A., Trejo-Téllez, L. I., García-Mata, R., & Salgado-Sosa, E. (2012). Caracterización fisicoquímica y sensorial de vinos tintos producidos en México. *Revista Fitotecnia Mexicana*, *35*(5), 89–94. https://www.scielo.org.mx/pdf/rfm/v35nspe5/v35nspe5a13.pdf
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An introduction to statistical learning: With applications in R* (2nd ed.). Springer. https://doi.org/10.1007/978-1-0716-1418-1
- Universidad Complutense de Madrid. (2018). *Evaluación de parámetros enológicos, compuestos fenólicos y capacidad antioxidante en vinos tintos*. Repositorio Institucional Docta Complutense. https://docta.ucm.es/bitstreams/56f1dd2a-e8c8-458e-a232-e3cc03834f98/download
- Universidad de La Rioja. (2017). *Evolución de la acidez volátil, sulfatos y calidad sensorial en vinos tintos de crianza*. Dialnet Documentos. https://dialnet.unirioja.es/descarga/articulo/6117897.pdf
- VanderPlas, J. (2022). *Python data science handbook: Essential tools for working with data* (2nd ed.). O'Reilly Media.
