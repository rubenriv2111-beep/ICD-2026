# Práctica 4: Métricas de Evaluación y Clasificación en Bank Marketing

## Descripción del Dataset
El conjunto de datos proviene del repositorio **UCI Machine Learning Repository** (Moro et al., 2014).
- **Nombre del archivo:** `bank-additional-full.csv`
- **Contenido original:** 41,188 observaciones telefónicas de clientes de una entidad bancaria portuguesa con 20 variables de entrada (demográficas, crediticias, operativas e indicadores macroeconómicos) y una variable objetivo binaria `y` ("yes" / "no") que indica la contratación de un depósito a plazo fijo.
- **Desbalance de clases:** 36,548 negativas ($88.73\%$) frente a 4,640 positivas ($11.27\%$), lo que equivale a una proporción aproximada de 1 aceptación por cada 9 contactos.
- **Licencia:** Dominio público / Uso académico abierto.

---

## Estructura Metodológica y Fases del Proyecto

### 1. Detección y Justificación de Fuga de Información (`duration`)
- **Variable identificada:** `duration` (duración del contacto en segundos).
- **Justificación técnica:** Representa la duración de la llamada en segundos, dato inexistente antes de establecer la comunicación. En la selección de prospectos este valor es desconocido, de modo que incluirlo generaría una fuga de datos (*data leakage*) irrealizable en producción (Domingos, 2012).

---

### 2. Partición Estratificada y Evaluación de la Línea Base
- **Partición:** 80% Entrenamiento ($32,950$ registros) / 20% Prueba ($8,238$ registros), estratificado por clase (`stratify=y`) con semilla fija (`random_state=42`).
- **Línea Base (Siempre "no"):** Exactitud de **$88.74\%$** en el conjunto de prueba.
- **Evaluación conceptual:** Un clasificador ingenuo que nunca llama logra casi $89\%$ de exactitud por el marcado desbalance de clases. Esto demuestra que la exactitud es engañosa en este escenario, pues oculta una sensibilidad de cero al no identificar a ningún cuentahabiente interesado.

---

### 3. Regresión Logística con `Pipeline` y Matriz de Confusión (Umbral 0.5)
- **Flujo de procesamiento:** `ColumnTransformer` que aplica `StandardScaler` a 9 variables cuantitativas y `OneHotEncoder(drop='first')` a 10 atributos categóricos, conectado a `LogisticRegression(max_iter=1000)`.
- **Matriz de Confusión en Prueba ($N = 8,238$):**
  - Verdaderos Negativos (TN): $7,218$
  - Falsos Positivos (FP): $92$
  - Falsos Negativos (FN): $720$
  - Verdaderos Positivos (TP): $208$
- **Métricas obtenidas:**
  - Exactitud: $90.14\%$
  - Precisión: $69.33\%$
  - Sensibilidad (Recall): $22.41\%$
  - Especificidad: $98.74\%$
  - Puntuación F1: $0.3388$
- **Métrica prioritaria:** Al banco le interesa el balance entre precisión y sensibilidad según los costos comerciales de contacto. Si los operadores tienen tiempo limitado y llamar cuesta, la precisión evita contactos inútiles; empero, perder clientes con alta intención reduce captación de capital, haciendo indispensable la evaluación económica del umbral.

---

### 4. Curva ROC frente a Curva Precisión-Sensibilidad (PR)
- **Área bajo la Curva ROC (AUC-ROC):** $0.8007$
- **Precisión Promedio (Average Precision / PR-AUC):** $0.4657$
- **Diagnóstico comparativo:** La curva ROC contrasta sensibilidad contra la tasa de falsos positivos, la cual permanece baja artificialmente por el enorme volumen de verdaderos negativos ($88.7\%$). La curva de precisión contra sensibilidad expone la caída abrupta de la precisión frente a los falsos positivos, reflejando el costo real de ampliar la búsqueda de clientes positivos (Saito & Rehmsmeier, 2015).

---

### 5. Modelo de Costo-Beneficio y Optimización del Umbral
- **Parámetros económicos:** Costo por contacto = $\$5$; Ingreso por contratación = $\$50$.
- **Deducción del umbral óptimo:** $\mathbb{E}[\text{Ganancia}] = 50 \cdot p - 5 \ge 0 \implies p^* = \frac{5}{50} = 0.10$.
- **Comparación de resultados económicos en prueba:**
  - **Umbral 0.50:** Realiza $300$ llamadas, concreta $208$ suscripciones y genera una **ganancia neta de $\$8,900$**.
  - **Umbral Óptimo 0.10:** Realiza $1,898$ llamadas, concreta $631$ suscripciones y genera una **ganancia neta de $\$22,060$**.
- **Impacto financiero:** El umbral teórico óptimo es 0.10, punto donde el ingreso esperado ($50 \times 0.10$) cubre el costo de la llamada ($5$). Cortar en 0.10 genera una ganancia neta superior frente al corte convencional de 0.50, captando un volumen significativamente mayor de contrataciones rentables.

---

### 6. Evaluación del Impacto de la Variable Omitida (`duration`)
- **Métricas con fuga de información:**
  - Exactitud: $91.62\%$
  - Precisión: $70.95\%$
  - Sensibilidad: $43.43\%$
  - Especificidad: $97.74\%$
  - Puntuación F1: $0.5388$
  - Área ROC (AUC): $0.9425$
- **Diagnóstico operativo:** Las métricas se incrementan artificialmente puesto que la duración actúa como sustituto del éxito de la interacción. Estas cifras son inútiles en la operación real, en vista de que el banco no dispone de la duración antes de marcar el número telefónico.

---

### 7. Tabla de Síntesis Metodológica y Hallazgos

| Etapa del Análisis | Procedimiento Metodológico | Hallazgo Central Documentado |
| :--- | :--- | :--- |
| **1. Auditoría de Variables** | Inspección de `names.txt` y exclusión preventiva de `duration`. | Se suprime una fuga de datos invalidante en la planificación comercial. |
| **2. Evaluación de Línea Base** | Partición estratificada 80/20 y ajuste de clasificador de clase mayoritaria. | Revela la falacia de la exactitud ($88.73\%$) ante el desbalance severo de clases. |
| **3. Clasificación Inicial** | Regresión logística en `Pipeline` evaluada con umbral probabilístico $0.50$. | Demuestra alta especificidad ($98.74\%$) empero baja sensibilidad ($22.41\%$). |
| **4. Diagnóstico de Curvas** | Análisis gráfico comparativo entre curva ROC ($AUC = 0.8007$) y curva PR ($AP = 0.4657$). | La curva PR delata la erosión de precisión oculta por los verdaderos negativos en ROC. |
| **5. Optimización Financiera** | Minimización de pérdidas esperadas con costo de $5$ e ingreso de $50$ por contacto. | El umbral $0.10$ maximiza el beneficio neto ($22,060$ frente a $8,900$ del umbral $0.50$). |
| **6. Simulación de Fuga** | Reincorporación de `duration` al modelo logístico de referencia. | Provoca un inflado espurio del rendimiento ($AUC = 0.9425$) inoperable en la realidad. |

---

## Referencias Bibliográficas (Normas APA 7.ª Edición)

- Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems*, *47*(4), 547–553. https://doi.org/10.1016/j.dss.2009.05.016
- Domingos, P. (2012). A few useful things to know about machine learning. *Communications of the ACM*, *55*(10), 78–87. https://doi.org/10.1145/2347736.2347755
- Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, *27*(8), 861–874. https://doi.org/10.1016/j.patrec.2005.10.010
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An introduction to statistical learning: With applications in R* (2nd ed.). Springer. https://doi.org/10.1007/978-1-0716-1418-1
- Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems*, *62*, 22–31. https://doi.org/10.1016/j.dss.2014.03.001
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, *10*(3), Article e0118432. https://doi.org/10.1371/journal.pone.0118432
- Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. *Information Processing & Management*, *45*(4), 427–437. https://doi.org/10.1016/j.ipm.2009.03.002
- VanderPlas, J. (2022). *Python data science handbook: Essential tools for working with data* (2nd ed.). O'Reilly Media.
