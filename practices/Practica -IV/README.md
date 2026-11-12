# Práctica 4 - Meta-aprendizaje y Ensambles de Clasificadores

**Introducción a la Ciencia de Datos 2026**  
Posgrado en Ciencias de la Computación - Centro de Investigación Científica y de Educación Superior de Ensenada (CICESE)  
**Autor -** Ruben Rivera Garcia  
**Repositorio -** [github.com/rubenriv2111-beep/ICD-2026](https://github.com/rubenriv2111-beep/ICD-2026)

---

## 1. Descripción del Estudio y Conjuntos de Datos

### Parte 1 - Deserción Estudiantil (`desercion-entrenamiento.csv`)
- **Fuente -** UCI Machine Learning Repository (https://doi.org/10.24432/C5MC89) - Realinho et al. (2021, 2022).
- **Instancias y atributos -** 3,539 registros de entrenamiento con 29 variables explicativas disponibles al cierre del primer semestre (factores demográficos, socioeconómicos, escolares, curriculares y macroeconómicos).
- **Variable objetivo (`Target`) -** Clasificación multiclase con tres estados (`Dropout`, `Enrolled`, `Graduate`).
- **Validación -** Partición estratificada idéntica a la Práctica 3 (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`) evaluando `f1_macro`, sensibilidad sobre `Enrolled` y tiempo de entrenamiento.

### Parte 2 - Bank Marketing (`bank-additional-full.csv`)
- **Fuente -** UCI Machine Learning Repository (https://doi.org/10.24432/C5K306) - Moro et al. (2014).
- **Instancias y atributos -** 41,188 contactos telefónicos bancarios con 19 variables predictoras reales tras la supresión obligatoria de `duration` frente a fuga de información.
- **Variable objetivo (`y`) -** Clasificación binaria desbalanceada (`no` = 0 con 36,548 registros [88.73 %] frente a `yes` = 1 con 4,640 registros [11.27 %], razón cercana a 7.88 a 1).
- **Validación -** `cross_validate` sobre `StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)` con `scoring="f1_macro"` y preprocesamiento/umbral íntegramente dentro de `Pipeline`.

---

## 2. Metodología Implementada

1. **Varianza y Sesgo -** Análisis progresivo de Bagging con árboles profundos (`warm_start=True` hasta 400 árboles) y AdaBoost con tocones (`staged_predict()` hasta 400 rondas).
2. **Integración de modelos base (Práctica 3) -** Votación dura, Votación suave y Stacking logístico con calibración sigmoidal de SVM (`CalibratedClassifierCV`) dentro de cada pliegue, acompañados del mapa de correlación de errores.
3. **Optimización de hiperparámetros (`RandomizedSearchCV`, 50 configuraciones) -** Exploración con curvas reales de meseta en Random Forest, Extra-Trees, XGBoost y LightGBM con ponderación de clases.
4. **Desbalance severo y ajuste de umbral en Bank Marketing -** Contraste experimental entre ponderación automática, sobremuestreo `SMOTENC`, pesos calibrados, ingeniería sobre `pdays`, `TargetEncoder`, ajuste interno del umbral (`TunedThresholdClassifierCV`) y combinación de ensambles.

---

## 3. Requisitos Técnicos y Versiones

Instalar dependencias mediante `requirements.txt`.

```bash
pip install -r requirements.txt
```

### Entorno validado en ejecución
- `Python` 3.13.7
- `numpy` 2.3.3
- `pandas` 2.3.3
- `scipy` 1.16.3
- `scikit-learn` 1.7.2
- `xgboost` 3.4.1
- `lightgbm` 4.7.0
- `imbalanced-learn` 0.14.2
- `matplotlib` 3.10.7
- `seaborn` 0.13.2
- `joblib` 1.5.2

---

## 4. Estructura de Archivos

```text
Practica -IV/
├── bank-additional-full.csv     # Conjunto completo Bank Marketing (41,188 instancias)
├── bank-additional-names.txt    # Diccionario oficial de variables UCI
├── desercion-entrenamiento.csv  # Entrenamiento Deserción Estudiantil (3,539 registros)
├── desercion-prueba.csv         # Conjunto auxiliar de columnas 1er semestre
├── practica-4.ipynb             # Cuaderno Jupyter autocontenido y reproducible
├── ruben_rivera-voting_duro.joblib              # Modelo campeón final de Bank Marketing
├── cache_practica4.pkl          # Caché opcional generada automáticamente al ejecutar
├── requirements.txt             # Especificación de dependencias
└── README.md                    # Documentación de la práctica
```

---

## 5. Instrucciones de Uso

1. Ubicar el cuaderno `practica-4.ipynb` junto con `desercion-entrenamiento.csv` y `bank-additional-full.csv` dentro del mismo directorio.
2. Iniciar Jupyter Notebook o JupyterLab y abrir `practica-4.ipynb`.
3. Ejecutar **Kernel -> Restart & Run All**. Si `cache_practica4.pkl` está presente, los resultados se recuperan al instante; si está ausente, el cuaderno ejecuta todas las búsquedas y validaciones cruzadas desde cero regenerando automáticamente la caché y exportando `ruben_rivera-voting_duro.joblib`.

---

## 6. Síntesis de Resultados

- **Parte 1 (Deserción Estudiantil) -** El modelo campeón resultó ser **XGBoost (Optimizado)** con **$F_1$-macro = 0.6966 +/- 0.0155** (sensibilidad `Enrolled` de 0.5165 y tiempo medio de ajuste de 1.19 s), seguido de cerca por **LightGBM (Optimizado)** con **0.6952 +/- 0.0115** (sensibilidad `Enrolled` de 0.5559 en 0.34 s) y **Votación dura** de los cinco clasificadores base (**0.6846 +/- 0.0058**).

- **Parte 2 (Bank Marketing sin `duration`) -** El mejor desempeño correspondió a **Voting Duro Optimizado (LGB+XGB+RF)** integrando tuberías especializadas por modelo (`TargetEncoder` + `Binarizer(threshold=998.5)` sobre `pdays`), alcanzando **$F_1$-macro = 0.7195 +/- 0.0081** (sensibilidad `yes` = 0.5384 y especificidad `no` = 0.9253), seguido por **LightGBM (Optimizado)** (**0.7191 +/- 0.0061**) y **Random Forest (Optimizado)** (**0.7191 +/- 0.0072**). El pipeline campeón final quedó exportado en `ruben_rivera-voting_duro.joblib` (5,345.41 KB).

---

## 7. Referencias Bibliográficas (Normas APA 7.ª Edición)

- Breiman, L. (1996). Bagging predictors. *Machine Learning*, *24*(2), 123–140. https://doi.org/10.1007/BF00058655
- Breiman, L. (2001). Random forests. *Machine Learning*, *45*(1), 5–32. https://doi.org/10.1023/A:1010933404324
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE - Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, *16*, 321–357. https://doi.org/10.1613/jair.953
- Chen, T., & Guestrin, C. (2016). XGBoost - A scalable tree boosting system. En *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). ACM. https://doi.org/10.1145/2939672.2939785
- Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. *Journal of Computer and System Sciences*, *55*(1), 119–139. https://doi.org/10.1006/jcss.1997.1504
- Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees. *Machine Learning*, *63*(1), 3–42. https://doi.org/10.1007/s10994-006-6226-1
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. Y. (2017). LightGBM - A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, *30*, 3146–3154.
- Kuncheva, L. I. (2004). *Combining pattern classifiers - Methods and algorithms*. John Wiley & Sons.
- Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems*, *62*, 22–31. https://doi.org/10.1016/j.dss.2014.03.001
- Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Conjunto de datos]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn - Machine learning in Python. *Journal of Machine Learning Research*, *12*, 2825–2830.
- Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). Predicting student dropout and academic success. *Data*, *7*(11), 146. https://doi.org/10.3390/data7110146
- Realinho, V., Martins, M. V., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Conjunto de datos]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89
- Schapire, R. E., & Freund, Y. (2012). *Boosting - Foundations and algorithms*. MIT Press.
- Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, *5*(2), 241–259. https://doi.org/10.1016/S0893-6080(05)80023-1
