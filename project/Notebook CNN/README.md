# Práctica: Redes Neuronales Convolucionales, Global Average Pooling y Tolerancia al Desplazamiento Espacial en MNIST

Este módulo forma parte de los proyectos del curso de Introducción a la Ciencia de Datos (ICD-2026). En este trabajo se investiga, diseña y evalúa una arquitectura de Red Neuronal Convolucional (CNN) basada en **Global Average Pooling (GAP)**, analizando la frontera de eficiencia paramétrica frente a las capas totalmente conectadas tradicionales (**Flatten**) y cuantificando la robustez ante perturbaciones espaciales (traslaciones) sin reentrenamiento.

---

## 1. Objetivos del Proyecto

1. **Optimización de Arquitectura:** Diseñar una CNN con Global Average Pooling que supere el **95 % de exactitud en validación** sobre el conjunto de datos MNIST, reduciendo drásticamente los parámetros entrenables respecto a la CNN base con Flatten (**54,330 parámetros**).
2. **Selección por Criterio de Pareto:** Entre todas las arquitecturas que satisfagan ambos requisitos, seleccionar la variante con el **menor número de parámetros entrenables**.
3. **Evaluación de Robustez Espacial:** Medir sistemáticamente la tolerancia de los modelos ante desplazamientos espaciales de los dígitos a **0, 2 y 4 píxeles** en cuatro direcciones cardinales (arriba, abajo, izquierda y derecha), comparando la degradación de exactitud entre Flatten y GAP.
4. **Comprensión del Mecanismo Físico:** Analizar la invariancia a la traslación, la regularización estructural de GAP y el papel del campo receptivo efectivo en la clasificación de imágenes.

---

## 2. Conjunto de Datos y Preprocesamiento

- **Dataset:** MNIST (LeCun et al., 1998), compuesto por 70,000 imágenes en escala de grises de dígitos manuscritos (0 a 9) de $28 \times 28$ píxeles.
- **Normalización:** Escalado lineal de intensidades de píxel al intervalo flotante $[0.0, 1.0]$.
- **Partición Estratificada:**
  - **Entrenamiento:** 48,000 imágenes (68.57 %).
  - **Validación:** 12,000 imágenes (17.14 %).
  - **Prueba Ciega:** 10,000 imágenes (14.29 %).
  - Semilla estocástica fija: `random_state = 42`.
- **Margen Negro Perimetral (Padding):** Se añade un relleno perimetral de cuatro píxeles de valor cero en los cuatro bordes mediante `np.pad`, transformando la resolución de $28 \times 28$ a **$36 \times 36$ píxeles**. Esto proporciona el espacio físico necesario para aplicar desplazamientos controlados sin recortar los trazos esenciales de los dígitos.

---

## 3. Modelos Base de Referencia

| Modelo Base | Topología del Extractor | Capa de Clasificación | Parámetros Entrenables | Exactitud de Validación (%) |
| :--- | :--- | :--- | :---: | :---: |
| **CNN + Flatten** | Conv(16, 3x3) + Conv(16, 3x3) + MaxPool(2x2) | Flatten + Dense(10, Softmax) | 54,330 | **96.87 %** |
| **CNN + GAP (Base)** | Conv(16, 3x3) + Conv(16, 3x3) + MaxPool(2x2) | GlobalAveragePooling2D + Dense(10, Softmax) | 2,650 | **43.80 %** |

### Diagnóstico del Colapso en GAP Base
La arquitectura con Flatten alcanza un 96.87 % de exactitud debido a la gran capacidad de memorización posicional de su matriz densa ($18 \times 18 \times 16 = 5,184$ conexiones de entrada, sumando 51,850 pesos solo en la capa final). En contraste, el GAP base colapsa en **43.80 %**. Conforme a Lin et al. (2013), esto ocurre porque el extractor es excesivamente superficial (solo dos capas con campo receptivo de $5 \times 5$ px) y el promedio global se aplica sobre una grilla espacial amplia ($18 \times 18$ px) dominada por el margen negro perimetral, diluyendo la activación de los trazos y eliminando el poder discriminativo.

---

## 4. Exploración Sistemática de Arquitecturas con GAP

Se implementaron y entrenaron sistemáticamente 10 variantes bajo un protocolo homogéneo (20 épocas, optimizador Adam con $\eta = 0.001$, pérdida de entropía cruzada categórica):

| Variante Arquitectónica | Parámetros Entrenables | Exactitud Validación (%) | ¿Supera 95 %? | ¿Menor a 54,330 params? |
| :--- | :---: | :---: | :---: | :---: |
| **1. GAP Base (16-16, pool fin)** | 2,650 | 43.80 % | No | Sí |
| **2. GAP Ancho 32 (32-32, pool fin)** | 9,898 | 49.20 % | No | Sí |
| **3. GAP Profundidad 3 (16-16-16, pool fin)** | 4,970 | 53.47 % | No | Sí |
| **4. GAP Pool Intermedio (16, pool, 16)** | 2,650 | 60.73 % | No | Sí |
| **5. GAP Kernel 5x5 (16-16 k5, pool fin)** | 7,002 | 73.27 % | No | Sí |
| **6. GAP Jerárquico 16-16-16 (2 pools)** | 4,970 | 86.73 % | No | Sí |
| **7. GAP Piramidal 16-32-32 (2 pools)** | 14,378 | 91.07 % | No | Sí |
| **8. GAP Profundo 16-16-32-32 (2 pools)** | 16,682 | 93.68 % | No | Sí |
| **9. GAP Jerárquico 4c (16-16-32-32, 3 pools)** | 16,682 | 95.83 % | **Sí** | **Sí** |
| **10. GAP Seleccionado (16-16-32-32-32, 3 pools)** | **25,946** | **96.13 %** | **Sí** | **Sí** |

### Hallazgos de la Exploración
1. **El ensanchamiento o profundización aislados son insuficientes:** Aumentar a 32 filtros o añadir una tercera capa manteniendo el pooling al final solo elevó la exactitud a 49.20 % y 53.47 %.
2. **Impacto de la reubicación de pooling:** Mover el pooling al punto intermedio (Variante 4) aumentó la exactitud en **+16.93 puntos porcentuales** (de 43.80 % a 60.73 %) **sin agregar un solo parámetro**.
3. **Transición crítica sobre el 95 %:** La condición necesaria para que GAP supere el 95 % consiste en intercalar al menos tres etapas de max pooling con múltiples capas convolucionales, reduciendo la dimensión espacial antes de GAP a $4 \times 4$ píxeles para evitar la dilución del fondo oscuro.

---

## 5. Selección de la Arquitectura Óptima

La arquitectura seleccionada corresponde a la **Variante 10 (CNN con GAP Jerárquica de 3 pools)**:
- **Topología Convolucional:**
  - Bloque 1: Conv2D(16, 3x3, ReLU) + Conv2D(16, 3x3, ReLU) + MaxPooling2D(2x2) $\to$ salida $16 \times 16 \times 16$.
  - Bloque 2: Conv2D(32, 3x3, ReLU) + MaxPooling2D(2x2) $\to$ salida $7 \times 7 \times 32$.
  - Bloque 3: Conv2D(32, 3x3, ReLU) + MaxPooling2D(2x2) $\to$ salida $2 \times 2 \times 32$.
  - Bloque 4: Conv2D(32, 3x3, ReLU, padding='same') $\to$ salida $2 \times 2 \times 32$.
  - Clasificador: GlobalAveragePooling2D + Dense(10, Softmax).
- **Parámetros Entrenables:** **25,946 parámetros**.
- **Ahorro Paramétrico:** Reducción de **28,384 parámetros** frente a Flatten (**-52.24 %** del modelo original eliminado).
- **Exactitud de Validación:** **96.13 %**.

---

## 6. Evaluación de Tolerancia al Desplazamiento Espacial

Se evaluó la capacidad de generalización sobre el conjunto de prueba independiente (10,000 imágenes) aplicando desplazamientos de 0, 2 y 4 píxeles en las cuatro direcciones cardinales (arriba, abajo, izquierda, derecha) sin reentrenamiento:

| Modelo Evaluado | Prueba a 0 px (%) | Prueba a 2 px (%) | Prueba a 4 px (%) | Caída a 4 px (puntos porcentuales) |
| :--- | :---: | :---: | :---: | :---: |
| **CNN + Flatten (Base)** | 97.15 % | 82.81 % | 37.25 % | **59.90 pp** |
| **CNN + GAP (Base)** | 46.25 % | 46.25 % | 46.16 % | **0.09 pp** |
| **CNN + GAP Seleccionado** | **97.50 %** | **92.82 %** | **79.90 %** | **17.60 pp** |

### Análisis de Robustez Espacial
- **Colapso de Flatten:** Aunque Flatten parte de 97.15 % a 0 px, se desploma a **37.25 % a 4 píxeles** (una pérdida catastrófica de casi 60 puntos porcentuales), degradándose a un nivel inferior al azar en dígitos específicos.
- **Superioridad de GAP Seleccionado:** El modelo GAP seleccionado no solo iguala y supera la exactitud inicial a 0 px (**97.50 %**), sino que conserva un **79.90 % de exactitud ante traslaciones extremas de 4 píxeles** (+42.65 puntos porcentuales de ventaja absoluta sobre Flatten).
- **Distancia de divergencia:** La superioridad se vuelve evidente a partir de **2 píxeles de desplazamiento** (92.82 % vs. 82.81 %) y se magnifica dramáticamente a **4 píxeles**.

---

## 7. Análisis Avanzados Desarrollados en el Cuaderno

El cuaderno incluye diagnósticos cuantitativos y visuales complementarios:
1. **Frontera de Pareto (Parámetros vs. Exactitud):** Gráfico de dispersión bidimensional que localiza las variantes dominantes y visualiza el límite de 54,330 parámetros y el umbral del 95 %.
2. **Anisotropía Espacial y Descomposición Direccional:** Comparativa del comportamiento direccional (arriba, abajo, izquierda, derecha) evidenciando que el desplazamiento hacia arriba genera mayor degradación debido a la asimetría de los bucles superiores en dígitos como el 8 y el 9.
3. **Sensibilidad por Morfología de Dígito:** Evaluación de exactitud desglosada por clase (0 al 9). Dígitos geométricamente compactos (como el 1) resisten mejor el desplazamiento, mientras que dígitos con terminaciones abiertas y curvaturas finas (como el 2, 4 y 5) sufren mayor colapso en Flatten.
4. **Inspección de Mapas de Características y Mecanismo Físico:** Visualización de activaciones intermedias que demuestra cómo GAP resume la presencia de rasgos con independencia de su posición absoluta, mientras que Flatten depende rígidamente de coordenadas fijas en la cuadrícula.
5. **Galería de Diagnóstico Visual:** Ejemplos individuales de prueba donde la CNN con Flatten se confunde a 4 píxeles de desplazamiento y la CNN con GAP clasifica correctamente.

---

## 8. Estructura de Archivos

```text
Notebook CNN/
├── README.md             # Documentación metodológica y reporte de resultados
└── practica-cnn.ipynb    # Cuaderno Jupyter interactivo completamente ejecutado
```

---

## 9. Referencias Bibliográficas (Normas APA 7.ª Edición)

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press. https://www.deeplearningbook.org/
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, *86*(11), 2278–2324. https://doi.org/10.1109/5.726791
- Lin, M., Chen, Q., & Yan, S. (2013). Network in network. *arXiv preprint arXiv:1312.4400*. https://doi.org/10.48550/arXiv.1312.4400
- VanderPlas, J. (2022). *Python data science handbook: Essential tools for working with data* (2nd ed.). O'Reilly Media.
