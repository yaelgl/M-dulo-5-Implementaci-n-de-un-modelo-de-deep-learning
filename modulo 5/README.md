# Reconocimiento de actividad física con señales de sensores

Implementación de un modelo de deep learning con **Keras + TensorFlow** que
clasifica seis actividades físicas a partir de las señales crudas del acelerómetro
y el giroscopio de un teléfono.

**Autor:** Yael Michel García López (A01750911)
**Materia:** Inteligencia artificial avanzada para la ciencia de datos
**Entregable:** Módulo 2 — Implementación de un modelo de deep learning

El desarrollo completo, ejecutado y con todas las salidas visibles, está en
[`reporte_deep_learning.ipynb`](reporte_deep_learning.ipynb).

## Resultado

**90.91% de exactitud** clasificando actividades de 9 personas que el modelo nunca
vio durante el entrenamiento.

| | Modelo base | Modelo final |
|---|---|---|
| Exactitud entrenamiento | 98.94% | 96.21% |
| Exactitud validación | 90.36% | 90.65% |
| **Exactitud prueba** | 89.62% | **90.91%** |
| Brecha entrenamiento − validación | 8.58 pts | 5.56 pts |
| F1 macro en prueba | 0.8972 | 0.9091 |
| Errores en prueba | 306 | 268 |
| Épocas | 60 (fijas) | 28 (paro temprano) |

Desempeño por clase del modelo final:

| Actividad | Precisión | Sensibilidad | F1 |
|---|---|---|---|
| LAYING | 0.9980 | 0.9497 | 0.9733 |
| WALKING | 0.9626 | 0.9859 | 0.9741 |
| WALKING_DOWNSTAIRS | 0.9354 | 1.0000 | 0.9666 |
| WALKING_UPSTAIRS | 0.9226 | 0.9108 | 0.9167 |
| STANDING | 0.7810 | 0.9117 | 0.8413 |
| SITTING | 0.8804 | 0.7047 | 0.7828 |

## El dataset

[Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones),
UCI Machine Learning Repository (dataset 240).

- 30 voluntarios de 19 a 48 años, con un Samsung Galaxy S II en la cintura
- 10,299 ventanas de 2.56 segundos muestreadas a 50 Hz
- 6 actividades: caminar, subir escaleras, bajar escaleras, sentado, de pie, acostado

Se usan **las 9 señales inerciales crudas** (128 lecturas × 9 canales por ventana),
no los 561 atributos precalculados que también incluye el dataset. Es una decisión
deliberada: con los atributos precalculados cualquier modelo clásico funciona, pero
alguien tuvo que diseñarlos. Partir de la señal cruda es donde el deep learning
aporta, porque la red aprende sola qué medir.

Son datos reales de personas y sensores físicos, y no es un *toy dataset* ni uno de
los ejemplos vistos en clase.

El dataset **no está incluido en el repositorio** porque pesa unos 350 MB
descomprimido. El notebook lo descarga automáticamente de UCI en su primera celda
de datos.

## La arquitectura

Red híbrida **Conv1D + LSTM bidireccional**, 153,433 parámetros:

```
Entrada (128, 9)
  └─ Normalization            estadísticos aprendidos del entrenamiento
  └─ Conv1D(64, kernel=7)  + BatchNorm + MaxPool + SpatialDropout(0.2)
  └─ Conv1D(128, kernel=5) + BatchNorm + MaxPool + SpatialDropout(0.2)
  └─ Bidirectional(LSTM(64, dropout=0.3))
  └─ Dropout(0.5) → Dense(64) → Dropout(0.3)
  └─ Dense(6, softmax)
```

Cada tipo de capa resuelve una parte distinta: las convoluciones detectan la forma
local de la señal (una zancada dura unas 7 lecturas) con invariancia a la posición
dentro de la ventana, y el LSTM bidireccional modela cómo se ordenan esos patrones
en el tiempo. El pooling reduce la secuencia de 128 a 32 pasos antes del LSTM, que
es costoso en secuencias largas.

La capa de normalización va **dentro** del modelo, así que el archivo `.keras`
guarda los estadísticos: quien use el modelo le pasa la ventana cruda sin replicar
ningún preprocesamiento.

## Hallazgos del trabajo

**La partición tiene que ser por persona, no por ventana.** Es la decisión más
consecuente del notebook. Las ventanas se traslapan 50% entre vecinas, así que una
partición aleatoria pondría una ventana en entrenamiento y su gemela en validación.
Además el modelo puede aprender a reconocer a la persona en lugar de la actividad.
Los tres conjuntos separan sujetos completos: 16 para entrenar, 5 para validar, 9
para probar, sin traslape.

**Regularizar y cambiar la arquitectura resuelven problemas distintos.** La
regularización cerró la brecha de sobreajuste de 8.58 a 5.56 puntos y bajó los
errores en clases dinámicas de 125 a 85, pero dejó la confusión SITTING/STANDING
intacta (181 contra 183). Hizo falta diagnosticar *dónde* estaban los errores, no
sólo cuántos.

**El error dominante tiene causa física.** El 59% de los errores del modelo base son
confusiones entre estar sentado y estar de pie. Las dos son inmóviles: lo único que
las distingue es la orientación del teléfono respecto a la gravedad, que es
información estática, y la arquitectura la descarta en el pooling. Se probó una
variante con una rama paralela de postura que confirmó la hipótesis —esos errores
bajaron de 183 a 169— pero no lo suficiente para justificar duplicar el modelo.

**Una sola partición de validación no alcanza para elegir modelo.** Con 5 sujetos de
validación, el candidato C ganaba en validación y el A en prueba. La validación
cruzada agrupada por sujeto (3 pliegues) mostró por qué: la exactitud por pliegue
varía de 86.86% a 98.36% según a quién le toque validar, un rango de 11 puntos,
mientras las diferencias entre candidatos son de 0.25 puntos. Son estadísticamente
equivalentes, y se eligió el A por parsimonia: 153,433 parámetros contra 289,473.

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
jupyter notebook reporte_deep_learning.ipynb
```

La primera ejecución descarga el dataset (58 MB) y tarda unos 20 minutos, repartidos
entre el modelo base, los tres candidatos y la validación cruzada. Las ejecuciones
siguientes reutilizan la caché `.npy` de los datos.

### Predicciones desde la consola

```bash
# Clasificar una ventana guardada en archivo
python predecir.py ejemplos/ventana_walking.npy

# Varias a la vez
python predecir.py ejemplos/ventana_sitting.npy ejemplos/ventana_laying.npy

# Tomar ejemplos del conjunto de prueba y verificar aciertos
python predecir.py --prueba 5
```

La salida muestra la actividad predicha, su confianza y la distribución completa de
probabilidad.

## Archivos

| Archivo | Contenido |
|---|---|
| `reporte_deep_learning.ipynb` | Desarrollo completo, ejecutado y con salidas |
| `modelo_final.keras` | Modelo entrenado: arquitectura, pesos y normalización |
| `predecir.py` | Aplicación de consola para clasificar ventanas nuevas |
| `ejemplos/` | Seis ventanas de ejemplo, una por actividad |
| `requirements.txt` | Dependencias |
| `datos/` | Dataset, descargado por el notebook (no versionado) |

## Estructura del notebook

1. **Introducción** — el problema, su relevancia y por qué deep learning
2. **Datos** — fuente, análisis exploratorio, separación por sujeto, preprocesamiento
3. **Desarrollo del modelo** — arquitectura, entrenamiento y diagnóstico
4. **Ajuste del modelo** — regularización, variantes de arquitectura y selección por validación cruzada
5. **Resultados** — evaluación final en prueba y guardado del modelo
6. **Conclusiones** — análisis y áreas de mejora
7. **Aplicación** — función de clasificación para datos nuevos

## Reproducibilidad

Semilla 42 fijada con `keras.utils.set_random_seed`, que cubre `numpy`, `random` y
TensorFlow. La partición de sujetos, la inicialización de pesos y los pliegues de
la validación cruzada son deterministas. Entrenado en CPU; no requiere GPU.
