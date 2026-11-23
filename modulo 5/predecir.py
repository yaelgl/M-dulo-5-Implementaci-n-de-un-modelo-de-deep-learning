"""
Aplicación de consola: clasifica una ventana de señales de sensores.

El modelo entrenado lleva la capa de normalización adentro, así que este script
no necesita replicar ningún preprocesamiento: basta con pasarle la ventana cruda
tal como la entrega el teléfono.

Formato de entrada esperado: una ventana de 128 lecturas x 9 canales, en el orden
    body_acc_x, body_acc_y, body_acc_z,
    body_gyro_x, body_gyro_y, body_gyro_z,
    total_acc_x, total_acc_y, total_acc_z

Se aceptan archivos .npy de forma (128, 9) o archivos de texto con 128 renglones
y 9 columnas separadas por espacios.

Ejemplos de uso:

    # Clasificar una ventana guardada en archivo
    python predecir.py ventana_ejemplo.npy

    # Clasificar varias a la vez
    python predecir.py ventana1.npy ventana2.txt

    # Tomar ejemplos del conjunto de prueba (para verificar la instalación)
    python predecir.py --prueba 5
"""

import argparse
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"       # silencia los avisos de TensorFlow

import numpy as np

CARPETA = Path(__file__).parent
RUTA_MODELO = CARPETA / "modelo_final.keras"
CACHE = CARPETA / "datos" / "cache"

ACTIVIDADES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
               "SITTING", "STANDING", "LAYING"]

NOMBRES_ES = {
    "WALKING": "caminando",
    "WALKING_UPSTAIRS": "subiendo escaleras",
    "WALKING_DOWNSTAIRS": "bajando escaleras",
    "SITTING": "sentado",
    "STANDING": "de pie",
    "LAYING": "acostado",
}

PASOS, CANALES = 128, 9


def cargar_ventana(ruta: Path) -> np.ndarray:
    """Lee una ventana de sensores desde .npy o texto y valida su forma."""
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo: {ruta}")

    if ruta.suffix == ".npy":
        x = np.load(ruta)
    else:
        import pandas as pd
        x = pd.read_csv(ruta, sep=r"\s+", header=None).to_numpy(dtype="float32")

    x = np.asarray(x, dtype="float32")

    # Se acepta la ventana transpuesta (9, 128) y se corrige sola
    if x.shape == (CANALES, PASOS):
        x = x.T
    if x.shape != (PASOS, CANALES):
        raise ValueError(
            f"Forma inválida en {ruta.name}: se esperaba ({PASOS}, {CANALES}) "
            f"y llegó {x.shape}")
    return x


def predecir(modelo, ventanas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (índices de clase, matriz de probabilidades)."""
    probabilidades = modelo.predict(ventanas, verbose=0)
    return probabilidades.argmax(axis=1), probabilidades


def clasificar_actividad(ruta_archivo, modelo=None) -> dict:
    """Clasifica UNA ventana y devuelve el resultado como diccionario.

    Es la función que pide el entregable: recibe una ruta y devuelve el
    resultado de la clasificación.
    """
    import keras

    if modelo is None:
        modelo = keras.models.load_model(RUTA_MODELO)

    ventana = cargar_ventana(Path(ruta_archivo))
    indice, probabilidades = predecir(modelo, ventana[np.newaxis, ...])
    i = int(indice[0])
    p = probabilidades[0]
    return {
        "archivo": str(ruta_archivo),
        "actividad": ACTIVIDADES[i],
        "actividad_es": NOMBRES_ES[ACTIVIDADES[i]],
        "confianza": float(p[i]),
        "probabilidades": {a: float(v) for a, v in zip(ACTIVIDADES, p)},
    }


def imprimir_resultado(r, real=None):
    print(f"  Archivo    : {Path(r['archivo']).name}")
    print(f"  Predicción : {r['actividad']}  ({r['actividad_es']})")
    print(f"  Confianza  : {r['confianza'] * 100:.2f}%")
    if real is not None:
        marca = "correcto" if real == r["actividad"] else "INCORRECTO"
        print(f"  Real       : {real}  ->  {marca}")
    print("  Distribución de probabilidad:")
    for actividad, p in sorted(r["probabilidades"].items(),
                               key=lambda kv: -kv[1]):
        barra = "#" * int(round(p * 40))
        print(f"    {actividad:20} {p * 100:6.2f}%  {barra}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Clasifica actividad física a partir de señales de sensores.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archivos", nargs="*", type=Path,
                        help="Rutas a ventanas .npy o .txt de forma (128, 9)")
    parser.add_argument("--prueba", type=int, metavar="N",
                        help="Clasifica N ejemplos tomados del conjunto de prueba")
    parser.add_argument("--modelo", type=Path, default=RUTA_MODELO,
                        help="Ruta al archivo .keras del modelo")
    args = parser.parse_args()

    if not args.archivos and args.prueba is None:
        parser.print_help()
        return 1

    if not args.modelo.exists():
        print(f"ERROR: no se encontró el modelo en {args.modelo}")
        print("Ejecuta primero el notebook reporte_deep_learning.ipynb para "
              "generarlo.")
        return 1

    import keras
    print(f"Cargando modelo: {args.modelo.name}")
    modelo = keras.models.load_model(args.modelo)
    print(f"Modelo cargado ({modelo.count_params():,} parámetros)")
    print()

    # ---- ventanas desde archivos
    for ruta in args.archivos:
        try:
            r = clasificar_actividad(ruta, modelo=modelo)
            imprimir_resultado(r)
        except (FileNotFoundError, ValueError) as e:
            print(f"  {type(e).__name__}: {e}")
            print()

    # ---- ejemplos del conjunto de prueba
    if args.prueba:
        if not (CACHE / "X_test.npy").exists():
            print("No hay datos de prueba en caché. Ejecuta primero el notebook.")
            return 1

        X = np.load(CACHE / "X_test.npy")
        y = np.load(CACHE / "y_test.npy")
        rng = np.random.default_rng(42)
        indices = rng.choice(len(X), size=min(args.prueba, len(X)),
                             replace=False)

        print(f"Clasificando {len(indices)} ventanas del conjunto de prueba")
        print("=" * 62)
        aciertos = 0
        for i in indices:
            idx, prob = predecir(modelo, X[i][np.newaxis, ...])
            j = int(idx[0])
            r = {
                "archivo": f"prueba[{i}]",
                "actividad": ACTIVIDADES[j],
                "actividad_es": NOMBRES_ES[ACTIVIDADES[j]],
                "confianza": float(prob[0][j]),
                "probabilidades": {a: float(v)
                                   for a, v in zip(ACTIVIDADES, prob[0])},
            }
            real = ACTIVIDADES[int(y[i])]
            aciertos += int(real == r["actividad"])
            imprimir_resultado(r, real=real)
        print(f"Aciertos: {aciertos} de {len(indices)} "
              f"({100 * aciertos / len(indices):.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
