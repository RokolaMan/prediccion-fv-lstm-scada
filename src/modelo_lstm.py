# -*- coding: utf-8 -*-
"""
Predicción de potencia fotovoltaica con redes LSTM (partición 70/30).

Pasos: lectura de datos, escalado (MinMaxScaler ajustado solo con
entrenamiento), ventanas de 24 pasos, entrenamiento con parada temprana,
evaluación y guardado de modelo, escaladores, métricas, resultados y figuras.

Uso:
    python src/modelo_lstm.py --datos data/solar_real.csv --salida resultados
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
FEATURES = ["voltaje_v", "corriente_a", "temp_c", "hum_pct", "lux", "potencia_w"]
TARGET = "potencia_w"

HORIZON = 1          # desplazamiento del objetivo (shift)
TIME_STEPS = 24      # longitud de la ventana de entrada
TRAIN_RATIO = 0.70   # partición cronológica 70/30
VAL_SPLIT = 0.10     # fracción final del entrenamiento usada como validación
EPOCHS = 50
BATCH_SIZE = 32
PATIENCE = 8
# Nota: con HORIZON = 1 y la ventana [i, i+23], el objetivo corresponde a la
# potencia en i+24+1 - 1 = t+2 respecto del último paso observado (t = i+23).


def cargar_datos(ruta):
    df = pd.read_csv(ruta)
    df.columns = df.columns.str.strip()
    faltan = [c for c in ["timestamp"] + FEATURES if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en el CSV: {faltan}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    data = df[["timestamp"] + FEATURES].copy()
    data["target_future"] = data[TARGET].shift(-HORIZON)
    data["ts_objetivo"] = data["timestamp"].shift(-HORIZON)
    return data.dropna().reset_index(drop=True)


def crear_secuencias(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def construir_modelo(time_steps, n_features):
    model = Sequential([
        Input(shape=(time_steps, n_features)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def guardar_figuras(y_real, y_pred, history, carpeta):
    carpeta.mkdir(parents=True, exist_ok=True)
    residuales = y_real - y_pred
    error_abs = np.abs(residuales)

    def guardar(nombre):
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(carpeta / nombre, dpi=200)
        plt.close()

    plt.figure(figsize=(14, 6))
    plt.plot(y_real, label="Valor real", linewidth=1.5)
    plt.plot(y_pred, label="Valor predicho", linewidth=1.5)
    plt.xlabel("Muestras"); plt.ylabel("Potencia (W)"); plt.legend()
    plt.title("Valores reales y predichos (conjunto de prueba)")
    guardar("real_vs_predicho.png")

    plt.figure(figsize=(14, 6))
    plt.plot(y_real[:48], label="Valor real", linewidth=2)
    plt.plot(y_pred[:48], label="Valor predicho", linewidth=2)
    plt.xlabel("Muestras"); plt.ylabel("Potencia (W)"); plt.legend()
    plt.title("Valores reales y predichos (primeras 48 muestras)")
    guardar("real_vs_predicho_48.png")

    plt.figure(figsize=(8, 8))
    plt.scatter(y_real, y_pred, alpha=0.5)
    lim = [y_real.min(), y_real.max()]
    plt.plot(lim, lim, "r--", linewidth=2)
    plt.xlabel("Valor real (W)"); plt.ylabel("Valor predicho (W)")
    plt.title("Dispersión real vs. predicho")
    guardar("dispersion.png")

    plt.figure(figsize=(14, 6))
    plt.scatter(range(len(residuales)), residuales, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Muestras"); plt.ylabel("Residual (real - predicho) (W)")
    plt.title("Residuales")
    guardar("residuales.png")

    plt.figure(figsize=(10, 6))
    plt.hist(residuales, bins=30, edgecolor="black")
    plt.xlabel("Error (W)"); plt.ylabel("Frecuencia")
    plt.title("Distribución de errores")
    guardar("histograma_errores.png")

    plt.figure(figsize=(8, 6))
    plt.boxplot(error_abs)
    plt.ylabel("Error absoluto (W)")
    plt.title("Boxplot del error absoluto")
    guardar("boxplot_error.png")

    plt.figure(figsize=(12, 6))
    plt.plot(history.history["loss"], label="Entrenamiento")
    plt.plot(history.history["val_loss"], label="Validación")
    plt.xlabel("Épocas"); plt.ylabel("Pérdida (MSE)"); plt.legend()
    plt.title("Curva de pérdida del modelo LSTM")
    guardar("curva_perdida.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datos", default="data/solar_real.csv", help="CSV de entrada")
    parser.add_argument("--salida", default="resultados", help="carpeta de salida")
    parser.add_argument("--semilla", type=int, default=42, help="semilla aleatoria")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.semilla)
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    data = cargar_datos(args.datos)
    n = len(data)
    train_end = int(n * TRAIN_RATIO)
    train_df = data.iloc[:train_end]
    test_df = data.iloc[train_end:]
    print(f"Registros: {n} | entrenamiento: {len(train_df)} | prueba: {len(test_df)}")
    print(f"Entrenamiento: {train_df.timestamp.iloc[0]} -> {train_df.timestamp.iloc[-1]}")
    print(f"Prueba:        {test_df.timestamp.iloc[0]} -> {test_df.timestamp.iloc[-1]}")

    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_train_s = scaler_X.fit_transform(train_df[FEATURES].values)
    y_train_s = scaler_y.fit_transform(train_df[["target_future"]].values)
    X_test_s = scaler_X.transform(test_df[FEATURES].values)
    y_test_s = scaler_y.transform(test_df[["target_future"]].values)

    X_train, y_train = crear_secuencias(X_train_s, y_train_s, TIME_STEPS)
    X_test, y_test = crear_secuencias(X_test_s, y_test_s, TIME_STEPS)
    print("X_train:", X_train.shape, "| X_test:", X_test.shape)

    model = construir_modelo(TIME_STEPS, len(FEATURES))
    model.summary()
    early_stop = EarlyStopping(monitor="val_loss", patience=PATIENCE,
                               restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_split=VAL_SPLIT,
                        epochs=EPOCHS, batch_size=BATCH_SIZE,
                        callbacks=[early_stop], verbose=1)

    y_real = scaler_y.inverse_transform(y_test).flatten()
    y_pred = scaler_y.inverse_transform(model.predict(X_test)).flatten()

    diurno = y_real > 1.0  # MAPE/sMAPE solo en horas con generación (> 1 W)
    metricas = {
        "MAE_W": float(mean_absolute_error(y_real, y_pred)),
        "RMSE_W": float(np.sqrt(mean_squared_error(y_real, y_pred))),
        "R2": float(r2_score(y_real, y_pred)),
        "MAPE_diurno_pct": float(np.mean(np.abs((y_real[diurno] - y_pred[diurno])
                                                 / y_real[diurno])) * 100),
        "sMAPE_diurno_pct": float(np.mean(2 * np.abs(y_real[diurno] - y_pred[diurno])
                                          / (np.abs(y_real[diurno]) + np.abs(y_pred[diurno])))
                                  * 100),
        "epocas_entrenadas": len(history.history["loss"]),
        "mejor_epoca": int(np.argmin(history.history["val_loss"]) + 1),
        "n_prueba": int(len(y_real)),
        "semilla": args.semilla,
    }
    for k, v in metricas.items():
        print(f"{k:>18}: {v}")

    # Marca de tiempo de cada objetivo (potencia en t+2)
    ts = test_df["ts_objetivo"].iloc[TIME_STEPS:].reset_index(drop=True)
    resultados = pd.DataFrame({"timestamp": ts, "real": y_real, "predicho": y_pred})
    resultados["error_abs"] = np.abs(resultados["real"] - resultados["predicho"])
    resultados["error_pct"] = np.where(
        resultados["real"] == 0, 0,
        np.abs((resultados["real"] - resultados["predicho"]) / resultados["real"]) * 100)

    resultados.to_csv(salida / "resultados_lstm_70_30.csv", index=False)
    (salida / "metricas.json").write_text(json.dumps(metricas, indent=2), encoding="utf-8")
    pd.DataFrame(history.history).to_csv(salida / "historial_entrenamiento.csv",
                                         index_label="epoca")
    model.save(salida / "modelo_lstm_70_30.keras")
    joblib.dump(scaler_X, salida / "scaler_X_70_30.save")
    joblib.dump(scaler_y, salida / "scaler_y_70_30.save")
    guardar_figuras(y_real, y_pred, history, salida / "figuras")
    print(f"Resultados guardados en: {salida.resolve()}")


if __name__ == "__main__":
    main()
