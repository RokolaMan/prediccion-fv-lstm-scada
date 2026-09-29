# -*- coding: utf-8 -*-
"""
Envía las predicciones del modelo LSTM a InfluxDB para su visualización en
Grafana (capa SCADA).

Las credenciales se leen de variables de entorno o de un archivo .env
(ver .env.example). Nunca escribas el token directamente en el código.

Uso:
    python src/enviar_influxdb.py --resultados resultados/resultados_lstm_70_30.csv
"""

import argparse
import os
import sys

import pandas as pd
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MEASUREMENT = "prediccion_lstm"


def config():
    faltan = [v for v in ("INFLUX_URL", "INFLUX_TOKEN", "INFLUX_ORG", "INFLUX_BUCKET")
              if not os.getenv(v)]
    if faltan:
        sys.exit(f"Faltan variables de entorno: {', '.join(faltan)}. "
                 "Copia .env.example a .env y complétalo.")
    return {k: os.getenv(k) for k in ("INFLUX_URL", "INFLUX_TOKEN",
                                      "INFLUX_ORG", "INFLUX_BUCKET")}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resultados", default="resultados/resultados_lstm_70_30.csv")
    parser.add_argument("--modelo", default="LSTM_70_30", help="etiqueta del modelo")
    parser.add_argument("--inicio", default="2025-01-01 00:00:00",
                        help="fecha inicial si el CSV no trae timestamp")
    parser.add_argument("--frecuencia", default="h",
                        help="frecuencia de muestreo si el CSV no trae timestamp")
    args = parser.parse_args()
    cfg = config()

    df = pd.read_csv(args.resultados)
    df.columns = df.columns.str.strip()
    if "timestamp" not in df.columns or df["timestamp"].isna().all():
        df["timestamp"] = pd.date_range(start=args.inicio, periods=len(df),
                                        freq=args.frecuencia)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Energía por intervalo (válida si el muestreo es horario: W·h -> kWh)
    df["energia_real_kwh"] = df["real"] / 1000.0
    df["energia_proyectada_kwh"] = df["predicho"] / 1000.0
    df["energia_real_acum_kwh"] = df["energia_real_kwh"].cumsum()
    df["energia_proyectada_acum_kwh"] = df["energia_proyectada_kwh"].cumsum()

    puntos = [
        Point(MEASUREMENT)
        .tag("modelo", args.modelo)
        .field("potencia_real_w", float(r.real))
        .field("potencia_predicha_w", float(r.predicho))
        .field("error_absoluto_w", float(r.error_abs))
        .field("error_porcentual", float(r.error_pct))
        .field("energia_real_kwh", float(r.energia_real_kwh))
        .field("energia_proyectada_kwh", float(r.energia_proyectada_kwh))
        .field("energia_real_acum_kwh", float(r.energia_real_acum_kwh))
        .field("energia_proyectada_acum_kwh", float(r.energia_proyectada_acum_kwh))
        .time(r.timestamp, WritePrecision.NS)
        for r in df.itertuples()
    ]

    with InfluxDBClient(url=cfg["INFLUX_URL"], token=cfg["INFLUX_TOKEN"],
                        org=cfg["INFLUX_ORG"]) as client:
        write_api = client.write_api(write_options=SYNCHRONOUS)
        for i in range(0, len(puntos), 5000):
            write_api.write(bucket=cfg["INFLUX_BUCKET"], org=cfg["INFLUX_ORG"],
                            record=puntos[i:i + 5000])
    print(f"{len(puntos)} puntos enviados a InfluxDB ({cfg['INFLUX_BUCKET']}).")


if __name__ == "__main__":
    main()
