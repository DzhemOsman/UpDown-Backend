from __future__ import annotations

import functools
import logging
import os
import pickle

import pandas as pd

from app.repositories.influx_repository import get_data_for_ticker_and_range

# Importiere eure bewährten Komponenten
from app.services.feature_engineering import build_features
from app.services.ingestion import DEFAULT_END, DEFAULT_START

logger = logging.getLogger(__name__)
MODEL_DIR = "models"


@functools.lru_cache(maxsize=4)
def get_model(ticker: str):
    """
    Lädt das trainierte LightGBM-Modell einmalig aus der .pkl-Datei.
    Dank lru_cache bleibt das Modell für die nächsten Anfragen im Arbeitsspeicher.
    """
    model_path = os.path.join(MODEL_DIR, f"lgbm_trend_{ticker.lower()}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modelldatei '{model_path}' nicht gefunden. "
            f"Bitte trainiere zuerst das Modell für {ticker}!"
        )

    logger.info(f"[{ticker}] Lade Modell aus Datei {model_path} in den Cache...")
    with open(model_path, "wb" if "w" in "rb" else "rb") as f:
        model = pickle.load(f)
    return model


def predict_ticker_trend(ticker: str, date_str: str) -> dict:
    """
    Sucht die historischen Daten aus der InfluxDB, berechnet die Features zur Laufzeit,
    extrahiert den exakten Tag und trifft die Vorhersage mit dem LightGBM-Modell.
    """
    # 1. Modell gecacht abrufen
    model = get_model(ticker)

    # 2. Rohdaten für die Berechnung aus InfluxDB ziehen (Zielaktie + Indizes)
    df_target = get_data_for_ticker_and_range(ticker, DEFAULT_START, DEFAULT_END)
    df_market = get_data_for_ticker_and_range("^GSPC", DEFAULT_START, DEFAULT_END)
    df_vix = get_data_for_ticker_and_range("^VIX", DEFAULT_START, DEFAULT_END)

    if df_target.empty or df_market.empty or df_vix.empty:
        raise ValueError(
            f"Unvollständige historische Daten in InfluxDB, "
            f"um Features für {ticker} zu berechnen."
        )

    # 3. Datenframes für das Feature-Engineering vereinheitlichen
    for df in [df_target, df_market, df_vix]:
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
        df.columns = [col.lower() for col in df.columns]
        df.sort_index(inplace=True)

    # 4. Transformationsschicht (Feature-Engineering) im RAM ausführen
    df_features = build_features(df_target)

    # Marktkontext-Spalten anfügen (analog zu train.py)
    market_returns = df_market["close"].pct_change(1).rename("market_return_1d")
    vix_close = df_vix["close"].rename("vix_close")

    df_features = df_features.join(market_returns, how="inner")
    df_features = df_features.join(vix_close, how="inner")

    # Nicht benötigte Preisspalten löschen (Das Modell kennt nur relative Indikatoren)
    columns_to_drop = [
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "bollinger_mid",
        "bollinger_std",
        "volume_sma_20",
    ]
    df_model_input = df_features.drop(
        columns=[col for col in columns_to_drop if col in df_features.columns]
    )

    # 5. Den angeforderten Tag aus der Zeitreihe heraussuchen
    target_date = pd.to_datetime(date_str).date()
    row_match = df_model_input[df_model_input.index.date == target_date]

    if row_match.empty:
        raise KeyError(
            f"Für das Datum {date_str} konnten keine ausreichenden Features "
            f"berechnet werden (z.B. wegen fehlender Historie davor)."
        )

    # 6. Einzelne Zeile isolieren und Vorhersage treffen
    X_pred = row_match.iloc[[0]]

    prediction = int(model.predict(X_pred)[0])
    probabilities = model.predict_proba(X_pred)[0]
    probability = float(probabilities[prediction])

    return {
        "ticker": ticker.upper(),
        "date": str(target_date),
        "prediction": prediction,
        "probability": round(probability, 4),
        "market_context": {
            "vix_level": float(X_pred["vix_close"].iloc[0]),
            "market_return_1d": float(X_pred["market_return_1d"].iloc[0]),
        },
    }
