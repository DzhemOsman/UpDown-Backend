from __future__ import annotations

import logging
import os
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score

# --- EIGENE IMPORTE ---
from app.services.feature_engineering import build_features
from app.repositories.influx_repository import get_data_for_ticker_and_range
from app.services.ingestion import DEFAULT_START, DEFAULT_END

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Konfiguration gemäss unserer Analyse
TARGET_TICKER = "AAPL"  # Die Aktie, auf die wir trainieren wollen
MARKET_INDEX = "^GSPC"  # S&P 500 als Benchmark
VIX_INDEX = "^VIX"      # Volatilitätsindex

HORIZON_N = 20          # 20 Handelstage Blick in die Zukunft
THRESHOLD_X = 0.05      # Mindestens +5% Rendite für ein Kaufsignal (1)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, f"lgbm_trend_{TARGET_TICKER.lower()}.pkl")


def load_data_from_influx(ticker: str) -> pd.DataFrame:
    """
    Lädt Rohdaten aus der InfluxDB über das offizielle Repository.
    Setzt den Datumsindex korrekt, damit Feature-Engineering und Merging funktionieren.
    """
    logger.info(f"[{ticker}] Lade Rohdaten aus InfluxDB...")

    # Echte Abfrage an die InfluxDB senden
    df = get_data_for_ticker_and_range(
        ticker=ticker,
        start_date=DEFAULT_START,
        end_date=DEFAULT_END
    )

    if df.empty:
        raise ValueError(f"Keine Daten für Ticker {ticker} in der InfluxDB gefunden! Bitte vorher ingesten.")

    # InfluxDB-Rückgabe aufbereiten (Zeitspalte als Index setzen)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

    # Spaltennamen klein schreiben, da unser Feature-Engineering 'close' statt 'Close' erwartet
    df.columns = [col.lower() for col in df.columns]

    # Nach Datum sortieren (absolut kritisch für Zeitreihen-Splits!)
    df = df.sort_index()

    return df


def prepare_training_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """
    Lädt alle benötigten Assets, berechnet Features, führt den Marktkontext
    zusammen und baut das zukunftsgerichtete Target ohne Look-Ahead-Bias.
    """
    # 1. Daten laden
    df_target = load_data_from_influx(TARGET_TICKER)
    df_market = load_data_from_influx(MARKET_INDEX)
    df_vix = load_data_from_influx(VIX_INDEX)

    # 2. Basis-Features für die Zielaktie berechnen
    logger.info(f"Berechne technische Indikatoren für {TARGET_TICKER}...")
    df_features = build_features(df_target)

    # 3. Marktkontext berechnen und hinzufügen
    logger.info("Führe globalen Marktkontext zusammen...")

    # Markt-Rendite berechnen (1-Tages-Rendite des S&P 500)
    df_market = df_market.sort_index()
    market_returns = df_market["close"].pct_change(1).rename("market_return_1d")

    # VIX-Stand extrahieren
    vix_close = df_vix["close"].rename("vix_close")

    # Über den Datums-Index an den Feature-DataFrame der Aktie anfügen
    df_features = df_features.join(market_returns, how="inner")
    df_features = df_features.join(vix_close, how="inner")

    # 4. Das mittelfristige Target bauen (20 Tage in die Zukunft, > 5% Gewinn)
    logger.info(f"Erstelle Target-Labels (Horizont: {HORIZON_N} Tage, Hürde: {THRESHOLD_X * 100}%)")

    # PROZENTUALE RENDITE IN N TAGEN (Zukunftsgerichtet)
    future_return = df_features["close"].pct_change(HORIZON_N).shift(-HORIZON_N)

    # Binäres Label erstellen: 1 wenn Rendite >= 5%, sonst 0
    target = (future_return >= THRESHOLD_X).astype(int)
    target.name = "target"

    # Da die letzten N Zeilen keine Zukunftsdaten haben können (NaN),
    # müssen wir diese aus den Features und dem Target herausschneiden.
    valid_indices = future_return.dropna().index
    X = df_features.loc[valid_indices]
    y = target.loc[valid_indices]

    # Sicherheitscheck: Kursspalten entfernen, die das Modell nicht sehen darf
    columns_to_drop = ["ticker", "open", "high", "low", "close", "volume", "bollinger_mid", "bollinger_std",
                       "volume_sma_20"]
    X = X.drop(columns=[col for col in columns_to_drop if col in X.columns])

    return X, y


def train_model():
    """
    Orchestriert den Daten-Load, den zeitlich korrekten Split,
    das Training von LightGBM und das Speichern der Model-Datei.
    """
    try:
        X, y = prepare_training_dataset()
    except Exception as e:
        logger.error(f"Fehler bei der Datenvorbereitung: {e}")
        return

    # 5. Zeitlich korrekter Train/Test-Split
    split_idx = int(len(X) * 0.8)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(f"Trainings-Set Grösse: {X_train.shape[0]} Zeilen (Zeitraum: {X_train.index.min().date()} bis {X_train.index.max().date()})")
    logger.info(f"Test-Set Grösse: {X_test.shape[0]} Zeilen (Zeitraum: {X_test.index.min().date()} bis {X_test.index.max().date()})")

    # 6. LightGBM initialisieren und trainieren
    logger.info("Starte LightGBM Modelltraining...")
    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        verbose=-1
    )

    model.fit(X_train, y_train)

    # 7. Evaluierung auf den unbekannten Testdaten
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    logger.info("=== EVALUIERUNG (UNBEKANNTE TESTDATEN) ===")
    logger.info(f"Klassifikations-Genauigkeit (Accuracy): {acc:.2%}")
    print(classification_report(y_test, y_pred))

    # 8. Modell als .pkl abspeichern
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    logger.info(f"🎉 Erfolg! Das trainierte Modell wurde unter '{MODEL_PATH}' gesichert.")


if __name__ == "__main__":
    train_model()