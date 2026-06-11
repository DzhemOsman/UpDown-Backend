from __future__ import annotations

import logging
import os
import pickle
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score

# --- EIGENE IMPORTE ---
from app.services.feature_engineering import build_features
from app.repositories.influx_repository import get_data_for_ticker_and_range
from app.services.ingestion import DEFAULT_START, DEFAULT_END
from app.ml.bulk_ingest import get_diversified_200_tickers

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MARKET_INDEX = "^GSPC"  # S&P 500 als Benchmark
VIX_INDEX = "^VIX"  # Volatilitätsindex

HORIZON_N = 20  # 20 Handelstage Blick in die Zukunft
THRESHOLD_X = 0.05  # Mindestens +5% Rendite für ein Kaufsignal (1)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_trend_diversified.pkl")


def load_data_from_influx(ticker: str) -> pd.DataFrame:
    """ Lädt Rohdaten aus InfluxDB und bereitet den Index vor. """
    df = get_data_for_ticker_and_range(ticker, DEFAULT_START, DEFAULT_END)
    if df.empty:
        raise ValueError(f"Keine Daten für Ticker {ticker} gefunden.")

    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

    df.columns = [col.lower() for col in df.columns]
    df = df.sort_index()
    return df


def prepare_multi_asset_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """
    Lädt das dynamische Universum aus der InfluxDB, berechnet Features
    und stapelt die Daten mathematisch korrekt untereinander.
    """
    # 1. Dynamische Tickerliste von Wikipedia holen (US-Teil)
    try:
        all_tickers = get_diversified_200_tickers()
        # Nur US-Ticker filtern (die ohne "-DE" Suffix), da wir die EU-Daten bewusst weglassen
        tickers = [t for t in all_tickers if "-DE" not in t]
    except Exception as e:
        logger.warning(f"Konnte dynamische Ticker nicht laden, nutze Fallback: {e}")
        tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "MMM", "CAT", "JPM", "AXP"]

    # 2. Globale Markt-Benchmarks vorab laden
    df_market = load_data_from_influx(MARKET_INDEX)
    df_vix = load_data_from_influx(VIX_INDEX)

    market_returns = df_market["close"].pct_change(1).rename("market_return_1d")
    vix_close = df_vix["close"].rename("vix_close")

    all_features = []
    all_targets = []

    # 3. Schleife über das gesamte geladene Universum
    for ticker in tickers:
        try:
            df_target = load_data_from_influx(ticker)

            # Feature Engineering im RAM
            df_features = build_features(df_target)

            # Marktkontext anfügen
            df_features = df_features.join(market_returns, how="inner")
            df_features = df_features.join(vix_close, how="inner")

            # Akademischer Proxy für relative Bewertung (Abstand zum 250-Tage-Hoch)
            df_features["dist_to_250d_high"] = df_features["close"] / df_features["close"].rolling(250).max()

            # Target erstellen (20 Tage in die Zukunft, > 5% Gewinn)
            future_return = df_features["close"].pct_change(HORIZON_N).shift(-HORIZON_N)
            target = (future_return >= THRESHOLD_X).astype(int)
            target.name = "target"

            # Letzte 20 Tage abschneiden, da dort das Target in die Zukunft greift
            valid_indices = future_return.dropna().index
            X_asset = df_features.loc[valid_indices]
            y_asset = target.loc[valid_indices]

            # Unwichtige Rohdatenspalten droppen
            columns_to_drop = ["ticker", "open", "high", "low", "close", "volume", "bollinger_mid", "bollinger_std",
                               "volume_sma_20"]
            X_asset = X_asset.drop(columns=[col for col in columns_to_drop if col in X_asset.columns])

            all_features.append(X_asset)
            all_targets.append(y_asset)
            logger.info(f"💾 Ticker erfolgreich verarbeitet und gestapelt: {ticker}")

        except Exception as e:
            # Deutsche Aktien oder fehlende Ticker werden hier einfach sauber übersprungen
            continue

    if not all_features:
        raise ValueError("Es konnten keine Features aus der InfluxDB generiert werden!")

    # 4. Zusammenheften aller Zeilen zu einer gigantischen Matrix
    X_total = pd.concat(all_features, axis=0).sort_index()
    y_total = pd.concat(all_targets, axis=0).sort_index()

    return X_total, y_total


def train_model():
    """ Trainiert das diversifizierte Multi-Asset-Modell. """
    logger.info("⏳ Starte Laden und Verarbeiten des Multi-Asset-Datensatzes...")
    X, y = prepare_multi_asset_dataset()

    # 5. Chronologischer Train/Test-Split über das gesamte gestapelte Universum
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(f"📊 Gesamt-Trainingsset: {X_train.shape[0]} Zeilen.")
    logger.info(f"📊 Gesamt-Testset: {X_test.shape[0]} Zeilen.")

    # 6. LightGBM trainieren
    logger.info("🚀 Starte LightGBM Multi-Asset-Modelltraining...")
    model = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=6,
        random_state=42,
        verbose=-1
    )

    model.fit(X_train, y_train)

    # 7. Evaluierung auf den unbekannten Testdaten
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    logger.info("=== EVALUIERUNG (DIVERSIFIZIERTES MODELL) ===")
    logger.info(f"Klassifikations-Genauigkeit (Accuracy): {acc:.2%}")
    print(classification_report(y_test, y_pred))

    # 8. Modell persistent abspeichern
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    logger.info(f"🎉 Erfolg! Das diversifizierte Modell wurde unter '{MODEL_PATH}' gesichert.")


if __name__ == "__main__":
    train_model()