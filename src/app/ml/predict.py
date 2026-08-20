import logging
import os
import pickle

import torch

from app.ml.dataset import MARKET_INDEX, SEQ_LEN, VIX_INDEX, load_raw_influx

# --- EIGENE IMPORTE ---
from app.ml.model import MarketLSTM
from app.services.feature_engineering import build_features

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- KONFIGURATION (Muss mit train.py übereinstimmen) ---
HIDDEN_SIZE = 128
NUM_LAYERS = 2
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "pytorch_lstm_market.pth")
SCALER_PATH = os.path.join(MODEL_DIR, "lstm_scaler.pkl")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_ticker(ticker: str):
    """Macht eine Vorhersage für eine einzelne Aktie basierend
    auf den letzten 60 Tagen."""
    device = get_device()

    # 1. Prüfen, ob Modell und Scaler existieren
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        logger.error(
            "Modell oder Scaler nicht gefunden! Bitte trainiere das Modell zuerst."
        )
        return

    # 2. Scaler laden
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # 3. Marktdaten (Benchmarks) laden
    logger.info(f"🔍 Lade Daten für {ticker}...")
    df_market = load_raw_influx(MARKET_INDEX)
    df_vix = load_raw_influx(VIX_INDEX)
    if df_market.empty or df_vix.empty:
        logger.error(
            "Fehler: Globale Marktdaten (^GSPC oder ^VIX) fehlen in der InfluxDB."
        )
        return

    market_returns = df_market["close"].pct_change(1).rename("market_return_1d")
    vix_close = df_vix["close"].rename("vix_close")

    # 4. Ticker-Daten laden und Features bauen (Exakt wie in dataset.py)
    df_target = load_raw_influx(ticker)
    if df_target.empty or len(df_target) < 250 + SEQ_LEN:
        logger.error(
            f"Nicht genug Historie für {ticker} "
            f"(Mindestens 310 Tage nötig für Indikatoren)."
        )
        return

    df_feat = build_features(df_target)
    df_feat = df_feat.join(market_returns, how="inner").join(vix_close, how="inner")

    # 250-Tage Hoch berechnen (Verursacht NaNs in den ersten 249 Tagen)
    df_feat["dist_to_250d_high"] = (
        df_feat["close"] / df_feat["close"].rolling(250).max()
    )

    # Unwichtige Spalten droppen
    cols_to_drop = [
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
    X_df = df_feat.drop(columns=[c for c in cols_to_drop if c in df_feat.columns])

    # NaNs entfernen (Die ersten ~250 Tage fliegen hier raus)
    X_df = X_df.dropna()

    if len(X_df) < SEQ_LEN:
        logger.error(
            f"Nach Bereinigung der NaNs bleiben für {ticker} "
            f"weniger als {SEQ_LEN} Tage übrig."
        )
        return

    # 5. Wir brauchen nur die allerletzten 60 Tage für die Live-Prognose!
    X_recent = X_df.iloc[-SEQ_LEN:]

    # 6. Skalieren mit dem GELERNTEN Maßstab aus dem Training
    X_scaled = scaler.transform(X_recent)

    # 7. In 3D Tensor umwandeln: Shape (Batch=1, Seq=60, Features=18)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    # 8. Modell initialisieren und Gewichte laden
    # Wir lesen die Input-Size dynamisch aus dem Tensor ab (bei euch 18 Features)
    model = MarketLSTM(
        input_size=X_tensor.shape[-1],
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout_rate=0.0,  # Beim Vorhersagen brauchen wir keinen Dropout
    ).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()  # Setzt Modell in den Inferenz-Modus

    # 9. Vorhersage generieren (ohne Gradientenberechnung)
    with torch.no_grad():
        logit = model(X_tensor)
        probability = torch.sigmoid(logit).item()

    # 10. Ergebnis formatieren
    prediction_class = "KAUFEN (1)" if probability >= 0.5 else "NICHT KAUFEN (0)"

    print("\n" + "=" * 50)
    print(f"📈 LIVE PROGNOSE FÜR: {ticker}")
    print("=" * 50)
    print(f"Letztes analysiertes Datum : {X_recent.index[-1].strftime('%Y-%m-%d')}")
    print(f"Modell Entscheidung        : {prediction_class}")
    print(f"Wahrscheinlichkeit für +5% : {probability:.2%}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    # Teste die Pipeline mit ein paar bekannten Tickern
    test_tickers = ["AAPL", "MSFT", "TSLA"]
    for t in test_tickers:
        predict_ticker(t)
