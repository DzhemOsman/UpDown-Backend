from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nimmt einen Rohdaten-DataFrame (OHLCV) aus InfluxDB entgegen und berechnet
    zur Laufzeit alle mathematischen Features für das LightGBM-Modell.

    WICHTIG: Vermeidet Look-Ahead-Bias durch strikte Nutzung historischer Daten.
    """
    # Kopie erstellen, um das Original nicht zu verändern
    features_df = df.copy().sort_index()

    if features_df.empty or len(features_df) < 50:
        logger.warning("DataFrame zu kurz für stabiles Feature-Engineering.")
        return features_df

    # ---------------------------------------------------------
    # 1. Momentum & Returns (Zeithorizonte: 1, 3, 5, 10, 20 Tage)
    # ---------------------------------------------------------
    for days in [1, 3, 5, 10, 20]:
        features_df[f"return_{days}d"] = features_df["close"].pct_change(days)

    # ---------------------------------------------------------
    # 2. Trend-Indikatoren (SMA & relatives Verhältnis zum Kurs)
    # ---------------------------------------------------------
    for window in [10, 20, 50]:
        features_df[f"sma_{window}"] = (
            features_df["close"].rolling(window=window).mean()
        )
        # Relatives Verhältnis: Wie weit ist der Kurs prozentual vom SMA entfernt?
        features_df[f"close_to_sma_{window}"] = (
            features_df["close"] / features_df[f"sma_{window}"]
        )

    # ---------------------------------------------------------
    # 3. Mean-Reversion: RSI (Relative Strength Index - 14 Tage)
    # ---------------------------------------------------------
    delta = features_df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)  # 1e-9 verhindert Division durch Null
    features_df["rsi_14"] = 100 - (100 / (1 + rs))

    # 4. Mean-Reversion: Bollinger Bänder (20 Tage, 2 Standardabweichungen)
    features_df["bollinger_mid"] = features_df["sma_20"]
    features_df["bollinger_std"] = features_df["close"].rolling(window=20).std()
    # Position im Band: Wo steht der Kurs relativ zu den Bändern?
    features_df["bollinger_position"] = (
        features_df["close"] - features_df["bollinger_mid"]
    ) / (2 * features_df["bollinger_std"] + 1e-9)

    # ---------------------------------------------------------
    # 5. Volatilität (Standardabweichung der täglichen Renditen)
    # ---------------------------------------------------------
    features_df["volatility_20d"] = features_df["return_1d"].rolling(window=20).std()

    # ---------------------------------------------------------
    # 6. Volumen-Features (Aktuelles Volumen relativ zum 20-Tage-Schnitt)
    # ---------------------------------------------------------
    features_df["volume_sma_20"] = features_df["volume"].rolling(window=20).mean()
    features_df["volume_ratio"] = features_df["volume"] / (
        features_df["volume_sma_20"] + 1e-9
    )

    # ---------------------------------------------------------
    # Bereinigung: Zeilen mit NaN-Werten löschen (entstehen durch rollierende Fenster)
    # ---------------------------------------------------------
    features_df = features_df.dropna()

    logger.info(
        f"Feature-Engineering abgeschlossen. {features_df.shape[1]} Spalten generiert."
    )
    return features_df
