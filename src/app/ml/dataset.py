from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

from app.ml.bulk_ingest import get_diversified_200_tickers
from app.repositories.influx_repository import get_data_for_ticker_and_range
from app.services.feature_engineering import build_features
from app.services.ingestion import DEFAULT_START

logger = logging.getLogger(__name__)

MARKET_INDEX = "^GSPC"
VIX_INDEX = "^VIX"
HORIZON_N = 20
THRESHOLD_X = 0.05
SEQ_LEN = 60  # Eure Lookback-Periode (S)

# Spalten, die zwar fürs Feature-Engineering/Labeling gebraucht werden, aber
# nicht Teil des trainierten Feature-Vektors sind. Muss exakt mit der
# Inferenz-Seite (backtest_ml.py) übereinstimmen, sonst driften Training und
# Inferenz auseinander.
COLS_TO_DROP = [
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


def load_raw_influx(ticker: str, end_date: datetime | None = None) -> pd.DataFrame:
    """Hilfsfunktion zum Laden der vorverarbeiteten Influx-Daten.

    end_date default ist "jetzt", NICHT ein hartkodiertes Datum - ein
    Trainingslauf darf sich nicht künstlich an einem alten Stichtag deckeln,
    sonst driftet das Modell wieder unbemerkt vom aktuellen Marktgeschehen ab.
    """
    end = end_date or datetime.now()
    df = get_data_for_ticker_and_range(ticker, DEFAULT_START, end)
    if df.empty:
        return pd.DataFrame()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
    df.columns = [col.lower() for col in df.columns]
    return df.sort_index()


def load_labeled_features_per_ticker(
    tickers: list[str], end_date: datetime | None = None
) -> list[tuple[str, pd.DataFrame, pd.Series]]:
    """Lädt & labelt Features pro Ticker (inkl. Markt-/VIX-Kontext und
    dist_to_250d_high), OHNE Scaling oder Windowing. Gemeinsame Grundlage für
    den LSTM-Tensor-Builder (build_tensor_dataset) und das LightGBM-Training
    (train_lgbm.py), damit Feature-/Label-Logik nicht an mehreren Stellen
    auseinanderdriftet.
    """
    df_market = load_raw_influx(MARKET_INDEX, end_date)
    df_vix = load_raw_influx(VIX_INDEX, end_date)
    if df_market.empty or df_vix.empty:
        raise RuntimeError("Globale Marktdaten (^GSPC/^VIX) fehlen in der InfluxDB.")
    market_returns = df_market["close"].pct_change(1).rename("market_return_1d")
    vix_close = df_vix["close"].rename("vix_close")

    results: list[tuple[str, pd.DataFrame, pd.Series]] = []

    logger.info("Berechne Features für das Universum...")
    for ticker in tickers:
        df_target = load_raw_influx(ticker, end_date)
        if df_target.empty or len(df_target) < (SEQ_LEN + HORIZON_N + 50):
            continue

        df_feat = build_features(df_target)
        df_feat = df_feat.join(market_returns, how="inner").join(vix_close, how="inner")
        df_feat["dist_to_250d_high"] = (
            df_feat["close"] / df_feat["close"].rolling(250).max()
        )

        future_return = df_feat["close"].pct_change(HORIZON_N).shift(-HORIZON_N)
        target = (future_return >= THRESHOLD_X).astype(int)

        valid_idx = future_return.dropna().index
        X_df = df_feat.loc[valid_idx]
        y_sr = target.loc[valid_idx]

        X_df = X_df.drop(columns=[c for c in COLS_TO_DROP if c in X_df.columns])

        # Sämtliche Zeilen entfernen, die durch Indikatoren (z.B. 250-Tage-Fenster)
        # NaNs enthalten - verhindert NaN-Loss/-Splits weiter unten.
        clean_idx = X_df.dropna().index
        X_df = X_df.loc[clean_idx]
        y_sr = y_sr.loc[clean_idx]

        if X_df.empty:
            continue

        results.append((ticker, X_df, y_sr))

    return results


def build_tensor_dataset() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, MinMaxScaler
]:
    """
    1. Lädt alle Aktien im RAM.
    2. Fitted den MinMaxScaler strikt NUR auf den chronologischen Train-Anteilen.
    3. Schneidet Sliding Windows pro Aktie isoliert aus.
    """
    try:
        all_tickers = [t for t in get_diversified_200_tickers() if "-DE" not in t]
    except Exception:
        all_tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "JPM"]

    valid_assets_data = [
        (X_df, y_sr) for _, X_df, y_sr in load_labeled_features_per_ticker(all_tickers)
    ]

    if not valid_assets_data:
        raise RuntimeError("Keine ausreichend langen Asset-Daten gefunden!")

    # --- SCHRITT 2: Leak-Free Scaler Fitting ---
    train_slices = []
    for X_df, _ in valid_assets_data:
        split_idx = int(len(X_df) * 0.8)
        train_slices.append(X_df.iloc[:split_idx])

    global_train_matrix = pd.concat(train_slices, axis=0)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(global_train_matrix)
    logger.info("MinMaxScaler erfolgreich NUR auf globalen Trainingsdaten gefittet.")

    # --- SCHRITT 3: Windowing pro Ticker isoliert ---
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    for X_df, y_sr in valid_assets_data:
        split_idx = int(len(X_df) * 0.8)

        # Skalierung anwenden
        X_scaled = scaler.transform(X_df)
        y_vals = y_sr.values

        num_windows = len(X_scaled) - SEQ_LEN + 1

        for i in range(num_windows):
            window_x = X_scaled[i : i + SEQ_LEN]  # Shape: (60, 24)
            window_y = y_vals[i + SEQ_LEN - 1]  # Target am Ende des Fensters

            # Zeitlicher Check: Gehört dieses Fenster ins Train- oder Testset?
            current_time_idx = i + SEQ_LEN - 1
            if current_time_idx < split_idx:
                X_train_list.append(window_x)
                y_train_list.append(window_y)
            else:
                X_test_list.append(window_x)
                y_test_list.append(window_y)

    # --- SCHRITT 4: Umwandlung in PyTorch 3D Tensors ---
    X_train_t = torch.tensor(np.array(X_train_list), dtype=torch.float32)
    y_train_t = torch.tensor(np.array(y_train_list), dtype=torch.float32).unsqueeze(1)

    X_test_t = torch.tensor(np.array(X_test_list), dtype=torch.float32)
    y_test_t = torch.tensor(np.array(y_test_list), dtype=torch.float32).unsqueeze(1)

    logger.info(f"Tensoren fertig! X_train Shape: {X_train_t.shape}")
    return X_train_t, y_train_t, X_test_t, y_test_t, scaler


class MarketDataset(Dataset):
    """
    Standard PyTorch Dataset für Zeitreihen-Tensoren.
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    @staticmethod
    def compute_pos_weight(y_train_tensor: torch.Tensor) -> torch.Tensor:
        """
        Berechnet das optimale 'pos_weight' für die PyTorch BCEWithLogitsLoss.
        Verhindert, dass das Netz bei ungleich verteilten Klassen faul wird.
        """
        positives = torch.sum(y_train_tensor == 1.0)
        negatives = torch.sum(y_train_tensor == 0.0)

        if positives == 0:
            return torch.tensor([1.0], dtype=torch.float32)

        weight = negatives / positives
        logger.info(
            f"Klassen-Verteilung berechnet: Positives={int(positives)}, "
            f"Negatives={int(negatives)}"
        )
        logger.info(f"Empfohlenes Loss pos_weight: {weight.item():.4f}")

        return weight.clone().detach().to(torch.float32)
