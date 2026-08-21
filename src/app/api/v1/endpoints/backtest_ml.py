import json
import logging
import os
import pickle
import traceback
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler

from app.core.exceptions import DataSourceError
from app.ml.dataset import COLS_TO_DROP, SEQ_LEN
from app.ml.model import MarketLSTM
from app.schemas.internal.strategy_result_dict import StrategyResultDict
from app.schemas.internal.trade_result_dict import TradeResultDict
from app.services.feature_engineering import build_features
from app.services.market_data import fetch_ticker_data
from app.services.mean_reversion_strategies.strategy_calculations import (
    calculate_strategy_result,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(redirect_slashes=True)

MARKET_INDEX = "^GSPC"
VIX_INDEX = "^VIX"

MODEL_DIR = "models"
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "pytorch_lstm_market.pth")
LSTM_SCALER_PATH = os.path.join(MODEL_DIR, "lstm_scaler.pkl")
LGBM_MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_trend_diversified.pkl")
LGBM_META_PATH = os.path.join(MODEL_DIR, "lgbm_trend_diversified.meta.json")
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2

# Ab wie vielen Monaten "Abstand" zwischen Modell-Trainingsende und
# Backtest-Ende die Staleness-Warnung greift.
STALENESS_WARNING_MONTHS = 6.0

DEFAULT_THRESHOLD = 0.5


class BacktestRequest(BaseModel):
    tickers: List[str]
    start_date: datetime
    end_date: datetime
    initial_capital: float
    fee_pct: float
    hold_days: int = 5
    take_profit_pct: float = 3.0
    strategy_name: str
    is_lstm: bool = False
    is_lightgbm: bool = False


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
    df.columns = [col.lower() for col in df.columns]
    df = df.sort_index()
    return df[~df.index.duplicated(keep="first")]


def _load_market_context(
    start: datetime, end: datetime
) -> tuple[pd.Series | None, pd.Series | None]:
    """Lädt ^GSPC/^VIX einmal pro Request, exakt wie in dataset.py/predict.py
    fürs Training."""
    try:
        df_market = fetch_ticker_data(MARKET_INDEX, start, end)
        df_vix = fetch_ticker_data(VIX_INDEX, start, end)
    except DataSourceError as exc:
        # Infra-Fehler (InfluxDB/yfinance) darf nicht den ganzen Request killen:
        # ohne Markt-Kontext bleiben die ML-Signale leer, der Aufrufer loggt das.
        logger.error(f"Markt-/VIX-Kontext nicht ladbar: {exc}")
        return None, None

    if df_market is None or df_market.empty or df_vix is None or df_vix.empty:
        return None, None

    df_market = _normalize_ohlcv(df_market)
    df_vix = _normalize_ohlcv(df_vix)

    market_returns = df_market["close"].pct_change(1).rename("market_return_1d")
    vix_close = df_vix["close"].rename("vix_close")
    return market_returns, vix_close


def _resolve_threshold_and_staleness_warning(
    meta: dict | None,
    request_end: datetime,
    staleness_months: float = STALENESS_WARNING_MONTHS,
) -> tuple[float, str | None]:
    """Liest den datengetrieben empfohlenen Threshold aus den Trainings-Metadaten
    (models/*.meta.json, siehe train_lgbm.py) statt einen hartkodierten Wert zu
    raten, und warnt, wenn der angefragte Backtest-Zeitraum deutlich über das
    Trainings-Enddatum des Modells hinausgeht (Concept-Drift-Risiko). Reine
    Funktion, ohne InfluxDB-/Modell-Zugriff testbar."""
    if not meta:
        return (
            DEFAULT_THRESHOLD,
            "Keine Modell-Metadaten gefunden - nutze Fallback-Threshold 0.5.",
        )

    threshold = float(meta.get("recommended_threshold", DEFAULT_THRESHOLD))
    warning = None

    data_end_raw = meta.get("data_end")
    if data_end_raw:
        data_end = datetime.fromisoformat(data_end_raw)
        gap_months = (request_end.replace(tzinfo=None) - data_end).days / 30.44
        if gap_months > staleness_months:
            warning = (
                f"Modell ist {gap_months:.1f} Monate älter als das Backtest-Ende "
                f"(trainiert bis {data_end.date()}, Backtest bis "
                f"{request_end.date()}). Vorhersagen ggf. weniger zuverlässig "
                f"(Concept Drift)."
            )

    return threshold, warning


class MLModelManager:
    """Lädt die von train.py erzeugten Artefakte (Scaler, LSTM-Gewichte, LightGBM)
    und führt die tatsächliche Inferenz durch. Fehlt ein Artefakt, bleibt das
    jeweilige Modell None und liefert bewusst KEINE Signale (statt Zufallswerten),
    damit ein Ladefehler nicht als "funktionierendes" Ergebnis missverstanden wird."""

    def __init__(self):
        self.seq_len = SEQ_LEN
        self.feature_names: list[str] | None = None
        self.scaler: MinMaxScaler | None = None
        self.lstm_model: MarketLSTM | None = None
        self.lgbm_model = None
        self.lgbm_meta: dict | None = None
        self._load_models()

    def _load_models(self):
        try:
            if os.path.exists(LSTM_SCALER_PATH):
                with open(LSTM_SCALER_PATH, "rb") as f:
                    self.scaler = pickle.load(f)
                self.feature_names = list(self.scaler.feature_names_in_)
                logger.info(
                    f"Scaler geladen ({len(self.feature_names)} Features): "
                    f"{self.feature_names}"
                )
            else:
                logger.error(f"Scaler nicht gefunden unter: {LSTM_SCALER_PATH}")
        except Exception:
            logger.error("Fehler beim Laden des Scalers:\n" + traceback.format_exc())

        try:
            if self.feature_names and os.path.exists(LSTM_MODEL_PATH):
                state_dict = torch.load(
                    LSTM_MODEL_PATH, map_location=torch.device("cpu")
                )
                model = MarketLSTM(
                    input_size=len(self.feature_names),
                    hidden_size=LSTM_HIDDEN_SIZE,
                    num_layers=LSTM_NUM_LAYERS,
                    dropout_rate=0.0,
                )
                model.load_state_dict(state_dict)
                model.eval()
                self.lstm_model = model
                logger.info("PyTorch LSTM-Modell geladen und einsatzbereit.")
            else:
                logger.error(
                    f"LSTM-Modell konnte nicht geladen werden (Scaler geladen: "
                    f"{self.feature_names is not None}, Datei vorhanden: "
                    f"{os.path.exists(LSTM_MODEL_PATH)})."
                )
        except Exception:
            logger.error(
                "Fehler beim Laden des LSTM-Modells:\n" + traceback.format_exc()
            )

        try:
            if os.path.exists(LGBM_MODEL_PATH):
                with open(LGBM_MODEL_PATH, "rb") as f:
                    self.lgbm_model = pickle.load(f)
                logger.info("LightGBM-Modell geladen und einsatzbereit.")
            else:
                logger.error(f"LightGBM-Modell nicht gefunden unter: {LGBM_MODEL_PATH}")
        except Exception:
            logger.error(
                "Fehler beim Laden des LightGBM-Modells:\n" + traceback.format_exc()
            )

        try:
            if os.path.exists(LGBM_META_PATH):
                with open(LGBM_META_PATH, "r", encoding="utf-8") as f:
                    self.lgbm_meta = json.load(f)
                logger.info(
                    f"LightGBM-Metadaten geladen (empfohlener Threshold: "
                    f"{self.lgbm_meta.get('recommended_threshold')}, Trainingsdaten "
                    f"bis: {self.lgbm_meta.get('data_end')})."
                )
            else:
                logger.warning(
                    f"Keine Trainings-Metadaten für LightGBM gefunden "
                    f"({LGBM_META_PATH}), nutze Fallback-Threshold "
                    f"{DEFAULT_THRESHOLD} (kein Retraining-Skript gelaufen? "
                    f"-> python -m app.ml.train_lgbm)."
                )
        except Exception:
            logger.error(
                "Fehler beim Laden der LightGBM-Metadaten:\n" + traceback.format_exc()
            )

    def _align_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_names:
            raise RuntimeError("Feature-Reihenfolge unbekannt (Scaler nicht geladen).")
        missing = [c for c in self.feature_names if c not in features_df.columns]
        if missing:
            raise RuntimeError(f"Fehlende Feature-Spalten für die Inferenz: {missing}")
        return features_df[self.feature_names]

    def predict_lstm(self, raw_features_df: pd.DataFrame) -> tuple[np.ndarray, int]:
        if self.lstm_model is None or self.scaler is None:
            return np.array([]), 0

        ordered = self._align_features(raw_features_df)
        scaled = self.scaler.transform(ordered).astype(np.float32)

        if len(scaled) < self.seq_len:
            return np.array([]), 0

        windows = np.stack(
            [
                scaled[i : i + self.seq_len]
                for i in range(len(scaled) - self.seq_len + 1)
            ]
        )
        tensor_data = torch.tensor(windows, dtype=torch.float32)

        with torch.no_grad():
            logits = self.lstm_model(tensor_data)
            probs = torch.sigmoid(logits).numpy().flatten()

        return probs, (self.seq_len - 1)

    def predict_lightgbm(self, raw_features_df: pd.DataFrame) -> tuple[np.ndarray, int]:
        if self.lgbm_model is None:
            return np.array([]), 0

        ordered = self._align_features(raw_features_df)
        probs = self.lgbm_model.predict_proba(ordered)[:, 1]
        return probs, 0


ml_manager = MLModelManager()


@router.post("/unified", response_model=StrategyResultDict)
def run_unified_backtest(req: BacktestRequest):
    if req.is_lstm and req.is_lightgbm:
        raise HTTPException(
            status_code=422,
            detail="Bitte genau ein ML-Modell wählen: is_lstm oder is_lightgbm.",
        )

    try:
        all_trades: List[TradeResultDict] = []
        ticker_cache: dict[str, pd.DataFrame] = {}

        # Kalender-Puffer vor dem eigentlichen Start-Datum. Muss groß genug sein für:
        # ~50 Handelstage Indikator-Warmup (build_features) + 250 Handelstage für
        # dist_to_250d_high + 60 Handelstage LSTM-Sequenzfenster (~360 Handelstage).
        lookback_start = req.start_date - timedelta(days=600)
        naive_start_date = req.start_date.replace(tzinfo=None)

        market_returns, vix_close = (None, None)
        if req.is_lstm or req.is_lightgbm:
            market_returns, vix_close = _load_market_context(
                lookback_start, req.end_date
            )
            if market_returns is None or vix_close is None:
                logger.error(
                    "Markt-/VIX-Kontext (^GSPC/^VIX) konnte nicht aus InfluxDB "
                    "geladen werden. ML-Signale bleiben für alle Ticker leer, bis "
                    "diese Daten vorhanden sind."
                )

        # Threshold + Staleness-Check einmal pro Request auflösen (nicht pro Ticker),
        # um Log-Spam zu vermeiden. LSTM hat noch kein eigenes meta.json (siehe Plan-
        # Ausblick), bleibt vorerst beim festen Default-Threshold.
        ml_threshold = DEFAULT_THRESHOLD
        if req.is_lightgbm:
            ml_threshold, staleness_warning = _resolve_threshold_and_staleness_warning(
                ml_manager.lgbm_meta, req.end_date
            )
            logger.info(f"LightGBM-Inferenz-Threshold: {ml_threshold:.3f}")
            if staleness_warning:
                logger.warning(f"{staleness_warning}")

        for ticker in req.tickers:
            # Service-Schicht statt rohem Repository-Read: fehlt der Zeitraum in
            # InfluxDB (neuer Ticker oder veralteter Cache), lädt fetch_ticker_data
            # ihn per yfinance nach. lookback_start sorgt dafür, dass auch der
            # Warmup-Puffer (Indikatoren, 250d-Fenster, LSTM-Sequenz) mitkommt.
            try:
                df = fetch_ticker_data(ticker, lookback_start, req.end_date)
            except DataSourceError as exc:
                logger.warning(f"[{ticker}] Daten nicht ladbar, übersprungen: {exc}")
                continue

            if df is None or df.empty:
                continue

            df = _normalize_ohlcv(df)

            if "volume" not in df.columns:
                logger.warning(
                    f"[{ticker}] Spalte 'volume' fehlt in der Datenbank! "
                    f"Überspringe Ticker."
                )
                continue

            df_index_naive = (
                df.index.tz_localize(None) if df.index.tz is not None else df.index
            )
            ticker_cache[ticker] = df[df_index_naive >= naive_start_date].copy()

            if req.is_lstm or req.is_lightgbm:
                features = build_features(df)
                if features.empty:
                    logger.warning(
                        f"[{ticker}] Zu wenig Historie für Feature-Engineering, "
                        f"übersprungen."
                    )
                    continue

                if market_returns is not None and vix_close is not None:
                    features = features.join(market_returns, how="inner").join(
                        vix_close, how="inner"
                    )
                features["dist_to_250d_high"] = (
                    features["close"] / features["close"].rolling(250).max()
                )
                features = features.dropna()

                if features.empty:
                    logger.warning(
                        f"[{ticker}] Nach Markt-Join und 250-Tage-Fenster keine "
                        f"Zeilen mehr übrig (zu kurze Historie für den gewählten "
                        f"Zeitraum), übersprungen."
                    )
                    continue

                inference_features = features.drop(
                    columns=[c for c in COLS_TO_DROP if c in features.columns]
                )

                try:
                    if req.is_lstm:
                        probabilities, offset = ml_manager.predict_lstm(
                            inference_features
                        )
                    else:
                        probabilities, offset = ml_manager.predict_lightgbm(
                            inference_features
                        )
                except RuntimeError as model_err:
                    logger.error(f"[{ticker}] ML-Inferenz fehlgeschlagen: {model_err}")
                    probabilities, offset = np.array([]), 0

                if len(probabilities) > 0:
                    logger.info(
                        f"[{ticker}] ML-Wahrscheinlichkeiten - "
                        f"Min: {probabilities.min():.4f}, "
                        f"Max: {probabilities.max():.4f}, "
                        f"Avg: {probabilities.mean():.4f}, Anzahl: {len(probabilities)}"
                    )
                else:
                    logger.warning(
                        f"[{ticker}] Keine Wahrscheinlichkeiten berechnet "
                        f"(Modell nicht geladen oder zu kurze Historie nach "
                        f"Feature-Engineering)."
                    )

                df["ml_signal"] = 0.0
                if len(probabilities) > 0:
                    target_indices = features.index[offset:]
                    df.loc[target_indices, "ml_signal"] = probabilities

                signal_indices = np.where(df["ml_signal"] > ml_threshold)[0]

            else:
                signal_indices = []

            open_prices = df["open"].to_numpy()
            high_prices = df["high"].to_numpy()
            close_prices = df["close"].to_numpy()
            dates = df_index_naive
            last_exit_index = -1

            for idx in signal_indices:
                entry_idx = idx + 1
                if entry_idx <= last_exit_index or entry_idx >= len(df):
                    continue

                entry_date = dates[entry_idx]
                if entry_date < naive_start_date:
                    continue

                raw_entry_price = open_prices[entry_idx]
                effective_entry_price = raw_entry_price * (1 + req.fee_pct)
                target_price = raw_entry_price * (1 + req.take_profit_pct / 100)

                raw_exit_price = 0.0
                exit_reason = ""
                days_held = 0
                found_exit = False

                for i in range(0, req.hold_days):
                    current_idx = entry_idx + i
                    if current_idx >= len(df):
                        break

                    current_high = high_prices[current_idx]
                    current_open = open_prices[current_idx]
                    current_close = close_prices[current_idx]

                    if current_high >= target_price:
                        raw_exit_price = (
                            current_open
                            if (i > 0 and current_open > target_price)
                            else target_price
                        )
                        exit_reason = "Take Profit"
                        days_held = i
                        found_exit = True
                    elif i == (req.hold_days - 1):
                        raw_exit_price = current_close
                        exit_reason = "Time Stop"
                        days_held = i
                        found_exit = True

                    if found_exit:
                        last_exit_index = current_idx
                        exit_date = dates[current_idx]
                        effective_exit_price = raw_exit_price * (1 - req.fee_pct)
                        profit_pct = (
                            effective_exit_price - effective_entry_price
                        ) / effective_entry_price
                        profit_abs = req.initial_capital * profit_pct

                        all_trades.append(
                            {
                                "ticker": ticker,
                                "buy_date": entry_date.strftime("%Y-%m-%d"),
                                "sell_date": exit_date.strftime("%Y-%m-%d"),
                                "days_held": int(days_held),
                                "exit_reason": exit_reason,
                                "entry_price": float(round(raw_entry_price, 2)),
                                "exit_price": float(round(raw_exit_price, 2)),
                                "profit_pct": float(round(profit_pct * 100, 2)),
                                "profit_abs": float(round(profit_abs, 2)),
                                "invested_capital": float(req.initial_capital),
                            }
                        )
                        break

        return calculate_strategy_result(
            all_trades, ticker_cache, int(req.initial_capital)
        )

    except Exception as e:
        logger.error("\n" + "=" * 60)
        logger.error("KRITISCHER PYTHON ABSTURZ IN BACKTEST_ML:")
        logger.error(traceback.format_exc())
        logger.error("=" * 60 + "\n")
        raise HTTPException(status_code=500, detail=f"Backend Error: {str(e)}")
