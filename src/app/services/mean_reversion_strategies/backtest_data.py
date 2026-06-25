import logging
from datetime import datetime

import pandas as pd

from app.services.market_data import fetch_ticker_data

logger = logging.getLogger(__name__)


def get_backtest_data(
    ticker: str, start_date: datetime, end_date: datetime, include_low: bool
) -> pd.DataFrame | None:
    """
    Lädt OHLC-Daten aus InfluxDB und bringt sie in das Format,
    das der Backtest erwartet: DatetimeIndex + Spalten Open/High/Low/Close.

    :param ticker: Ticker-Symbol, welches geladen werden soll z.B.: 'PLTR'
    :param start_date: Datum, ab wann Daten gelesen werden sollen
    :param end_date: Datum, bis wann Daten gelesen werden sollen
    :param include_low: Boolean, ob ursprünglicher oder neuer Algorithmus
    :return: Wenn Daten geliefert werden ein DataFrame ansonsten None
    """
    df = fetch_ticker_data(ticker, start_date, end_date)

    if df.empty:
        # KEIN Infra-Fehler: Ticker existiert nicht / hat keine Daten im Zeitraum.
        # None = "diesen Ticker überspringen" → wichtig für Multi-Ticker-Optimierung.
        logger.warning(f"Keine Daten für Ticker: {ticker}")
        return None

    try:
        # InfluxDB liefert Zeitzonen mit, deshalb muss die Lokalisierung zuerst
        # entfernt werden.
        df.index = pd.DatetimeIndex(df["time"]).tz_localize(None)

        columns_to_keep = ["open", "high", "close"]
        if include_low:
            columns_to_keep.append("low")

        clean_df = df[columns_to_keep].copy()
        clean_df = clean_df.sort_index()

        # Tage ohne handelbaren Kurs (NaN in einer Preisspalt-) entfernen.
        # Begründung: Ein fehlender Kurs ist kein auffüllbarer Wert, sondern ein
        # Tag, an dem real nicht gehandelt werden konnte. Würden wir ffill/bfill
        # nutzen, erlaubten wir dem Backtest Trades zu erfundenen Preisen.
        # dropna entfernt genau diese Zeilen → die NumPy-Arrays im Hot-Loop
        # enthalten danach garantiert keine NaN mehr.
        clean_df = clean_df.dropna(subset=columns_to_keep)

        if clean_df.empty:
            # Nach dem Bereinigen blieb nichts übrig → wie "keine Daten" behandeln.
            logger.warning(
                f"Keine handelbaren Kursdaten für Ticker {ticker} nach NaN-Bereinigung."
            )
            return None

        return clean_df
    except (KeyError, ValueError) as e:
        logger.error(f"Fehler beim Aufbereiten von {ticker}: {e}", exc_info=True)
        return None
