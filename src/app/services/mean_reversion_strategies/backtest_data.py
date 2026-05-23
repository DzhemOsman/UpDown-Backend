import logging
from datetime import datetime

import pandas as pd

from app.services.market_data import fetch_ticker_data

logger = logging.getLogger(__name__)


def get_backtest_data(
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        is_optimized: bool
) -> pd.DataFrame | None:
    """
    Lädt OHLC-Daten aus InfluxDB und bringt sie in das Format,
    das der Backtest erwartet: DatetimeIndex + Spalten Open/High/Low/Close.

    :param ticker: Ticker-Symbol, welches geladen werden soll z.B.: 'PLTR'
    :param start_date: Datum, ab wann Daten gelesen werden sollen
    :param end_date: Datum, bis wann Daten gelesen werden sollen
    :param is_optimized: Boolean, ob ursprünglicher oder neuer Algorithmus
    :return: Wenn Daten geliefert werden ein DataFrame ansonsten None
    """
    try:
        df = fetch_ticker_data(ticker, start_date, end_date)
    except Exception as e:
        logger.error(f"Fehler beim Laden von {ticker}: {e}", exc_info=True)
        return None

    if df is None or df.empty:
        logger.warning(f"Keine Daten für Ticker: {ticker}")
        return None

    try:
        # InfluxDB liefert Zeitzonen mit, deshalb muss die Lokalisierung zuerst entfernt werden.
        df.index = pd.DatetimeIndex(df["time"]).tz_localize(None)

        # Benötigte Spalten markieren für Extrahierung
        columns_to_keep = ["open", "high", "close"]
        if is_optimized:
            columns_to_keep.append("low")
            
        clean_df = df[columns_to_keep].copy()

        # Fehlende Werte mit letztem bekannten Wert füllen und df nach Index sortieren
        clean_df = clean_df.sort_index().ffill()
        
        return clean_df
    except Exception as e:
        logger.error(f"Fehler beim Aufbereiten von {ticker}: {e}", exc_info=True)
        return None
