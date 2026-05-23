from __future__ import annotations

from datetime import datetime

import pandas as pd

from app.repositories.influx_repository import get_data_for_ticker_and_range
from app.services.ingestion import ingest_ticker


def fetch_ticker_data(
        ticker: str,
        start_date: datetime,
        end_date: datetime,
) -> pd.DataFrame:
    """
    Startet Leseprozess der angeforderten Daten aus der Datenbankn, wenn sie nicht in der Datenbank vorhanden sind,
    wird der Schreibprozess gestartet, um die Daten zu laden. Anschließend wird erneut versucht die Daten zu lesen.

    :param ticker: Ticker-Symbol, welches geladen werden soll z.B.: 'PLTR'
    :param start_date: Datum, ab wann Daten gelesen werden sollen
    :param end_date: Datum, bis wann Daten gelesen werden sollen
    :return: Wenn Daten erfolgreich gelesen, gefülltes DataFrame ansonsten leeres DataFrame
    """
    if not ticker:
        raise ValueError("ticker must not be empty")
    if start_date > end_date:
        raise ValueError("start_date must be before or equal to end_date")

    needs_ingestion = False
    df: pd.DataFrame | None = None
    try:
        df = get_data_for_ticker_and_range(ticker, start_date, end_date)
    except Exception as e:
        print(f"{ticker}: {e}")
        needs_ingestion = True

    if df is None or df.empty:
        needs_ingestion = True
    else:
        if "time" in df.columns:
            earliest_available = pd.to_datetime(df["time"]).min()
            if start_date < earliest_available:
                needs_ingestion = True

    if needs_ingestion:
        written = ingest_ticker(
            ticker,
            start=start_date,
            end=end_date,
        )

        if written == 0:
            return pd.DataFrame()

        df = get_data_for_ticker_and_range(ticker, start_date, end_date)

    return df if df is not None else pd.DataFrame()
