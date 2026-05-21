from datetime import datetime

import pandas as pd
import time

from app.repositories.influx_repository import get_data_for_ticker_and_range
from app.services.ingestion import ingest_ticker


def fetch_ticker_data(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    if not ticker:
        raise ValueError("ticker must not be empty")
    if start_date > end_date:
        raise ValueError("start_date must be before or equal to end_date")

    df = get_data_for_ticker_and_range(ticker, start_date, end_date)
    if df is None or df.empty:
        ingest_ticker(ticker, start=start_date.strftime("%Y-%m-%d"))
        time.sleep(1)
        df = get_data_for_ticker_and_range(ticker, start_date, end_date)

    return df