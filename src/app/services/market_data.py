from datetime import datetime

import pandas as pd

from app.repositories.influx_repository import get_data_for_ticker_and_range


def fetch_ticker_data(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    if not ticker:
        raise ValueError("ticker must not be empty")
    if start_date > end_date:
        raise ValueError("start_date must be before or equal to end_date")

    return get_data_for_ticker_and_range(ticker, start_date, end_date)