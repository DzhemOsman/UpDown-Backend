from datetime import datetime

import pandas as pd
from influxdb_client_3 import Point

from app.config import MEASUREMENT
from app.core.influx import get_client
from app.schemas.api.query_request import QueryRequest


def write_points(points: list[Point]) -> None:
    """
    Schreibt gelieferte Points in Datenbank

    :param points: OHLCV-Daten eines Tickers
    :return: None
    """
    client = get_client()
    client.write(record=points)


def get_data_for_ticker_and_range(
        ticker: str,
        start_date: datetime,
        end_date: datetime,
) -> pd.DataFrame:
    """
    Liest Daten für angeforderten Ticker aus der Datenbank, die im angegebenen Zeitraum liegen.

    :param ticker: Ticker-Symbol, welches geladen werden soll z.B.: 'PLTR'
    :param start_date: Datum, ab wann Daten gelesen werden sollen
    :param end_date: Datum, bis wann Daten gelesen werden sollen
    :return: Wenn Daten erfolgreich gelesen, gefülltes DataFrame ansonsten leeres DataFrame
    """
    client = get_client()

    start_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end_date.strftime("%Y-%m-%d %H:%M:%S")

    request = QueryRequest(
        sql=f"SELECT * FROM '{MEASUREMENT}' WHERE ticker = '{ticker}' AND time >= '{start_str}' AND time <= '{end_str}' ORDER BY time")

    result = client.query(request.sql)
    if result is None:
        return pd.DataFrame()

    data = result.to_pydict()
    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data)
