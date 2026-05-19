from datetime import datetime

import pandas as pd
from influxdb_client_3 import Point
from pandas import DatetimeIndex, Series, Timestamp

from app.core.influx import get_client
from app.config import MEASUREMENT
from app.schemas.schemas import QueryRequest


def write_points(points: list[Point]) -> None:
    client = get_client()
    client.write(record=points)


def get_latest_timestamp(ticker: str) -> None | Timestamp | Series | DatetimeIndex:
    client = get_client()

    request = QueryRequest(f"""SELECT MAX(time) AS latest_time FROM {MEASUREMENT}WHERE ticker = '{ticker}'""")

    result = client.query(request.sql)
    if result is None:
        return None

    data = result.to_pydict()
    if not data or not data.get("latest_time"):
        return None

    latest = pd.to_datetime(data["latest_time"][0], utc=True)
    if pd.isna(latest):
        return None

    return pd.to_datetime(data["latest_time"][0], utc=True)

def get_data_for_ticker_and_range(
    ticker: str,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    client = get_client()

    start_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end_date.strftime("%Y-%m-%d %H:%M:%S")

    request = QueryRequest(f"""SELECT * FROM {MEASUREMENT} WHERE ticker = '{ticker}' AND time >= '{start_str}' AND time <= '{end_str}' ORDER BY time""")

    result = client.query(request.sql)
    if result is None:
        return pd.DataFrame()

    data = result.to_pydict()
    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data)