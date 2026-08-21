from datetime import datetime

import pandas as pd
from influxdb_client_3 import Point

from app.config import MEASUREMENT
from app.core.influx import get_client


def write_points(points: list[Point]) -> None:
    """
    Schreibt gelieferte Points in Datenbank

    :param points: OHLCV-Daten eines Tickers
    :return: None
    """
    client = get_client()
    client.write(record=points)


def get_ingested_tickers() -> list[str]:
    """Liest alle Ticker-Symbole, für die bereits Daten in InfluxDB liegen.

    Dient als robuster Fallback für Trainingsskripte, wenn externe
    Ticker-Quellen (z.B. Wikipedia in bulk_ingest.py) nicht erreichbar sind -
    das bereits ingestierte Universum ist für ein Retraining ohnehin die
    relevante Grundmenge.
    """
    client = get_client()
    sql = f"SELECT DISTINCT ticker FROM '{MEASUREMENT}'"
    result = client.query(query=sql)

    if result is None:
        return []

    data = result.to_pydict()
    return list(data.get("ticker", []))


def _build_ticker_query(ticker, start_str, end_str) -> tuple[str, dict]:
    sql = (
        f"SELECT * FROM '{MEASUREMENT}' WHERE ticker = $ticker "
        f"AND time >= $start AND time <= $end ORDER BY time"
    )
    return sql, {"ticker": ticker, "start": start_str, "end": end_str}


def get_data_for_ticker_and_range(
    ticker: str,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """
    Liest Daten für angeforderten Ticker aus der Datenbank, die im angegebenen
    Zeitraum liegen.

    :param ticker: Ticker-Symbol, welches geladen werden soll z.B.: 'PLTR'
    :param start_date: Datum, ab wann Daten gelesen werden sollen
    :param end_date: Datum, bis wann Daten gelesen werden sollen
    :return: Wenn Daten erfolgreich gelesen, gefülltes DataFrame
    ansonsten leeres DataFrame
    """
    client = get_client()

    start_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end_date.strftime("%Y-%m-%d %H:%M:%S")

    sql, params = _build_ticker_query(ticker, start_str, end_str)
    result = client.query(query=sql, query_parameters=params)

    if result is None:
        return pd.DataFrame()

    data = result.to_pydict()
    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data)
