from __future__ import annotations

from datetime import datetime
from typing import Iterable

import pandas as pd
import yfinance as yf
from influxdb_client_3 import Point, WritePrecision

from app.repositories.influx_repository import (
    MEASUREMENT,
    write_points,
)

# Standardwerte
DEFAULT_START = datetime(2000, 1, 1)
DEFAULT_END = datetime(2026, 5, 1)


def ingest_ticker(
        ticker: str,
        start: datetime = DEFAULT_START,
        end: datetime = DEFAULT_END,
) -> int:
    """
    Lädt Daten für einen Ticker für den angegebenen Zeitraum aus Yahoo Finance und startet den Schreibprozess
    in InfluxDB.

    :param ticker: Ticker-Symbol für Asset z.B.: 'TSLA'
    :param start: Datum ab den Daten geschrieben werden sollen
    :param end: Datum bis wann die Daten geschrieben werden sollen
    :return: Anzahl der in InfluxDB geschriebenen Ticker (Datenpunkte)
    """
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    if start_str >= end_str:
        print(f"[{ticker}] Startdatum liegt nicht vor Enddatum, nichts zu tun.")
        return 0

    print(f"[{ticker}] Voll-Reload {start_str} bis {end_str}...")

    df = yf.download(
        ticker,
        start=start_str,
        end=end_str,
        progress=False,
        auto_adjust=True,
    )

    if df is None or df.empty:
        print(f"[{ticker}] Keine Daten.")
        return 0

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    points: list[Point] = []
    for ts, row in df.iterrows():
        ts_utc = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
        points.append(
            Point(MEASUREMENT)
            .tag("ticker", ticker)
            .field("open", float(row["Open"]))
            .field("high", float(row["High"]))
            .field("low", float(row["Low"]))
            .field("close", float(row["Close"]))
            .field("volume", int(row["Volume"]))
            .time(ts_utc.to_pydatetime(), WritePrecision.S)
        )

    write_points(points)
    print(f"[{ticker}] {len(points)} Points geschrieben.")
    return len(points)


def ingest_all(tickers: Iterable[str]) -> int:
    """
    Lädt Daten aus Yahoo Finance für eine Liste an Ticker-Symbolen für InfluxDb in dem es für jeden
    Ticker ingest_ticker aufruft und die Anzahl der geschriebenen Datenpunkte summiert.

    :param tickers: List an Ticker-Symbolen z.B., ['MSFT', 'AAPL']
    :return: Anzahl der in InfluxDB geschriebenen Ticker (Datenpunkte)
    """
    total = 0
    for ticker in tickers:
        try:
            total += ingest_ticker(ticker)
        except Exception as exc:
            print(f"[{ticker}] FEHLER: {exc}")
    print(f"Fertig. Gesamt: {total} neue Points")
    return total
