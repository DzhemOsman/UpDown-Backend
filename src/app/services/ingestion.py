from __future__ import annotations

from datetime import datetime
from typing import Iterable

import pandas as pd
import yfinance as yf
from influxdb_client_3 import Point, WritePrecision

from app.repositories.influx_repository import MEASUREMENT, get_latest_timestamp, write_points

START_DATE = "2000-01-01"


def ingest_ticker(ticker: str, start: str = START_DATE) -> int:
    print(f"[DEBUG] ingest_ticker aufgerufen mit: ticker='{ticker}'")  # NEU

    if not ticker or not isinstance(ticker, str):
        print(f"[{ticker}] FEHLER: Ungültiger Ticker")


    latest = get_latest_timestamp(ticker)
    if latest is not None:
        start = (latest + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"[{ticker}] Cache vorhanden bis {latest.date()}, lade ab {start}...")
    else:
        print(f"[{ticker}] Erste Befüllung ab {start}...")

    end = datetime.today().strftime("%Y-%m-%d")
    if start >= end:
        print(f"[{ticker}] Bereits aktuell, nichts zu tun.")
        return 0

    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
    )

    if df is None or df.empty:
        print(f"[{ticker}] Keine neuen Daten.")
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


def ingest_all(tickers: Iterable[str]) -> None:
    total = 0
    for ticker in tickers:
        try:
            total += ingest_ticker(ticker,start=start)
        except Exception as exc:
            print(f"[{ticker}] FEHLER: {exc}")
    print(f"\n--- Fertig. Gesamt: {total} neue Points ---")
