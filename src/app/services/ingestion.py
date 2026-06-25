from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterable

import pandas as pd
import yfinance as yf
from influxdb_client_3 import Point, WritePrecision

from app.config import MEASUREMENT
from app.repositories.influx_repository import write_points
from app.services.mean_reversion_strategies.mean_reversion_defaults import (
    DEFAULT_END,
    DEFAULT_START,
)

logger = logging.getLogger(__name__)


def ingest_ticker(
    ticker: str,
    start: datetime = DEFAULT_START,
    end: datetime = DEFAULT_END,
) -> int:
    """
    Lädt Daten für einen Ticker für den angegebenen Zeitraum aus Yahoo Finance
    und startet den Schreibprozess in InfluxDB.

    :param ticker: Ticker-Symbol für Asset z.B.: 'TSLA'
    :param start: Datum ab den Daten geschrieben werden sollen
    :param end: Datum bis wann die Daten geschrieben werden sollen
    :return: Anzahl der in InfluxDB geschriebenen Datenpunkte
    """
    if start >= end:
        logger.warning(
            f"[{ticker}] Startdatum liegt nicht vor Enddatum, nichts zu tun."
        )
        return 0
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    logger.info(f"[{ticker}] Voll-Reload {start_str} bis {end_str}...")

    df = yf.download(
        ticker,
        start=start_str,
        end=end_str,
        progress=False,
        auto_adjust=True,
    )

    if df.empty:
        logger.warning(f"[{ticker}] Keine Daten.")
        return 0

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    # Zeitzonen-Konvertierung, InfluxDB speichert diese mit
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    points: list[Point] = []
    # Erstellt eine List an Punkt mit Yahoo Finance Daten, die in die
    # Datenbank geschrieben werden sollen
    for row in df.itertuples(index=True):
        points.append(
            Point(MEASUREMENT)
            .tag("ticker", ticker)
            .field("open", float(row.Open))
            .field("high", float(row.High))
            .field("low", float(row.Low))
            .field("close", float(row.Close))
            .field("volume", int(row.Volume))
            .time(row.Index.to_pydatetime(), WritePrecision.S)
        )

    write_points(points)
    logger.info(f"[{ticker}] {len(points)} Points geschrieben.")
    return len(points)


def ingest_all(tickers: Iterable[str]) -> int:
    """
    Lädt Daten aus Yahoo Finance für eine Liste an Ticker-Symbolen für InfluxDB
    in dem es für jeden Ticker ingest_ticker aufruft und die Anzahl der
    geschriebenen Datenpunkte summiert.

    :param tickers: List an Ticker-Symbolen z.B., ['MSFT', 'AAPL']
    :return: Anzahl der in InfluxDB geschriebenen Ticker (Datenpunkte)
    """
    total = 0
    for ticker in tickers:
        try:
            total += ingest_ticker(ticker)
        except Exception as exc:
            logger.error(f"[{ticker}] FEHLER beim Ingest: {exc}", exc_info=True)
    logger.info(f"Fertig. Gesamt: {total} neue Points in InfluxDB geschrieben.")
    return total
