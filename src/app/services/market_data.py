from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from app.core.exceptions import DataSourceError
from app.repositories.influx_repository import get_data_for_ticker_and_range
from app.services.ingestion import ingest_ticker

logger = logging.getLogger(__name__)


def _cache_is_complete(
    df: pd.DataFrame | None,
    start_date: datetime,
    end_date: datetime,
) -> bool:
    """
    Prüft, ob das aus der DB gelesene DataFrame den angefragten Zeitraum
    vollständig abdeckt (Anfang UND Ende, mit Toleranz für Wochenenden/Feiertage).

    :param df: Das aus der DB gelesene DataFrame (oder None bei Lesefehler).
    :param start_date: Angefragtes Startdatum.
    :param end_date: Angefragtes Enddatum.
    :return: True, wenn der Cache den Zeitraum abdeckt, sonst False.
    """
    if df is None or df.empty or "time" not in df.columns:
        return False

    times = pd.to_datetime(df["time"])
    earliest_naive = times.min().tz_localize(None)
    latest_naive = times.max().tz_localize(None)

    start_naive = pd.to_datetime(start_date).tz_localize(None)
    end_naive = pd.to_datetime(end_date).tz_localize(None)

    # Toleranz, da an Wochenenden und Feiertagen keine Börsendaten existieren
    tolerance = pd.Timedelta(days=3)

    start_covered = start_naive >= (earliest_naive - tolerance)
    end_covered = end_naive <= (latest_naive + tolerance)

    return start_covered and end_covered


def fetch_ticker_data(
    ticker: str,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """
    Startet Leseprozess der angeforderten Daten aus der Datenbank. Wenn sie nicht
    (vollständig) vorhanden sind, wird der Schreibprozess gestartet, um die Daten zu
    laden. Anschließend wird erneut versucht, die Daten zu lesen.

    :param ticker: Ticker-Symbol, welches geladen werden soll z.B.: 'PLTR'
    :param start_date: Datum, ab wann Daten gelesen werden sollen
    :param end_date: Datum, bis wann Daten gelesen werden sollen
    :return: Wenn Daten erfolgreich gelesen, gefülltes DataFrame
    ansonsten leeres DataFrame
    """
    if not ticker:
        raise ValueError("ticker must not be empty")
    if start_date >= end_date:
        raise ValueError("start_date must be before end_date")

    # 1. Erster Lese-Versuch: DB-Fehler ist Infrastruktur, NICHT "keine Daten".
    try:
        df = get_data_for_ticker_and_range(ticker, start_date, end_date)
    except Exception as exc:
        logger.error(f"InfluxDB-Lesefehler für {ticker}: {exc}", exc_info=True)
        raise DataSourceError(
            f"Marktdaten für '{ticker}' konnten nicht gelesen werden."
        ) from exc

    if _cache_is_complete(df, start_date, end_date):
        return df

    # 2. Ingestion-Pfad: yfinance-Download + DB-Write absichern.
    logger.info(f"Daten für {ticker} unvollständig oder fehlen. Starte Ingestion...")
    try:
        written = ingest_ticker(ticker, start=start_date, end=end_date)
    except Exception as exc:
        logger.error(f"Ingestion für {ticker} fehlgeschlagen: {exc}", exc_info=True)
        raise DataSourceError(
            f"Marktdaten für '{ticker}' konnten nicht geladen werden."
        ) from exc

    if written == 0:
        # yfinance lieferte nichts -> Ticker existiert vermutlich nicht.
        # Das ist KEIN Infra-Fehler -> leeres DataFrame, später 404.
        return pd.DataFrame()

    # 3. Zweiter Lese-Versuch nach Ingestion ebenfalls absichern.
    try:
        df = get_data_for_ticker_and_range(ticker, start_date, end_date)
    except Exception as exc:
        logger.error(
            f"InfluxDB-Lesefehler nach Ingestion für {ticker}: {exc}", exc_info=True
        )
        raise DataSourceError(
            f"Marktdaten für '{ticker}' konnten nach dem Laden nicht gelesen werden."
        ) from exc

    return df if df is not None else pd.DataFrame()
