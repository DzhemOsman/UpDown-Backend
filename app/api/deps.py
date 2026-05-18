"""Dependency-injection helpers shared across routes."""

from __future__ import annotations

from typing import Generator

from app.core.influxdb import InfluxDBRepository, get_client


def get_influxdb_repo() -> Generator[InfluxDBRepository, None, None]:
    """FastAPI dependency that yields an ``InfluxDBRepository`` and closes it after the request."""
    client = get_client()
    repo = InfluxDBRepository(client)
    try:
        yield repo
    finally:
        repo.close()
