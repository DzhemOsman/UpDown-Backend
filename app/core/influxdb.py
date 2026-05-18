"""InfluxDB Core 3 client wrapper."""

from __future__ import annotations

from typing import Any

import influxdb_client_3 as influxdb3

from app.config import settings


def get_client() -> influxdb3.InfluxDBClient3:
    """Create and return a configured InfluxDB Core 3 client."""
    return influxdb3.InfluxDBClient3(
        host=settings.influxdb_host,
        token=settings.influxdb_token,
        database=settings.influxdb_database,
    )


class InfluxDBRepository:
    """Thin repository layer for reading/writing InfluxDB data."""

    def __init__(self, client: influxdb3.InfluxDBClient3) -> None:
        self._client = client

    def write_point(self, record: Any) -> None:
        """Write a single record.

        ``record`` can be a ``Point``, ``dict``, or line-protocol string.
        See the influxdb3-python docs for supported types.
        """
        self._client.write(record=record)

    def query(self, sql: str) -> list[dict]:
        """Execute a SQL query and return rows as a list of dicts."""
        table = self._client.query(sql)
        return table.to_pydict() if table is not None else []

    def close(self) -> None:
        """Close the underlying client connection."""
        self._client.close()
