from functools import lru_cache

import influxdb_client_3 as influxdb3

from app.config import get_settings


@lru_cache(maxsize=1)
def get_client() -> influxdb3.InfluxDBClient3:
    settings = get_settings()
    return influxdb3.InfluxDBClient3(
        host=settings.INFLUXDB_HOST,
        token=settings.INFLUXDB_TOKEN.get_secret_value(),
        database=settings.INFLUXDB_DATABASE,
    )
