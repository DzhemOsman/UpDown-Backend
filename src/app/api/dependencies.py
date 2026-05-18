import influxdb_client_3 as influxdb3
from app.config import settings

_client = None

def get_influx_client() -> influxdb3.InfluxDBClient3:
    global _client
    if _client is None:
        _client = influxdb3.InfluxDBClient3(
            host=settings.INFLUXDB_HOST,
            token=settings.INFLUXDB_TOKEN,
            database=settings.INFLUXDB_DATABASE,
        )
    return _client