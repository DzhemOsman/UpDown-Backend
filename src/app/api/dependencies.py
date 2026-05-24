import influxdb_client_3 as influxdb3

from app.core.influx import get_client


def get_influx_client() -> influxdb3.InfluxDBClient3:
    return get_client()