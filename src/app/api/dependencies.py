import influxdb_client_3 as influxdb3
from fastapi import Request


def get_influx_client(request: Request) -> influxdb3.InfluxDBClient3:
    return request.app.state.influx_client
