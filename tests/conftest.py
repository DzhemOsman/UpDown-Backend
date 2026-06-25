import pandas as pd
import pytest

import os

# Dummy-Settings, damit "import app.main" nicht an fehlenden Env-Vars scheitert.
# setdefault → überschreibt KEINE echten lokalen Werte in der .env.
os.environ.setdefault("INFLUXDB_HOST", "http://localhost:8086")
os.environ.setdefault("INFLUXDB_TOKEN", "test-token")
os.environ.setdefault("INFLUXDB_DATABASE", "test")


@pytest.fixture
def make_ohlc_df():
    """
    Factory-Fixture: baut synthetische OHLC-DataFrames mit DatetimeIndex.

    Aufruf im Test:
        df = make_ohlc_df([(100, 100, 100, 100), (90, 92, 88, 90)])
    """
    def _factory(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
        index = pd.date_range(start="2020-01-01", periods=len(rows), freq="B")
        return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index)
    return _factory