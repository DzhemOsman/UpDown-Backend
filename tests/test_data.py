"""Tests for data endpoints (using a mocked InfluxDB repository)."""

from unittest.mock import MagicMock, patch

import pytest

from app.api.deps import get_influxdb_repo
from app.main import app


@pytest.fixture
def mock_repo():
    repo = MagicMock()
    app.dependency_overrides[get_influxdb_repo] = lambda: repo
    yield repo
    app.dependency_overrides.clear()


def test_write_data_point(client, mock_repo):
    payload = {
        "measurement": "temperature",
        "tags": {"location": "office"},
        "fields": {"value": 22.5},
    }
    response = client.post("/api/v1/data/write", json=payload)
    assert response.status_code == 201
    assert response.json()["message"] == "Data point written successfully"
    mock_repo.write_point.assert_called_once()


def test_write_data_point_influxdb_error(client, mock_repo):
    mock_repo.write_point.side_effect = RuntimeError("connection refused")
    payload = {
        "measurement": "temperature",
        "tags": {},
        "fields": {"value": 1.0},
    }
    response = client.post("/api/v1/data/write", json=payload)
    assert response.status_code == 502


def test_query_data(client, mock_repo):
    mock_repo.query.return_value = [{"time": "2024-01-01", "value": 42.0}]
    response = client.post(
        "/api/v1/data/query",
        json={"sql": "SELECT * FROM temperature LIMIT 1"},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["value"] == 42.0


def test_query_data_influxdb_error(client, mock_repo):
    mock_repo.query.side_effect = RuntimeError("query failed")
    response = client.post(
        "/api/v1/data/query",
        json={"sql": "SELECT * FROM nonexistent"},
    )
    assert response.status_code == 502
