"""Tests for the ML prediction endpoint."""

import pytest


def test_predict_returns_result(client):
    response = client.post("/api/v1/predict", json={"features": [1.0, 2.0, 3.0]})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["model_name"] == "dummy-v0"
    assert data["prediction"] == pytest.approx(2.0)


def test_predict_empty_features_returns_422(client):
    response = client.post("/api/v1/predict", json={"features": []})
    assert response.status_code == 422
