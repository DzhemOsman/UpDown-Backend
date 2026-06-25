import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.v1.endpoints import strategy as strategy_endpoint

# Kein "with" → lifespan läuft NICHT → kein InfluxDB-Verbindungsversuch beim Start.
client = TestClient(app)


def _valid_body(**overrides) -> dict:
    body = {
        "tickers": ["AAPL"],
        "drop_option": 5.0,
        "lookback_days": 3,
        "hold_option": 5,
        "take_profit_option": 3.0,
    }
    body.update(overrides)
    return body


def test_mean_reversion_returns_200_and_valid_schema(monkeypatch):
    """
    Gültiger Request → 200, Response erfüllt das StrategyResult-Schema.

    Die Strategie-Klasse wird ersetzt, damit KEINE DB noetig ist – wir testen
    nur Routing, Validierung und Serialisierung der Antwort.
    """
    fake_result = {
        "total_profit": 400.0,
        "roi_pct": 4.0,
        "win_rate": 66.67,
        "total_number_of_trades": 3,
        "equity_curve_data": [],
        "trades": [],
    }

    class FakeBot:
        def __init__(self, **kwargs):
            pass

        def run_portfolio_single(self, tickers, params):
            return fake_result

    # Den Namen ERSETZEN, den der Endpoint benutzt (bot = MeanReversionStrategy(...)).
    monkeypatch.setattr(strategy_endpoint, "MeanReversionStrategy", FakeBot)

    response = client.post("/v1/strategy/mean-reversion", json=_valid_body())

    assert response.status_code == 200
    data = response.json()
    # Die Pflichtfelder des Response-Schemas muessen vorhanden und korrekt sein.
    assert data["total_profit"] == 400.0
    assert data["roi_pct"] == 4.0
    assert data["total_number_of_trades"] == 3
    assert data["trades"] == []


def test_mean_reversion_rejects_invalid_drop_option():
    """
    drop_option = 0 verletzt die Regel gt=0 → Pydantic gibt 422 zurück.

    KEIN Mock nötig: Die Validierung greift VOR dem Endpoint-Code, die DB
    wird nie angefasst.
    """
    response = client.post("/v1/strategy/mean-reversion", json=_valid_body(drop_option=0))

    assert response.status_code == 422


def test_mean_reversion_rejects_empty_tickers():
    """
    Leere Tickerliste verletzt min_length=1 → 422.
    """
    response = client.post("/v1/strategy/mean-reversion", json=_valid_body(tickers=[]))

    assert response.status_code == 422