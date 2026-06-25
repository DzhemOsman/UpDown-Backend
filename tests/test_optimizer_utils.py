import pytest

from app.services.mean_reversion_strategies.optimizer_utils import trades_to_metrics


def _trade(profit_abs: float) -> dict:
    """
    Minimaler Trade – trades_to_metrics liest NUR profit_abs.
    """
    return {"profit_abs": profit_abs}


def test_empty_trades_returns_zeroes():
    """
    Ohne Trades darf NICHTS gerechnet werden (sonst ZeroDivision).
    """
    roi, profit, win_rate, n = trades_to_metrics([], initial_capital=10_000)

    assert roi == 0.0
    assert profit == 0.0
    assert win_rate == 0.0
    assert n == 0


def test_mixed_trades_metrics():
    """
    2 Gewinner, 1 Verlierer auf 10.000 Kapital.

    Summe = 300 + 200 - 100 = 400
    ROI = 400 / 10.000 · 100 = 4,0 %
    WinRate = 2 von 3 > 0 = 66,67 %
    """
    trades = [_trade(300.0), _trade(200.0), _trade(-100.0)]

    roi, profit, win_rate, n = trades_to_metrics(trades, initial_capital=10_000)

    assert profit == pytest.approx(400.0)
    assert roi == pytest.approx(4.0)
    assert win_rate == pytest.approx(66.6667, abs=1e-3)
    assert n == 3
