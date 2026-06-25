import pytest

from app.services.mean_reversion_strategies.strategy_calculations import (
    calculate_strategy_result,
)


def _trade(profit_abs: float, sell_date: str = "2020-01-10") -> dict:
    """
    Trade-Stub. calculate_strategy_result aggregiert profit_abs;
    sell_date wird nur von der (hier leeren) Equity-Kurve gebraucht.
    """
    return {"profit_abs": profit_abs, "sell_date": sell_date}


def test_empty_trades_no_division_error():
    """
    Ohne Trades: win_rate=0, keine ZeroDivision, sauberes Null-Ergebnis.
    """
    result = calculate_strategy_result(
        trades=[], ticker_data={}, initial_capital=10_000
    )

    assert result["total_profit"] == 0.0
    assert result["roi_pct"] == 0.0
    assert result["win_rate"] == 0.0
    assert result["total_number_of_trades"] == 0


def test_aggregates_profit_and_roi():
    """
    3 Trades, ticker_data leer -> Equity-Kurve ist [], aber die Kennzahlen
    müssen trotzdem korrekt aggregiert werden.
    """
    trades = [_trade(300.0), _trade(200.0), _trade(-100.0)]

    result = calculate_strategy_result(
        trades=trades, ticker_data={}, initial_capital=10_000
    )

    assert result["total_profit"] == pytest.approx(400.0)
    assert result["roi_pct"] == pytest.approx(4.0)
    assert result["win_rate"] == pytest.approx(66.67, abs=1e-2)
    assert result["total_number_of_trades"] == 3
    assert result["equity_curve_data"] == []  # leere ticker_data -> leere Kurve
