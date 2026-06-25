import pandas as pd
import pytest

from app.services.mean_reversion_strategies.mean_reversion_strategy import (
    MeanReversionStrategy,
)


def _run_single_ticker(df: pd.DataFrame, **params) -> list[dict]:
    """
    Injiziert df in den Cache (kein DB-Zugriff) und ruft den Kern-Backtest auf.

    Jeder Test bekommt seinen EIGENEN df, weil _backtest die Spalte 'change'
    in-place ergänzt – so kann keine Mutation zwischen Tests durchsickern.
    """
    bot = MeanReversionStrategy()           # initial_capital = 10_000 (Default)
    bot._ticker_cache["TEST"] = df
    return bot._backtest("TEST", **params)


def test_take_profit_exit(make_ohlc_df):
    # Tag 1 fällt um 10 % → Signal. Einstieg Tag 2 (open=90).
    # Ziel = 90 · 1.10 = 99. Tag 3 erreicht high=100 → Exit bei 99.
    df = make_ohlc_df([
        (100, 100, 100, 100),
        (90,   92,  88,  90),
        (90,   95,  89,  94),   # Einstiegstag (i=0)
        (96,  100,  95,  99),   # Take Profit erreicht (i=1)
        (99,   99,  99,  99),
        (99,   99,  99,  99),
    ])

    trades = _run_single_ticker(
        df,
        drop_threshold_pct=5,
        lookback_days=1,
        hold_days=5,
        take_profit_pct=10,
        fee_rate=0.0,
    )

    assert len(trades) == 1
    trade = trades[0]
    assert trade["exit_reason"] == "Take Profit"
    assert trade["entry_price"] == 90.0
    assert trade["exit_price"] == 99.0
    assert trade["profit_pct"] == pytest.approx(10.0)
    assert trade["profit_abs"] == pytest.approx(1000.0)


def test_time_stop_exit(make_ohlc_df):
    # Signal an Tag 1, Take-Profit (50 %) wird nie erreicht.
    # hold_days=3 -> Exit am letzten Haltetag zum Schlusskurs (close=85).
    df = make_ohlc_df([
        (100, 100, 100, 100),
        (90,   92,  88,  90),
        (90,   91,  89,  90),   # Einstiegstag (i=0)
        (90,   92,  88,  88),   # i=1
        (88,   89,  84,  85),   # i=2 = letzter Haltetag -> Time Stop
    ])

    trades = _run_single_ticker(
        df,
        drop_threshold_pct=5,
        lookback_days=1,
        hold_days=3,
        take_profit_pct=50,
        fee_rate=0.0,
    )

    assert len(trades) == 1
    trade = trades[0]
    assert trade["exit_reason"] == "Time Stop"
    assert trade["days_held"] == 2
    assert trade["entry_price"] == 90.0
    assert trade["exit_price"] == 85.0
    assert trade["profit_abs"] == pytest.approx(-555.56)


def test_no_signal_no_trades(make_ohlc_df):
    # Kurs steigt monoton → nie ein Drop > 5 % → kein Signal → keine Trades.
    df = make_ohlc_df([
        (100, 101,  99, 100),
        (101, 102, 100, 101),
        (102, 103, 101, 102),
        (103, 104, 102, 103),
        (104, 105, 103, 104),
    ])

    trades = _run_single_ticker(
        df,
        drop_threshold_pct=5,
        lookback_days=1,
        hold_days=5,
        take_profit_pct=10,
        fee_rate=0.0,
    )

    assert trades == []