import pytest

from app.schemas.internal.best_parameter_combination_dict import (
    ParameterCombinationDict,
)
from app.services.mean_reversion_strategies.money_management_reversion import (
    MeanReversionWithMoneyManagement,
)


def _backtest_single(df, **kwargs):
    """
    Injiziert df in den Cache (kein DB-Zugriff) und ruft den Kern-Backtest auf.
    """
    bot = MeanReversionWithMoneyManagement()  # initial_capital = 10_000 (Default)
    bot.set_ticker_cache({"TEST": df})
    return bot._backtest_with_money_management("TEST", **kwargs)


def _trade(buy_date: str, sell_date: str, profit_pct: float) -> dict:
    """
    Minimaler 'potenzieller Trade', wie ihn der innere Backtest liefert.

    Die Orchestrierung liest nur buy_date, sell_date, profit_pct und SETZT
    invested_capital + profit_abs selbst – der Rest ist Beiwerk.
    """
    return {
        "ticker": "TEST",
        "buy_date": buy_date,
        "sell_date": sell_date,
        "days_held": 1,
        "exit_reason": "Take Profit",
        "entry_price": 100.0,
        "exit_price": 110.0,
        "profit_pct": profit_pct,
        "profit_abs": 0.0,  # wird neu berechnet
    }


def _params_mm(**overrides) -> ParameterCombinationDict:
    base = dict(
        drop_threshold=5,
        lookback_days=1,
        hold_days=5,
        take_profit_pct=5,
        fee_pct=0.0,
        stop_loss_pct=10.0,
        max_positions=1,
        allocation_pct=100.0,
    )
    base.update(overrides)
    return ParameterCombinationDict(**base)


def test_entry_day_spike_does_not_take_profit(make_ohlc_df):
    """
    REGRESSION-CHARAKTERISIERUNG: Am Einstiegstag (i=0) ist KEIN Exit möglich.

    Tag 2 (Einstieg) hat ein High von 100 – das würde das Take-Profit-Ziel (94.5)
    locker reissen. Weil can_exit_intraday erst ab i>0 True wird, wird dieser Gewinn
    IGNORIERT. Der Trade läuft weiter und schliesst per Time Stop im Minus.
    Genau dieses Verhalten ist der Verdacht hinter den negativen Trades.
    """
    df = make_ohlc_df(
        [
            (100, 100, 100, 100),
            (90, 92, 88, 90),  # -10 % -> Signal (idx=1)
            (
                90,
                100,
                89,
                91,
            ),  # Einstieg (i=0): High 100 → wäre Take Profit, wird ignoriert
            (91, 92, 89, 90),  # i=1
            (90, 91, 86, 88),  # i=2 = letzter Haltetag -> Time Stop @ close 88
        ]
    )

    trades = _backtest_single(
        df,
        drop_threshold_pct=5,
        lookback_days=1,
        hold_days=3,
        take_profit_pct=5,
        fee_rate=0.0,
        stop_loss_pct=20,  # bewusst weit weg -> Stop darf nicht stören
    )

    assert len(trades) == 1
    trade = trades[0]
    # Der springende Punkt: KEIN "Take Profit", obwohl Tag 2 das Ziel erreicht hätte.
    assert trade["exit_reason"] == "Time Stop"
    assert trade["exit_price"] == 88.0
    assert trade["profit_abs"] < 0


def test_stop_loss_exit(make_ohlc_df):
    """
    Stop-Loss feuert ab i>0, wenn das Low unter den Stop-Preis faellt.

    Einstieg @ 90, stop_loss_pct=5 → Stop-Preis = 85.5. An Tag 3 (i=1) fällt das
    Low auf 85. Da der Open (89) NICHT unter dem Stop liegt, wird zum Stop-Preis
    85.5 verkauft (nicht zum schlechteren Open).
    """
    df = make_ohlc_df(
        [
            (100, 100, 100, 100),
            (90, 92, 88, 90),  # Signal
            (90, 91, 89, 90),  # Einstieg (i=0)
            (89, 90, 85, 86),  # i=1: Low 85 <= Stop 85.5 -> Stop Loss @ 85.5
            (86, 87, 85, 86),
        ]
    )

    trades = _backtest_single(
        df,
        drop_threshold_pct=5,
        lookback_days=1,
        hold_days=3,
        take_profit_pct=50,  # bewusst hoch → Take Profit darf NICHT zuerst feuern
        fee_rate=0.0,
        stop_loss_pct=5,
    )

    assert len(trades) == 1
    trade = trades[0]
    assert trade["exit_reason"] == "Stop Loss"
    assert trade["exit_price"] == 85.5
    assert trade["days_held"] == 1
    assert trade["profit_pct"] == pytest.approx(-5.0)


def test_money_management_requires_risk_params(make_ohlc_df):
    """
    run_portfolio_with_money_management lehnt fehlende MM-Pflichtfelder ab.

    Ohne stop_loss_pct/max_positions/allocation_pct ist Money Management nicht
    definiert → die Methode wirft bewusst ValueError, statt still Unsinn zu rechnen.
    """
    bot = MeanReversionWithMoneyManagement()
    bot.set_ticker_cache({"TEST": make_ohlc_df([(100, 100, 100, 100)])})

    params = ParameterCombinationDict(
        drop_threshold=5,
        lookback_days=1,
        hold_days=3,
        take_profit_pct=5,
        fee_pct=0.0,
        stop_loss_pct=None,  # <- fehlt absichtlich
        max_positions=1,
        allocation_pct=10.0,
    )

    with pytest.raises(ValueError):
        bot.run_portfolio_with_money_management(["TEST"], params)


def test_max_positions_blocks_overlapping_trade():
    """
    Bei max_positions=1 wird ein zweites, zeitlich überlappendes Signal verworfen.
    """
    bot = MeanReversionWithMoneyManagement(initial_capital=10_000)

    potential = [
        _trade("2020-01-01", "2020-01-10", profit_pct=10.0),  # läuft bis 10.01.
        _trade(
            "2020-01-05", "2020-01-15", profit_pct=10.0
        ),  # startet WÄHREND Trade 1 läuft
    ]
    # Inneren Backtest durch die kontrollierten Trades ersetzen → wir testen
    # NUR die Orchestrierung, keine Marktdaten, keine DB.
    bot._backtest_with_money_management = lambda ticker, **kwargs: potential

    executed = bot.run_portfolio_with_money_management(
        ["TEST"], _params_mm(max_positions=1, allocation_pct=50.0)
    )

    assert len(executed) == 1
    assert (
        executed[0]["buy_date"] == "2020-01-01"
    )  # nur der erste, früher startende Trade


def test_capital_release_enables_compounding():
    """
    Schliesst Trade 1, fliesst sein Kapital INKL. Gewinn zurück und wird
    voll in Trade 2 reinvestiert (Compounding).
    """
    bot = MeanReversionWithMoneyManagement(initial_capital=10_000)

    potential = [
        _trade("2020-01-01", "2020-01-05", profit_pct=10.0),  # schliesst VOR Trade 2
        _trade("2020-01-10", "2020-01-20", profit_pct=10.0),
    ]
    bot._backtest_with_money_management = lambda ticker, **kwargs: potential

    executed = bot.run_portfolio_with_money_management(
        ["TEST"], _params_mm(max_positions=1, allocation_pct=100.0)
    )

    assert len(executed) == 2
    # Trade 1: all-in auf 10.000 → +10 % = 1.000
    assert executed[0]["invested_capital"] == 10_000.0
    assert executed[0]["profit_abs"] == pytest.approx(1000.0)
    # Trade 2: 10.000 + 1.000 wurden freigegeben → 11.000 all-in → +10 % = 1.100
    assert executed[1]["invested_capital"] == 11_000.0
    assert executed[1]["profit_abs"] == pytest.approx(1100.0)
