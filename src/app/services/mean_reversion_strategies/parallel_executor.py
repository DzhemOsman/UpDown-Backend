import pandas as pd

from app.schemas.internal.best_parameter_combination_dict import ParameterCombinationDict, Combo
from app.services.mean_reversion_strategies.mean_reversion_defaults import DEFAULT_FEE_RATE, DEFAULT_LOOKBACK_DAYS
from app.services.mean_reversion_strategies.money_management_reversion import MeanReversionWithMoneyManagement

_WORKER_DATA: dict[str, pd.DataFrame] = {}
_WORKER_CFG: tuple = ()


def _init_worker(ticker_data, initial_capital, is_kadane, is_trend):
    global _WORKER_DATA, _WORKER_CFG
    _WORKER_DATA = ticker_data
    _WORKER_CFG = (initial_capital, is_kadane, is_trend)


def _evaluate(combo: Combo) -> tuple[Combo, float, float, float, int]:
    """
    Wertet EINE Parameterkombination in einem Worker-Prozess aus.
    Greift auf die durch _init_worker gesetzten Modul-Globals zu und gibt
    nur Kennzahlen zurück (keine Trades -> minimaler Pickle-Overhead).

    :param combo: Tupel (drop, hold, tp, sl, max_pos, alloc), Parameter Kombinationen.
    :return: (combo, roi_pct, profit_abs, win_rate_pct, anzahl_trades)
    """
    initial_capital, is_kadane, is_trend = _WORKER_CFG

    bot = MeanReversionWithMoneyManagement(initial_capital=initial_capital)
    # Vorgeladene Daten injizieren -> _get_ticker_data_for_backtest trifft den Cache,
    # es findet KEIN InfluxDB-Zugriff im Worker statt.
    bot.set_ticker_cache(_WORKER_DATA)

    current_params = ParameterCombinationDict(
        drop_threshold=combo.drop_threshold,
        lookback_days=DEFAULT_LOOKBACK_DAYS,
        hold_days=combo.hold_days,
        take_profit_pct=combo.take_profit_pct,
        stop_loss_pct=combo.stop_loss_pct,
        max_positions=combo.max_positions,
        allocation_pct=combo.allocation_pct,
        fee_pct=DEFAULT_FEE_RATE,
    )

    trades = bot.run_portfolio_with_money_management(
        list(_WORKER_DATA), current_params, is_kadane, is_trend
    )

    if trades:
        current_trades_df = pd.DataFrame(trades)
        current_profit = float(current_trades_df['profit_abs'].sum())
        current_roi = (current_profit / initial_capital) * 100
        # Pandas Best Practice: Wahrheitswerte (True/False) als Mean berechnen
        current_win_rate = float((current_trades_df['profit_abs'] > 0).mean() * 100)
        n_trades = len(trades)
    else:
        current_profit = 0.0
        current_roi = 0.0
        current_win_rate = 0.0
        n_trades = 0

    return combo, current_roi, current_profit, current_win_rate, n_trades
