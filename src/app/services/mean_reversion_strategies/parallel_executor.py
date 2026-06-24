import pandas as pd

from app.schemas.internal.best_parameter_combination_dict import Combo
from app.services.mean_reversion_strategies.money_management_reversion import MeanReversionWithMoneyManagement
from app.services.mean_reversion_strategies.optimizer_utils import trades_to_metrics, combo_to_params

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

    current_params = combo_to_params(combo)

    trades = bot.run_portfolio_with_money_management(
        list(_WORKER_DATA), current_params, is_kadane, is_trend
    )

    current_roi, current_profit, current_win_rate, n_trades = trades_to_metrics(trades, initial_capital)

    return combo, current_roi, current_profit, current_win_rate, n_trades
