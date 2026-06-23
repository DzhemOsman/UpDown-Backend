import itertools
import logging
import random
from datetime import datetime
from typing import Iterable

import pandas as pd

from app.schemas.internal.best_parameter_combination_dict import (
    BestParameterCombinationDict,
    ParameterCombinationDict,
    BestResultDict
)
from app.services.mean_reversion_strategies.mean_reversion_defaults import (
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_START,
    DEFAULT_END,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_FEE_RATE
)
from app.services.mean_reversion_strategies.money_management_reversion import MeanReversionWithMoneyManagement
from app.services.mean_reversion_strategies.strategy_calculations import calculate_comparison_curves

logger = logging.getLogger(__name__)


def optimize_money_management_with_grid_search(
        tickers: list[str],
        drop_options: list[float],
        hold_options: list[int],
        take_profit_options: list[float],
        stop_loss_options: list[float],
        max_positions_options: list[int],
        allocation_options: list[float],
        initial_capital: int = DEFAULT_INITIAL_CAPITAL,
        start: datetime = DEFAULT_START,
        end: datetime = DEFAULT_END,
        is_kadane: bool = False,
        is_trend: bool = False
) -> BestParameterCombinationDict | None:
    """
    Führt eine Grid-Search, alle Parameterkombinationen, zur Optimierung der Mean-Reversion-Strategie
    über die angegebenen Ticker und Parameterbereiche durch.

    :param is_trend: Bool, ob Mean-Reversion mithilfe des SMA berechnet werden soll.
    :param is_kadane: Bool, ob der Kadane-Algorithmus für die Signalgenerierung verwendet werden soll.
    :param allocation_options: Liste von Allokationsprozentsätzen (Anteil des verfügbaren Kapitals pro Trade).
    :param max_positions_options: Liste von maximalen gleichzeitigen Positionen im Portfolio.
    :param stop_loss_options: Liste von Stop-Loss-Schwellenwerten in Prozent (maximaler Verlust pro Position).
    :param tickers: Liste von Ticker-Symbolen, z.B., ['AAPL', 'MSFT'], die getestet werden.
    :param drop_options: Liste von Schwellenwerten (in Prozent) für den prozentualen Rückgang, der ein Kaufsignal auslöst.
    :param hold_options: Liste von Halteperioden in Tagen (Anzahl Tage, die eine Position maximal gehalten wird).
    :param take_profit_options: Liste von Take-Profit-Zielen in Prozent (Gewinnziel zum Verkauf).
    :param initial_capital: Startkapital zur Berechnung des absoluten Gewinns und ROI.
    :param start: Startdatum des Backtests
    :param end: Enddatum des Backtests.
    :return: Die beste Kombination (höchster ROI), der getesteten Parameter
    """
    # itertools.product verhindert 6-fache Schleifen-Verschachtelung (Arrow Anti-Pattern)
    combinations = itertools.product(
        drop_options, hold_options, take_profit_options,
        stop_loss_options, max_positions_options, allocation_options,
    )
    return _find_best_combination(combinations, tickers, initial_capital, start, end, is_kadane, is_trend)


def optimize_money_management_with_randomized_grid_search(
        n_trials: int,
        tickers: list[str],
        drop_options: list[float],
        hold_options: list[int],
        take_profit_options: list[float],
        stop_loss_options: list[float],
        max_positions_options: list[int],
        allocation_options: list[float],
        initial_capital: int = DEFAULT_INITIAL_CAPITAL,
        start: datetime = DEFAULT_START,
        end: datetime = DEFAULT_END,
        is_kadane: bool = False,
        is_trend: bool = False,
        seed: int | None = None,
) -> BestParameterCombinationDict | None:
    """
    Führt eine Grid-Search, über n_trials zufällige Parameterkombinationen, zur Optimierung der Mean-Reversion-Strategie
    über die angegebenen Ticker und Parameterbereiche durch.

    :param n_trials: Anzahl der zufällig ausgewählten Parameterkombinationen
    :param seed: Seed um Zufallsgenerierte Kombinationen reproduzieren zu können
    :param tickers: Liste von Ticker-Symbolen, z.B., ['AAPL', 'MSFT'], die getestet werden.
    :param drop_options: Liste von Schwellenwerten (in Prozent) für den prozentualen Rückgang, der ein Kaufsignal auslöst.
    :param hold_options: Liste von Halteperioden in Tagen (Anzahl Tage, die eine Position maximal gehalten wird).
    :param take_profit_options: Liste von Take-Profit-Zielen in Prozent (Gewinnziel zum Verkauf).
    :param stop_loss_options: Liste von Stop-Loss-Schwellenwerten in Prozent (maximaler Verlust pro Position).
    :param max_positions_options: Liste von maximalen gleichzeitigen Positionen im Portfolio.
    :param allocation_options: Liste von Allokationsprozentsätzen (Anteil des verfügbaren Kapitals pro Trade).
    :param initial_capital: Startkapital zur Berechnung des absoluten Gewinns und ROI.
    :param start: Startdatum des Backtests
    :param end: Enddatum des Backtests
    :param is_kadane: Bool, ob der Kadane-Algorithmus für die Signalgenerierung verwendet werden soll.
    :param is_trend: Bool, ob Mean-Reversion mithilfe des SMA berechnet werden soll.
    :return: Die beste Kombination (höchster ROI), der getesteten Parameter
    """
    rng = random.Random(seed)
    all_combinations = list(itertools.product(
        drop_options, hold_options, take_profit_options,
        stop_loss_options, max_positions_options, allocation_options
    ))
    sampled = rng.sample(all_combinations, min(n_trials, len(all_combinations)))

    return _find_best_combination(sampled, tickers, initial_capital, start, end, is_kadane, is_trend)


def _find_best_combination(
        combinations: Iterable[tuple],
        tickers: list[str],
        initial_capital: int,
        start: datetime,
        end: datetime,
        is_kadane: bool,
        is_trend: bool,
) -> BestParameterCombinationDict | None:
    bot = MeanReversionWithMoneyManagement(initial_capital=initial_capital, start_date=start, end_date=end)

    best_roi = float('-inf')
    best_result = None
    best_trades = None
    best_params = None

    for drop, hold, tp, sl, max_pos, alloc in combinations:

        current_params = ParameterCombinationDict(
            drop_threshold=drop,
            lookback_days=DEFAULT_LOOKBACK_DAYS,
            hold_days=hold,
            take_profit_pct=tp,
            stop_loss_pct=sl,
            max_positions=max_pos,
            allocation_pct=alloc,
            fee_pct=DEFAULT_FEE_RATE
        )
        trades = bot.run_portfolio_with_money_management(tickers, current_params, is_kadane, is_trend)

        if trades:
            current_trades_df = pd.DataFrame(trades)
            current_profit = current_trades_df['profit_abs'].sum()
            current_roi = (current_profit / initial_capital) * 100
            #  Pandas Best Practice: Wahrheitswerte (True/False) als Mean berechnen
            current_win_rate = (current_trades_df['profit_abs'] > 0).mean() * 100
        else:
            current_profit = 0
            current_roi = 0
            current_win_rate = 0

        if current_roi > best_roi:
            best_roi = current_roi
            best_trades = trades
            best_params = current_params

            best_result = BestResultDict(
                profit=current_profit,
                win_rate=current_win_rate,
                total_number_of_trades=len(trades)
            )

    if best_params is None or best_result is None or best_trades is None:
        return None

    equity_data = calculate_comparison_curves(best_trades, bot.get_cached_ticker_data(), initial_capital)

    return BestParameterCombinationDict(
        best_drop_threshold=float(best_params['drop_threshold']),
        best_hold_days=int(best_params['hold_days']),
        best_take_profit_pct=float(best_params['take_profit_pct']),
        best_stop_loss_pct=float(best_params['stop_loss_pct']),
        best_max_positions=int(best_params['max_positions']),
        best_allocation_pct=float(best_params['allocation_pct']),
        total_profit=float(round(best_result['profit'], 2)),
        roi_pct=float(round(best_roi, 2)),
        win_rate=float(round(best_result['win_rate'], 2)),
        total_number_of_trades=int(best_result['total_number_of_trades']),
        equity_curve_data=equity_data,
        trades=best_trades
    )
