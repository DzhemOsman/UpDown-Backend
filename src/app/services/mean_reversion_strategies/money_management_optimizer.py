import itertools
import logging
import random
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import pandas as pd

from app.config import get_settings
from app.schemas.internal.best_parameter_combination_dict import (
    BestParameterCombinationDict,
    ParameterCombinationDict,
    BestResultDict, Combo
)
from app.services.mean_reversion_strategies.backtest_data import get_backtest_data
from app.services.mean_reversion_strategies.mean_reversion_defaults import (
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_START,
    DEFAULT_END,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_FEE_RATE
)
from app.services.mean_reversion_strategies.money_management_reversion import MeanReversionWithMoneyManagement
from app.services.mean_reversion_strategies.parallel_executor import _init_worker, _evaluate
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
    combinations = _build_combinations(
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
    all_combinations = _build_combinations(
        drop_options, hold_options, take_profit_options,
        stop_loss_options, max_positions_options, allocation_options,
    )
    rng = random.Random(seed)
    sampled = rng.sample(all_combinations, min(n_trials, len(all_combinations)))

    return _find_best_combination(sampled, tickers, initial_capital, start, end, is_kadane, is_trend)


def _build_combinations(
        drop_options: list[float],
        hold_options: list[int],
        take_profit_options: list[float],
        stop_loss_options: list[float],
        max_positions_options: list[int],
        allocation_options: list[float],
) -> list[Combo]:
    return [
        Combo(
            drop_threshold=drop,
            hold_days=hold,
            take_profit_pct=tp,
            stop_loss_pct=sl,
            max_positions=max_pos,
            allocation_pct=alloc,
        )
        for drop, hold, tp, sl, max_pos, alloc in itertools.product(
            drop_options, hold_options, take_profit_options,
            stop_loss_options, max_positions_options, allocation_options,
        )
    ]


def _find_best_combination(
        combinations: list[Combo],
        tickers: list[str],
        initial_capital: int,
        start: datetime,
        end: datetime,
        is_kadane: bool,
        is_trend: bool,
) -> BestParameterCombinationDict | None:
    # 1. Tickerdaten EINMAL im Hauptprozess laden.
    ticker_data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = get_backtest_data(ticker, start_date=start, end_date=end, is_optimized=True)
        if df is not None and not df.empty:
            ticker_data[ticker] = df

    if not ticker_data:
        logger.warning("Keine Tickerdaten geladen - Optimierung abgebrochen.")
        return None

    # 2. Kombinationen parallel auswerten (Prozesse wegen GIL, nicht Threads).
    #    map() verlangt eine materialisierte Sequenz -> beim Grid den Lazy-Iterator vorher in eine Liste ziehen.
    best_roi = float('-inf')
    best_combo = None
    best_result = None

    with ProcessPoolExecutor(
            max_workers=get_settings().OPTIMIZER_MAX_WORKERS,
            initializer=_init_worker,
            initargs=(ticker_data, initial_capital, is_kadane, is_trend),
    ) as executor:
        for combo, roi, profit, win_rate, n_trades in executor.map(_evaluate, combinations):
            if roi > best_roi:
                best_roi = roi
                best_combo = combo
                best_result = BestResultDict(
                    profit=profit,
                    win_rate=win_rate,
                    total_number_of_trades=n_trades,
                )

    if best_combo is None or best_result is None:
        return None

    best_params = ParameterCombinationDict(
        drop_threshold=best_combo.drop_threshold,
        lookback_days=DEFAULT_LOOKBACK_DAYS,
        hold_days=best_combo.hold_days,
        take_profit_pct=best_combo.take_profit_pct,
        stop_loss_pct=best_combo.stop_loss_pct,
        max_positions=best_combo.max_positions,
        allocation_pct=best_combo.allocation_pct,
        fee_pct=DEFAULT_FEE_RATE,
    )

    bot = MeanReversionWithMoneyManagement(
        initial_capital=initial_capital, start_date=start, end_date=end
    )
    bot.set_ticker_cache(ticker_data)  # vorgeladene Daten injizieren -> kein erneuter DB-Zugriff
    best_trades = bot.run_portfolio_with_money_management(
        list(ticker_data), best_params, is_kadane, is_trend
    )

    equity_data = calculate_comparison_curves(best_trades, ticker_data, initial_capital)

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
        trades=best_trades,
    )
