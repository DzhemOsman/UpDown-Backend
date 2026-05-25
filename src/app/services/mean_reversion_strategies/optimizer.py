from datetime import datetime

import pandas as pd

from app.schemas.internal.best_parameter_combination_dict import (
    BestParameterCombinationDict,
    ParameterCombinationDict,
    BestResultDict
)
from app.services.mean_reversion_strategies.strategy_calculations import calculate_comparison_curves
from app.services.mean_reversion_strategies.mean_reversion_defaults import (
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_START,
    DEFAULT_END,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_FEE_RATE
)
from app.services.mean_reversion_strategies.mean_reversion_strategy import MeanReversionStrategy


def optimize_grid_search(
        tickers: list[str],
        drop_options: list[int],
        hold_options: list[int],
        take_profit_options: list[float],
        initial_capital: int = DEFAULT_INITIAL_CAPITAL,
        start: datetime = DEFAULT_START,
        end: datetime = DEFAULT_END
) -> BestParameterCombinationDict | None:
    """
    Führt eine Grid-Search zur Optimierung der Mean-Reversion-Strategie
    über die angegebenen Ticker und Parameterbereiche durch.

    :param tickers: Liste von Ticker-Symbolen, z.B., ['AAPL', 'MSFT'], die getestet werden.
    :param drop_options: Liste von Schwellenwerten (in Prozent) für den prozentualen Rückgang, der ein Kaufsignal auslöst.
    :param hold_options: Liste von Halteperioden in Tagen (Anzahl Tage, die eine Position maximal gehalten wird).
    :param take_profit_options: Liste von Take-Profit-Zielen in Prozent (Gewinnziel zum Verkauf).
    :param initial_capital: Startkapital zur Berechnung des absoluten Gewinns und ROI.
    :param start: Startdatum des Backtests
    :param end: Enddatum des Backtests.
    :return: Die beste Kombination (höchster ROI), der angegebenen Parameter
    """
    bot = MeanReversionStrategy(initial_capital=initial_capital, start_date=start, end_date=end)

    best_roi = float('-inf')
    best_result = None
    best_trades = None
    best_params = None

    for drop in drop_options:
        for hold in hold_options:
            for tp in take_profit_options:
                current_params = ParameterCombinationDict(
                    drop_threshold=drop,
                    lookback_days=DEFAULT_LOOKBACK_DAYS,
                    hold_days=hold,
                    take_profit_pct=tp,
                    fee_pct=DEFAULT_FEE_RATE
                )
                trades = bot.run_portfolio(tickers, current_params)

                if trades is None:
                    break

                if trades:
                    current_trades_df = pd.DataFrame(trades)
                    current_profit = current_trades_df['profit_abs'].sum()
                    current_roi = (current_profit / initial_capital) * 100
                    current_win_rate = len(current_trades_df[current_trades_df['profit_abs'] > 0]) / len(
                        current_trades_df) * 100
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
        total_profit=float(round(best_result['profit'], 2)),
        roi_pct=float(round(best_roi, 2)),
        win_rate=float(round(best_result['win_rate'], 2)),
        total_number_of_trades=int(best_result['total_number_of_trades']),
        equity_curve_data=equity_data,
        trades=best_trades
    )
