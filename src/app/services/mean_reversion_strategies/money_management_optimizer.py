import itertools
import logging
from datetime import datetime

import optuna
import pandas as pd

from app.schemas.internal.best_parameter_combination_dict import (
    BestParameterCombinationDict,
    ParameterCombinationDict,
    BestResultDict
)
from app.services.mean_reversion_strategies.compare_to_buy_and_hold import calculate_comparison_curves
from app.services.mean_reversion_strategies.mean_reversion_defaults import (
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_START,
    DEFAULT_END,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_FEE_RATE
)
from app.services.mean_reversion_strategies.money_management_reversion import MeanReversionWithMoneyManagement

logger = logging.getLogger(__name__)


def optimize_money_management_with_grid_search(
        tickers: list[str],
        drop_options: list[int],
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
    Führt eine Grid-Search zur Optimierung der Mean-Reversion-Strategie
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
    :return: Die beste Kombination (höchster ROI), der angegebenen Parameter
    """
    bot = MeanReversionWithMoneyManagement(initial_capital=initial_capital, start_date=start, end_date=end)

    best_roi = float('-inf')
    best_result = None
    best_trades = None
    best_params = None

    # itertools.product verhindert 6-fache Schleifen-Verschachtelung (Arrow Anti-Pattern)
    for drop, hold, tp, sl, max_pos, alloc in itertools.product(
            drop_options, hold_options, take_profit_options, stop_loss_options, max_positions_options,
            allocation_options
    ):
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
            # Pandas Best Practice: Wahrheitswerte (True/False) als Mean berechnen
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
        best_drop_threshold=best_params['drop_threshold'],
        best_hold_days=best_params['hold_days'],
        best_take_profit_pct=best_params['take_profit_pct'],
        best_stop_loss_pct=best_params['stop_loss_pct'],
        best_max_positions=best_params['max_positions'],
        best_allocation_pct=best_params['allocation_pct'],
        total_profit=round(best_result['profit'], 2),
        roi_pct=round(best_roi, 2),
        win_rate=round(best_result['win_rate'], 2),
        total_number_of_trades=best_result['total_number_of_trades'],
        equity_curve_data=equity_data,
        trades=best_trades
    )


def objective(
        trial: optuna.Trial,
        tickers: list[str],
        initial_capital: int,
        is_kadane: bool,
        is_trend: bool,
        bot: MeanReversionWithMoneyManagement
) -> float:
    """
    Die Objective-Funktion für Optuna. Sie definiert die Suchräume und gibt den ROI zurück,
    den Optuna maximieren soll.

    :param trial: Optuna Trial-Objekt, das Parametervorschläge macht.
    :param tickers: Liste von Ticker-Symbolen, z.B., ['AAPL', 'MSFT'], die getestet werden.
    :param initial_capital: Startkapital zur Berechnung des ROI.
    :param is_kadane: Bool, ob der Kadane-basierte Signalgenerator verwendet werden soll.
    :param is_trend: Bool, ob Trend-basierte (SMA) Signale verwendet werden sollen.
    :param bot: Instanz von MeanReversionWithMoneyManagement, die den Backtest ausführt.
    :return: ROI in Prozent (float), den Optuna maximieren möchte.
    """
    # 1. Optuna schlägt Parameter vor (Suchräume definieren)
    drop = trial.suggest_int('drop_threshold', 3, 10)
    hold = trial.suggest_int('hold_days', 2, 10)

    # Für Floats definieren wir Ober- und Untergrenze. step bestimmt die Schrittweite.
    tp = trial.suggest_float('take_profit_pct', 1.0, 5.0, step=0.5)
    sl = trial.suggest_float('stop_loss_pct', 1.0, 10.0, step=0.5)

    max_pos = trial.suggest_int('max_positions', 1, 5)
    alloc = trial.suggest_float('allocation_pct', 10.0, 50.0, step=5.0)

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

    # 2. Backtest durchführen
    trades = bot.run_portfolio_with_money_management(tickers, current_params, is_kadane, is_trend)

    # 3. ROI berechnen (das ist der Wert, den Optuna maximieren soll)
    if not trades:
        return 0.0

    current_profit = sum(trade['profit_abs'] for trade in trades)
    current_roi = (current_profit / initial_capital) * 100

    return current_roi


def optimize_bayesian(
        tickers: list[str],
        n_trials: int = 50,
        initial_capital: int = DEFAULT_INITIAL_CAPITAL,
        start: datetime = DEFAULT_START,
        end: datetime = DEFAULT_END,
        is_kadane: bool = False,
        is_trend: bool = False
) -> BestParameterCombinationDict | None:
    """
    Führt eine Bayesianische Optimierung mit Optuna durch.

    :param tickers: Liste von Ticker-Symbolen, z.B., ['AAPL', 'MSFT'], die getestet werden.
    :param n_trials: Anzahl der Optuna‑Trials (Anzahl der Optimierungsdurchläufe).
    :param initial_capital: Startkapital zur Berechnung von Gewinn und ROI.
    :param start: Startdatum des Backtests.
    :param end: Enddatum des Backtests.
    :param is_kadane: Wenn True, wird der Kadane‑basierte Signalgenerator verwendet.
    :param is_trend: Wenn True, werden trendbasierte Signale (SMA‑Crossover) verwendet.
    :return: BestParameterCombinationDict mit den besten Parametern, Performance‑Metriken und Trades oder None, falls keine valide Lösung gefunden wurde.
    """
    bot = MeanReversionWithMoneyManagement(initial_capital=initial_capital, start_date=start, end_date=end)

    # Optuna so konfigurieren, dass es weniger Logs in die Konsole spuckt
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # 1. Study erstellen (direction='maximize' bedeutet, wir wollen den höchsten ROI)
    study = optuna.create_study(direction='maximize')

    # 2. Optimierung starten. Wir übergeben die objective-Funktion mit einer lambda-Funktion,
    # damit wir unsere zusätzlichen Argumente (tickers, bot, etc.) mitgeben können.
    logger.info(f"Starte Bayesian Optimization mit {n_trials} Trials...")
    study.optimize(lambda trial: objective(
        trial,
        tickers,
        initial_capital,
        is_kadane,
        is_trend,
        bot
    ), n_trials=n_trials)

    # 3. Beste Parameter auswerten
    best_params_optuna = study.best_params
    best_roi = study.best_value

    if best_roi == 0.0:
        return None

    # 4. Den besten Durchlauf noch EINMAL simulieren, um alle Trade-Daten und Equity-Kurven zu bekommen
    final_params = ParameterCombinationDict(
        drop_threshold=best_params_optuna['drop_threshold'],
        lookback_days=DEFAULT_LOOKBACK_DAYS,
        hold_days=best_params_optuna['hold_days'],
        take_profit_pct=best_params_optuna['take_profit_pct'],
        stop_loss_pct=best_params_optuna['stop_loss_pct'],
        max_positions=best_params_optuna['max_positions'],
        allocation_pct=best_params_optuna['allocation_pct'],
        fee_pct=DEFAULT_FEE_RATE
    )

    best_trades = bot.run_portfolio_with_money_management(tickers, final_params, is_kadane, is_trend)

    if not best_trades:
        return None

    best_trades_df = pd.DataFrame(best_trades)
    current_profit = best_trades_df['profit_abs'].sum()
    # Pandas Best Practice für Win Rate
    current_win_rate = (best_trades_df['profit_abs'] > 0).mean() * 100

    equity_data = calculate_comparison_curves(best_trades, bot.get_cached_ticker_data(), initial_capital)

    return BestParameterCombinationDict(
        best_drop_threshold=final_params['drop_threshold'],
        best_hold_days=final_params['hold_days'],
        best_take_profit_pct=final_params['take_profit_pct'],
        best_stop_loss_pct=final_params['stop_loss_pct'],
        best_max_positions=final_params['max_positions'],
        best_allocation_pct=final_params['allocation_pct'],
        total_profit=round(current_profit, 2),
        roi_pct=round(best_roi, 2),
        win_rate=round(current_win_rate, 2),
        total_number_of_trades=len(best_trades),
        equity_curve_data=equity_data,
        trades=best_trades
    )
