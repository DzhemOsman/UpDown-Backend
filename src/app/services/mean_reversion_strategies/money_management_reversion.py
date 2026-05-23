from __future__ import annotations

import itertools
import logging
from datetime import datetime

import numpy as np
import optuna
import pandas as pd

from app.schemas.internal.best_parameter_combination_dict import (
    BestParameterCombinationDict,
    ParameterCombinationDict,
    BestResultDict
)
from app.schemas.internal.trade_result_dict import TradeResultDict
from app.services.mean_reversion_strategies.backtest_data import get_backtest_data
from app.services.mean_reversion_strategies.compare_to_buy_and_hold import calculate_comparison_curves
from app.services.mean_reversion_strategies.mean_reversion_defaults import (
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_START,
    DEFAULT_END,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_FEE_RATE
)

logger = logging.getLogger(__name__)


class MeanReversionWithMoneyManagement:
    """
    Mean-Reversion-Algorithmus erweitert mit Money Management.
    """

    def __init__(
            self,
            initial_capital: int = DEFAULT_INITIAL_CAPITAL,
            start_date: datetime = DEFAULT_START,
            end_date: datetime = DEFAULT_END,
    ):
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date
        self._ticker_cache: dict[str, pd.DataFrame] = {}

    def run_portfolio_with_money_management(
            self,
            tickers: list[str],
            params: ParameterCombinationDict,
            is_kadane: bool = False,
            is_trend: bool = False
    ) -> list[TradeResultDict]:
        """
        Starte den backtest() für ein Portfolio(Eine Sammlung von Assets) und speichert alle gemachten Transaktionen
        (trades).
        :param is_trend: Bool, ob Mean-Reversion mithilfe des SMA berechnet werden soll.
        :param is_kadane: Bool, ob für die Suche die minimale Abschnittssuche implementiert werden soll.
        :param tickers: Liste an Tickern, z.B., ['TSLA', 'MSFT', 'PLTR'].
        :param params: Parameter mit denen der Backtest durchgeführt wir.
        :return: Eine Liste aller Transaktionen.
        """
        all_potential_trades = []

        for ticker in tickers:
            trades = self._backtest_with_money_management(
                ticker,
                drop_threshold_pct=params['drop_threshold'],
                lookback_days=params['lookback_days'],
                hold_days=params['hold_days'],
                take_profit_pct=params['take_profit_pct'],
                fee_rate=params['fee_pct'],
                stop_loss_pct=params['stop_loss_pct'],
                is_kadane=is_kadane,
                is_trend=is_trend
            )
            if trades:
                all_potential_trades.extend(trades)

        all_potential_trades.sort(key=lambda x: x['buy_date'])

        current_capital = self.initial_capital
        active_trades: list[TradeResultDict] = []
        executed_trades: list[TradeResultDict] = []

        for trade in all_potential_trades:
            trade_date = trade['buy_date']

            # A. Geschlossene Positionen abrechnen
            # Prüfen, ob offene Positionen am heutigen trade_date bereits verkauft wurden
            still_active = []
            for active in active_trades:
                # Wenn das Verkaufsdatum in der Vergangenheit oder am selben Tag (morgens/abends) liegt:
                if active['sell_date'] <= trade_date:
                    # Kapital wird freigegeben (Initiales Investment + erwirtschafteter Profit)
                    current_capital += (active['invested_capital'] + active['profit_abs'])
                else:
                    still_active.append(active)
            active_trades = still_active  # Nur noch die laufenden Trades behalten

            # B. Prüfen, ob neues Signal angenommen wird (Positions-Limit)
            if len(active_trades) < params['max_positions']:
                # C. Positionsgröße berechnen (Compounding)
                # Anteil vom momentan verfügbaren Kapital
                allocation = current_capital * (params['allocation_pct'] / 100)

                # Trade mit echten Geldwerten aktualisieren
                actual_profit = allocation * (trade['profit_pct'] / 100)

                trade['invested_capital'] = round(allocation, 2)
                trade['profit_abs'] = round(actual_profit, 2)

                # D. Kapital für diesen Trade blockieren
                current_capital -= allocation

                active_trades.append(trade)
                executed_trades.append(trade)

        return executed_trades

    def get_cached_ticker_data(self) -> dict[str, pd.DataFrame]:
        return self._ticker_cache

    def _get_ticker_data_for_backtest(self, ticker: str) -> pd.DataFrame:
        """
       Startet das Laden von Tickerdaten aus der Datenbank und speichert sie im Cache.

        :param ticker: Ticker-Symbol, z.B., 'AAPL' der von der Datenbank abgefragt wird.
        :return: Daten des angefragten Tickers als pd.DataFrame.
        """
        if ticker in self._ticker_cache:
            return self._ticker_cache[ticker]

        data = get_backtest_data(
            ticker,
            start_date=self.start_date,
            end_date=self.end_date,
            is_optimized=True
        )

        # if data is not None and not data.empty:
        self._ticker_cache[ticker] = data

        return data

    def _calculate_max_loss_kadane(self, returns_array: np.ndarray) -> float:
        """
        Invertierter Kadane-Algorithmus: Sucht den stärksten zusammenhängenden Kursverlust
         innerhalb eines Arrays von Tagesrenditen.

        :param returns_array: Array von Tagesrenditen (prozentuale Änderungen), die analysiert werden sollen.
        :return: Der maximale Verlust (negativer Wert) als Dezimalzahl, z.B. -0.15 für -15%.
        """
        total_min = 0.0
        end_sum = 0.0
        for r in returns_array:
            # Addiere die heutige Rendite zur bisherigen Summe
            s = end_sum + r

            # Invertierte Logik:
            # Wenn die Summe positiv wird, haben wir keinen durchgehenden Abwärtstrend mehr,
            # also setzen wir den Zähler auf 0 zurück. Bei Verlusten behalten wir die Summe (s).
            end_sum = s if s < 0 else 0.0

            # Merken des bisher tiefsten Sturzes
            if end_sum < total_min:
                total_min = end_sum

        return total_min

    def _backtest_with_money_management(
            self, ticker: str,
            drop_threshold_pct: int,
            lookback_days: int,
            hold_days: int,
            take_profit_pct: float,
            fee_rate: float,
            stop_loss_pct: float,
            is_kadane: bool = False,
            is_trend: bool = False
    ) -> list[TradeResultDict]:
        """
        Führt den Backtest für einen Ticker mit den mitgelieferten Parametern durch.

        :param is_trend: Bool, die Signale mithilfe des SMA berechnet werden sollen.
        :param is_kadane: Bool, ob für die Suche die minimale Abschnittssuche implementiert werden soll.
        :param ticker: Ticker-Symbol, z.B., 'AAPL' der getestet wird.
        :param drop_threshold_pct: Prozentualer-Fall über die lookback Periode das ein Signal markiert.
        :param lookback_days: Anzahl an Tagen über die Preisveränderungen berechnet werden.
        :param hold_days: Wie viele Tage die Position gehalten wird.
        :param take_profit_pct: Profit-Ziel, wenn diese erreicht wird, wird die Position verkauft.
        :param fee_rate: Gebühren die pro Transaktion (Verkauf und Kauf) anfallen.
        :param stop_loss_pct: Kursfall, ab dem wieder verkauft wird, um Verluste zu minimieren.
        :return: Eine Liste aller ausgeführten Transkationen.
        """
        ticker_df = self._get_ticker_data_for_backtest(ticker)
        if ticker_df is None or ticker_df.empty:
            return []

        threshold_decimal = -(drop_threshold_pct / 100)

        if is_trend:
            # Berechne den Simple Moving Average (SMA) über die lookback_days
            ticker_df['sma'] = ticker_df['close'].rolling(window=lookback_days).mean()

            # Um einen sauberen Crossover zu erkennen, brauchen wir die Werte vom Vortag
            ticker_df['prev_close'] = ticker_df['close'].shift(1)
            ticker_df['prev_sma'] = ticker_df['sma'].shift(1)

            # Kaufsignal: Der gestrige Kurs war unter dem SMA, der heutige Kurs ist über dem SMA
            signal_indices = np.where(
                (ticker_df['close'] > ticker_df['sma']) &
                (ticker_df['prev_close'] <= ticker_df['prev_sma'])
            )[0]

        elif is_kadane:
            # 1. Tägliche prozentuale Veränderung berechnen (Tag zu Tag)
            ticker_df['daily_return'] = ticker_df['close'].pct_change(periods=1)

            # 2. Den Kadane-Algorithmus als rollierendes Fenster anwenden.
            ticker_df['kadane_max_loss'] = ticker_df['daily_return'].rolling(window=lookback_days).apply(
                self._calculate_max_loss_kadane, raw=True
            )

            # 3. Signale generieren
            signal_indices = np.where(ticker_df['kadane_max_loss'] < threshold_decimal)[0]

        else:
            # --- ALTE POINT-TO-POINT LOGIK ---
            ticker_df['change'] = ticker_df['close'].pct_change(periods=lookback_days)
            signal_indices = np.where(ticker_df['change'] < threshold_decimal)[0]

        trades = []
        last_exit_index = -1

        open_prices = ticker_df['open'].to_numpy()
        high_prices = ticker_df['high'].to_numpy()
        low_prices = ticker_df['low'].to_numpy()
        close_prices = ticker_df['close'].to_numpy()
        dates = ticker_df.index

        for idx in signal_indices:
            # Trade wird erst am nächsten Tag ausgeführt, da erst bei Börsenschluss der Tagesschluss bekannt ist.
            entry_idx = idx + 1

            if entry_idx <= last_exit_index:
                continue
            if entry_idx >= len(ticker_df):
                continue

            entry_date = dates[entry_idx]
            raw_entry_price = open_prices[entry_idx]
            raw_exit_price = 0.0
            effective_entry_price = raw_entry_price * (1 + fee_rate)
            target_price = raw_entry_price * (1 + take_profit_pct / 100)
            stop_price = raw_entry_price * (1 - stop_loss_pct / 100)
            exit_reason = ""
            days_held = 0

            found_exit = False

            for i in range(0, hold_days):
                current_idx = entry_idx + i

                if current_idx >= len(ticker_df):
                    break

                current_high = high_prices[current_idx]
                current_open = open_prices[current_idx]
                current_low = low_prices[current_idx]
                current_close = close_prices[current_idx]
                current_date = dates[current_idx]

                # Take-Profit Logik
                can_sell_at_open = (i > 0)
                # Stop-Loss Logik
                if current_low <= stop_price:
                    # Wenn Eröffnung schon unter Stop, dann zur Eröffnung raus
                    if i > 0 and current_open < stop_price:
                        raw_exit_price = current_open
                    else:
                        raw_exit_price = stop_price

                    exit_reason = "Stop Loss"
                    days_held = i
                    found_exit = True
                elif current_high >= target_price:
                    if can_sell_at_open and current_open > target_price:
                        raw_exit_price = current_open
                    else:
                        raw_exit_price = target_price

                    exit_reason = "Take Profit"
                    days_held = i
                    found_exit = True

                # Stopp, wenn Haltedauer erreicht wurde
                elif i == (hold_days - 1):
                    raw_exit_price = current_close
                    exit_reason = "Time Stop"
                    days_held = i
                    found_exit = True

                if found_exit:
                    last_exit_index = current_idx
                    exit_date = current_date

                    effective_exit_price = raw_exit_price * (1 - fee_rate)

                    profit_pct = (effective_exit_price - effective_entry_price) / effective_entry_price
                    profit_abs = self.initial_capital * profit_pct

                    trades.append(
                        TradeResultDict(
                            ticker=ticker,
                            buy_date=entry_date.strftime('%Y-%m-%d'),
                            sell_date=exit_date.strftime('%Y-%m-%d'),
                            days_held=days_held,
                            exit_reason=exit_reason,
                            entry_price=round(raw_entry_price, 2),
                            exit_price=round(raw_exit_price, 2),
                            profit_pct=round(profit_pct * 100, 2),
                            profit_abs=round(profit_abs, 2),
                            invested_capital=0.0
                        )
                    )
                    break

        return trades


def optimize_money_management_with_grid_search(
        tickers,
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

    current_trades_df = pd.DataFrame(trades)
    current_profit = current_trades_df['profit_abs'].sum()
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
    study.optimize(lambda trial: objective(trial, tickers, initial_capital, bot), n_trials=n_trials)

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
