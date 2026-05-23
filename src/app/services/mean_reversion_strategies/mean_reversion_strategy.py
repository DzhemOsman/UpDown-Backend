from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
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


class MeanReversionStrategy:
    """
    Ursprünglicher Mean-Reversion-Algorithmus aus dem alten Projekt. Logik ist unverändert aber Code wurde angepasst,
    um die Code-Quality zu erhöhen und die Performance zu verbessern.
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

    def run_portfolio(
            self,
            tickers: list[str],
            params: ParameterCombinationDict
    ) -> list[TradeResultDict]:
        """
        Starte den backtest() für ein Portfolio(Eine Sammlung von Assets) und speichert alle gemachten Transaktionen
        (trades).
        :param tickers: Liste an Tickern, z.B., ['TSLA', 'MSFT', 'PLTR']
        :param params: Parameter mit denen der Backtest durchgeführt wir
        :return: Eine Liste aller Transaktionen
        """
        all_trades = []

        for ticker in tickers:
            trades = self._backtest(
                ticker,
                drop_threshold_pct=params['drop_threshold'],
                lookback_days=params['lookback_days'],
                hold_days=params['hold_days'],
                take_profit_pct=params['take_profit_pct'],
                fee_rate=params['fee_pct']
            )
            if trades:
                all_trades.extend(trades)
        return all_trades

    def get_cached_ticker_data(self) -> dict[str, pd.DataFrame]:
        return self._ticker_cache

    def _get_ticker_data_for_backtest(self, ticker: str) -> pd.DataFrame:
        """
       Startet das Laden von Tickerdaten aus der Datenbank und speichert sie im Cache.

        :param ticker: Ticker-Symbol, z.B., 'AAPL' der von der Datenbank abgefragt wird
        :return: Daten des angefragten Tickers als pd.DataFrame
        """
        if ticker in self._ticker_cache:
            return self._ticker_cache[ticker]

        data = get_backtest_data(
            ticker,
            start_date=self.start_date,
            end_date=self.end_date,
            is_optimized=False
        )

        # if data is not None and not data.empty:
        self._ticker_cache[ticker] = data

        return data

    def _backtest(
            self, ticker: str,
            drop_threshold_pct: int,
            lookback_days: int,
            hold_days: int,
            take_profit_pct: float,
            fee_rate: float,
    ) -> list[TradeResultDict]:
        """
        Führt den Backtest für einen Ticker mit den mitgelieferten Parametern durch

        :param ticker: Ticker-Symbol, z.B., 'AAPL' der getestet wird
        :param drop_threshold_pct: Prozentualer-Fall über die lookback Periode das ein Signal markiert
        :param lookback_days: Anzahl an Tagen über die Preisveränderungen berechnet werden
        :param hold_days: Wie viele Tage die Position gehalten wird
        :param take_profit_pct: Profit-Ziel, wenn diese erreicht wird, wird die Position verkauft
        :param fee_rate: Gebühren die pro Transaktion (Verkauf und Kauf) anfallen
        :return: Eine Liste aller ausgeführten Transkationen
        """
        ticker_df = self._get_ticker_data_for_backtest(ticker)
        if ticker_df is None or ticker_df.empty:
            return []

        ticker_df['change'] = ticker_df['close'].pct_change(periods=lookback_days)

        threshold_decimal = -(drop_threshold_pct / 100)

        signal_indices = np.where(ticker_df['change'] < threshold_decimal)[0]

        trades = []
        last_exit_index = -1

        open_prices = ticker_df['open'].to_numpy()
        high_prices = ticker_df['high'].to_numpy()
        close_prices = ticker_df['open'].to_numpy()
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
            exit_reason = ""
            days_held = 0

            found_exit = False

            for i in range(0, hold_days):
                current_idx = entry_idx + i

                if current_idx >= len(ticker_df):
                    break

                current_high = high_prices[current_idx]
                current_open = open_prices[current_idx]
                current_close = close_prices[current_idx]
                current_date = dates[current_idx]

                # Take-Profit Logik
                can_sell_at_open = (i > 0)

                if current_high >= target_price:
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
                            profit_abs=round(profit_abs, 2)
                        )
                    )
                    break

        return trades


def optimize_grid_search(
        tickers,
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
        best_drop_threshold=best_params['drop_threshold'],
        best_hold_days=best_params['hold_days'],
        best_take_profit_pct=best_params['take_profit_pct'],
        total_profit=round(best_result['profit'], 2),
        roi_pct=round(best_roi, 2),
        win_rate=round(best_result['win_rate'], 2),
        total_number_of_trades=best_result['total_number_of_trades'],
        equity_curve_data=equity_data,
        trades=best_trades
    )
