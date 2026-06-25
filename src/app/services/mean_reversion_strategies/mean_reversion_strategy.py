from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from app.schemas.internal.best_parameter_combination_dict import (
    ParameterCombinationDict,
)
from app.schemas.internal.strategy_result_dict import StrategyResultDict
from app.schemas.internal.trade_result_dict import TradeResultDict
from app.services.mean_reversion_strategies.backtest_data import get_backtest_data
from app.services.mean_reversion_strategies.mean_reversion_defaults import (
    DEFAULT_END,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_START,
)
from app.services.mean_reversion_strategies.strategy_calculations import (
    calculate_strategy_result,
)

logger = logging.getLogger(__name__)


class MeanReversionStrategy:
    """
    Ursprünglicher Mean-Reversion-Algorithmus aus dem alten Projekt.
    Logik ist unverändert aber Code wurde angepasst, um die Code-Quality
    zu erhöhen und die Performance zu verbessern.
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
        params: ParameterCombinationDict,
    ) -> list[TradeResultDict]:
        """
        Führt den Backtest für ein Portfolio durch und sammelt alle Transaktionen.

        :param tickers: Liste an Tickern, z.B., ['TSLA', 'MSFT', 'PLTR'].
        :param params: Parameter, mit denen der Backtest durchgeführt wird.
        :return: Liste aller ausgeführten Transaktionen (ggf. leer).
        """
        all_trades = []
        for ticker in tickers:
            trades = self._backtest(
                ticker,
                drop_threshold_pct=params["drop_threshold"],
                lookback_days=params["lookback_days"],
                hold_days=params["hold_days"],
                take_profit_pct=params["take_profit_pct"],
                fee_rate=params["fee_pct"],
            )
            if trades:
                all_trades.extend(trades)
        return all_trades

    def run_portfolio_single(
        self,
        tickers: list[str],
        params: ParameterCombinationDict,
    ) -> StrategyResultDict:
        """
        Führt den Backtest durch und gibt das aggregierte Strategie-Ergebnis zurück.

        :param tickers: Liste an Tickern.
        :param params: Parameter mit denen der Backtest durchgeführt wird.
        :return: Aggregiertes Ergebnis (ROI, Win-Rate, Equity-Kurve, Trades).
        """
        trades = self.run_portfolio(tickers, params)
        return calculate_strategy_result(
            trades, self._ticker_cache, self.initial_capital
        )

    def get_cached_ticker_data(self) -> dict[str, pd.DataFrame]:
        return self._ticker_cache

    def _get_ticker_data_for_backtest(self, ticker: str) -> pd.DataFrame | None:
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
            include_low=False,
        )

        if data is not None and not data.empty:
            self._ticker_cache[ticker] = data

        return data

    def _backtest(
        self,
        ticker: str,
        drop_threshold_pct: float,
        lookback_days: int,
        hold_days: int,
        take_profit_pct: float,
        fee_rate: float,
    ) -> list[TradeResultDict]:
        """
        Führt den Backtest für einen Ticker mit den mitgelieferten Parametern durch

        :param ticker: Ticker-Symbol, z.B., 'AAPL' der getestet wird
        :param drop_threshold_pct: Prozentualer-Fall über die lookback Periode
        das ein Signal markiert
        :param lookback_days: Anzahl an Tagen über die Preisveränderungen
        berechnet werden
        :param hold_days: Wie viele Tage die Position gehalten wird
        :param take_profit_pct: Profit-Ziel, wenn diese erreicht wird,
        wird die Position verkauft
        :param fee_rate: Gebühren die pro Transaktion (Verkauf und Kauf) anfallen
        :return: Eine Liste aller ausgeführten Transkationen
        """
        ticker_df = self._get_ticker_data_for_backtest(ticker)
        if ticker_df is None or ticker_df.empty:
            return []

        ticker_df["change"] = ticker_df["close"].pct_change(periods=lookback_days)

        threshold_decimal = -(drop_threshold_pct / 100)

        signal_indices = np.where(ticker_df["change"] < threshold_decimal)[0]

        trades = []
        last_exit_index = -1

        open_prices = ticker_df["open"].to_numpy()
        high_prices = ticker_df["high"].to_numpy()
        close_prices = ticker_df["close"].to_numpy()
        dates = ticker_df.index

        for idx in signal_indices:
            # Trade wird erst am nächsten Tag ausgeführt, da erst bei Börsenschluss der
            # Tagesschluss bekannt ist.
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
                can_sell_at_open = i > 0

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

                    profit_pct = (
                        effective_exit_price - effective_entry_price
                    ) / effective_entry_price
                    profit_abs = self.initial_capital * profit_pct

                    trades.append(
                        TradeResultDict(
                            ticker=ticker,
                            buy_date=entry_date.strftime("%Y-%m-%d"),
                            sell_date=exit_date.strftime("%Y-%m-%d"),
                            days_held=int(days_held),
                            exit_reason=exit_reason,
                            entry_price=float(round(raw_entry_price, 2)),
                            exit_price=float(round(raw_exit_price, 2)),
                            profit_pct=float(round(profit_pct * 100, 2)),
                            profit_abs=float(round(profit_abs, 2)),
                        )
                    )
                    break

        return trades
