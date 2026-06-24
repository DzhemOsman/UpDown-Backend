from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from app.schemas.internal.best_parameter_combination_dict import (
    ParameterCombinationDict
)
from app.schemas.internal.strategy_result_dict import StrategyResultDict
from app.schemas.internal.trade_result_dict import TradeResultDict
from app.services.mean_reversion_strategies.backtest_data import get_backtest_data
from app.services.mean_reversion_strategies.mean_reversion_defaults import (
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_START,
    DEFAULT_END
)
from app.services.mean_reversion_strategies.strategy_calculations import calculate_strategy_result


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
            is_kadane: bool | None = False,
            is_trend: bool | None = False,
            is_single: bool = False
    ) -> list[TradeResultDict] | StrategyResultDict:
        """
        Starte den backtest() für ein Portfolio(Eine Sammlung von Assets) und speichert alle gemachten Transaktionen
        (trades).
        
        :param is_single: Bool, ob Funktion nur eine einzelne Kombination ausprobiert oder mehrere.
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

        if is_single:
            result = calculate_strategy_result(executed_trades, self._ticker_cache, self.initial_capital)
            return result
        else:
            return executed_trades

    def get_cached_ticker_data(self) -> dict[str, pd.DataFrame]:
        return self._ticker_cache

    def set_ticker_cache(self, ticker_cache: dict[str, pd.DataFrame]) -> None:
        self._ticker_cache = ticker_cache

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
            sum = end_sum + r

            # Invertierte Logik:
            # Wenn die Summe positiv wird, haben wir keinen durchgehenden Abwärtstrend mehr,
            # also setzen wir den Zähler auf 0 zurück. Bei Verlusten behalten wir die Summe (s).
            end_sum = sum if sum < 0 else 0.0

            # Merken des bisher tiefsten Sturzes
            if end_sum < total_min:
                total_min = end_sum

        return total_min

    def _backtest_with_money_management(
            self, ticker: str,
            drop_threshold_pct: float,
            lookback_days: int,
            hold_days: int,
            take_profit_pct: float,
            fee_rate: float,
            stop_loss_pct: float| None,
            is_kadane: bool | None = False,
            is_trend: bool | None = False
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
                            days_held=int(days_held),
                            exit_reason=exit_reason,
                            entry_price=float(round(effective_entry_price, 2)),
                            exit_price=float(round(effective_exit_price, 2)),
                            profit_pct=float(round(profit_pct * 100, 2)),
                            profit_abs=float(round(profit_abs, 2)),
                        )
                    )
                    break

        return trades
