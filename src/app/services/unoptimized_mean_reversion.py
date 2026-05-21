from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from app.services import market_data

# Standard-Backtest-Fenster (entspricht dem bisherigen DataManager-Default).
DEFAULT_START = datetime(2000, 1, 1)
DEFAULT_END = datetime(2025, 1, 1)


class MeanReversionStrategy:
    """Mean-Reversion-Backtest auf Tagesbasis.

    Der Backtest-Algorithmus ist unverändert aus der ursprünglichen
    ``strategy.py`` übernommen. Lediglich die Datenbeschaffung wurde von
    lokalen CSV-Dateien (DataManager) auf InfluxDB
    (``market_data.fetch_ticker_data``) umgestellt.
    """

    def __init__(
        self,
        initial_capital: float = 10000,
        start_date: datetime = DEFAULT_START,
        end_date: datetime = DEFAULT_END,
    ):
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date

    def load_and_clean_data(self, ticker: str) -> pd.DataFrame | None:
        """Lädt OHLC-Daten aus InfluxDB und bringt sie in das Format,
        das der Backtest erwartet: DatetimeIndex + Spalten Open/High/Close.

        Gibt ``None`` zurück, wenn keine Daten vorhanden sind – identisches
        Verhalten wie zuvor bei fehlender CSV-Datei.
        """
        try:
            df = market_data.fetch_ticker_data(ticker, self.start_date, self.end_date)
        except Exception as e:  # noqa: BLE001 - bewusst defensiv wie zuvor
            print(f"Fehler beim Laden von {ticker}: {e}")
            return None

        if df is None or df.empty:
            print(f"WARNUNG: Keine Daten für Ticker: {ticker}")
            return None

        try:
            # Influx liefert flache, kleingeschriebene Spalten + 'time'.
            ts = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)

            clean_df = pd.DataFrame(index=pd.DatetimeIndex(ts))
            clean_df["Close"] = df["close"].to_numpy()
            clean_df["Open"] = df["open"].to_numpy()
            clean_df["High"] = df["high"].to_numpy()

            clean_df.sort_index(inplace=True)

            # Lücken füllen
            clean_df.ffill(inplace=True)
            return clean_df
        except Exception as e:  # noqa: BLE001
            print(f"Fehler beim Aufbereiten von {ticker}: {e}")
            return None

    def backtest(self, ticker, drop_threshold_pct, lookback_days, hold_days, take_profit_pct, fee_rate):
        """
        Führt den Backtest durch
        """
        df = self.load_and_clean_data(ticker)
        if df is None or df.empty:
            return []

        df['change'] = df['Close'].pct_change(periods=lookback_days)

        threshold_decimal = -(drop_threshold_pct / 100)

        signal_indices = np.where(df['change'] < threshold_decimal)[0]

        trades = []
        last_exit_index = -1

        for idx in signal_indices:
            """
            Trade wird erst am nächsten Tag ausgeführt, da erst bei Börsenschluss der Tagesschluss bekannt ist.
            """
            entry_idx = idx + 1

            if entry_idx <= last_exit_index:
                continue
            if entry_idx >= len(df):
                continue

            entry_date = df.index[entry_idx]
            raw_entry_price = df['Open'].iloc[entry_idx]

            effective_entry_price = raw_entry_price * (1 + fee_rate)

            target_price = raw_entry_price * (1 + take_profit_pct / 100)

            exit_price = None
            exit_date = None
            exit_reason = ""
            days_held = 0

            found_exit = False

            for i in range(0, hold_days):
                current_idx = entry_idx + i

                if current_idx >= len(df):
                    break

                current_high = df['High'].iloc[current_idx]
                current_open = df['Open'].iloc[current_idx]
                current_close = df['Close'].iloc[current_idx]
                current_date = df.index[current_idx]

                # Take Profit Logik
                can_sell_at_open = (i > 0)

                if current_high >= target_price:
                    if can_sell_at_open and current_open > target_price:
                        raw_exit_price = current_open
                    else:
                        raw_exit_price = target_price

                    exit_reason = "Take Profit"
                    days_held = i
                    found_exit = True

                # Haltedauer wurde erreicht
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

                    trades.append({
                        "ticker": ticker,
                        "buy_date": entry_date.strftime('%Y-%m-%d'),
                        "sell_date": exit_date.strftime('%Y-%m-%d'),
                        "days_held": days_held,
                        "exit_reason": exit_reason,
                        "entry_price": round(raw_entry_price, 2),  # Chart-Preis anzeigen
                        "exit_price": round(raw_exit_price, 2),  # Chart-Preis anzeigen
                        "profit_pct": round(profit_pct * 100, 2),  # Netto-Profit %
                        "profit_abs": round(profit_abs, 2)  # Netto-Profit €
                    })
                    break

        return trades

    def run_portfolio(self, tickers, params):
        all_trades = []
        fee = params.get('fee', 0.001)

        for ticker in tickers:
            trades = self.backtest(
                ticker,
                drop_threshold_pct=params['drop'],
                lookback_days=params['lookback'],
                hold_days=params['hold'],
                take_profit_pct=params['take_profit'],
                fee_rate=fee
            )
            if trades:
                all_trades.extend(trades)
        return all_trades


def _load_close_series(ticker: str, all_dates: pd.DatetimeIndex,
                       start: datetime, end: datetime) -> pd.Series | None:
    """Lädt eine Close-Preisreihe aus InfluxDB, auf den gemeinsamen
    Zeitstrahl reindiziert. Ersetzt den früheren data_dict-Zugriff.
    """
    df = market_data.fetch_ticker_data(ticker, start, end)
    if df is None or df.empty:
        return None

    ts = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    series = pd.Series(df["close"].to_numpy(), index=pd.DatetimeIndex(ts)).sort_index()
    return series.reindex(all_dates).ffill().bfill()


def calculate_comparison_curves(trades, tickers, initial_capital,
                                start: datetime, end: datetime):
    """
    Berechnet tagesgenau die Strategie-Equity vs. Buy & Hold Benchmark.
    """
    close_series: dict[str, pd.Series] = {}
    all_dates = pd.DatetimeIndex([])
    for t in tickers:
        df = market_data.fetch_ticker_data(t, start, end)
        if df is not None and not df.empty:
            ts = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
            s = pd.Series(df["close"].to_numpy(), index=pd.DatetimeIndex(ts)).sort_index()
            close_series[t] = s
            all_dates = all_dates.union(s.index)

    if len(all_dates) == 0:
        return []

    all_dates = all_dates.sort_values().unique()
    df_curve = pd.DataFrame(index=all_dates)

    df_curve['strategy_equity'] = 0.0
    daily_profits = pd.Series(0.0, index=all_dates)

    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df['sell_date'] = pd.to_datetime(trades_df['sell_date'])
        grouped_profits = trades_df.groupby('sell_date')['profit_abs'].sum()
        daily_profits = daily_profits.add(grouped_profits, fill_value=0)

    df_curve['strategy_equity'] = initial_capital + daily_profits.cumsum()
    df_curve['strategy_equity'] = df_curve['strategy_equity'].ffill()

    # BUY & HOLD BENCHMARK
    allocation_per_ticker = initial_capital / len(tickers)
    df_curve['benchmark_equity'] = 0.0

    for t in tickers:
        if t in close_series:
            prices = close_series[t].reindex(all_dates).ffill().bfill()
            start_price = prices.iloc[0]
            if start_price > 0:
                val_series = (prices / start_price) * allocation_per_ticker
                df_curve['benchmark_equity'] += val_series
            else:
                df_curve['benchmark_equity'] += allocation_per_ticker

    chart_data = []
    df_curve = df_curve.ffill().fillna(initial_capital)

    for date, row in df_curve.iterrows():
        chart_data.append({
            "date": date.strftime('%Y-%m-%d'),
            "equity": round(row['strategy_equity'], 2),
            "buy_and_hold": round(row['benchmark_equity'], 2)
        })

    return chart_data


def optimize_grid_search(tickers, drop_options, hold_options, take_profit_options,
                         initial_capital=10000.0, start: datetime = DEFAULT_START,
                         end: datetime = DEFAULT_END):
    """Grid-Search über alle Parameter-Kombinationen. Liefert die beste
    Konfiguration nach ROI samt Trades und Vergleichskurven.

    Die Such- und Bewertungslogik ist unverändert aus der ursprünglichen
    ``main.py`` übernommen.
    """
    bot = MeanReversionStrategy(initial_capital=initial_capital, start_date=start, end_date=end)

    best_roi = -999999.0
    best_result = None
    best_trades = []
    best_params = {}

    for drop in drop_options:
        for hold in hold_options:
            for tp in take_profit_options:
                current_params = {
                    "drop": drop, "lookback": 3, "hold": hold, "take_profit": tp, "fee": 0.001
                }
                trades = bot.run_portfolio(tickers, current_params)

                if trades:
                    df = pd.DataFrame(trades)
                    profit = df['profit_abs'].sum()
                    roi = (profit / initial_capital) * 100
                    win_rate = len(df[df['profit_abs'] > 0]) / len(df) * 100
                else:
                    profit = 0
                    roi = 0
                    win_rate = 0

                if roi > best_roi:
                    best_roi = roi
                    best_trades = trades
                    best_params = {"drop": drop, "hold": hold, "tp": tp}
                    best_result = {"profit": profit, "win_rate": win_rate, "count": len(trades)}

    if not best_params:
        return None

    equity_data = calculate_comparison_curves(
        best_trades, tickers, initial_capital, start, end
    )

    return {
        "best_drop": best_params['drop'],
        "best_hold": best_params['hold'],
        "best_tp": best_params['tp'],
        "total_profit": round(best_result['profit'], 2),
        "roi_pct": round(best_roi, 2),
        "win_rate": round(best_result['win_rate'], 2),
        "total_trades": best_result['count'],
        "equity_curve_data": equity_data,
        "trades": best_trades,
    }