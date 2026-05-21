from __future__ import annotations

from datetime import datetime
from pprint import pprint

import numpy as np
import pandas as pd

from app.services import market_data

# Standard-Backtest-Fenster (entspricht dem bisherigen DataManager-Default).
DEFAULT_START = datetime(2000, 1, 1)
DEFAULT_END = datetime.now()


class OptimizedMeanReversionStrategy:
    """Optimierte Mean-Reversion-Strategie.

    Basiert direkt auf ``unoptimized_mean_reversion.MeanReversionStrategy``.
    Struktur, Methodennamen, Signal- und Entry-Logik sind unverändert.
    Hinzugekommen sind vier Erweiterungen aus dem Projektstrukturplan:

    1. Gleitender Durchschnitt (SMA) als zusätzliche Spalte.
    2. Trend-Filter: Es wird nur gekauft, wenn der Kurs über dem SMA liegt
       (Mean Reversion nur im Aufwärtstrend -> kein "fallendes Messer").
    3. Stopp-Loss: zusätzlicher Ausstieg, wenn der Kurs zu weit fällt.
    4. Maximale Abschnittssuche (Kadane) für das Exit-Timing: ein laufender
       Trailing-Exit, der aussteigt, sobald der bis dahin aufgebaute
       Gewinn-Lauf wieder aufgezehrt wird. Arbeitet ohne Blick in die
       Zukunft, passt also sauber in die bestehende Halte-Schleife.
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
            # NEU: Low wird für den Stopp-Loss gebraucht.
            clean_df["Low"] = df["low"].to_numpy()

            clean_df.sort_index(inplace=True)

            # Lücken füllen
            clean_df.ffill(inplace=True)
            return clean_df
        except Exception as e:  # noqa: BLE001
            print(f"Fehler beim Aufbereiten von {ticker}: {e}")
            return None

    def backtest(self, ticker, drop_threshold_pct, lookback_days, hold_days,
                 take_profit_pct, fee_rate,
                 stop_loss_pct=10.0, sma_window=200):
        """
        Führt den Backtest durch.

        Neue Parameter (mit Standardwerten, damit alte Aufrufe weiter
        funktionieren):
          stop_loss_pct : Verlustgrenze in % unter dem Einstiegspreis.
          sma_window    : Fenster für den gleitenden Durchschnitt (Trend).
        """
        df = self.load_and_clean_data(ticker)
        if df is None or df.empty:
            return []

        df['change'] = df['Close'].pct_change(periods=lookback_days)

        # NEU (1) Gleitender Durchschnitt als Trend-Indikator.
        df['sma'] = df['Close'].rolling(window=sma_window).mean()

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

            # NEU (2) Trend-Filter: Nur kaufen, wenn der Kurs am Tag des
            # Signals über dem gleitenden Durchschnitt liegt. Liegt er
            # darunter (Abwärtstrend) oder ist der SMA noch nicht definiert
            # (zu wenig Historie), wird das Signal übersprungen.
            sma_value = df['sma'].iloc[idx]
            signal_close = df['Close'].iloc[idx]
            if np.isnan(sma_value) or signal_close < sma_value:
                continue

            entry_date = df.index[entry_idx]
            raw_entry_price = df['Open'].iloc[entry_idx]

            effective_entry_price = raw_entry_price * (1 + fee_rate)

            target_price = raw_entry_price * (1 + take_profit_pct / 100)
            # NEU (3) Stopp-Loss-Preis.
            stop_price = raw_entry_price * (1 - stop_loss_pct / 100)

            exit_price = None
            exit_date = None
            exit_reason = ""
            days_held = 0

            found_exit = False

            # NEU (4) Kadane / maximale Abschnittssuche für das Exit-Timing.
            # Wir verfolgen tagesweise die Tagesrenditen seit dem Einstieg.
            # "run_sum" ist der laufende Gewinn-Abschnitt (wie bei Kadane).
            # Solange der Kurs steigt, wächst er; bricht der Kurs ein, fällt
            # er. Sinkt er deutlich unter sein bisheriges Maximum, ist der
            # beste zusammenhängende Gewinn-Lauf vorbei -> wir steigen aus.
            run_sum = 0.0          # aktueller Abschnitt (Summe der Renditen)
            max_run_sum = 0.0      # bestes bisher gesehenes Maximum
            prev_close = raw_entry_price
            # Wie viel Rückgang vom Hoch wir tolerieren, bevor wir verkaufen.
            kadane_giveback = 0.02  # 2 Prozentpunkte

            for i in range(0, hold_days):
                current_idx = entry_idx + i

                if current_idx >= len(df):
                    break

                current_high = df['High'].iloc[current_idx]
                current_low = df['Low'].iloc[current_idx]
                current_open = df['Open'].iloc[current_idx]
                current_close = df['Close'].iloc[current_idx]
                current_date = df.index[current_idx]

                can_sell_at_open = (i > 0)

                # --- Kadane-Update zuerst (mit dem heutigen Tagesschluss) ---
                # So wird ein Einbruch noch am selben Tag erkannt und nicht
                # erst am nächsten. Wir verkaufen frühestens zum Close dieses
                # Tages -> kein Blick in die Zukunft.
                daily_return = (current_close - prev_close) / prev_close
                run_sum = max(0.0, run_sum + daily_return)
                if run_sum > max_run_sum:
                    max_run_sum = run_sum
                prev_close = current_close

                # --- Exit-Prüfungen in Prioritäts-Reihenfolge ---

                # (a) Stopp-Loss: zuerst, weil Risikobegrenzung Vorrang hat.
                if current_low <= stop_price:
                    # Konservativ: am Open verkaufen, wenn dieses schon
                    # unter dem Stopp lag (Gap-Down), sonst zum Stopp-Preis.
                    if can_sell_at_open and current_open < stop_price:
                        raw_exit_price = current_open
                    else:
                        raw_exit_price = stop_price
                    exit_reason = "Stop Loss"
                    days_held = i
                    found_exit = True

                # (b) Take Profit (unveränderte Original-Logik).
                elif current_high >= target_price:
                    if can_sell_at_open and current_open > target_price:
                        raw_exit_price = current_open
                    else:
                        raw_exit_price = target_price
                    exit_reason = "Take Profit"
                    days_held = i
                    found_exit = True

                # (c) Kadane-Exit: Gewinn-Lauf aufgebraucht. Sobald der
                # laufende Abschnitt mehr als "giveback" unter sein bisheriges
                # Maximum gefallen ist, ist der beste zusammenhängende
                # Gewinn-Lauf vorbei -> wir verkaufen zum heutigen Close.
                elif i > 0 and max_run_sum > 0 and (max_run_sum - run_sum) >= kadane_giveback:
                    raw_exit_price = current_close
                    exit_reason = "Kadane Exit"
                    days_held = i
                    found_exit = True

                # (d) Haltedauer erreicht (unveränderte Original-Logik).
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
                fee_rate=fee,
                # Neue Parameter mit Fallback auf die Standardwerte.
                stop_loss_pct=params.get('stop_loss', 10.0),
                sma_window=params.get('sma_window', 200),
            )
            if trades:
                all_trades.extend(trades)
        return all_trades

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
                         end: datetime = DEFAULT_END,
                         stop_loss_options=None, sma_window=200):
    """Grid-Search über alle Parameter-Kombinationen. Liefert die beste
    Konfiguration nach ROI samt Trades und Vergleichskurven.

    Such- und Bewertungslogik unverändert aus dem Original. Neu ist nur,
    dass optional auch über Stopp-Loss-Werte gesucht wird; wird nichts
    übergeben, bleibt der Standard (10 %) gesetzt.
    """
    bot = OptimizedMeanReversionStrategy(
        initial_capital=initial_capital, start_date=start, end_date=end
    )

    if stop_loss_options is None:
        stop_loss_options = [10.0]

    best_roi = -999999.0
    best_result = None
    best_trades = []
    best_params = {}

    for drop in drop_options:
        for hold in hold_options:
            for tp in take_profit_options:
                for sl in stop_loss_options:
                    current_params = {
                        "drop": drop, "lookback": 3, "hold": hold,
                        "take_profit": tp, "fee": 0.001,
                        "stop_loss": sl, "sma_window": sma_window,
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
                        best_params = {"drop": drop, "hold": hold, "tp": tp, "stop_loss": sl}
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
        "best_stop_loss": best_params['stop_loss'],
        "total_profit": round(best_result['profit'], 2),
        "roi_pct": round(best_roi, 2),
        "win_rate": round(best_result['win_rate'], 2),
        "stop_loss": stop_loss_options,
        "total_trades": best_result['count'],
        "equity_curve_data": equity_data,
        "trades": best_trades,
    }

def main():
    tickers = ["AAPL", "MSFT", "DBK"]

    result = optimize_grid_search(
        tickers=tickers,
        drop_options=[3, 4, 5, 6, 7],
        hold_options=[2, 3, 4, 5, 6],
        take_profit_options=[1.5, 2.0, 2.5, 3.0],
        stop_loss_options=[5.0, 7.5, 10.0],
        initial_capital=10000.0,
        start=datetime(2018, 1, 1),
        end=datetime(2024, 12, 31),
    )

    if result is None:
        print("Keine gültige Konfiguration gefunden.")
        return

    print("\n=== BESTE KONFIGURATION ===")
    print(f"Drop Threshold:   {result['best_drop']}%")
    print(f"Hold Days:        {result['best_hold']}")
    print(f"Take Profit:      {result['best_tp']}%")
    print(f"Stop Loss:        {result['best_stop_loss']}%")

    print("\n=== PERFORMANCE ===")
    print(f"ROI:              {result['roi_pct']}%")
    print(f"Total Profit:     {result['total_profit']}")
    print(f"Win Rate:         {result['win_rate']}%")
    print(f"Total Trades:     {result['total_trades']}")
    print(f"Search Type:      {result.get('search_type', 'grid')}")
    print(f"Trials:           {result.get('trials', 'n/a')}")

    """print("\n=== ERSTE 5 TRADES ===")
    for trade in result["trades"][:5]:
        pprint(trade)

    print("\n=== ERSTE 5 EQUITY-PUNKTE ===")
    for point in result["equity_curve_data"][:5]:
        pprint(point)
    """

if __name__ == "__main__":
    main()