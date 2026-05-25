import pandas as pd

from app.schemas.internal.chart_data_dict import ChartDataDict
from app.schemas.internal.strategy_result_dict import StrategyResultDict
from app.schemas.internal.trade_result_dict import TradeResultDict
from app.services.mean_reversion_strategies.mean_reversion_defaults import DEFAULT_INITIAL_CAPITAL


def calculate_comparison_curves(
        trades: list[TradeResultDict],
        ticker_data: dict[str, pd.DataFrame],
        initial_capital: int
) -> list[ChartDataDict]:
    """
    Berechnet die Vergleichskurven zwischen einer Handelsstrategie und einer Buy & Hold Benchmark.
    Die Funktion ermittelt für jeden Handelstag die kumulierte Equity der Strategie
    und vergleicht sie mit der Equity einer Buy & Hold Strategie, die gleichmäßig
    auf alle Ticker aufgeteilt ist.

    :param trades: Liste der durchgeführten Transaktionen mit Verkaufsdatum und Gewinn
    :param ticker_data: Dictionary mit Ticker-Symbolen als Keys und DataFrames mit Kursdaten als Values
    :param initial_capital: Startkapital
    :return: Liste von ChartDataDict Objekten mit Datum, Strategie-Equity und Buy & Hold Equity
    """
    # Berechnet tages genau die Strategie-Equity vs. Buy & Hold Benchmark.
    close_series: dict[str, pd.Series] = {}
    all_dates = pd.DatetimeIndex([])
    for ticker, ticker_df in ticker_data.items():
        if ticker_df is not None and not ticker_df.empty:
            timestamp = pd.to_datetime(ticker_df.index, utc=True).tz_localize(None)
            series = pd.Series(ticker_df["close"].to_numpy(), index=timestamp).sort_index()
            close_series[ticker] = series
            all_dates = all_dates.union(series.index)

    if len(all_dates) == 0:
        return []

    all_dates = all_dates.sort_values().unique()
    curve_df = pd.DataFrame(index=all_dates)

    curve_df['strategy_equity'] = 0.0
    daily_profits = pd.Series(0.0, index=all_dates)

    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df['sell_date'] = pd.to_datetime(trades_df['sell_date'])
        grouped_profits = trades_df.groupby('sell_date')['profit_abs'].sum()
        daily_profits = daily_profits.add(grouped_profits, fill_value=0)

    curve_df['strategy_equity'] = initial_capital + daily_profits.cumsum()
    curve_df['strategy_equity'] = curve_df['strategy_equity'].ffill()

    # BUY & HOLD BENCHMARK
    allocation_per_ticker = initial_capital / len(ticker_data)
    curve_df['benchmark_equity'] = 0.0

    for ticker, ticker_df in ticker_data.items():
        if ticker in close_series:
            prices = close_series[ticker].reindex(all_dates).ffill().bfill()
            start_price = prices.iloc[0]
            if start_price > 0:
                val_series = (prices / start_price) * allocation_per_ticker
                curve_df['benchmark_equity'] += val_series
            else:
                curve_df['benchmark_equity'] += allocation_per_ticker

    chart_data = []
    curve_df = curve_df.ffill().fillna(initial_capital)

    for row in curve_df.itertuples(index=True):
        chart_data.append(
            ChartDataDict(
                date=row.Index.strftime('%Y-%m-%d'),
                equity=float(round(row.strategy_equity, 2)),
                buy_and_hold=float(round(row.benchmark_equity, 2))
            )
        )

    return chart_data

def calculate_strategy_result(
        trades: list[TradeResultDict],
        ticker_data: dict[str, pd.DataFrame],
        initial_capital: int = DEFAULT_INITIAL_CAPITAL
) -> StrategyResultDict:
    total_profit = sum(trade['profit_abs'] for trade in trades)
    roi_pct = (total_profit / initial_capital) * 100
    win_rate = (sum(1 for trade in trades if trade['profit_abs'] > 0) / len(trades)) * 100 if trades else 0
    total_trades = len(trades)

    equity_curve_data = calculate_comparison_curves(trades, ticker_data, initial_capital)

    return StrategyResultDict(
        total_profit=float(round(total_profit, 2)),
        roi_pct=float(round(roi_pct, 2)),
        win_rate=float(round(win_rate, 2)),
        total_number_of_trades=int(total_trades),
        equity_curve_data=equity_curve_data,
        trades=trades
    )
