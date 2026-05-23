from datetime import datetime

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.schemas.api.best_strategy_response import BestStrategyResponse
from app.schemas.api.optimization_request import OptimizationRequest
from app.services import market_data, mean_reversion_strategy

router = APIRouter()

DEFAULT_START = datetime(2000, 1, 1)
DEFAULT_END = datetime.now()


@router.post("/optimize/grid-search", response_model=BestStrategyResponse)
def get_grid_search_strategy(request: OptimizationRequest):
    result = mean_reversion_strategy.optimize_grid_search(
        tickers=request.tickers,
        drop_options=request.drop_options,
        hold_options=request.hold_options,
        take_profit_options=request.take_profit_options,
        initial_capital=request.initial_capital,
    )

    if result is None:
        raise HTTPException(status_code=404, detail="Keine profitablen Trades gefunden.")

    return result

@router.get("/chart/{ticker}")
def get_chart_data(ticker: str):
    df = market_data.fetch_ticker_data(ticker, DEFAULT_START, DEFAULT_END)

    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="Ticker nicht gefunden.")

    ts = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    clean_df = pd.DataFrame(index=pd.DatetimeIndex(ts))
    clean_df["open"] = df["open"].to_numpy()
    clean_df["high"] = df["high"].to_numpy()
    clean_df["low"] = df["low"].to_numpy()
    clean_df["close"] = df["close"].to_numpy()
    clean_df.sort_index(inplace=True)
    clean_df = clean_df.ffill().bfill().fillna(0.0)

    chart_data = []
    for index, row in clean_df.iterrows():
        chart_data.append({
            "date": index.strftime('%Y-%m-%d'),
            "open": float(round(row['open'], 2)),
            "high": float(round(row['high'], 2)),
            "low": float(round(row['low'], 2)),
            "close": float(round(row['close'], 2)),
        })
    return chart_data