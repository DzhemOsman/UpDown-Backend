from datetime import datetime

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.schemas.api.best_strategy_response import BestStrategyResponse
from app.schemas.api.optimization_request import (
    OptimizationRequest,
    MoneyManagementOptimizationRequest,
)
from app.schemas.api.strategy_request import StrategyRequest
from app.schemas.api.strategy_result import StrategyResult
from app.schemas.internal.best_parameter_combination_dict import ParameterCombinationDict
from app.services import market_data
from app.services.mean_reversion_strategies.mean_reversion_defaults import DEFAULT_START, DEFAULT_END
from app.services.mean_reversion_strategies.mean_reversion_strategy import MeanReversionStrategy
from app.services.mean_reversion_strategies.money_management_optimizer import (
    optimize_money_management_with_grid_search,
    optimize_money_management_with_randomized_grid_search,
)
from app.services.mean_reversion_strategies.money_management_reversion import MeanReversionWithMoneyManagement
from app.services.mean_reversion_strategies.optimizer import optimize_grid_search

router = APIRouter()


def _require_result(result):
    """
    Gibt das Ergebnis zurück oder wirft 422, wenn der Backtest kein Ergebnis lieferte.
    422 (nicht 404): Der Request war valide und wurde verarbeitet, es gab nur kein
    profitables Resultat — die Ressource 'fehlt' nicht, sie ist leer.
    """
    if result is None:
        raise HTTPException(
            status_code=422,
            detail="Keine profitablen Trades für die angegebenen Parameter gefunden.",
        )
    return result


@router.post("/mean-reversion", response_model=StrategyResult)
def get_mean_reversion_result(request: StrategyRequest) -> StrategyResult:
    bot = MeanReversionStrategy(
        initial_capital=request.initial_capital,
        start_date=request.start_date,
        end_date=request.end_date,
    )
    result = bot.run_portfolio_single(
        tickers=request.tickers,
        params=ParameterCombinationDict(
            drop_threshold=request.drop_option,
            lookback_days=request.lookback_days,
            hold_days=request.hold_option,
            take_profit_pct=request.take_profit_option,
            fee_pct=request.fee_pct,
        ),
    )
    return _require_result(result)


@router.post("/mean-reversion/money-management", response_model=StrategyResult)
def get_money_management_result(request: StrategyRequest) -> StrategyResult:
    bot = MeanReversionWithMoneyManagement(
        initial_capital=request.initial_capital,
        start_date=request.start_date,
        end_date=request.end_date,
    )
    result = bot.run_portfolio_with_money_management_single(
        tickers=request.tickers,
        params=ParameterCombinationDict(
            drop_threshold=request.drop_option,
            lookback_days=request.lookback_days,
            hold_days=request.hold_option,
            take_profit_pct=request.take_profit_option,
            fee_pct=request.fee_pct,
            stop_loss_pct=request.stop_loss,
            max_positions=request.max_positions,
            allocation_pct=request.allocation_pct,
        ),
        is_kadane=request.is_kadane,
        is_trend=request.is_trend,
    )
    return _require_result(result)


@router.post("/optimize/grid-search", response_model=BestStrategyResponse)
def get_optimized_grid_search_strategy(request: OptimizationRequest) -> BestStrategyResponse:
    result = optimize_grid_search(
        tickers=request.tickers,
        drop_options=request.drop_options,
        hold_options=request.hold_options,
        take_profit_options=request.take_profit_options,
        initial_capital=request.initial_capital,
        start=request.start_date,
        end=request.end_date,
    )
    return _require_result(result)


@router.post("/optimize/money-management/grid-search", response_model=BestStrategyResponse)
def get_optimized_strategy_with_money_management_and_grid_search(
        request: MoneyManagementOptimizationRequest,
) -> BestStrategyResponse:
    result = optimize_money_management_with_grid_search(
        tickers=request.tickers,
        drop_options=request.drop_options,
        hold_options=request.hold_options,
        take_profit_options=request.take_profit_options,
        stop_loss_options=request.stop_loss,
        max_positions_options=request.max_positions,
        allocation_options=request.allocation_pct,
        initial_capital=request.initial_capital,
        start=request.start_date,
        end=request.end_date,
        is_kadane=request.is_kadane,
        is_trend=request.is_trend,
    )
    return _require_result(result)


@router.post("/optimize/money-management/randomized-grid-search", response_model=BestStrategyResponse)
def get_optimized_strategy_with_money_management_and_randomized_grid_search(
        request: MoneyManagementOptimizationRequest,
) -> BestStrategyResponse:
    result = optimize_money_management_with_randomized_grid_search(
        tickers=request.tickers,
        drop_options=request.drop_options,
        hold_options=request.hold_options,
        take_profit_options=request.take_profit_options,
        stop_loss_options=request.stop_loss,
        max_positions_options=request.max_positions,
        allocation_options=request.allocation_pct,
        initial_capital=request.initial_capital,
        start=request.start_date,
        end=request.end_date,
        is_kadane=request.is_kadane,
        is_trend=request.is_trend,
        n_trials=request.n_trials,
    )
    return _require_result(result)


@router.get("/chart/{ticker}")
def get_chart_data(
        ticker: str,
        start_date: datetime = DEFAULT_START,
        end_date: datetime = DEFAULT_END,
):
    df = market_data.fetch_ticker_data(ticker, start_date, end_date)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="Ticker nicht gefunden.")

    clean_df = _prepare_ohlc_for_chart(df)

    return [
        {
            "date": row.Index.strftime('%Y-%m-%d'),
            "open": float(round(row.open, 2)),
            "high": float(round(row.high, 2)),
            "low": float(round(row.low, 2)),
            "close": float(round(row.close, 2)),
        }
        for row in clean_df.itertuples(index=True)
    ]


def _prepare_ohlc_for_chart(df: pd.DataFrame) -> pd.DataFrame:
    """Bringt rohe InfluxDB-OHLC-Daten in die fürs Frontend-Chart erwartete Form."""
    ts = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    df = df.copy()
    df.index = ts
    return df[["open", "high", "low", "close"]].sort_index().ffill().bfill().fillna(0.0)
