from datetime import datetime

from pydantic import BaseModel, Field

from app.services.mean_reversion_strategies.mean_reversion_defaults import (
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_START,
    DEFAULT_END, DEFAULT_N_TRIALS,
)


class OptimizationRequest(BaseModel):
    """Request für den Grid-Search OHNE Money Management (/optimize/grid-search)."""
    tickers: list[str] = Field(..., min_length=1)
    start_date: datetime = DEFAULT_START
    end_date: datetime = DEFAULT_END
    drop_options: list[int] = Field(..., min_length=1)
    hold_options: list[int] = Field(..., min_length=1)
    take_profit_options: list[float] = Field(..., min_length=1)
    initial_capital: int = DEFAULT_INITIAL_CAPITAL


class MoneyManagementOptimizationRequest(OptimizationRequest):
    """
    Request für die Money-Management-Optimierung
    (/optimize/money-management/grid-search und /randomized-grid-search).
    Erbt die Basisfelder und ergänzt die MM-Pflichtfelder.
    """
    stop_loss: list[float] = Field(..., min_length=1)
    max_positions: list[int] = Field(..., min_length=1)
    allocation_pct: list[float] = Field(..., min_length=1)
    is_kadane: bool = False
    is_trend: bool = False
    n_trials: int = Field(default=DEFAULT_N_TRIALS, ge=1)
