from datetime import datetime

from pydantic import BaseModel, Field

from app.services.mean_reversion_strategies.mean_reversion_defaults import (
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_START,
    DEFAULT_END
)


class OptimizationRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=1)
    start_date: datetime = DEFAULT_START
    end_date: datetime = DEFAULT_END
    drop_options: list[int] = None
    hold_options: list[int] = None
    take_profit_options: list[float] = None
    initial_capital: int = DEFAULT_INITIAL_CAPITAL
    stop_loss: list[float] = None
    max_positions: list[int] = None
    allocation_pct: list[float] = None
    is_kadane: bool = None
    is_trend: bool = None
    n_trials: int = None
