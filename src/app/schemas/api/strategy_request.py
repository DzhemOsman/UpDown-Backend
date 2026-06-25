from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.services.mean_reversion_strategies.mean_reversion_defaults import (
    DEFAULT_END,
    DEFAULT_FEE_RATE,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_N_TRIALS,
    DEFAULT_START,
)


class StrategyRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=1)
    start_date: datetime = DEFAULT_START
    end_date: datetime = DEFAULT_END
    drop_option: float = Field(..., gt=0)
    lookback_days: int = Field(..., ge=1)
    hold_option: int = Field(..., ge=1)
    take_profit_option: float = Field(..., gt=0)
    initial_capital: int = DEFAULT_INITIAL_CAPITAL
    fee_pct: float = DEFAULT_FEE_RATE
    stop_loss: Optional[float] = None
    max_positions: Optional[int] = None
    allocation_pct: Optional[float] = None
    is_kadane: bool = False
    is_trend: bool = False
    n_trials: int = DEFAULT_N_TRIALS
