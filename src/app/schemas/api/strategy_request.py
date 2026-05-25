from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.services.mean_reversion_strategies.mean_reversion_defaults import DEFAULT_START, DEFAULT_END, \
    DEFAULT_INITIAL_CAPITAL


class StrategyRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=1)
    start_date: datetime = DEFAULT_START
    end_date: datetime = DEFAULT_END
    drop_option: int = 0
    lookback_days: int = 0
    hold_option: int = 0
    take_profit_option: float = 0.0
    initial_capital: int = DEFAULT_INITIAL_CAPITAL
    fee_pct: float = 0.01
    stop_loss: Optional[float] = None
    max_positions: Optional[int] = None
    allocation_pct: Optional[float] = None
    is_kadane: Optional[bool] = None
    is_trend: Optional[bool] = None
    n_trials: Optional[int] = None