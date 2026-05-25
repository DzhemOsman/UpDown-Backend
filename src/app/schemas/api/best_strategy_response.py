from typing import Optional

from pydantic import BaseModel

from app.schemas.api.trade_result import TradeResult


class BestStrategyResponse(BaseModel):
    best_drop_threshold: float
    best_hold_days: int
    best_take_profit_pct: float
    best_stop_loss_pct: Optional[float] = None
    best_max_positions: Optional[int] = None
    best_allocation_pct: Optional[float] = None
    total_profit: float
    roi_pct: float
    win_rate: float
    best_stop_loss_pct: float
    total_number_of_trades: int
    equity_curve_data: list[dict]
    trades: list[TradeResult]
