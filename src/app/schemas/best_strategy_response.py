from pydantic import BaseModel
from app.schemas.trade_result import TradeResult

class BestStrategyResponse(BaseModel):
    best_drop: float
    best_hold: int
    best_tp: float
    total_profit: float
    roi_pct: float
    win_rate: float
    stop_loss: float
    total_trades: int
    equity_curve_data: list[dict]
    trades: list[TradeResult]
