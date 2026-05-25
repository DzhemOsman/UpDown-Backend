from pydantic import BaseModel

from app.schemas.api.trade_result import TradeResult


class StrategyResult(BaseModel):
    total_profit: float
    roi_pct: float
    win_rate: float
    total_number_of_trades: int
    equity_curve_data: list[dict]
    trades: list[TradeResult]