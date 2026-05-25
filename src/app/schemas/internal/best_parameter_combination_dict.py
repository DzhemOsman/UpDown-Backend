from typing import TypedDict, NotRequired, Optional

from app.schemas.internal.chart_data_dict import ChartDataDict
from app.schemas.internal.trade_result_dict import TradeResultDict


class BestParameterCombinationDict(TypedDict):
    best_drop_threshold: float
    best_hold_days: int
    best_take_profit_pct: float
    best_stop_loss_pct: NotRequired[Optional[float]]
    best_max_positions: NotRequired[Optional[int]]
    best_allocation_pct: NotRequired[Optional[float]]
    total_profit: float
    roi_pct: float
    win_rate: float
    total_number_of_trades: int
    equity_curve_data: list[ChartDataDict]
    trades: list[TradeResultDict]


class ParameterCombinationDict(TypedDict):
    drop_threshold: float
    lookback_days: int
    hold_days: int
    take_profit_pct: float
    fee_pct: float
    stop_loss_pct: float | None
    max_positions: int | None
    allocation_pct: float| None


class BestResultDict(TypedDict):
    profit: float
    win_rate: float
    total_number_of_trades: int
