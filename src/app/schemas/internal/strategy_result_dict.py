from typing import TypedDict

from app.schemas.internal.chart_data_dict import ChartDataDict
from app.schemas.internal.trade_result_dict import TradeResultDict


class StrategyResultDict(TypedDict):
    total_profit: float
    roi_pct: float
    win_rate: float
    total_number_of_trades: int
    equity_curve_data: list[ChartDataDict]
    trades: list[TradeResultDict]
