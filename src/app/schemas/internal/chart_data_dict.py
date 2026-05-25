from typing import TypedDict


class ChartDataDict(TypedDict):
    date: str
    equity: float
    buy_and_hold: float
