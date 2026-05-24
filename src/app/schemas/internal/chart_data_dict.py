from typing import TypedDict


class ChartDataDict(TypedDict):
    date: str
    equity: int
    buy_and_hold: int
