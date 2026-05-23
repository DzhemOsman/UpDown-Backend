from typing import TypedDict


class TradeResultDict(TypedDict):
    ticker: str
    buy_date: str
    sell_date: str
    days_held: int
    exit_reason: str
    entry_price: float
    exit_price: float
    profit_pct: float
    profit_abs: float
    invested_capital: float
