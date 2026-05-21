from pydantic import BaseModel


class TradeResult(BaseModel):
    ticker: str
    buy_date: str
    sell_date: str
    entry_price: float
    exit_price: float
    profit_abs: float
    exit_reason: str