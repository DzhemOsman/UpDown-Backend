from pydantic import BaseModel


class OptimizationRequest(BaseModel):
    tickers: list[str]
    drop_options: list[float]
    hold_options: list[int]
    take_profit_options: list[float]
    initial_capital: float = 10000.0