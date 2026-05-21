from pydantic import BaseModel, Field


class OptimizationRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=1)
    drop_options: list[float] = Field(..., min_length=1)
    hold_options: list[int] = Field(..., min_length=1)
    take_profit_options: list[float] = Field(..., min_length=1)
    initial_capital: float = 10000.0
    stop_loss: float = None