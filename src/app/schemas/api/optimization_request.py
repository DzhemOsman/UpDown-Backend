from pydantic import BaseModel, Field


DEFAULT_INITIAL_CAPITAL = 10_000

class OptimizationRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=1)
    drop_options: list[int] = Field(..., min_length=1)
    hold_options: list[int] = Field(..., min_length=1)
    take_profit_options: list[float] = Field(..., min_length=1)
    initial_capital: int = DEFAULT_INITIAL_CAPITAL
    stop_loss: float = None
