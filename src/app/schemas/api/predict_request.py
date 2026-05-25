from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    ticker: str = Field(..., description="Das Ticker-Symbol der Aktie, z.B. 'AAPL'", example="AAPL")
    date: str = Field(..., description="Das gewünschte Analysedatum im Format YYYY-MM-DD", example="2024-01-15")