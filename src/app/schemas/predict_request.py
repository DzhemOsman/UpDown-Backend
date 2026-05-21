from pydantic import BaseModel


class PredictRequest(BaseModel):
    features: list[float]