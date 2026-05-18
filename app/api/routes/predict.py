"""ML prediction endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.ml.model import model
from app.models.schemas import PredictionRequest, PredictionResponse

router = APIRouter(prefix="/predict", tags=["ml"])


@router.post(
    "",
    response_model=PredictionResponse,
    summary="Run the ML model on a feature vector",
)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Pass a feature vector to the ML model and return the prediction."""
    if not request.features:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="features list must not be empty",
        )
    try:
        result = model.predict(request.features)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc

    return PredictionResponse(prediction=result, model_name=model.name)
