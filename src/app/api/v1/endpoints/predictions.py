from fastapi import APIRouter, HTTPException
from app.schemas.schemas import PredictBody
from app.services.prediction import predict_features

router = APIRouter()

@router.post("/predict")
def run_predict(body: PredictBody):
    if not body.features:
        raise HTTPException(422, detail="features must not be empty")
    return {"prediction": predict_features(body.features)}