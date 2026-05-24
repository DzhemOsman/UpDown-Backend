from fastapi import APIRouter

from app.api.v1.endpoints import data, predictions, strategy

api_router = APIRouter()

api_router.include_router(data.router, prefix="/data", tags=["Database"])
api_router.include_router(predictions.router, prefix="/predictions", tags=["Machine Learning"])
api_router.include_router(strategy.router, prefix="/strategy", tags=["Strategy"])
