from fastapi import APIRouter

from app.api.v1.endpoints import predictions, strategy

api_router = APIRouter()

api_router.include_router(
    predictions.router, prefix="/predictions", tags=["Machine Learning"]
)
api_router.include_router(strategy.router, prefix="/strategy", tags=["Strategy"])
