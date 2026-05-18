from fastapi import APIRouter
from app.api.v1.endpoints import data, predictions

api_router = APIRouter()

api_router.include_router(data.router, prefix="/data", tags=["Database"])
api_router.include_router(predictions.router, tags=["Machine Learning"])