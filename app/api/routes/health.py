"""Health-check endpoints."""

from fastapi import APIRouter

from app.config import settings
from app.models.schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse, summary="Liveness probe")
async def health_check() -> HealthResponse:
    """Return the application status and current version."""
    return HealthResponse(status="ok", version=settings.app_version)
