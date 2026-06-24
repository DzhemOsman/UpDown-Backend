from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.config import get_settings
from app.core.influx import get_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_client()
    try:
        yield
    finally:
        get_client().close()
        get_client.cache_clear()


app = FastAPI(title="UpDown Backend", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(api_router, prefix="/v1")
