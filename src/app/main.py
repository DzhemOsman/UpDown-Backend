from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.config import get_settings
from app.core.exceptions import DataSourceError
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


@app.exception_handler(DataSourceError)
async def data_source_error_handler(
    request: Request, exc: DataSourceError
) -> JSONResponse:
    """
    Externe Datenquelle (InfluxDB/Yahoo Finance) nicht erreichbar.
    -> 503 Service Unavailable: Das Problem liegt NICHT beim Client-Request,
    sondern an einer Abhängigkeit, die gerade nicht verfügbar ist.
    """
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """
    Ungültige Eingabe, die Pydantic nicht abfängt (z.B. start_date >= end_date
    in fetch_ticker_data). -> 422 Unprocessable Entity: Request war syntaktisch
    ok, aber semantisch nicht verarbeitbar.
    """
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/health", tags=["System"])
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(api_router, prefix="/v1")
