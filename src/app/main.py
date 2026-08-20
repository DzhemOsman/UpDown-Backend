from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# --- BESTEHENDE IMPORTS ---
from app.api.v1.router import api_router
from app.config import get_settings
from app.core.exceptions import DataSourceError
from app.core.influx import get_client

# --- NEUER IMPORT FÜR ML ---
from app.api.v1.endpoints import backtest_ml


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_client()
    try:
        yield
    finally:
        get_client().close()
        get_client.cache_clear()


app = FastAPI(title="UpDown Backend", version="0.1.0", lifespan=lifespan)

# --- CORS ERWEITERUNG ---
# Wir laden deine bestehenden Origins und fügen die Standard-Entwicklungs-Ports
# hinzu, falls sie fehlen. Das verhindert CORS-Blockaden beim Testen.
cors_origins = get_settings().CORS_ORIGINS
if isinstance(cors_origins, str):
    # Falls Pydantic es als String lädt
    cors_origins = [cors_origins]

additional_origins = ["http://localhost:5173", "http://localhost:8080", "http://127.0.0.1:5173"]
for origin in additional_origins:
    if origin not in cors_origins:
        cors_origins.append(origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
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


# --- ROUTER REGISTRIERUNG ---

# 1. Bestehende klassische Endpunkte (Kadane, SMA, Reversion) bleiben unberührt
app.include_router(api_router, prefix="/v1")

# 2. NEU: Der isolierte Unified Backtest Router für LSTM und LightGBM
app.include_router(backtest_ml.router, prefix="/v1/strategy/backtest", tags=["Unified Backtest"])
