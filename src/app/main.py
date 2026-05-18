from contextlib import asynccontextmanager

import influxdb_client_3 as influxdb3
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.influx_client = influxdb3.InfluxDBClient3(
        host=settings.INFLUXDB_HOST,
        token=settings.INFLUXDB_TOKEN,
        database=settings.INFLUXDB_DATABASE,
    )
    try:
        yield
    finally:
        app.state.influx_client.close()


app = FastAPI(title="UpDown Backend", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
def health():
    return {"status": "ok"}


app.include_router(api_router, prefix="/v1")
