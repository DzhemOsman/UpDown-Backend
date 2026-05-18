import os

import influxdb_client_3 as influxdb3
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config (set via environment variables or .env)
# ---------------------------------------------------------------------------
INFLUXDB_HOST = os.getenv("INFLUXDB_HOST", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "")
INFLUXDB_DATABASE = os.getenv("INFLUXDB_DATABASE", "updown")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="UpDown Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# InfluxDB helper
# ---------------------------------------------------------------------------
def get_db() -> influxdb3.InfluxDBClient3:
    return influxdb3.InfluxDBClient3(
        host=INFLUXDB_HOST,
        token=INFLUXDB_TOKEN,
        database=INFLUXDB_DATABASE,
    )


# ---------------------------------------------------------------------------
# ML placeholder — replace with your real model
# ---------------------------------------------------------------------------
def predict(features: list[float]):
    return float(np.mean(features))


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class WriteBody(BaseModel):
    measurement: str
    tags: dict = {}
    fields: dict


class QueryBody(BaseModel):
    sql: str


class PredictBody(BaseModel):
    features: list[float]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/data/write", status_code=201)
def write_data(body: WriteBody):
    db = get_db()
    try:
        db.write(record={"measurement": body.measurement, "tags": body.tags, "fields": body.fields})
    except Exception as exc:
        raise HTTPException(502, detail=str(exc)) from exc
    finally:
        db.close()
    return {"message": "written"}


@app.post("/data/query")
def query_data(body: QueryBody):
    db = get_db()
    try:
        result = db.query(body.sql)
        return {"results": result.to_pydict() if result is not None else []}
    except Exception as exc:
        raise HTTPException(502, detail=str(exc)) from exc
    finally:
        db.close()


@app.post("/predict")
def run_predict(body: PredictBody):
    if not body.features:
        raise HTTPException(422, detail="features must not be empty")
    return {"prediction": predict(body.features)}
