# UpDown Backend

FastAPI + InfluxDB Core 3 starter template for a Vite frontend.

## Stack

- [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/)
- [InfluxDB Core 3](https://docs.influxdata.com/influxdb3/core/) (`influxdb3-python`)
- NumPy (ML placeholder)

## Quick start

```bash
cp .env.example .env        # fill in INFLUXDB_TOKEN
pip install -r requirements.txt
uvicorn app.main:app --reload --app-dir src
```

`INFLUXDB_HOST`, `INFLUXDB_TOKEN`, and `INFLUXDB_DATABASE` are required — the app will refuse to start without them.

Docs at <http://localhost:8000/docs>.

## Docker

```bash
docker compose up --build
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/v1/data/write` | Write a point to InfluxDB |
| `POST` | `/v1/data/query` | SQL query against InfluxDB |
| `POST` | `/v1/predictions` | Run the ML model |

## ML model

Replace `predict_features` in `src/app/services/prediction.py` with your real model.

