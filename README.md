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
uvicorn main:app --reload
```

Docs at <http://localhost:8000/docs>.

## Docker

```bash
docker compose up --build
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/data/write` | Write a point to InfluxDB |
| `POST` | `/data/query` | SQL query against InfluxDB |
| `POST` | `/predict` | Run the ML model |

## ML model

Replace the `predict` function in `main.py` with your real model.
