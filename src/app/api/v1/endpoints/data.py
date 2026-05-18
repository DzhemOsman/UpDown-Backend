import influxdb_client_3 as influxdb3
from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_influx_client
from app.schemas.schemas import DataWriteRequest, QueryRequest

router = APIRouter()


@router.post("/write", status_code=201)
def write_data(body: DataWriteRequest, db: influxdb3.InfluxDBClient3 = Depends(get_influx_client)):
    try:
        db.write(record={"measurement": body.measurement, "tags": body.tags, "fields": body.fields})
        return {"message": "written"}
    except Exception as exc:
        raise HTTPException(502, detail=str(exc)) from exc


@router.post("/query")
def query_data(body: QueryRequest, db: influxdb3.InfluxDBClient3 = Depends(get_influx_client)):
    try:
        result = db.query(body.sql)
        return {"results": result.to_pydict() if result is not None else []}
    except Exception as exc:
        raise HTTPException(502, detail=str(exc)) from exc
