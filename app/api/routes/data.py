"""Data endpoints for reading from and writing to InfluxDB."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_influxdb_repo
from app.core.influxdb import InfluxDBRepository
from app.models.schemas import (
    DataPoint,
    DataPointResponse,
    QueryRequest,
    QueryResponse,
)

router = APIRouter(prefix="/data", tags=["data"])

InfluxDBDep = Annotated[InfluxDBRepository, Depends(get_influxdb_repo)]


@router.post(
    "/write",
    response_model=DataPointResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Write a data point to InfluxDB",
)
async def write_data_point(point: DataPoint, repo: InfluxDBDep) -> DataPointResponse:
    """Write a single data point (measurement + tags + fields) to InfluxDB Core 3."""
    try:
        record = {
            "measurement": point.measurement,
            "tags": point.tags,
            "fields": point.fields,
        }
        if point.timestamp is not None:
            record["time"] = point.timestamp
        repo.write_point(record)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to write to InfluxDB: {exc}",
        ) from exc

    return DataPointResponse(message="Data point written successfully")


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Execute a SQL query against InfluxDB",
)
async def query_data(request: QueryRequest, repo: InfluxDBDep) -> QueryResponse:
    """Run an arbitrary SQL query against the InfluxDB Core 3 database."""
    try:
        results = repo.query(request.sql)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Query failed: {exc}",
        ) from exc

    return QueryResponse(results=results)
