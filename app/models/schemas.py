"""Shared Pydantic schemas used across the API."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str


class DataPoint(BaseModel):
    measurement: str = Field(..., description="InfluxDB measurement name")
    tags: dict[str, str] = Field(default_factory=dict, description="Tag key/value pairs")
    fields: dict[str, Any] = Field(..., description="Field key/value pairs")
    timestamp: datetime | None = Field(
        default=None,
        description="Optional timestamp (uses server time when omitted)",
    )


class DataPointResponse(BaseModel):
    message: str


class QueryRequest(BaseModel):
    sql: str = Field(..., description="SQL query to execute against InfluxDB")


class QueryResponse(BaseModel):
    results: list[dict[str, Any]]


class PredictionRequest(BaseModel):
    features: list[float] = Field(..., description="Input feature vector for the ML model")


class PredictionResponse(BaseModel):
    prediction: Any
    model_name: str
