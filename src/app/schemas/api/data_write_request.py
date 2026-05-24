from typing import Any

from pydantic import BaseModel, Field


class DataWriteRequest(BaseModel):
    measurement: str
    fields: dict[str, Any]
    tags: dict[str, str] = Field(default_factory=dict)
