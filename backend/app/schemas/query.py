from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question")
    appliance_hint: Optional[str] = Field(
        None, description="Optional detected appliance manual filename, e.g. from the vision endpoint"
    )
    k: int = Field(3, ge=1, le=10, description="Number of chunks to retrieve")


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


class DetectionResult(BaseModel):
    appliance: Optional[str]
    confidence: Optional[float]
    manual_source: Optional[str]
