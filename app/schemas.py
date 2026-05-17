from pydantic import BaseModel
from typing import List

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    probabilities: List[float]
    model_version: str = "1.0.0"

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str = "1.0.0"

class ErrorResponse(BaseModel):
    error: str
    detail: str
