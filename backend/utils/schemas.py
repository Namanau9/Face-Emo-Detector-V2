from typing import List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    labels: List[str]
    architecture: str
    image_size: int
    opencv_version: str


class PredictionResponse(BaseModel):
    emotion: str = Field(..., description="Predicted emotion label or no_face/uncertain")
    confidence: float = Field(..., ge=0.0, le=1.0)
    face_detected: bool
    face_count: int = Field(..., ge=0)
    message: Optional[str] = None
    box: Optional[List[int]] = None
