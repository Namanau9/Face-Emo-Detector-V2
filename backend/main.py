import os
from contextlib import asynccontextmanager
from typing import Optional

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from utils.schemas import HealthResponse, PredictionResponse
from utils.service import EmotionService


service: Optional[EmotionService] = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global service
    model_path = os.getenv("MODEL_PATH", os.path.join("..", "saved_model", "emotion_model.keras"))
    labels_path = os.getenv("LABELS_PATH", os.path.join("..", "saved_model", "labels.json"))
    image_size = int(os.getenv("IMAGE_SIZE", "64"))
    confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.30"))

    service = EmotionService(
        model_path=model_path,
        labels_path=labels_path,
        image_size=image_size,
        confidence_threshold=confidence_threshold,
    )
    yield


app = FastAPI(
    title="Face Emotion Recognition API",
    version="1.0.0",
    description="FastAPI backend for CPU-friendly face emotion recognition.",
    lifespan=lifespan,
)

cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in cors_origins if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    assert service is not None
    return HealthResponse(
        status="ok",
        model_loaded=service.model is not None,
        labels=service.labels,
        architecture=service.architecture,
        image_size=service.image_size,
        opencv_version=cv2.__version__,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    assert service is not None

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    result = service.predict_from_bytes(contents)
    return PredictionResponse(**result)
