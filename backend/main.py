import os
import threading
from contextlib import asynccontextmanager
from typing import Optional

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from utils.schemas import HealthResponse, PredictionResponse
from utils.service import EmotionService


service: Optional[EmotionService] = None
service_error: Optional[str] = None
service_lock = threading.Lock()


def build_service() -> EmotionService:
    model_path = os.getenv("MODEL_PATH", os.path.join("..", "saved_model", "emotion_model.keras"))
    labels_path = os.getenv("LABELS_PATH", os.path.join("..", "saved_model", "labels.json"))
    image_size = int(os.getenv("IMAGE_SIZE", "64"))
    confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.30"))
    return EmotionService(
        model_path=model_path,
        labels_path=labels_path,
        image_size=image_size,
        confidence_threshold=confidence_threshold,
    )


def load_service_background() -> None:
    global service, service_error
    try:
        loaded_service = build_service()
        with service_lock:
            service = loaded_service
            service_error = None
    except Exception as exc:
        with service_lock:
            service_error = str(exc)


def get_service_or_raise() -> EmotionService:
    if service is not None:
        return service
    if service_error:
        raise HTTPException(status_code=500, detail=f"Model failed to load: {service_error}")
    raise HTTPException(status_code=503, detail="Model is still loading. Please retry in a few seconds.")


@asynccontextmanager
async def lifespan(_: FastAPI):
    thread = threading.Thread(target=load_service_background, daemon=True)
    thread.start()
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
    current_service = service
    if current_service is None:
        return HealthResponse(
            status="error" if service_error else "loading",
            model_loaded=False,
            labels=[],
            architecture="unknown",
            image_size=int(os.getenv("IMAGE_SIZE", "64")),
            opencv_version=cv2.__version__,
        )
    return HealthResponse(
        status="ok",
        model_loaded=current_service.model is not None,
        labels=current_service.labels,
        architecture=current_service.architecture,
        image_size=current_service.image_size,
        opencv_version=cv2.__version__,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    current_service = get_service_or_raise()

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    result = current_service.predict_from_bytes(contents)
    return PredictionResponse(**result)
