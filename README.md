# Face Emo Detector V2

A production-ready face emotion recognition web app built with TensorFlow/Keras, FastAPI, Next.js, TailwindCSS, and OpenCV.

It trains on a FER-style dataset, serves predictions through a FastAPI backend, and delivers a modern webcam-based frontend with live emotion feedback, confidence smoothing, screenshot capture, and CSV logging.

## Highlights

- EfficientNetB0 and MobileNetV2 training options for CPU-friendly deployment
- Spatial attention layer for stronger feature focus
- FastAPI backend with singleton model loading and face detection
- Next.js frontend with webcam streaming, live prediction updates, and animated emotion UI
- Rolling-average smoothing to reduce flickering predictions
- Emotion history graph, screenshot capture, CSV log download, and dark/light mode
- Render deployment config included

## Demo Features

- Live webcam emotion detection
- Confidence score output
- Graceful no-face handling
- Multiple-face handling by selecting the largest visible face
- Emotion-reactive avatar and glow theme
- Last-30-seconds confidence history chart

## Tech Stack

- Model training: TensorFlow / Keras
- Backend: FastAPI + Uvicorn
- Face detection: OpenCV Haarcascade
- Frontend: Next.js + TailwindCSS
- Deployment: Render

## Project Structure

```text
root/
|-- backend/
|   |-- main.py
|   |-- requirements.txt
|   |-- tests/
|   `-- utils/
|-- dataset/
|-- docs/
|-- frontend/
|   |-- components/
|   |-- pages/
|   |-- styles/
|   `-- package.json
|-- saved_model/
|-- training/
|   |-- dataset_loader.py
|   |-- requirements.txt
|   `-- train.py
|-- render.yaml
`-- README.md
```

## Dataset Format

```text
dataset/
  train/
    angry/
    disgust/
    fear/
    happy/
    neutral/
    sad/
    surprise/
  test/
    angry/
    disgust/
    fear/
    happy/
    neutral/
    sad/
    surprise/
```

## Local Setup

### 1. Training environment

```bash
cd training
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Backend environment

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Frontend environment

```bash
cd frontend
npm install
```

Create `frontend/.env.local`:

```env
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

## Training

### Recommended model

EfficientNetB0 is the current best default for accuracy:

```bash
python train.py --architecture efficientnetb0 --batch-size 48 --epochs 24 --fine-tune-epochs 12
```

### Faster model

Use MobileNetV2 if you want lower CPU inference cost:

```bash
python train.py --architecture mobilenetv2 --image-size 96 --batch-size 48 --epochs 24 --fine-tune-epochs 12
```

### Training outputs

Artifacts are written to `saved_model/`:

- `emotion_model.keras`
- `labels.json`
- `training_log.csv`
- `training_summary.json`
- `checkpoints/best_model.keras`

`labels.json` stores the trained model architecture and image size, and the backend reads those automatically during inference.

## Run the App

### Backend

From `backend/`:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Useful URLs:

- [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Important backend environment variables:

```env
MODEL_PATH=../saved_model/emotion_model.keras
LABELS_PATH=../saved_model/labels.json
IMAGE_SIZE=128
CONFIDENCE_THRESHOLD=0.30
CORS_ORIGINS=http://localhost:3000
```

### Frontend

From `frontend/`:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## API Contract

### `POST /predict`

Example success response:

```json
{
  "emotion": "happy",
  "confidence": 0.92,
  "face_detected": true,
  "face_count": 1,
  "message": null,
  "box": [120, 84, 166, 166]
}
```

Example no-face response:

```json
{
  "emotion": "no_face",
  "confidence": 0.0,
  "face_detected": false,
  "face_count": 0,
  "message": "No face detected. Try moving closer to the camera.",
  "box": null
}
```

### `GET /health`

Returns model availability, labels, architecture, image size, and OpenCV version.

## API Test Script

From `backend/`:

```bash
python tests/test_api.py --image ..\dataset\test\happy\PrivateTest_10714097.jpg
```

## Deployment on Render

This repository includes [render.yaml](/F:/Projects/Face%20Emotion%20V2/render.yaml) for:

1. `face-emotion-api`
2. `face-emotion-frontend`

### Backend service

- Root directory: `backend`
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Frontend service

- Root directory: `frontend`
- Build command: `npm install && npm run build`
- Start command: `npm run start`

### Required environment variables

- `MODEL_PATH=../saved_model/emotion_model.keras`
- `LABELS_PATH=../saved_model/labels.json`
- `IMAGE_SIZE=128`
- `CONFIDENCE_THRESHOLD=0.30`
- `CORS_ORIGINS=https://your-frontend-domain.onrender.com`
- `NEXT_PUBLIC_API_BASE_URL=https://your-backend-domain.onrender.com`

### Important deployment note

The repository currently ignores `dataset/` and generated trained model artifacts. That keeps GitHub clean, but it also means Render will not automatically receive your local trained model unless you:

- commit the model artifacts intentionally, or
- upload them from managed storage at deploy/start time, or
- retrain within your deployment workflow

## Performance Notes

- EfficientNetB0 is the best current default for accuracy
- MobileNetV2 is the better option if you need faster CPU inference
- Sampling every 500ms gives a good balance of responsiveness and backend load
- Lower frontend frame size if latency becomes noticeable on weaker systems

## Edge Cases Covered

- No face detected
- Multiple faces in frame
- Low-confidence predictions
- Webcam permission denied
- Backend unavailable during live capture

## Screenshots

Add screenshots here to improve the GitHub landing page:

- [Live Dashboard Placeholder](/F:/Projects/Face%20Emotion%20V2/docs/live-dashboard-placeholder.txt)
- [Prediction Panel Placeholder](/F:/Projects/Face%20Emotion%20V2/docs/prediction-panel-placeholder.txt)

## Suggested GitHub Repo Description

Production-ready face emotion recognition web app using EfficientNetB0, FastAPI, Next.js, TailwindCSS, and OpenCV with live webcam inference.

## Suggested GitHub Topics

`emotion-recognition`, `face-detection`, `computer-vision`, `fastapi`, `nextjs`, `tensorflow`, `keras`, `opencv`, `tailwindcss`, `render`

## Future Improvements

- Add a confusion matrix and per-class metrics report
- Replace Haarcascade with a lightweight DNN face detector
- Add Grad-CAM visualization for explainability
- Store prediction history in a database
