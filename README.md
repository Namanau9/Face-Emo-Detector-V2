# Face Emotion Recognition Web App

A production-ready full-stack face emotion recognition app built with TensorFlow/Keras, FastAPI, Next.js, TailwindCSS, and OpenCV. It trains on the dataset already present in this repository and serves low-latency webcam predictions through a modern UI.

## Features

- MobileNetV2 and EfficientNetB0 transfer learning options tuned for CPU-friendly inference
- Spatial attention block on top of backbone features
- Data augmentation, validation split, checkpointing, early stopping, fine-tuning, and model export
- FastAPI backend that loads the model once and exposes `/health` and `/predict`
- OpenCV Haarcascade face detection with graceful `no_face` handling
- Frontend webcam capture loop with rolling-average smoothing to reduce prediction flicker
- Emotion confidence meter, animated emoji avatar, adaptive glow theme, dark/light mode
- Emotion confidence history chart for the last 30 seconds
- Screenshot capture and CSV log download
- Render deployment config for frontend and backend

## Project Structure

```text
root/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── tests/
│   └── utils/
├── dataset/
├── frontend/
│   ├── components/
│   ├── pages/
│   ├── styles/
│   └── package.json
├── saved_model/
├── training/
│   ├── dataset_loader.py
│   ├── requirements.txt
│   └── train.py
├── render.yaml
└── README.md
```

## 1. Local Setup

### Backend and training

```bash
cd training
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm install
```

Create `frontend/.env.local`:

```env
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

## 2. Train the Model

The dataset is expected at:

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
    same structure
```

Run training from the `training/` folder:

Recommended stronger baseline:

```bash
python train.py --architecture efficientnetb0 --batch-size 48 --epochs 24 --fine-tune-epochs 12
```

Faster alternative:

```bash
python train.py --architecture mobilenetv2 --image-size 96 --batch-size 48 --epochs 24 --fine-tune-epochs 12
```

Artifacts are written to `saved_model/`:

- `emotion_model.keras`
- `labels.json`
- `training_log.csv`
- `training_summary.json`
- `checkpoints/best_model.keras`

`labels.json` now also stores the trained `architecture` and `image_size`, and the backend reads that automatically.

## 3. Run the Backend

From `backend/`:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Important environment variables:

```env
MODEL_PATH=../saved_model/emotion_model.keras
LABELS_PATH=../saved_model/labels.json
IMAGE_SIZE=128
CONFIDENCE_THRESHOLD=0.30
CORS_ORIGINS=http://localhost:3000
```

### API contract

`POST /predict`

Returns:

```json
{
  "emotion": "happy",
  "confidence": 0.92,
  "face_detected": true,
  "face_count": 1,
  "message": null,
  "box": [x, y, w, h]
}
```

When no face is found:

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

## 4. Run the Frontend

From `frontend/`:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## 5. Test the API

From `backend/`:

```bash
python tests/test_api.py --image ..\dataset\test\happy\PrivateTest_10714097.jpg
```

## 6. Render Deployment

This repo includes [`render.yaml`](/F:/Projects/Face%20Emotion%20V2/render.yaml) for two services:

1. `face-emotion-api`
2. `face-emotion-frontend`

### Backend Render settings

- Root directory: `backend`
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Frontend Render settings

- Root directory: `frontend`
- Build command: `npm install && npm run build`
- Start command: `npm run start`

### Required env vars

- `MODEL_PATH=../saved_model/emotion_model.keras`
- `LABELS_PATH=../saved_model/labels.json`
- `IMAGE_SIZE=128`
- `CONFIDENCE_THRESHOLD=0.30`
- `CORS_ORIGINS=https://your-frontend-domain.onrender.com`
- `NEXT_PUBLIC_API_BASE_URL=https://your-backend-domain.onrender.com`

### Deployment note

Render needs access to the trained model artifacts in `saved_model/`. Train locally first and push the exported model files you want deployed, or change deployment to download them from managed storage at build/start time.

## 7. Screenshots

Add screenshots here after running the app:

- `docs/screenshot-dashboard.png`
- `docs/screenshot-live-prediction.png`

## 8. Edge Cases Covered

- No face detected
- Multiple faces in frame: the largest face is used for prediction
- Low-confidence predictions return `uncertain`
- Frontend smoothing reduces rapid flicker between adjacent classes
- Backend model is loaded once on startup for low latency

## 9. Performance Tips

- `EfficientNetB0` is the strongest current default in this project for accuracy while staying reasonably lightweight
- `MobileNetV2` remains the better choice if you need faster CPU inference
- Use `opencv-python-headless` in deployment
- Sample frames at 500ms to balance responsiveness and backend load
- If latency is high, lower the client capture resolution from `256x256`

## 10. Future Improvements

- Replace Haarcascade with a lightweight face detector DNN
- Add Grad-CAM visualization page for model explainability
- Store logs in a database for long-term analytics
