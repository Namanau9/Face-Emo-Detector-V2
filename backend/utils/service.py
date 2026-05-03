import json
import os
from typing import Dict, List

import cv2
import numpy as np
import tensorflow as tf

from .image import crop_and_resize_face, decode_image, largest_face


@tf.keras.utils.register_keras_serializable()
class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, kernel_size: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=kernel_size,
            padding="same",
            activation="sigmoid",
            name="attention_map",
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        attention = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.conv(attention)
        return inputs * attention

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


@tf.keras.utils.register_keras_serializable()
class ModelPreprocessor(tf.keras.layers.Layer):
    def __init__(self, architecture: str = "mobilenetv2", **kwargs):
        super().__init__(**kwargs)
        self.architecture = architecture

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        if self.architecture == "efficientnetb0":
            return tf.keras.applications.efficientnet.preprocess_input(inputs)
        return tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"architecture": self.architecture})
        return config


class EmotionService:
    def __init__(self, model_path: str, labels_path: str, image_size: int, confidence_threshold: float):
        self.model_path = os.path.abspath(model_path)
        self.labels_path = os.path.abspath(labels_path)
        self.confidence_threshold = confidence_threshold
        metadata = self._load_labels()
        self.labels = metadata["labels"]
        self.architecture = metadata.get("architecture", "mobilenetv2")
        self.image_size = int(metadata.get("image_size", image_size))
        self.model = tf.keras.models.load_model(
            self.model_path,
            custom_objects={
                "SpatialAttention": SpatialAttention,
                "ModelPreprocessor": ModelPreprocessor,
                "preprocess_input": tf.keras.applications.efficientnet.preprocess_input
                if self.architecture == "efficientnetb0"
                else tf.keras.applications.mobilenet_v2.preprocess_input,
            },
            compile=False,
        )
        self.face_detector = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        )
        if self.face_detector.empty():
            raise RuntimeError("OpenCV Haar cascade failed to load.")

    def _load_labels(self) -> Dict:
        with open(self.labels_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _preprocess_face(self, face: np.ndarray) -> np.ndarray:
        if self.architecture == "efficientnetb0":
            return tf.keras.applications.efficientnet.preprocess_input(face)
        return tf.keras.applications.mobilenet_v2.preprocess_input(face)

    def predict_from_bytes(self, file_bytes: bytes) -> Dict:
        image = decode_image(file_bytes)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(48, 48))

        if len(faces) == 0:
            return {
                "emotion": "no_face",
                "confidence": 0.0,
                "face_detected": False,
                "face_count": 0,
                "message": "No face detected. Try moving closer to the camera.",
                "box": None,
            }

        box = largest_face(faces)
        assert box is not None
        face = crop_and_resize_face(image, box, self.image_size)
        face = self._preprocess_face(face)
        face = np.expand_dims(face, axis=0)

        probs = self.model.predict(face, verbose=0)[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        emotion = self.labels[idx]
        uncertain = confidence < self.confidence_threshold

        return {
            "emotion": "uncertain" if uncertain else emotion,
            "confidence": round(confidence, 4),
            "face_detected": True,
            "face_count": int(len(faces)),
            "message": "Low confidence prediction." if uncertain else None,
            "box": [int(v) for v in box],
        }
