from typing import Optional, Tuple

import cv2
import numpy as np


def decode_image(file_bytes: bytes) -> np.ndarray:
    buffer = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image bytes.")
    return image


def largest_face(faces: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if len(faces) == 0:
        return None
    return max(faces, key=lambda rect: int(rect[2] * rect[3]))


def crop_and_resize_face(image: np.ndarray, box: Tuple[int, int, int, int], image_size: int) -> np.ndarray:
    x, y, w, h = box
    x0 = max(x, 0)
    y0 = max(y, 0)
    x1 = min(x + w, image.shape[1])
    y1 = min(y + h, image.shape[0])
    face = image[y0:y1, x0:x1]
    if face.size == 0:
        raise ValueError("Detected face crop was empty.")
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return face.astype("float32")
