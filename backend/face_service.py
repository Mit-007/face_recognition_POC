import cv2
import numpy as np
from insightface.app import FaceAnalysis

EMBEDDING_DIM = 512
MIN_FACE_SIZE_PX = 20
BLUR_VARIANCE_THRESHOLD = 5.0

_app = None


def get_face_app():
    global _app
    if _app is None:
        _app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _app.prepare(ctx_id=-1, det_size=(640, 640))
    return _app


def decode_image(image_bytes: bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Uploaded file is not a valid image.")
    return img


def check_blur(img) -> None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance < BLUR_VARIANCE_THRESHOLD:
        raise ValueError("Image is too blurry. Please upload a clearer photo.")


def extract_embedding(image_bytes: bytes, require_single_face: bool = True) -> np.ndarray:
    """Validates the uploaded image with OpenCV, then returns the embedding
    of the detected face. Raises ValueError on any validation failure.
    """
    img = decode_image(image_bytes)
    check_blur(img)

    app = get_face_app()
    faces = app.get(img)

    if len(faces) == 0:
        raise ValueError("No face detected in the uploaded image.")

    if require_single_face and len(faces) > 1:
        raise ValueError(
            f"Multiple faces ({len(faces)}) detected. Please upload an image with a single face."
        )

    face = faces[0]
    x1, y1, x2, y2 = face.bbox
    if (x2 - x1) < MIN_FACE_SIZE_PX or (y2 - y1) < MIN_FACE_SIZE_PX:
        raise ValueError("Detected face is too small. Please upload a closer photo.")

    return face.embedding.astype(np.float32)
