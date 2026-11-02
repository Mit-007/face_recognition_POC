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
        _app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        _app.prepare(ctx_id=-1, det_size=(640, 640))
    return _app


def decode_image(image_bytes: bytes, max_dim: int = 1280):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Uploaded file is not a valid image.")
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def check_blur(img) -> None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance < BLUR_VARIANCE_THRESHOLD:
        raise ValueError("Image is too blurry. Please upload a clearer photo.")


def extract_face_data(image_bytes: bytes, require_single_face: bool = True) -> dict:
    """Validates the uploaded image with OpenCV, detects the face, and returns
    a dictionary containing the 512-d embedding, bounding box, 5 keypoints,
    detection score, and image dimensions. Raises ValueError on any validation failure.
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
    x1, y1, x2, y2 = [float(v) for v in face.bbox]
    if (x2 - x1) < MIN_FACE_SIZE_PX or (y2 - y1) < MIN_FACE_SIZE_PX:
        raise ValueError("Detected face is too small. Please upload a closer photo.")

    kps = face.kps.tolist() if hasattr(face, "kps") and face.kps is not None else []
    det_score = float(face.det_score) if hasattr(face, "det_score") and face.det_score is not None else 1.0

    return {
        "embedding": face.embedding.astype(np.float32),
        "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        "kps": [[round(pt[0], 1), round(pt[1], 1)] for pt in kps],
        "det_score": round(det_score, 3),
        "image_dimensions": {
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
        },
    }


def extract_embedding(image_bytes: bytes, require_single_face: bool = True) -> np.ndarray:
    """Validates the uploaded image with OpenCV, then returns the embedding
    of the detected face. Raises ValueError on any validation failure.
    """
    return extract_face_data(image_bytes, require_single_face=require_single_face)["embedding"]
