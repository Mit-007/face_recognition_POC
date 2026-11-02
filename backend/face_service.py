import os
import cv2
import numpy as np

EMBEDDING_DIM = 512
MIN_FACE_SIZE_PX = 20
BLUR_VARIANCE_THRESHOLD = 5.0

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
YUNET_MODEL = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_MODEL = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec_int8.onnx")
SFACE_FALLBACK = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

_detector = None
_recognizer = None


def get_detector(input_size: tuple[int, int] = (320, 320)):
    global _detector
    if _detector is None:
        if not os.path.exists(YUNET_MODEL):
            raise FileNotFoundError(
                f"YuNet model not found at {YUNET_MODEL}. Please run download_models.py."
            )
        _detector = cv2.FaceDetectorYN.create(
            model=YUNET_MODEL,
            config="",
            input_size=input_size,
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000,
        )
    return _detector


def get_recognizer():
    global _recognizer
    if _recognizer is None:
        model_path = SFACE_MODEL if os.path.exists(SFACE_MODEL) else SFACE_FALLBACK
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"SFace model not found at {model_path}. Please run download_models.py."
            )
        _recognizer = cv2.FaceRecognizerSF.create(model=model_path, config="")
    return _recognizer


# Backward compatibility alias
def get_face_app():
    return get_detector()


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
    """Validates the uploaded image with OpenCV, detects the face using YuNet,
    extracts the SFace (int8) embedding, pads to 512-d for Pinecone compatibility,
    and returns bounding box, keypoints, and dimensions.
    """
    img = decode_image(image_bytes)
    check_blur(img)

    h, w = img.shape[:2]
    detector = get_detector((w, h))
    detector.setInputSize((w, h))
    ret, faces = detector.detect(img)

    if faces is None or len(faces) == 0:
        raise ValueError("No face detected in the uploaded image.")

    if require_single_face and len(faces) > 1:
        raise ValueError(
            f"Multiple faces ({len(faces)}) detected. Please upload an image with a single face."
        )

    face = faces[0]
    x, y, box_w, box_h = [float(v) for v in face[0:4]]
    if box_w < MIN_FACE_SIZE_PX or box_h < MIN_FACE_SIZE_PX:
        raise ValueError("Detected face is too small. Please upload a closer photo.")

    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(w), x + box_w)
    y2 = min(float(h), y + box_h)

    # 5 facial landmarks from YuNet (right eye, left eye, nose, right mouth, left mouth)
    kps = face[4:14].reshape(5, 2).tolist()
    det_score = float(face[14])

    # Crop and align face for SFace
    recognizer = get_recognizer()
    aligned_face = recognizer.alignCrop(img, face)
    feature_128 = recognizer.feature(aligned_face)

    # L2 normalize the 128-d feature
    norm = np.linalg.norm(feature_128)
    if norm > 0:
        feature_128 = feature_128 / norm

    # Zero-pad normalized 128-d vector to 512 dimensions for Pinecone 512-d index compatibility.
    # Note: dot(p1, p2) == dot(u1, u2), preserving 100% of the SFace cosine similarity metric.
    embedding_512 = np.pad(feature_128.flatten(), (0, 384)).astype(np.float32)

    return {
        "embedding": embedding_512,
        "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        "kps": [[round(pt[0], 1), round(pt[1], 1)] for pt in kps],
        "det_score": round(det_score, 3),
        "image_dimensions": {
            "width": int(w),
            "height": int(h),
        },
    }


def extract_embedding(image_bytes: bytes, require_single_face: bool = True) -> np.ndarray:
    return extract_face_data(image_bytes, require_single_face=require_single_face)["embedding"]
