import math
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog

# The pipeline runs on the CPU, and OpenCL support costs about 100 MB of RAM
cv2.ocl.setUseOpenCL(False)

# Thresholds and configuration
GRID_SIZE = 8
CELL_SIZE = 16  # 128 / 8
TARGET_SIZE = (128, 128)

# YuNet face detector settings
DETECTION_SCORE_THRESHOLD = 0.7
DETECTION_NMS_THRESHOLD = 0.3
DETECTION_TOP_K = 50
MAX_DETECTION_SIDE = 640  # Larger photos are shrunk for detection, which is more reliable

# Every face is warped so these landmarks land as close as possible to the same
# spots in the output image: eyes, nose tip, then mouth corners (as fractions of its size)
ALIGN_WORK_SIZE = 256  # Warped at twice the final size, then shrunk, to avoid aliasing
LANDMARK_TEMPLATE = np.array(
    [
        [0.300, 0.360],
        [0.700, 0.360],
        [0.500, 0.587],
        [0.334, 0.821],
        [0.666, 0.821],
    ],
    dtype=np.float64,
)

# Another face this large relative to the main face means the photo shows two people
SECOND_FACE_AREA_RATIO = 0.30
MAX_EYE_TILT_DEGREES = 25.0


@dataclass(frozen=True)
class QualityLimits:
    min_face_px: int
    min_sharpness: float
    min_brightness: float
    max_brightness: float
    max_side_imbalance: float
    max_nose_offset: float


# The registered photo is used for every later match, so it has the stricter limits
REGISTRATION_LIMITS = QualityLimits(
    min_face_px=90,
    min_sharpness=60.0,
    min_brightness=60.0,
    max_brightness=200.0,
    max_side_imbalance=0.25,
    max_nose_offset=0.15,
)
DETECTION_LIMITS = QualityLimits(
    min_face_px=70,
    min_sharpness=40.0,
    min_brightness=45.0,
    max_brightness=220.0,
    max_side_imbalance=0.35,
    max_nose_offset=0.25,
)

# Path to the YuNet face detection model
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
FACE_DETECTOR_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")

_face_detector: Optional[cv2.FaceDetectorYN] = None

# In-memory feature cache for registered users: {image_name: features_dict}
_feature_cache: Dict[str, Dict[str, Any]] = {}


def get_face_detector() -> cv2.FaceDetectorYN:
    global _face_detector
    if _face_detector is None:
        if not os.path.exists(FACE_DETECTOR_PATH):
            raise FileNotFoundError(
                f"YuNet face detector not found at {FACE_DETECTOR_PATH}. Run download_models.py."
            )
        _face_detector = cv2.FaceDetectorYN.create(
            FACE_DETECTOR_PATH,
            "",
            (320, 320),
            DETECTION_SCORE_THRESHOLD,
            DETECTION_NMS_THRESHOLD,
            DETECTION_TOP_K,
        )
    return _face_detector


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decodes raw image bytes into a BGR NumPy array."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image file. Could not decode byte stream.")
    return img


def _detect_faces(img: np.ndarray) -> List[Dict[str, Any]]:
    """
    Runs YuNet and returns faces largest first, in original image coordinates.
    Each has a box clipped to the image and landmarks ordered by image position:
    eyes left to right, nose tip, then mouth corners left to right.
    """
    h_img, w_img = img.shape[:2]
    scale = min(1.0, MAX_DETECTION_SIDE / max(h_img, w_img))
    small = img
    if scale < 1.0:
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    detector = get_face_detector()
    detector.setInputSize((small.shape[1], small.shape[0]))
    _, detections = detector.detect(small)
    if detections is None:
        return []

    faces: List[Dict[str, Any]] = []
    for detection in detections:
        values = detection[:14] / scale
        x, y = max(0, int(values[0])), max(0, int(values[1]))
        w = min(w_img - x, int(values[0] + values[2]) - x)
        h = min(h_img - y, int(values[1] + values[3]) - y)
        if w <= 0 or h <= 0:
            continue

        points = values[4:14].reshape(5, 2).astype(np.float64)
        eyes = points[np.argsort(points[:2, 0])]
        mouth = points[3:][np.argsort(points[3:, 0])]
        landmarks = np.vstack([eyes, points[2:3], mouth])
        faces.append({"box": (x, y, w, h), "landmarks": landmarks})

    faces.sort(key=lambda f: f["box"][2] * f["box"][3], reverse=True)
    return faces


def _has_second_face(boxes: List[Tuple[int, int, int, int]]) -> bool:
    """Checks for another large face outside the main face. `boxes` is sorted largest first."""
    x, y, w, h = boxes[0]
    for fx, fy, fw, fh in boxes[1:]:
        center_x, center_y = fx + fw / 2, fy + fh / 2
        inside_main_face = x <= center_x <= x + w and y <= center_y <= y + h
        if not inside_main_face and fw * fh >= SECOND_FACE_AREA_RATIO * w * h:
            return True
    return False


def _check_quality(
    gray: np.ndarray,
    box: Tuple[int, int, int, int],
    landmarks: np.ndarray,
    limits: QualityLimits,
) -> None:
    """Rejects photos likely to match badly, with a message saying how to fix them."""
    x, y, w, h = box
    # Measured at a fixed size so scores don't depend on how close the person stands
    face = cv2.resize(gray[y : y + h, x : x + w], TARGET_SIZE, interpolation=cv2.INTER_AREA)

    # Brightness comes first because a dark face also scores low on sharpness
    brightness = float(face.mean())
    if brightness < limits.min_brightness:
        raise ValueError("Face is too dark. Move to a brighter place and try again.")
    if brightness > limits.max_brightness:
        raise ValueError("Face is too bright. Avoid direct light on your face and try again.")

    sharpness = float(cv2.Laplacian(face, cv2.CV_64F).var())
    if sharpness < limits.min_sharpness:
        raise ValueError(
            f"Face is too blurry (sharpness {sharpness:.0f} < {limits.min_sharpness:.0f}). "
            "Hold still and try again."
        )

    half = TARGET_SIZE[0] // 2
    left_side, right_side = float(face[:, :half].mean()), float(face[:, half:].mean())
    imbalance = abs(left_side - right_side) / ((left_side + right_side) / 2 + 1e-7)
    if imbalance > limits.max_side_imbalance:
        raise ValueError("One side of your face is in shadow. Face the light and try again.")

    # A turned head moves the nose away from the middle of the eyes
    (lx, ly), (rx, ry), (nose_x, _) = landmarks[0], landmarks[1], landmarks[2]
    eye_distance = max(math.hypot(rx - lx, ry - ly), 1.0)
    nose_offset = abs(nose_x - (lx + rx) / 2) / eye_distance
    tilt = abs(math.degrees(math.atan2(ry - ly, rx - lx)))
    if nose_offset > limits.max_nose_offset or tilt > MAX_EYE_TILT_DEGREES:
        raise ValueError("Your head is turned. Look straight at the camera and try again.")


def _align_face(gray: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Rotates, scales, and shifts the image so the five landmarks best fit
    LANDMARK_TEMPLATE, using a least-squares fit, then crops to the face.
    """
    target = LANDMARK_TEMPLATE * ALIGN_WORK_SIZE

    # Solves x' = a*x - b*y + tx and y' = b*x + a*y + ty for all landmarks
    equations = np.zeros((10, 4), dtype=np.float64)
    equations[0::2] = np.column_stack([landmarks[:, 0], -landmarks[:, 1], np.ones(5), np.zeros(5)])
    equations[1::2] = np.column_stack([landmarks[:, 1], landmarks[:, 0], np.zeros(5), np.ones(5)])
    a, b, tx, ty = np.linalg.lstsq(equations, target.ravel(), rcond=None)[0]
    matrix = np.array([[a, -b, tx], [b, a, ty]], dtype=np.float64)

    warped = cv2.warpAffine(
        gray,
        matrix,
        (ALIGN_WORK_SIZE, ALIGN_WORK_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return cv2.resize(warped, TARGET_SIZE, interpolation=cv2.INTER_AREA)


def detect_and_align_face(
    img: np.ndarray,
    limits: Optional[QualityLimits] = DETECTION_LIMITS,
) -> Tuple[np.ndarray, List[int], List[List[int]]]:
    """
    Pipeline Steps 1, 2, 3:
    1. Detect faces and landmarks using YuNet, select largest face, and check photo quality.
    2. Align face so its five landmarks land on the same spots in every photo.
    3. Normalize: 128x128 grayscale with CLAHE.
    Pass limits=None to skip the quality checks.
    """
    faces = _detect_faces(img)
    if not faces:
        raise ValueError("No face detected. Please face the camera directly.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    box, landmarks = faces[0]["box"], faces[0]["landmarks"]

    if limits:
        if _has_second_face([face["box"] for face in faces]):
            raise ValueError("More than one face found. Only one person should be in the photo.")
        if min(box[2], box[3]) < limits.min_face_px:
            raise ValueError("Face is too small. Move closer to the camera and try again.")
        _check_quality(gray, box, landmarks, limits)

    aligned = _align_face(gray, landmarks)

    # CLAHE for illumination invariance
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm_face = clahe.apply(aligned)

    x, y, w, h = box
    bbox = [x, y, x + w, y + h]
    return norm_face, bbox, [[int(px), int(py)] for px, py in landmarks]


def extract_features_from_normalized(norm_face: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pipeline Step 4:
    - LBP on an 8x8 grid (local texture, keeps spatial layout).
    - HOG (edge and shape structure).
    """
    # 1. Uniform LBP (P=8, R=1 -> 10 histogram bins per cell)
    lbp_map = local_binary_pattern(norm_face, P=8, R=1, method="uniform")
    cell_hists: List[np.ndarray] = []

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cell = lbp_map[
                r * CELL_SIZE : (r + 1) * CELL_SIZE,
                c * CELL_SIZE : (c + 1) * CELL_SIZE,
            ]
            hist, _ = np.histogram(cell.ravel(), bins=10, range=(0, 10))
            # L1 normalize cell histogram
            hist_sum = hist.sum()
            hist_norm = (hist.astype(np.float32) / (hist_sum + 1e-7))
            cell_hists.append(hist_norm)

    lbp_feature = np.concatenate(cell_hists).astype(np.float32)  # 640-dim vector

    # 2. HOG (8 orientations, 16x16 cell, 1x1 block -> 512-dim vector)
    hog_feature = hog(
        norm_face,
        orientations=8,
        pixels_per_cell=(CELL_SIZE, CELL_SIZE),
        cells_per_block=(1, 1),
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True,
    ).astype(np.float32)

    # L2 normalize HOG vector
    h_norm = float(np.linalg.norm(hog_feature))
    if h_norm > 0:
        hog_feature = hog_feature / h_norm

    return lbp_feature, hog_feature


def extract_classical_face_data(
    image_bytes: bytes,
    limits: Optional[QualityLimits] = DETECTION_LIMITS,
) -> Dict[str, Any]:
    """
    End-to-end extraction from raw bytes:
    Decoding -> YuNet Detect -> Quality Checks -> Landmark Alignment -> 128x128 CLAHE -> LBP + HOG.
    """
    img = decode_image(image_bytes)

    h_img, w_img = img.shape[:2]
    norm_face, bbox, landmarks = detect_and_align_face(img, limits)
    lbp_feat, hog_feat = extract_features_from_normalized(norm_face)

    return {
        "lbp": lbp_feat,
        "hog": hog_feat,
        "bbox": bbox,
        "kps": landmarks,
        "image_dimensions": {"width": int(w_img), "height": int(h_img)},
    }


def compare_classical_features(
    f1: Dict[str, np.ndarray], f2: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Pipeline Step 5:
    - Chi-square distance for LBP (averaged across 64 cells, scaled to [0, 1]).
    - Cosine distance for HOG.
    - Final Distance = 0.6 * LBP + 0.4 * HOG.
    Smaller distance = more similar.
    """
    lbp1, lbp2 = f1["lbp"], f2["lbp"]
    hog1, hog2 = f1["hog"], f2["hog"]

    # 1. Chi-square distance for LBP across all 64 cells
    diff = lbp1 - lbp2
    denom = lbp1 + lbp2 + 1e-7
    # Summing 0.5 * (P-Q)^2 / (P+Q) per cell sums to at most 1.0 per cell when disjoint
    chi2_raw = float(0.5 * np.sum((diff ** 2) / denom))
    lbp_dist = min(1.0, max(0.0, chi2_raw / float(GRID_SIZE * GRID_SIZE)))

    # 2. Cosine distance for HOG: 1.0 - (u . v) / (||u|| * ||v||)
    dot_prod = float(np.dot(hog1, hog2))
    norm_prod = float(np.linalg.norm(hog1) * np.linalg.norm(hog2) + 1e-7)
    cos_sim = dot_prod / norm_prod
    hog_dist = min(1.0, max(0.0, 1.0 - cos_sim))

    # 3. Weighted combination: 0.6 * LBP + 0.4 * HOG
    final_distance = float(0.6 * lbp_dist + 0.4 * hog_dist)
    similarity_score = max(0.0, 1.0 - final_distance)

    return {
        "final_distance": round(final_distance, 4),
        "similarity_score": round(similarity_score, 4),
        "lbp_distance": round(lbp_dist, 4),
        "hog_distance": round(hog_dist, 4),
    }


def cache_user_features(image_name: str, features: Dict[str, Any]):
    """Caches extracted features for a newly registered user portrait."""
    _feature_cache[image_name] = {"lbp": features["lbp"], "hog": features["hog"]}


def get_cached_features_for_image(
    image_name: str, images_dir: str
) -> Optional[Dict[str, np.ndarray]]:
    """Retrieves or calculates classical features for a registered image."""
    if image_name in _feature_cache:
        return _feature_cache[image_name]

    img_path = os.path.join(images_dir, image_name)
    if not os.path.exists(img_path):
        return None

    try:
        with open(img_path, "rb") as f:
            data = f.read()
        # This portrait is already registered, so only detection has to succeed
        feats = extract_classical_face_data(data, limits=None)
        _feature_cache[image_name] = {"lbp": feats["lbp"], "hog": feats["hog"]}
        return _feature_cache[image_name]
    except Exception as e:
        print(f"[Warning] Skipping registered portrait {image_name}: {e}")
        return None


def invalidate_feature_cache(image_name: Optional[str] = None):
    """Clears cached features when a user is modified or deleted."""
    if image_name:
        _feature_cache.pop(image_name, None)
    else:
        _feature_cache.clear()


def rank_candidates(
    query_features: Dict[str, np.ndarray],
    registered_users: List[Dict[str, Any]],
    images_dir: str,
) -> List[Dict[str, Any]]:
    """
    Pipeline Step 6:
    Compare query features against all registered images and rank by distance ascending.
    """
    ranked_results: List[Dict[str, Any]] = []

    for user in registered_users:
        img_name = user.get("image_name")
        if not img_name:
            continue

        ref_features = get_cached_features_for_image(img_name, images_dir)
        if ref_features is None:
            continue

        cmp = compare_classical_features(query_features, ref_features)
        ranked_results.append(
            {
                "person_id": user.get("person_id"),
                "name": user.get("name"),
                "image_name": img_name,
                "final_distance": cmp["final_distance"],
                "similarity_score": cmp["similarity_score"],
                "lbp_distance": cmp["lbp_distance"],
                "hog_distance": cmp["hog_distance"],
            }
        )

    # Sort ascending by final distance (smaller distance = more similar)
    ranked_results.sort(key=lambda x: x["final_distance"])
    return ranked_results
