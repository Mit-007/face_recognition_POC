import itertools
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
MIN_FACE_SIZE_PX = 30
GRID_SIZE = 8
CELL_SIZE = 16  # 128 / 8
TARGET_SIZE = (128, 128)

# Every face is warped so the eyes land on the same pixels of the output image
ALIGN_WORK_SIZE = 256  # Warped at twice the final size, then shrunk, to avoid aliasing
EYE_DISTANCE_RATIO = 0.40
EYE_Y_RATIO = 0.36

# Limits for accepting a pair of eye detections
MIN_EYE_DISTANCE_RATIO = 0.25
MAX_EYE_DISTANCE_RATIO = 0.65
MAX_EYE_TILT_DEGREES = 25.0
MAX_EYE_SIZE_RATIO = 1.6
MAX_EYE_PAIR_OFFSET = 0.25
MIN_EYE_SIZE_RATIO = 0.08
SMALL_FACE_PX = 200  # Faces narrower than this also get an enlarged eye search

# Another face this large relative to the main face means the photo shows two people
SECOND_FACE_AREA_RATIO = 0.30


@dataclass(frozen=True)
class QualityLimits:
    min_face_px: int
    min_sharpness: float
    min_brightness: float
    max_brightness: float
    max_side_imbalance: float
    max_eye_offset: float


# The registered photo is used for every later match, so it has the stricter limits
REGISTRATION_LIMITS = QualityLimits(
    min_face_px=100,
    min_sharpness=100.0,
    min_brightness=60.0,
    max_brightness=200.0,
    max_side_imbalance=0.25,
    max_eye_offset=0.08,
)
DETECTION_LIMITS = QualityLimits(
    min_face_px=80,
    min_sharpness=60.0,
    min_brightness=45.0,
    max_brightness=220.0,
    max_side_imbalance=0.35,
    max_eye_offset=0.12,
)

# Paths to Haar Cascades
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
FACE_CASCADE_PATH = os.path.join(MODELS_DIR, "haarcascade_frontalface_default.xml")
EYE_CASCADE_PATH = os.path.join(MODELS_DIR, "haarcascade_eye.xml")

_face_cascade: Optional[cv2.CascadeClassifier] = None
_eye_cascade: Optional[cv2.CascadeClassifier] = None

# In-memory feature cache for registered users: {image_name: features_dict}
_feature_cache: Dict[str, Dict[str, Any]] = {}


def get_face_cascade() -> cv2.CascadeClassifier:
    global _face_cascade
    if _face_cascade is None:
        if not os.path.exists(FACE_CASCADE_PATH):
            raise FileNotFoundError(
                f"Haar face cascade not found at {FACE_CASCADE_PATH}. Run download_models.py."
            )
        _face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        if _face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar face cascade from {FACE_CASCADE_PATH}.")
    return _face_cascade


def get_eye_cascade() -> cv2.CascadeClassifier:
    global _eye_cascade
    if _eye_cascade is None:
        if not os.path.exists(EYE_CASCADE_PATH):
            raise FileNotFoundError(
                f"Haar eye cascade not found at {EYE_CASCADE_PATH}. Run download_models.py."
            )
        _eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
        if _eye_cascade.empty():
            raise RuntimeError(f"Failed to load Haar eye cascade from {EYE_CASCADE_PATH}.")
    return _eye_cascade


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decodes raw image bytes into a BGR NumPy array."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image file. Could not decode byte stream.")
    return img


def _has_second_face(faces: List[Tuple[int, int, int, int]]) -> bool:
    """Checks for another large face outside the main face. `faces` is sorted largest first."""
    x, y, w, h = faces[0]
    for fx, fy, fw, fh in faces[1:]:
        center_x, center_y = fx + fw / 2, fy + fh / 2
        inside_main_face = x <= center_x <= x + w and y <= center_y <= y + h
        if not inside_main_face and fw * fh >= SECOND_FACE_AREA_RATIO * w * h:
            return True
    return False


def _find_eye_pair(
    gray: np.ndarray, box: Tuple[int, int, int, int]
) -> Optional[List[List[int]]]:
    """
    Finds the most believable pair of eyes in the upper 60% of the face,
    returned left to right in full-image coordinates. Starts with the most
    confident detector settings and only relaxes them if no valid pair is found.
    """
    x, y, w, h = box
    roi = gray[y : y + int(h * 0.60), x : x + w]
    eye_cascade = get_eye_cascade()

    attempts = [(roi, 1), (cv2.equalizeHist(roi), 1)]
    if w < SMALL_FACE_PX:
        enlarged = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        attempts += [(enlarged, 2), (cv2.equalizeHist(enlarged), 2)]

    for (image, scale), min_neighbors in itertools.product(attempts, (3, 2, 1)):
        min_eye = (int(w * scale * MIN_EYE_SIZE_RATIO), int(h * scale * MIN_EYE_SIZE_RATIO))
        found = eye_cascade.detectMultiScale(
            image, scaleFactor=1.1, minNeighbors=min_neighbors, minSize=min_eye
        )
        eyes = np.asarray(found, dtype=np.float64) / scale
        best_score = float("inf")
        best_pair = None
        for first, second in itertools.combinations(eyes, 2):
            left, right = sorted((first, second), key=lambda e: e[0])
            lx, ly = left[0] + left[2] / 2, left[1] + left[3] / 2
            rx, ry = right[0] + right[2] / 2, right[1] + right[3] / 2
            dx, dy = rx - lx, ry - ly
            if dx <= 0:
                continue

            tilt = abs(math.degrees(math.atan2(dy, dx)))
            offset = abs((lx + rx) / 2 - w / 2) / w
            size_ratio = max(left[2], right[2]) / min(left[2], right[2])
            if (
                not MIN_EYE_DISTANCE_RATIO <= dx / w <= MAX_EYE_DISTANCE_RATIO
                or tilt > MAX_EYE_TILT_DEGREES
                or size_ratio > MAX_EYE_SIZE_RATIO
                or offset > MAX_EYE_PAIR_OFFSET
            ):
                continue

            # Prefer the pair most centered in the face and closest to level
            score = offset + abs(dy) / dx
            if score < best_score:
                best_score = score
                best_pair = [[int(x + lx), int(y + ly)], [int(x + rx), int(y + ry)]]
        if best_pair:
            return best_pair
    return None


def _check_quality(
    gray: np.ndarray,
    box: Tuple[int, int, int, int],
    eyes: List[List[int]],
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

    eye_center_x = (eyes[0][0] + eyes[1][0]) / 2
    if abs(eye_center_x - (x + w / 2)) / w > limits.max_eye_offset:
        raise ValueError("Your head is turned. Look straight at the camera and try again.")


def _align_face_to_eyes(gray: np.ndarray, eyes: List[List[int]]) -> np.ndarray:
    """Rotates, scales, and shifts the image so the eyes land on fixed pixels, then crops."""
    (lx, ly), (rx, ry) = eyes
    angle = math.degrees(math.atan2(ry - ly, rx - lx))
    scale = ALIGN_WORK_SIZE * EYE_DISTANCE_RATIO / math.hypot(rx - lx, ry - ly)
    center = ((lx + rx) / 2.0, (ly + ry) / 2.0)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += ALIGN_WORK_SIZE / 2 - center[0]
    matrix[1, 2] += ALIGN_WORK_SIZE * EYE_Y_RATIO - center[1]

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
    1. Detect face using Haar cascade, select largest face, and check photo quality.
    2. Align face so both eyes land on the same pixels in every photo.
    3. Normalize: 128x128 grayscale with CLAHE.
    Pass limits=None to skip the quality checks. Both eyes are always required.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = get_face_cascade()
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(MIN_FACE_SIZE_PX, MIN_FACE_SIZE_PX),
    )

    if len(faces) == 0:
        raise ValueError("No face detected by Haar cascade. Please face the camera directly.")

    # 1. Use the largest face
    faces = sorted(
        ([int(v) for v in face] for face in faces),
        key=lambda f: f[2] * f[3],
        reverse=True,
    )
    x, y, w, h = faces[0]

    if limits:
        if _has_second_face(faces):
            raise ValueError("More than one face found. Only one person should be in the photo.")
        if min(w, h) < limits.min_face_px:
            raise ValueError("Face is too small. Move closer to the camera and try again.")

    # 2. Eye-based alignment
    eyes = _find_eye_pair(gray, (x, y, w, h))
    if eyes is None:
        raise ValueError(
            "Could not find both eyes. Look straight at the camera and remove any sunglasses."
        )

    if limits:
        _check_quality(gray, (x, y, w, h), eyes, limits)

    aligned = _align_face_to_eyes(gray, eyes)

    # 3. CLAHE for illumination invariance
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm_face = clahe.apply(aligned)

    bbox = [x, y, x + w, y + h]
    return norm_face, bbox, eyes


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
    Decoding -> Haar Detect -> Quality Checks -> Eye Alignment -> 128x128 CLAHE -> LBP + HOG.
    """
    img = decode_image(image_bytes)

    h_img, w_img = img.shape[:2]
    norm_face, bbox, eyes_kps = detect_and_align_face(img, limits)
    lbp_feat, hog_feat = extract_features_from_normalized(norm_face)

    return {
        "lbp": lbp_feat,
        "hog": hog_feat,
        "bbox": bbox,
        "kps": eyes_kps,
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
        # This portrait is already registered, so only the alignment has to succeed
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
