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
LBP_BINS = 10
HOG_ORIENTATIONS = 8
# The worst boxes are ignored, so a shadow or hair can't dominate the distance
BOXES_KEPT = int(GRID_SIZE * GRID_SIZE * 0.8)
# Share of the final distance from texture (LBP) and from edges and shape (HOG)
LBP_WEIGHT = 0.4
HOG_WEIGHT = 0.6

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

# Another face at this confidence, at any size, means the photo is invalid. It's
# above the detection threshold because tiny false detections can score just over that
SECOND_FACE_SCORE_THRESHOLD = 0.8
MAX_EYE_TILT_DEGREES = 25.0

# The face box must stay this share of its size away from every photo edge. YuNet shrinks
# its box to the visible part of a cut-off face, so that box ends up against the edge
MIN_EDGE_MARGIN = 0.05

# Face areas used by the covering checks, in the 128x128 aligned face: (x0, x1, y0, y1)
LEFT_EYE_AREA = (26, 50, 38, 54)
RIGHT_EYE_AREA = (78, 102, 38, 54)
LEFT_CHEEK_AREA = (22, 40, 62, 76)
RIGHT_CHEEK_AREA = (88, 106, 62, 76)
FOREHEAD_AREA = (44, 84, 18, 30)
NOSE_AREA = (56, 72, 64, 82)
CHIN_AREA = (44, 84, 108, 124)
MOUTH_AREA = (44, 84, 92, 112)

# Sunglasses: the brighter eye area is this dark compared with the cheeks. Its bright
# part is used instead of its average, so a thick glasses frame doesn't look like a dark lens
MAX_SUNGLASSES_EYE_RATIO = 0.35
EYE_BRIGHTNESS_PERCENTILE = 75
# Mask: at least two of these signals must agree, so a beard alone doesn't count
MASK_SIGNALS_REQUIRED = 2
MIN_MASK_NOSE_COLOR_DIFFERENCE = 20.0  # The nose area's color differs from the forehead
MAX_MASK_CHIN_DETAIL = 30.0  # The chin area is unusually smooth, while a beard is rough
MAX_MASK_LIP_EDGES = 18.0  # The lips have no visible edges


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
    Each has a confidence score, a box clipped to the image, whether the face is cut off
    by the photo edge, and landmarks ordered by image position: eyes left to right,
    nose tip, then mouth corners left to right.
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

        margin_x = MIN_EDGE_MARGIN * values[2]
        margin_y = MIN_EDGE_MARGIN * values[3]
        box_cut_off = (
            values[0] < margin_x
            or values[1] < margin_y
            or values[0] + values[2] > w_img - margin_x
            or values[1] + values[3] > h_img - margin_y
        )
        landmark_outside = bool(
            np.any((points[:, 0] < 0) | (points[:, 0] >= w_img) | (points[:, 1] < 0) | (points[:, 1] >= h_img))
        )

        faces.append(
            {
                "box": (x, y, w, h),
                "landmarks": landmarks,
                "score": float(detection[14]),
                "cut_off": bool(box_cut_off) or landmark_outside,
            }
        )

    faces.sort(key=lambda f: f["box"][2] * f["box"][3], reverse=True)
    return faces


def _has_second_face(faces: List[Dict[str, Any]]) -> bool:
    """Checks for another confident face outside the main face. `faces` is sorted largest first."""
    x, y, w, h = faces[0]["box"]
    for face in faces[1:]:
        fx, fy, fw, fh = face["box"]
        center_x, center_y = fx + fw / 2, fy + fh / 2
        inside_main_face = x <= center_x <= x + w and y <= center_y <= y + h
        if not inside_main_face and face["score"] >= SECOND_FACE_SCORE_THRESHOLD:
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


def _check_coverings(img: np.ndarray, landmarks: np.ndarray) -> None:
    """Rejects a face covered by sunglasses or a mask, by comparing areas of the aligned face."""
    # Converted after alignment, so only the small face image is processed
    light, red_color, blue_color = cv2.split(cv2.cvtColor(_align_face(img, landmarks), cv2.COLOR_BGR2YCrCb))

    def area(channel: np.ndarray, bounds: Tuple[int, int, int, int]) -> np.ndarray:
        x0, x1, y0, y1 = bounds
        return channel[y0:y1, x0:x1]

    def mean_of(channel: np.ndarray, bounds: Tuple[int, int, int, int]) -> float:
        return float(area(channel, bounds).mean())

    cheeks = (mean_of(light, LEFT_CHEEK_AREA) + mean_of(light, RIGHT_CHEEK_AREA)) / 2
    brighter_eye = max(
        float(np.percentile(area(light, LEFT_EYE_AREA), EYE_BRIGHTNESS_PERCENTILE)),
        float(np.percentile(area(light, RIGHT_EYE_AREA), EYE_BRIGHTNESS_PERCENTILE)),
    )
    if brighter_eye / (cheeks + 1e-7) < MAX_SUNGLASSES_EYE_RATIO:
        raise ValueError("Please remove your sunglasses and try again.")

    nose_color_difference = math.hypot(
        mean_of(red_color, NOSE_AREA) - mean_of(red_color, FOREHEAD_AREA),
        mean_of(blue_color, NOSE_AREA) - mean_of(blue_color, FOREHEAD_AREA),
    )
    chin_detail = float(cv2.Laplacian(area(light, CHIN_AREA), cv2.CV_64F).var())
    lip_edges = float(np.abs(cv2.Sobel(area(light, MOUTH_AREA), cv2.CV_64F, 0, 1, ksize=3)).mean())
    mask_signals = (
        int(nose_color_difference > MIN_MASK_NOSE_COLOR_DIFFERENCE)
        + int(chin_detail < MAX_MASK_CHIN_DETAIL)
        + int(lip_edges < MAX_MASK_LIP_EDGES)
    )
    if mask_signals >= MASK_SIGNALS_REQUIRED:
        raise ValueError("Please remove your mask or anything covering your mouth and try again.")


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
        if _has_second_face(faces):
            raise ValueError("More than one face found. Only one person should be in the photo.")
        if faces[0]["cut_off"]:
            raise ValueError("Your full face isn't visible. Move to the center of the frame and try again.")
        if min(box[2], box[3]) < limits.min_face_px:
            raise ValueError("Face is too small. Move closer to the camera and try again.")
        _check_quality(gray, box, landmarks, limits)
        _check_coverings(img, landmarks)

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
    Pipeline Step 5, comparing the same grid box in both faces:
    - Chi-square distance for LBP in each box.
    - Cosine distance for HOG in each box.
    - For each, averages the closest 80% of the 64 boxes and ignores the rest.
    - Final Distance = 0.4 * LBP + 0.6 * HOG.
    Smaller distance = more similar.
    """
    lbp1, lbp2 = f1["lbp"].reshape(-1, LBP_BINS), f2["lbp"].reshape(-1, LBP_BINS)
    hog1, hog2 = f1["hog"].reshape(-1, HOG_ORIENTATIONS), f2["hog"].reshape(-1, HOG_ORIENTATIONS)

    # 1. Chi-square distance for LBP: each box is at most 1.0 when disjoint
    lbp_boxes = 0.5 * np.sum((lbp1 - lbp2) ** 2 / (lbp1 + lbp2 + 1e-7), axis=1)
    lbp_dist = min(1.0, max(0.0, float(np.sort(lbp_boxes)[:BOXES_KEPT].mean())))

    # 2. Cosine distance for HOG: 1.0 - (u . v) / (||u|| * ||v||)
    dot_prods = np.sum(hog1 * hog2, axis=1)
    norm_prods = np.linalg.norm(hog1, axis=1) * np.linalg.norm(hog2, axis=1) + 1e-7
    hog_boxes = 1.0 - dot_prods / norm_prods
    hog_dist = min(1.0, max(0.0, float(np.sort(hog_boxes)[:BOXES_KEPT].mean())))

    # 3. Weighted combination
    final_distance = float(LBP_WEIGHT * lbp_dist + HOG_WEIGHT * hog_dist)
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


def nearest_other_distance(ranked: List[Dict[str, Any]], images_dir: str) -> Optional[float]:
    """Distance from the closest person's registered photo to the nearest other registered photo."""
    top_features = get_cached_features_for_image(ranked[0]["image_name"], images_dir)
    distances = []
    for candidate in ranked[1:]:
        other_features = get_cached_features_for_image(candidate["image_name"], images_dir)
        if top_features is not None and other_features is not None:
            distances.append(compare_classical_features(top_features, other_features)["final_distance"])
    return min(distances) if distances else None
