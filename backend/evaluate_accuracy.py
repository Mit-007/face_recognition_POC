"""
Measures face matching accuracy for every approach listed in APPROACHES.

To test a new idea, add one Approach entry and run this file again. It only reads
photos: it never touches registered_users.json, detection_history.json, or database_images/.

Usage (from backend/):
    python evaluate_accuracy.py
    python evaluate_accuracy.py --dataset D:\\photos --enroll 3 --output results.json
    python evaluate_accuracy.py --approach current_app --approach inner_crop_80

Dataset layouts:
    photos/alice/1.jpg, photos/alice/2.jpg, photos/bob/1.jpg   (one folder per person)
    photos/alice-1.jpg, photos/alice-2.jpg, photos/bob-1.jpg   (name-number filenames)
"""
import argparse
import itertools
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv
from skimage.feature import hog, local_binary_pattern

from classical_face_service import (
    CELL_SIZE,
    DETECTION_LIMITS,
    GRID_SIZE,
    REGISTRATION_LIMITS,
    TARGET_SIZE,
    compare_classical_features,
    decode_image,
    detect_and_align_face,
    extract_classical_face_data,
    extract_features_from_normalized,
)

load_dotenv()

DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "data_images"
DEFAULT_CUTOFF = float(os.getenv("MATCH_DISTANCE_CUTOFF", "0.22"))
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RAM_LIMIT_MB = 100.0
TARGET_FAR = 0.01
MIN_RELIABLE_PEOPLE = 20

Features = Dict[str, Any]

# Higher weight on the eye, nose, and mouth cells of the 8x8 grid
REGION_WEIGHTS = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 2, 2, 2, 2, 2, 2, 1],
        [1, 4, 4, 2, 2, 4, 4, 1],
        [1, 4, 4, 2, 2, 4, 4, 1],
        [1, 1, 2, 3, 3, 2, 1, 1],
        [1, 1, 2, 3, 3, 2, 1, 1],
        [1, 1, 2, 2, 2, 2, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 0],
    ],
    dtype=np.float32,
).ravel()


# ---------- Approaches ----------


def _normalized_face(image_bytes: bytes, inner: float = 1.0) -> Tuple[np.ndarray, bool]:
    img = decode_image(image_bytes)
    norm_face, _, eyes = detect_and_align_face(img, DETECTION_LIMITS)
    if inner < 1.0:
        margin = int(TARGET_SIZE[0] * (1.0 - inner) / 2)
        norm_face = cv2.resize(norm_face[margin:-margin, margin:-margin], TARGET_SIZE)
    return norm_face, eyes is not None


def _lbp_cells(norm_face: np.ndarray, radius: int) -> np.ndarray:
    lbp_map = local_binary_pattern(norm_face, P=8, R=radius, method="uniform")
    cells = np.zeros((GRID_SIZE * GRID_SIZE, 10), dtype=np.float32)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cell = lbp_map[r * CELL_SIZE : (r + 1) * CELL_SIZE, c * CELL_SIZE : (c + 1) * CELL_SIZE]
            hist, _ = np.histogram(cell.ravel(), bins=10, range=(0, 10))
            cells[r * GRID_SIZE + c] = hist / (hist.sum() + 1e-7)
    return cells


def _hog_vector(norm_face: np.ndarray, cell: int, block: int) -> np.ndarray:
    vector = hog(
        norm_face,
        orientations=8,
        pixels_per_cell=(cell, cell),
        cells_per_block=(block, block),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)
    return vector / (np.linalg.norm(vector) + 1e-7)


def _describe(
    inner: float = 1.0,
    radii: Tuple[int, ...] = (1,),
    hog_cell: int = 16,
    hog_block: int = 1,
) -> Callable[[bytes], Features]:
    def describe(image_bytes: bytes) -> Features:
        norm_face, eyes_found = _normalized_face(image_bytes, inner)
        return {
            "lbp": [_lbp_cells(norm_face, r) for r in radii],
            "hog": _hog_vector(norm_face, hog_cell, hog_block),
            "eyes_found": eyes_found,
        }

    return describe


def _distance(
    lbp_weight: float = 0.6,
    cell_weights: Optional[np.ndarray] = None,
) -> Callable[[Features, Features], float]:
    weights = np.ones(GRID_SIZE * GRID_SIZE, dtype=np.float32) if cell_weights is None else cell_weights

    def distance(a: Features, b: Features) -> float:
        lbp_scales = []
        for p, q in zip(a["lbp"], b["lbp"]):
            per_cell = 0.5 * np.sum((p - q) ** 2 / (p + q + 1e-7), axis=1)
            lbp_scales.append(float(np.sum(per_cell * weights) / np.sum(weights)))
        lbp_dist = float(np.mean(lbp_scales))
        hog_dist = 1.0 - float(np.dot(a["hog"], b["hog"]))
        return lbp_weight * lbp_dist + (1.0 - lbp_weight) * hog_dist

    return distance


def _describe_current_app(image_bytes: bytes) -> Features:
    data = extract_classical_face_data(image_bytes)
    return {"lbp": data["lbp"], "hog": data["hog"], "eyes_found": data["kps"] is not None}


def _distance_current_app(a: Features, b: Features) -> float:
    return compare_classical_features(a, b)["final_distance"]


def _describe_yunet_no_checks(image_bytes: bytes) -> Features:
    norm_face, _, _ = detect_and_align_face(decode_image(image_bytes), limits=None)
    lbp, hog_feature = extract_features_from_normalized(norm_face)
    return {"lbp": lbp, "hog": hog_feature, "eyes_found": True}


def _find_haar_eye_pair(
    gray: np.ndarray, box: Tuple[int, int, int, int], eye_cascade: cv2.CascadeClassifier
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    x, y, w, h = box
    roi = gray[y : y + int(h * 0.60), x : x + w]
    attempts = [(roi, 1), (cv2.equalizeHist(roi), 1)]
    if w < 200:
        enlarged = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        attempts += [(enlarged, 2), (cv2.equalizeHist(enlarged), 2)]

    for (image, scale), neighbors in itertools.product(attempts, (3, 2, 1)):
        found = eye_cascade.detectMultiScale(
            image, scaleFactor=1.1, minNeighbors=neighbors, minSize=(int(w * scale * 0.08), int(h * scale * 0.08))
        )
        best_score, best_pair = float("inf"), None
        for first, second in itertools.combinations(np.asarray(found, dtype=np.float64) / scale, 2):
            left, right = sorted((first, second), key=lambda e: e[0])
            lx, ly = left[0] + left[2] / 2, left[1] + left[3] / 2
            rx, ry = right[0] + right[2] / 2, right[1] + right[3] / 2
            dx, dy = rx - lx, ry - ly
            if dx <= 0:
                continue
            offset = abs((lx + rx) / 2 - w / 2) / w
            if (
                not 0.25 <= dx / w <= 0.65
                or abs(math.degrees(math.atan2(dy, dx))) > 25.0
                or max(left[2], right[2]) / min(left[2], right[2]) > 1.6
                or offset > 0.25
            ):
                continue
            score = offset + abs(dy) / dx
            if score < best_score:
                best_score, best_pair = score, ((x + lx, y + ly), (x + rx, y + ry))
        if best_pair:
            return best_pair
    return None


def _describe_haar_previous(image_bytes: bytes) -> Features:
    """Frozen copy of the previous detector, without photo checks: Haar face and eyes, eye alignment."""
    gray = cv2.cvtColor(decode_image(image_bytes), cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    if len(faces) == 0:
        raise ValueError("No face detected by Haar cascade.")
    box = tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3]))

    eyes = _find_haar_eye_pair(gray, box, eye_cascade)
    if eyes is None:
        raise ValueError("Could not find both eyes with Haar cascade.")
    (lx, ly), (rx, ry) = eyes

    work_size = 256
    center = ((lx + rx) / 2.0, (ly + ry) / 2.0)
    matrix = cv2.getRotationMatrix2D(
        center, math.degrees(math.atan2(ry - ly, rx - lx)), work_size * 0.40 / math.hypot(rx - lx, ry - ly)
    )
    matrix[0, 2] += work_size / 2 - center[0]
    matrix[1, 2] += work_size * 0.36 - center[1]
    warped = cv2.warpAffine(
        gray, matrix, (work_size, work_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    aligned = cv2.resize(warped, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    norm_face = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(aligned)
    lbp, hog_feature = extract_features_from_normalized(norm_face)
    return {"lbp": lbp, "hog": hog_feature, "eyes_found": True}


@dataclass(frozen=True)
class Approach:
    name: str
    description: str
    describe: Callable[[bytes], Features]
    distance: Callable[[Features, Features], float]
    uses_app_cutoff: bool = False


APPROACHES: List[Approach] = [
    Approach(
        "haar_previous",
        "Previous detector, for comparison: Haar face and eyes, eye alignment, no photo checks",
        _describe_haar_previous,
        _distance_current_app,
    ),
    Approach(
        "yunet_no_checks",
        "YuNet and landmark alignment without the photo quality checks",
        _describe_yunet_no_checks,
        _distance_current_app,
    ),
    Approach(
        "current_app",
        "The code the app runs today: YuNet, photo checks, landmark alignment, 0.6 LBP + 0.4 HOG",
        _describe_current_app,
        _distance_current_app,
        uses_app_cutoff=True,
    ),
    Approach("lbp_only", "LBP texture alone", _describe(), _distance(lbp_weight=1.0)),
    Approach("hog_only", "HOG shape alone", _describe(), _distance(lbp_weight=0.0)),
    Approach("inner_crop_80", "Central 80% of the face, less hair and background", _describe(inner=0.8), _distance()),
    Approach("multiscale_lbp", "LBP at radius 1, 2, and 3", _describe(radii=(1, 2, 3)), _distance()),
    Approach("region_weighted", "More LBP weight on eyes, nose, and mouth", _describe(), _distance(cell_weights=REGION_WEIGHTS)),
    Approach("finer_hog", "HOG with 8px cells and 2x2 blocks", _describe(hog_cell=8, hog_block=2), _distance()),
    Approach(
        "combined",
        "Inner crop + multi-scale LBP + region weights + finer HOG",
        _describe(inner=0.8, radii=(1, 2, 3), hog_cell=8, hog_block=2),
        _distance(cell_weights=REGION_WEIGHTS),
    ),
]


# ---------- Dataset ----------


def load_dataset(dataset: Path) -> List[Tuple[str, Path]]:
    """Returns (person, photo path), using folder names or else name-number filenames."""
    samples: List[Tuple[str, Path]] = []
    for path in sorted(dataset.rglob("*"), key=lambda p: str(p).lower()):
        if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
            continue
        if path.parent != dataset:
            person = path.relative_to(dataset).parts[0]
        else:
            person = re.sub(r"[-_ ]*\d+$", "", path.stem)
        samples.append((person.lower(), path))
    return samples


# ---------- Metrics ----------


def _eer(genuine: np.ndarray, impostor: np.ndarray) -> Tuple[float, float]:
    """Returns (threshold, rate) where false accepts and false rejects are closest to equal."""
    thresholds = np.unique(np.concatenate([genuine, impostor]))
    far = np.searchsorted(np.sort(impostor), thresholds, side="right") / len(impostor)
    frr = 1.0 - np.searchsorted(np.sort(genuine), thresholds, side="right") / len(genuine)
    best = int(np.argmin(np.abs(far - frr)))
    return float(thresholds[best]), float((far[best] + frr[best]) / 2)


def _auc(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """Chance that a same-person pair is closer than a different-people pair (0.5 = guessing)."""
    ordered = np.sort(impostor)
    above = len(ordered) - np.searchsorted(ordered, genuine, side="right")
    ties = np.searchsorted(ordered, genuine, side="right") - np.searchsorted(ordered, genuine, side="left")
    return float(np.mean((above + 0.5 * ties) / len(ordered)))


def _frr_at_far(genuine: np.ndarray, impostor: np.ndarray, target: float) -> Tuple[float, float]:
    """Returns (false reject rate, threshold) for the loosest threshold keeping FAR within target."""
    thresholds = np.unique(np.concatenate([genuine, impostor]))
    far = np.searchsorted(np.sort(impostor), thresholds, side="right") / len(impostor)
    allowed = thresholds[far <= target]
    threshold = float(allowed[-1]) if len(allowed) else float("-inf")
    return float(np.mean(genuine > threshold)), threshold


def _held_out(
    distances: np.ndarray, same: np.ndarray, involved: Dict[str, np.ndarray]
) -> Optional[Dict[str, float]]:
    """Chooses each threshold without one person, then tests it on that person's pairs."""
    false_accepts = false_rejects = impostors = genuines = 0
    for mask in involved.values():
        train_gen, train_imp = distances[~mask & same], distances[~mask & ~same]
        if len(train_gen) == 0 or len(train_imp) == 0:
            continue
        threshold, _ = _eer(train_gen, train_imp)
        false_rejects += int(np.sum(distances[mask & same] > threshold))
        false_accepts += int(np.sum(distances[mask & ~same] <= threshold))
        genuines += int(np.sum(mask & same))
        impostors += int(np.sum(mask & ~same))
    if genuines == 0 or impostors == 0:
        return None
    return {"far": false_accepts / impostors, "frr": false_rejects / genuines}


def _closest_person(
    matrix: np.ndarray,
    probe: int,
    photos_by_person: Dict[str, List[int]],
    excluded: set,
    enroll: int,
) -> Optional[Tuple[str, float]]:
    """Finds the closest person the way the app does, using up to `enroll` photos per person."""
    best: Optional[Tuple[str, float]] = None
    for person, photos in photos_by_person.items():
        if person in excluded:
            continue
        enrolled = [i for i in photos if i != probe][:enroll]
        if not enrolled:
            continue
        score = float(np.min(matrix[probe, enrolled]))
        if best is None or score < best[1]:
            best = (person, score)
    return best


def _identification(
    matrix: np.ndarray,
    labels: List[str],
    unsure: Dict[str, set],
    enroll: int,
    threshold: float,
) -> Dict[str, Any]:
    photos_by_person: Dict[str, List[int]] = {}
    for i, person in enumerate(labels):
        photos_by_person.setdefault(person, []).append(i)

    known_probes = closest_correct = accepted_correct = 0
    unknown_probes = unknown_accepted = 0
    for probe, person in enumerate(labels):
        partners = unsure.get(person, set())

        # This person is enrolled through their other photos
        if len(photos_by_person[person]) > 1:
            found = _closest_person(matrix, probe, photos_by_person, partners, enroll)
            if found:
                known_probes += 1
                closest_correct += found[0] == person
                accepted_correct += found[0] == person and found[1] <= threshold

        # This person is not enrolled, so any acceptance is an error
        found = _closest_person(matrix, probe, photos_by_person, partners | {person}, enroll)
        if found:
            unknown_probes += 1
            unknown_accepted += found[1] <= threshold

    return {
        "known_probes": known_probes,
        "closest_correct": closest_correct / known_probes if known_probes else None,
        "correctly_accepted": accepted_correct / known_probes if known_probes else None,
        "unknown_probes": unknown_probes,
        "unknown_accepted": unknown_accepted / unknown_probes if unknown_probes else None,
    }


def compute_metrics(
    matrix: np.ndarray,
    labels: List[str],
    unsure_pairs: List[Tuple[str, str]],
    enroll: int,
    app_cutoff: Optional[float],
) -> Dict[str, Any]:
    """Calculates all accuracy numbers from a symmetric photo-to-photo distance matrix."""
    label_array = np.array(labels)
    unsure: Dict[str, set] = {}
    for a, b in unsure_pairs:
        unsure.setdefault(a, set()).add(b)
        unsure.setdefault(b, set()).add(a)

    first, second = np.triu_indices(len(labels), k=1)
    usable = np.array([labels[j] not in unsure.get(labels[i], set()) for i, j in zip(first, second)], dtype=bool)
    first, second = first[usable], second[usable]
    distances = matrix[first, second]
    same = label_array[first] == label_array[second]
    genuine, impostor = distances[same], distances[~same]
    if len(genuine) == 0 or len(impostor) == 0:
        raise ValueError("Need at least one same-person pair and one different-people pair.")

    eer_threshold, eer = _eer(genuine, impostor)
    frr_at_target, strict_threshold = _frr_at_far(genuine, impostor, TARGET_FAR)
    involved = {
        person: (label_array[first] == person) | (label_array[second] == person)
        for person in sorted(set(labels))
    }

    metrics: Dict[str, Any] = {
        "genuine_pairs": len(genuine),
        "impostor_pairs": len(impostor),
        "genuine_range": [float(genuine.min()), float(genuine.max())],
        "impostor_range": [float(impostor.min()), float(impostor.max())],
        "ranges_overlap": bool(genuine.max() >= impostor.min()),
        "auc": _auc(genuine, impostor),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "frr_at_far_1pct": frr_at_target,
        "strict_threshold": strict_threshold,
        "held_out": _held_out(distances, same, involved),
        "identification": _identification(matrix, labels, unsure, enroll, eer_threshold),
    }
    if app_cutoff is not None:
        metrics["app_cutoff"] = {
            "cutoff": app_cutoff,
            "far": float(np.mean(impostor <= app_cutoff)),
            "frr": float(np.mean(genuine > app_cutoff)),
            "identification": _identification(matrix, labels, unsure, enroll, app_cutoff),
        }
    return metrics


# ---------- Running one approach ----------


def _peak_memory_mb() -> float:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return counters.PeakWorkingSetSize / 2**20

    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def _feature_bytes(features: Features) -> int:
    total = 0
    for value in features.values():
        for part in value if isinstance(value, list) else [value]:
            if isinstance(part, np.ndarray):
                total += part.nbytes
    return total


def run_approach(approach: Approach, args: argparse.Namespace) -> Dict[str, Any]:
    samples = load_dataset(Path(args.dataset))
    if args.only_photos:
        allowed = set(json.loads(Path(args.only_photos).read_text(encoding="utf-8")))
        samples = [sample for sample in samples if str(sample[1]) in allowed]

    features: List[Features] = []
    labels: List[str] = []
    accepted: List[str] = []
    rejected: List[Dict[str, str]] = []
    registration_rejected = 0

    describe_seconds = 0.0
    for person, path in samples:
        data = path.read_bytes()
        start = time.perf_counter()
        try:
            result = approach.describe(data)
        except ValueError as e:
            rejected.append({"photo": path.name, "reason": str(e)})
            continue
        finally:
            describe_seconds += time.perf_counter() - start
        features.append(result)
        labels.append(person)
        accepted.append(str(path))
        if approach.uses_app_cutoff:
            try:
                extract_classical_face_data(data, REGISTRATION_LIMITS)
            except ValueError:
                registration_rejected += 1

    summary = {
        "name": approach.name,
        "description": approach.description,
        "photos": len(samples),
        "people": len(set(labels)),
        "accepted": accepted,
        "rejected": rejected,
        "registration_rejected": registration_rejected,
    }

    size = len(features)
    if size < 2:
        summary["error"] = "Not enough usable photos to compare."
        return summary
    matrix = np.zeros((size, size), dtype=np.float64)
    start = time.perf_counter()
    for i in range(size):
        for j in range(i + 1, size):
            matrix[i, j] = matrix[j, i] = approach.distance(features[i], features[j])
    compare_seconds = time.perf_counter() - start
    comparisons = size * (size - 1) / 2

    try:
        metrics = compute_metrics(
            matrix,
            labels,
            [tuple(pair) for pair in args.unsure_pair],
            args.enroll,
            args.cutoff if approach.uses_app_cutoff else None,
        )
    except ValueError as e:
        summary["error"] = str(e)
        return summary

    metrics.update(summary)
    metrics.update(
        {
            "eyes_found_rate": float(np.mean([f["eyes_found"] for f in features])),
            "describe_ms": describe_seconds / len(samples) * 1000,
            "compare_ms": compare_seconds / comparisons * 1000 if comparisons else 0.0,
            "feature_kb": _feature_bytes(features[0]) / 1024,
            "peak_memory_mb": _peak_memory_mb(),
        }
    )
    return metrics


# ---------- Report ----------


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def print_report(all_results: List[Dict[str, Any]], args: argparse.Namespace, title: str) -> None:
    print(f"\n========== {title} ==========")
    print(f"Dataset: {args.dataset}")
    print(f"Registered photos per person: {args.enroll}")
    if args.unsure_pair:
        print(f"Not compared: {', '.join(' / '.join(pair) for pair in args.unsure_pair)}")

    for r in all_results:
        if "error" in r:
            print(
                f"NOT MEASURED {r['name']}: {r['error']} "
                f"({len(r['accepted'])} of {r['photos']} photos accepted, {r['people']} people)"
            )
    results = [r for r in all_results if "error" not in r]

    if any(r["people"] < MIN_RELIABLE_PEOPLE for r in results):
        print(f"WARNING: under {MIN_RELIABLE_PEOPLE} people. Treat these numbers as a rough check only.")

    print("\nPHOTOS USED")
    header = f"{'Approach':<18}{'Accepted':>10}{'Rejected':>10}{'People':>8}{'Same pairs':>12}{'Diff pairs':>12}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['name']:<18}{len(r['accepted']):>10}{len(r['rejected']):>10}{r['people']:>8}"
            f"{r['genuine_pairs']:>12}{r['impostor_pairs']:>12}"
        )

    print("\nACCURACY (thresholds chosen per approach; higher AUC and lower EER are better)")
    header = (
        f"{'Approach':<18}{'AUC':>7}{'EER':>8}{'FRR@FAR1%':>11}{'Test FAR':>10}{'Test FRR':>10}"
        f"{'Closest':>9}{'Accepted':>10}{'Unknown':>9}{'Overlap':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        held = r["held_out"] or {}
        ident = r["identification"]
        print(
            f"{r['name']:<18}{r['auc']:>7.3f}{_pct(r['eer']):>8}{_pct(r['frr_at_far_1pct']):>11}"
            f"{_pct(held.get('far')):>10}{_pct(held.get('frr')):>10}"
            f"{_pct(ident['closest_correct']):>9}{_pct(ident['correctly_accepted']):>10}"
            f"{_pct(ident['unknown_accepted']):>9}{'yes' if r['ranges_overlap'] else 'no':>9}"
        )

    print("\nCOST (peak memory is this test process, not the web server)")
    header = f"{'Approach':<18}{'Peak MB':>9}{'Under limit':>13}{'Feature KB':>12}{'Photo ms':>10}{'Compare ms':>12}{'Eyes':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['name']:<18}{r['peak_memory_mb']:>9.1f}{'yes' if r['peak_memory_mb'] <= RAM_LIMIT_MB else 'NO':>13}"
            f"{r['feature_kb']:>12.1f}{r['describe_ms']:>10.1f}{r['compare_ms']:>12.3f}"
            f"{_pct(r['eyes_found_rate']):>8}"
        )

    print("\nSUGGESTED THRESHOLDS (distance)")
    for r in results:
        print(f"{r['name']:<18}balanced={r['eer_threshold']:.4f}  strict(FAR<=1%)={r['strict_threshold']:.4f}")

    for r in results:
        if "app_cutoff" in r:
            app = r["app_cutoff"]
            ident = app["identification"]
            print(f"\nAPP AS CONFIGURED TODAY ({r['name']}, cutoff {app['cutoff']:.2f})")
            print(f"  Different-people pairs accepted: {_pct(app['far'])}")
            print(f"  Same-person pairs rejected:      {_pct(app['frr'])}")
            print(f"  Known person correctly verified: {_pct(ident['correctly_accepted'])}")
            print(f"  Unknown person accepted:         {_pct(ident['unknown_accepted'])}")
            print(f"  Accepted photos failing the stricter registration checks: {r['registration_rejected']}")

    for r in all_results:
        if r["name"] == "current_app":
            for item in r["rejected"]:
                print(f"Rejected by {r['name']}: {item['photo']}: {item['reason']}")


HOW_TO_READ = """
HOW TO READ THIS
  AUC        Chance a same-person pair is closer than a different-people pair. 0.5 = guessing, 1.0 = perfect.
  EER        Error rate where wrong acceptances and wrong rejections are equal. Lower is better.
  FRR@FAR1%  Real people rejected when at most 1% of different-people pairs may pass.
  Test FAR/FRR  Errors when the threshold is chosen without the person being tested. Most honest pair numbers.
  Closest    Known person's closest registered match is the right person, with no threshold.
  Accepted   Closest match is right and also passes the balanced threshold.
  Unknown    Person who is not registered still gets accepted as someone. Lower is better.
  Overlap    yes = same-person and different-people distances overlap, so no threshold separates them fully.
  Compare approaches using the SAME PHOTOS section. Rejecting hard photos can make any approach look better."""


def _run_workers(names: List[str], args: argparse.Namespace, only_photos: Optional[str] = None) -> List[Dict[str, Any]]:
    # A separate process per approach keeps each peak memory reading independent
    results = []
    for name in names:
        print(f"Testing {name}...", file=sys.stderr)
        command = [
            sys.executable, __file__,
            "--worker", name,
            "--dataset", args.dataset,
            "--enroll", str(args.enroll),
            "--cutoff", str(args.cutoff),
        ]
        for a, b in args.unsure_pair:
            command += ["--unsure-pair", a, b]
        if only_photos:
            command += ["--only-photos", only_photos]
        done = subprocess.run(command, capture_output=True, text=True)
        if done.returncode != 0:
            sys.exit(f"{name} failed:\n{done.stderr}")
        results.append(json.loads(done.stdout.strip().splitlines()[-1]))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure face matching accuracy for each approach.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Folder of test photos")
    parser.add_argument("--enroll", type=int, default=1, help="Registered photos per person")
    parser.add_argument("--cutoff", type=float, default=DEFAULT_CUTOFF, help="Cutoff the app uses")
    parser.add_argument("--approach", action="append", help="Run only this approach (repeatable)")
    parser.add_argument(
        "--unsure-pair",
        action="append",
        nargs=2,
        default=[],
        metavar=("PERSON_A", "PERSON_B"),
        help="Skip comparisons between two labels that may be the same person (repeatable)",
    )
    parser.add_argument("--output", help="Also save full results to this JSON file")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--only-photos", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.unsure_pair = [[a.lower(), b.lower()] for a, b in args.unsure_pair]

    by_name = {approach.name: approach for approach in APPROACHES}

    if args.worker:
        print(json.dumps(run_approach(by_name[args.worker], args)))
        return

    names = args.approach or list(by_name)
    unknown = [name for name in names if name not in by_name]
    if unknown:
        parser.error(f"Unknown approach: {', '.join(unknown)}. Choose from: {', '.join(by_name)}")
    if not load_dataset(Path(args.dataset)):
        parser.error(f"No photos found in {args.dataset}")

    results = _run_workers(names, args)
    print_report(results, args, "EACH APPROACH ON THE PHOTOS IT ACCEPTS")

    # Approaches that reject photos are compared again on the photos all of them accepted
    saved: Dict[str, Any] = {"each_approach": results}
    common = set(results[0]["accepted"])
    for result in results[1:]:
        common &= set(result["accepted"])
    if any(len(result["rejected"]) > 0 for result in results) and len(common) >= 2:
        with tempfile.TemporaryDirectory() as folder:
            photo_list = Path(folder) / "photos.json"
            photo_list.write_text(json.dumps(sorted(common)), encoding="utf-8")
            same_photos = _run_workers(names, args, str(photo_list))
        print_report(same_photos, args, f"SAME PHOTOS ({len(common)} accepted by every approach)")
        saved["same_photos"] = same_photos

    print(HOW_TO_READ)

    if args.output:
        Path(args.output).write_text(json.dumps(saved, indent=2), encoding="utf-8")
        print(f"\nSaved full results to {args.output}")


if __name__ == "__main__":
    main()
