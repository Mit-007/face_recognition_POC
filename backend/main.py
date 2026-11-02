import os
import time
import uuid

from dotenv import load_dotenv

load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from classical_face_service import (
    REGISTRATION_LIMITS,
    extract_classical_face_data,
    rank_candidates,
    cache_user_features,
    invalidate_feature_cache,
    get_face_detector,
)
from history_service import add_detection_record, get_detection_history, clear_detection_history
from user_service import add_registered_user, get_registered_users, delete_registered_user

# A face matches when its LBP + HOG distance is at or below this cutoff.
# Resemblance is 1 - distance, so the score threshold follows from the cutoff.
MATCH_DISTANCE_CUTOFF: float = float(os.getenv("MATCH_DISTANCE_CUTOFF", "0.22"))
MATCH_SCORE_THRESHOLD: float = round(1.0 - MATCH_DISTANCE_CUTOFF, 4)

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "database_images")
os.makedirs(IMAGES_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm the YuNet face detector on startup
    try:
        get_face_detector()
    except Exception:
        pass
    yield


app = FastAPI(title="Face Recognition (YuNet + LBP + HOG)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")


def _record_rejection(message: str, latency_ms: float) -> None:
    add_detection_record(
        matched=False,
        status="REJECTED",
        name=None,
        score=0.0,
        threshold=MATCH_SCORE_THRESHOLD,
        image_name=None,
        latency_ms=latency_ms,
        message=message,
    )


def _no_match_response(message: str, latency_ms: float, face_data: dict | None = None) -> dict:
    return {
        "face_detected": face_data is not None,
        "matched": False,
        "score": 0.0,
        "threshold": MATCH_SCORE_THRESHOLD,
        "name": None,
        "image_name": None,
        "message": message,
        "bbox": face_data["bbox"] if face_data else None,
        "kps": face_data["kps"] if face_data else None,
        "image_dimensions": face_data["image_dimensions"] if face_data else None,
        "latency_ms": latency_ms,
        "ranked_candidates": [],
    }


@app.post("/register_new_person")
async def register_new_person(
    first_name: str = "",
    last_name: str = "",
    file: UploadFile = File(...)
):
    """
    Registers a new person:
    1. Applies the strict registration photo checks and aligns the face using its landmarks.
    2. Saves reference portrait to database_images/.
    3. Persists user profile in registered_users.json.
    4. Caches LBP & HOG feature descriptors in memory for instant matching.
    """
    display_name = f"{first_name} {last_name}".strip()
    if not display_name:
        raise HTTPException(status_code=400, detail="First name is required.")

    image_bytes = await file.read()

    try:
        face_data = extract_classical_face_data(image_bytes, REGISTRATION_LIMITS)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    person_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename or "")[1] or ".jpg"
    image_name = f"{person_id}{ext}"
    image_path = os.path.join(IMAGES_DIR, image_name)

    with open(image_path, "wb") as f:
        f.write(image_bytes)

    # Persist in local registered users catalog
    add_registered_user(
        person_id=person_id,
        first_name=first_name,
        last_name=last_name,
        name=display_name,
        image_name=image_name,
    )

    # Pre-cache LBP & HOG features for instant recognition
    cache_user_features(image_name, face_data)

    return {
        "person_id": person_id,
        "first_name": first_name,
        "last_name": last_name,
        "name": display_name,
        "image_name": image_name,
        "bbox": face_data["bbox"],
        "kps": face_data["kps"],
        "image_dimensions": face_data["image_dimensions"],
    }


@app.post("/detect_person")
async def detect_person(file: UploadFile = File(...)):
    """
    Detects and identifies a person using hand-crafted face features:
    1. Detects face and landmarks with YuNet and applies the detection photo checks.
    2. Aligns the face using its landmarks and normalizes to 128x128 CLAHE grayscale.
    3. Extracts 8x8 grid LBP (texture, 640-d) and HOG (edges, 512-d).
    4. Compares against all registered faces using:
       Final Distance = 0.6 * ChiSquare(LBP) + 0.4 * Cosine(HOG).
    5. Returns closest matching identity and candidate distance rankings.
    """
    image_bytes = await file.read()
    start_time = time.time()

    try:
        face_data = extract_classical_face_data(image_bytes)
    except ValueError as e:
        latency_ms = round((time.time() - start_time) * 1000, 1)
        _record_rejection(str(e), latency_ms)
        return _no_match_response(str(e), latency_ms)

    users = get_registered_users()
    latency_ms = round((time.time() - start_time) * 1000, 1)

    if not users:
        msg = "No registered faces in database."
        _record_rejection(msg, latency_ms)
        return _no_match_response(msg, latency_ms, face_data)

    ranked = rank_candidates(face_data, users, IMAGES_DIR)
    latency_ms = round((time.time() - start_time) * 1000, 1)

    if not ranked:
        msg = "Could not compute distance against registered portraits."
        return _no_match_response(msg, latency_ms, face_data)

    best = ranked[0]
    final_dist = float(best["final_distance"])
    similarity = float(best["similarity_score"])
    is_matched = final_dist <= MATCH_DISTANCE_CUTOFF

    matched_name = best["name"] if is_matched else None
    matched_image = best["image_name"] if is_matched else None

    if is_matched:
        msg = f"Match verified: {matched_name} (Distance: {final_dist:.3f}, Resemblance: {(similarity * 100):.1f}%)"
        status_label = "MATCHED"
    else:
        msg = f"Unknown person. Distance ({final_dist:.3f}) above cutoff ({MATCH_DISTANCE_CUTOFF:.2f})."
        status_label = "REJECTED"

    add_detection_record(
        matched=is_matched,
        status=status_label,
        name=matched_name or best["name"],
        score=similarity,
        threshold=MATCH_SCORE_THRESHOLD,
        image_name=matched_image,
        latency_ms=latency_ms,
        message=msg,
    )

    return {
        "face_detected": True,
        "matched": is_matched,
        "score": similarity,
        "threshold": MATCH_SCORE_THRESHOLD,
        "final_distance": final_dist,
        "lbp_distance": best["lbp_distance"],
        "hog_distance": best["hog_distance"],
        "name": matched_name,
        "image_name": matched_image,
        "message": msg,
        "bbox": face_data["bbox"],
        "kps": face_data["kps"],
        "image_dimensions": face_data["image_dimensions"],
        "latency_ms": latency_ms,
        "ranked_candidates": ranked[:5],
    }


@app.get("/detection_history")
async def get_history(limit: int = 100):
    records = get_detection_history(limit=limit)
    return {"history": records, "count": len(records)}


@app.delete("/detection_history")
async def clear_history():
    clear_detection_history()
    return {"cleared": True, "message": "Detection history cleared successfully."}


@app.get("/registered_users")
async def list_registered_users():
    users = get_registered_users()
    return {"users": users, "count": len(users)}


@app.delete("/registered_users/{person_id}")
async def remove_registered_user(person_id: str):
    users = get_registered_users()
    target_user = next((u for u in users if u.get("person_id") == person_id), None)
    if target_user and "image_name" in target_user:
        invalidate_feature_cache(target_user["image_name"])

    success = delete_registered_user(person_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found.")
    return {"deleted": True, "person_id": person_id, "message": "User deleted successfully."}
