import os
import time
import uuid

from dotenv import load_dotenv

load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from face_service import extract_face_data, get_face_app
from pinecone_service import upsert_face, search_face, get_index, delete_face
from history_service import add_detection_record, get_detection_history, clear_detection_history
from user_service import add_registered_user, get_registered_users, delete_registered_user

MATCH_SCORE_THRESHOLD: float = float(os.getenv("MATCH_SCORE_THRESHOLD", "0.50"))

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "database_images")
os.makedirs(IMAGES_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm InsightFace and Pinecone on server startup to eliminate cold start
    try:
        get_face_app()
        get_index()
    except Exception:
        pass
    yield


app = FastAPI(title="Face Recognition POC", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")


@app.post("/register_new_person")
async def register_new_person(
    first_name: str = "",
    last_name: str = "",
    name: str = "",
    file: UploadFile = File(...)
):
    display_name = f"{first_name} {last_name}".strip() if (first_name or last_name) else name.strip()
    if not display_name:
        raise HTTPException(status_code=400, detail="First name is required.")

    image_bytes = await file.read()

    try:
        face_data = extract_face_data(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    person_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    image_name = f"{person_id}{ext}"
    image_path = os.path.join(IMAGES_DIR, image_name)

    with open(image_path, "wb") as f:
        f.write(image_bytes)

    upsert_face(
        vector_id=person_id,
        embedding=face_data["embedding"],
        name=display_name,
        image_name=image_name
    )

    add_registered_user(
        person_id=person_id,
        first_name=first_name,
        last_name=last_name,
        name=display_name,
        image_name=image_name,
    )

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
    threshold = MATCH_SCORE_THRESHOLD
    image_bytes = await file.read()
    start_time = time.time()

    try:
        face_data = extract_face_data(image_bytes)
    except ValueError as e:
        latency_ms = round((time.time() - start_time) * 1000, 1)
        err_msg = str(e)
        add_detection_record(
            matched=False,
            status="REJECTED",
            name=None,
            score=0.0,
            threshold=threshold,
            image_name=None,
            latency_ms=latency_ms,
            message=err_msg,
        )
        return {
            "face_detected": False,
            "matched": False,
            "score": 0.0,
            "threshold": threshold,
            "name": None,
            "image_name": None,
            "message": err_msg,
            "bbox": None,
            "kps": None,
            "image_dimensions": None,
            "latency_ms": latency_ms,
        }

    embedding = face_data["embedding"]
    matches = search_face(embedding, top_k=1)
    latency_ms = round((time.time() - start_time) * 1000, 1)

    if not matches:
        msg = "No registered faces in database."
        add_detection_record(
            matched=False,
            status="REJECTED",
            name=None,
            score=0.0,
            threshold=threshold,
            image_name=None,
            latency_ms=latency_ms,
            message=msg,
        )
        return {
            "face_detected": True,
            "matched": False,
            "score": 0.0,
            "threshold": threshold,
            "name": None,
            "image_name": None,
            "message": msg,
            "bbox": face_data["bbox"],
            "kps": face_data["kps"],
            "image_dimensions": face_data["image_dimensions"],
            "latency_ms": latency_ms,
        }

    best = matches[0]
    score = float(best["score"])
    is_matched = score >= threshold

    matched_name = best["metadata"].get("name") if is_matched else None
    matched_image = best["metadata"].get("image_name") if is_matched else None

    if is_matched:
        msg = f"Match verified: {matched_name}"
        status_label = "MATCHED"
    else:
        msg = f"Unknown person. Similarity ({score:.3f}) below threshold ({threshold:.2f})."
        status_label = "REJECTED"

    add_detection_record(
        matched=is_matched,
        status=status_label,
        name=matched_name or best["metadata"].get("name"),
        score=score,
        threshold=threshold,
        image_name=matched_image,
        latency_ms=latency_ms,
        message=msg,
    )

    return {
        "face_detected": True,
        "matched": is_matched,
        "score": score,
        "threshold": threshold,
        "name": matched_name,
        "image_name": matched_image,
        "message": msg,
        "bbox": face_data["bbox"],
        "kps": face_data["kps"],
        "image_dimensions": face_data["image_dimensions"],
        "latency_ms": latency_ms,
    }


@app.get("/detection_history")
async def get_history(limit: int = Query(100, ge=1, le=500)):
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
    success = delete_registered_user(person_id)
    delete_face(person_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found.")
    return {"deleted": True, "person_id": person_id, "message": "User deleted successfully."}
