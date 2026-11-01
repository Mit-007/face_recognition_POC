import os
import uuid

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from face_service import extract_embedding
from pinecone_service import upsert_face, search_face

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "database_images")
os.makedirs(IMAGES_DIR, exist_ok=True)

app = FastAPI(title="Face Recognition POC")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")


@app.post("/register_new_person")
async def register_new_person(name: str, file: UploadFile = File(...)):
    image_bytes = await file.read()

    try:
        embedding = extract_embedding(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    person_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    image_name = f"{person_id}{ext}"
    image_path = os.path.join(IMAGES_DIR, image_name)

    with open(image_path, "wb") as f:
        f.write(image_bytes)

    upsert_face(vector_id=person_id, embedding=embedding, name=name, image_name=image_name)

    return {"person_id": person_id, "name": name, "image_name": image_name}


@app.post("/detect_person")
async def detect_person(file: UploadFile = File(...)):
    image_bytes = await file.read()

    try:
        embedding = extract_embedding(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    matches = search_face(embedding, top_k=1)

    if not matches:
        return {"matched": False, "message": "No person found in database."}

    best = matches[0]
    return {
        "matched": True,
        "score": best["score"],
        "name": best["metadata"].get("name"),
        "image_name": best["metadata"].get("image_name"),
    }
