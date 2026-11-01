# API & Image Validation Configuration

Covers both endpoints (`/register_new_person`, `/detect_person`) and the
image-validation settings they share, defined in `backend/face_service.py`.
Both endpoints call the same `extract_embedding()` function, so **the same
validation rules apply to both** — there's no separate config per endpoint
today.

---

## 1. API Reference

### `POST /register_new_person`

| | |
|---|---|
| Query param | `name` (string, required) — person's display name |
| Body | `file` (multipart/form-data, required) — one image |
| Success response | `{"person_id": "<uuid>", "name": "...", "image_name": "<uuid>.jpg"}` |
| Validation failure | `400` — `{"detail": "<reason>"}` |
| Server error | `500` — e.g. Pinecone write failure |

### `POST /detect_person`

| | |
|---|---|
| Body | `file` (multipart/form-data, required) — one image |
| Match found | `{"matched": true, "score": 0.0-1.0, "name": "...", "image_name": "..."}` |
| No match | `{"matched": false, "message": "No person found in database."}` |
| Validation failure | `400` — `{"detail": "<reason>"}` |

`score` is a **cosine similarity** value from Pinecone (higher = more similar, max 1.0). There is currently **no minimum-score threshold applied** — any top-1 result is returned as `matched: true`, even a weak match. See [Section 3](#3-score-threshold-not-yet-enforced) below.

---

## 2. Image Validation Settings (`face_service.py`)

| Constant | Current Value | Valid Range | Effect of Increasing | Effect of Decreasing |
|---|---|---|---|---|
| `BLUR_VARIANCE_THRESHOLD` | `5.0` | `0` – `150` typical | Stricter — rejects more images as "too blurry", fewer false matches from unclear photos | Looser — accepts blurrier images, risk of poor-quality embeddings reducing match accuracy |
| `MIN_FACE_SIZE_PX` | `20` | `20` – `150` typical | Stricter — requires closer/larger face in frame, better embedding quality | Looser — accepts smaller/farther faces, embedding quality may degrade |
| `require_single_face` | `True` (hardcoded default in both endpoints) | `True` / `False` | `True`: rejects multi-face images | `False`: silently uses the first detected face even if multiple people are in frame |

### How these were tuned
- Originally set stricter (`BLUR_VARIANCE_THRESHOLD=30.0`, `MIN_FACE_SIZE_PX=40`) per general best practice.
- Lowered to current values (`5.0` / `20`) because real test uploads (webcam/phone photos) were being rejected as "too blurry" — current values are intentionally **loose for development/testing**.
- **Before production**, revisit these — recommended production starting point: `BLUR_VARIANCE_THRESHOLD=15-20`, `MIN_FACE_SIZE_PX=40`, tuned against your actual camera/upload source.

### Where to change them
Edit the constants at the top of `backend/face_service.py`:
```python
MIN_FACE_SIZE_PX = 20
BLUR_VARIANCE_THRESHOLD = 5.0
```
Restart the backend (`uvicorn`) after changing — no other config needed.

---

## 3. Score Threshold (Not Yet Enforced)

`/detect_person` currently returns whatever Pinecone's top-1 match is, with **no minimum similarity cutoff**. This means a genuinely unrecognized face could still return `matched: true` with a low score (e.g. 0.3) if that's the closest vector in the database.

**Recommended addition** (not yet implemented):
```python
MATCH_SCORE_THRESHOLD = 0.5   # tune based on testing; typical ArcFace cosine range: 0.3 (same person, poor conditions) to 0.9+ (same person, good conditions)

if not matches or matches[0]["score"] < MATCH_SCORE_THRESHOLD:
    return {"matched": False, "message": "No confident match found."}
```

| Threshold | Behavior |
|---|---|
| `0.3 - 0.4` | Permissive — more false positives (wrong person matched) |
| `0.5 - 0.6` | Balanced — reasonable starting point for most use cases |
| `0.7+` | Strict — fewer false positives, but may reject valid matches under poor lighting/angle |

This should be added before relying on `/detect_person` for any real identification decision (e.g. attendance marking).

---

## 4. Other Implicit Limits

| Limit | Current Behavior | Configurable? |
|---|---|---|
| Max upload file size | None enforced by the API — limited only by FastAPI/Uvicorn defaults | Not currently configured; add `python-multipart` size limits or a manual `Content-Length` check if needed |
| Allowed file types | None enforced explicitly — relies on OpenCV's `imdecode` failing on non-image data | Add explicit extension/MIME check if stricter control is needed |
| Multiple images per person | Not supported — one image per registration call | Would require API changes (accept `List[UploadFile]`) |
