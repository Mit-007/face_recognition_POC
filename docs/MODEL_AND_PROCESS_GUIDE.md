# Model & Process Guide — Face Recognition POC

This documents exactly which models run, what each one does, and the full
step-by-step process for both APIs. Verified directly against the actual
files InsightFace downloaded to this machine
(`C:\Users\MIT\.insightface\models\buffalo_l\`).

---

## 1. The Model Pack: `buffalo_l`

`FaceAnalysis(name="buffalo_l")` in `face_service.py` downloads and loads **5 ONNX models**:

| File | Size | Purpose | Used by our app? |
|---|---|---|---|
| `det_10g.onnx` | 17 MB | **Face detection** — SCRFD detector, finds face bounding boxes + 5-point landmarks | **Yes** — required |
| `w600k_r50.onnx` | 174 MB | **Face recognition / embedding** — ResNet-50 trained with ArcFace loss on the WebFace600K dataset, outputs the 512-d vector | **Yes** — this is the core model |
| `2d106det.onnx` | 5 MB | 106-point 2D facial landmark detection (detailed face mesh points) | Loaded automatically, **not used** by our code |
| `1k3d68.onnx` | 144 MB | 68-point 3D facial landmark detection | Loaded automatically, **not used** by our code |
| `genderage.onnx` | 1.3 MB | Age + gender estimation | Loaded automatically, **not used** by our code |

**Only 2 of the 5 models actually matter for what this app does** — detection (`det_10g`) and recognition (`w600k_r50`). The other 3 load into memory because `FaceAnalysis` loads the whole pack by default, but we only ever read `face.embedding` from the result — we never read `face.landmark_2d_106`, `face.landmark_3d_68`, `face.gender`, or `face.age`.

> Optimization note: if RAM/startup time ever becomes a concern, you could load only the `detection` and `recognition` modules instead of the full pack — not done currently since it wasn't a priority yet.

---

## 2. Model Details

### Detection — `det_10g.onnx` (SCRFD)
- **Architecture**: SCRFD (Sample and Computation Redistribution for Face Detection) — a lightweight anchor-based detector.
- **Input**: resized image (`det_size=(640, 640)` as configured in `face_service.py`).
- **Output per face**: bounding box (`bbox`), detection confidence (`det_score`), and 5 facial keypoints (`kps`) — left eye, right eye, nose, left mouth corner, right mouth corner.
- **Why it matters here**: those 5 keypoints are what `insightface` uses internally to **align** the face (rotate/scale/crop to a standard 112x112 orientation) before it ever reaches the recognition model. Bad detection → bad alignment → bad embedding, even if the recognition model itself is fine.

### Recognition — `w600k_r50.onnx` (ArcFace / ResNet-50)
- **Architecture**: ResNet-50 backbone trained with **ArcFace loss** (Additive Angular Margin Loss) on WebFace600K (~600K identities).
- **Input**: the aligned 112x112 face crop (output of the detection+alignment step).
- **Output**: a single **512-dimensional float vector** — the embedding.
- **Why ArcFace loss matters**: unlike standard classification loss, ArcFace explicitly optimizes so that embeddings of the *same* identity are pulled angularly close together, and *different* identities are pushed angularly apart — which is exactly what makes **cosine similarity** a meaningful distance metric on these vectors later. This is why the matching step (Section 4) uses cosine similarity, not Euclidean distance.

---

## 3. Full Process — Step by Step

### A. Register New Person (`POST /register_new_person`)

```
1. Client sends: name (query param) + image file (multipart)
        │
2. Backend reads image bytes
        │
3. face_service.decode_image()
   → cv2.imdecode() turns bytes into a pixel array
   → fails here if file isn't a valid image
        │
4. face_service.check_blur()
   → grayscale + Laplacian variance
   → fails here if image is too blurry (BLUR_VARIANCE_THRESHOLD)
        │
5. InsightFace FaceAnalysis.get(img)
   a) det_10g.onnx scans the image → finds face bbox + 5 keypoints
   b) internal alignment: crops/warps face to 112x112 using those keypoints
   c) w600k_r50.onnx runs on the aligned crop → outputs 512-d embedding
        │
6. Validation on the detection result:
   → 0 faces found → reject
   → >1 face found (require_single_face=True) → reject
   → face bbox smaller than MIN_FACE_SIZE_PX → reject
        │
7. Generate person_id (UUID)
        │
8. Save original image bytes to disk:
   backend/database_images/{person_id}.jpg
        │
9. pinecone_service.upsert_face()
   → sends {id: person_id, values: embedding, metadata: {name, image_name}}
   → stored in Pinecone index "face-recognition-poc" (512-dim, cosine)
        │
10. Return {person_id, name, image_name} to client
```

### B. Detect Person (`POST /detect_person`)

```
1. Client sends: image file (multipart)
        │
2-6. Same as Register steps 2-6 above
   (decode → blur check → detect+align+embed → validate)
        │
7. pinecone_service.search_face(embedding, top_k=1)
   → Pinecone computes cosine similarity between this embedding
     and every stored embedding in the index
   → returns the single closest match + its metadata + similarity score
        │
8. Backend returns:
   - matched: true, name, image_name, score   (if any match exists)
   - matched: false, "No person found"        (if index is empty)
        │
   NOTE: there is currently no minimum score threshold —
   any top-1 result is returned as a "match" even if the score is low.
```

---

## 4. Why Cosine Similarity, and What the Score Means

Pinecone's index (`pinecone_service.py`) was created with `metric="cosine"`:
```python
_pc.create_index(..., metric="cosine", ...)
```

Cosine similarity measures the **angle** between two 512-d vectors, ignoring their magnitude:
- `1.0` → vectors point in exactly the same direction → extremely likely the same person
- `~0.5-0.7` → somewhat similar → could be the same person under poor conditions, or a different but similar-looking person
- `~0.0-0.3` → very different directions → almost certainly a different person

Because the recognition model (`w600k_r50.onnx`) was trained specifically with ArcFace's angular margin loss, the angle between embeddings is a meaningful, trained signal for identity — this isn't an arbitrary choice, it's matched to how the model was trained.

---

## 5. Summary: What Is and Isn't Used

| Component | Role | Status |
|---|---|---|
| SCRFD detector (`det_10g.onnx`) | Find + locate the face, provide alignment keypoints | Active, required |
| ArcFace recognizer (`w600k_r50.onnx`) | Convert aligned face → 512-d embedding | Active, the core of matching |
| 2D/3D landmark models, gender/age model | Loaded with the pack, outputs unused | Inactive (dead weight currently) |
| OpenCV (`cv2`) | Image decode + blur detection only | Active, pre-validation only |
| Pinecone cosine search | Compare embedding against stored embeddings | Active, the actual matching step |
| Minimum score threshold | Reject low-confidence matches | **Not implemented yet** — known gap |
