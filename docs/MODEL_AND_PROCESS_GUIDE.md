# Model & Process Guide — Face Recognition POC (YuNet + SFace int8)

This document provides a technical guide to the neural models running in the Face Recognition POC system, explaining model architectures, memory footprints, input/output specifications, and the zero-padding strategy for 512-d vector compatibility.

---

## 1. Overview: The OpenCV DNN Model Suite

The backend utilizes native **OpenCV Zoo** models running directly inside OpenCV's C++ DNN engine via `cv2.FaceDetectorYN` and `cv2.FaceRecognizerSF`. This eliminates heavy PyTorch/ONNX Runtime overhead.

| Model | File | Disk Size | RAM Footprint | Output | Purpose |
|---|---|---|---|---|---|
| **YuNet** | `face_detection_yunet_2023mar.onnx` | **~232 KB** | ~5 MB | BBox + 5 Landmarks + Confidence | Real-time face detection |
| **SFace (int8)** | `face_recognition_sface_2021dec_int8.onnx` | **~9.8 MB** | ~20–30 MB | 128-d L2-normalized feature vector | Face recognition & embedding |

**Total Model Memory**: **~25–35 MB total RAM** (vs. ~500 MB for InsightFace Buffalo_L).

---

## 2. Model Deep Dive

### 2.1 Detection — YuNet (`face_detection_yunet_2023mar.onnx`)
- **Developer**: OpenCV Zoo / Shiqi Yu team.
- **Architecture**: Ultra-lightweight anchor-based convolutional network optimized for edge and low-power hardware.
- **Input**:
  - Dynamically sized BGR image (input dimensions must be supplied before detection: `detector.setInputSize((w, h))`).
  - Score threshold: `0.70` (filters out noise and non-faces).
  - NMS threshold: `0.30` (suppresses overlapping duplicate boxes).
- **Output**:
  For each detected face, an array of 15 elements:
  - `[0:4]`: Bounding box `[x, y, w, h]`
  - `[4:14]`: 5 facial landmarks (10 coordinates: right eye, left eye, nose tip, right mouth corner, left mouth corner)
  - `[14]`: Detection confidence score (between 0.0 and 1.0)
- **Coordinate Conversion**:
  YuNet provides `[x, y, w, h]`. The backend automatically converts this to `[x1, y1, x2, y2]` = `[x, y, x + w, y + h]` for standardized frontend SVG overlays (`FaceOverlay.jsx`).

---

### 2.2 Recognition — SFace int8 (`face_recognition_sface_2021dec_int8.onnx`)
- **Developer**: OpenCV Zoo.
- **Architecture**: Deep convolutional feature extractor quantized to **INT8** precision for minimal memory consumption and fast CPU inference.
- **Facial Alignment (`alignCrop`)**:
  - `recognizer.alignCrop(img, face)` uses the 5 landmarks detected by YuNet to compute a similarity transform.
  - Rotates, scales, and crops the face into a standardized 112×112 px image.
- **Feature Extraction**:
  - `recognizer.feature(aligned_crop)` evaluates the aligned crop.
  - Outputs a **128-dimensional float32 vector**.
- **L2 Normalization**:
  The 128-d feature vector is normalized to unit length:
  $$\mathbf{u} = \frac{\mathbf{f}}{\|\mathbf{f}\|_2}$$

---

## 3. 512-Dimensional Vector Strategy for Pinecone

### The Requirement:
The Pinecone serverless index (`face-embeddings-poc`) was provisioned with:
- `dimension = 512`
- `metric = "cosine"`

### The Solution: Zero-Padding
Instead of changing or dropping the index, we zero-pad the 128-d unit vector $\mathbf{u}$ with 384 zeros to form a 512-dimensional vector $\mathbf{u}'$:

```python
# L2 normalize 128-d vector
norm = np.linalg.norm(feature_128)
if norm > 0:
    feature_128 = feature_128 / norm

# Zero-pad to 512 dimensions
embedding_512 = np.pad(feature_128.flatten(), (0, 384)).astype(np.float32)
```

### Mathematical Proof of Metric Invariance:
For two unit vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^{128}$:
$$\|\mathbf{u}'\| = \sqrt{\sum_{i=1}^{128} u_i^2 + \sum_{j=1}^{384} 0^2} = \|\mathbf{u}\| = 1$$
The dot product in 512 dimensions:
$$\mathbf{u}' \cdot \mathbf{v}' = \sum_{i=1}^{128} u_i v_i + \sum_{j=1}^{384} (0 \cdot 0) = \mathbf{u} \cdot \mathbf{v}$$
Since cosine similarity between unit vectors is identical to their dot product:
$$\text{Cosine}(\mathbf{u}', \mathbf{v}') \equiv \text{Cosine}(\mathbf{u}, \mathbf{v})$$
This preserves exact cosine distance precision to $< 10^{-9}$ numerical accuracy.

---

## 4. Matching Threshold Calibration

- SFace is trained using SphereFace/ArcFace margin principles.
- Under cosine similarity metric:
  - **Identical Identity**: Typically scores between **`0.45` and `0.90`**.
  - **Threshold**: **`0.363`** (calibrated for False Acceptance Rate $< 0.001$).
  - Scores below `0.363` represent different individuals and are rejected as **Unknown**.
