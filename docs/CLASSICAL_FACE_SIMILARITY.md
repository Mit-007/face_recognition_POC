# Classical Face Similarity (No Embeddings)

This document provides a complete technical guide to the face similarity pipeline implemented in the application. A small YuNet model finds the face, and **hand-crafted classical features** compare identity.

---

## 1. Core Idea

Unlike deep learning approaches (such as ArcFace or SFace) that project facial crops into latent metric embedding hyperspheres, this method describes faces using **fixed, hand-crafted texture and edge descriptors**:
- **Local Binary Patterns (LBP)**: Captures microscopic skin texture while preserving spatial layout via an $8 \times 8$ grid.
- **Histogram of Oriented Gradients (HOG)**: Captures macroscopic edge structures, eye sockets, nose contours, and facial contours.
- **Distance-Based Matching**: Computes mathematical distance between descriptors ($0.4 \times \text{LBP}_{\chi^2} + 0.6 \times \text{HOG}_{\text{cosine}}$). Smaller distance = higher resemblance.
- **No Embeddings or Large Models**: Runs on the CPU with OpenCV and scikit-image. The only model is the YuNet face detector of about 232 KB, which locates faces but doesn't identify anyone.

---

## 2. Six-Stage Pipeline

```mermaid
flowchart TD
    Img["Raw Image (BGR)"] --> Detect["1. Detect Largest Face & Check Photo\nYuNet (face_detection_yunet_2023mar.onnx)"]
    Detect --> Align["2. Landmark Alignment\nEyes, Nose, Mouth Corners to Fixed Spots -> cv2.warpAffine()"]
    Align --> Norm["3. Normalization\n128x128 Grayscale -> CLAHE -> Slight Blur"]
    Norm --> Feat["4. Feature Extraction\n- LBP (8x8 grid -> 640-dim)\n- HOG (8 bins, 8px cell -> 2048-dim)"]
    Feat --> Comp["5. Distance Comparison\n- LBP: Chi-Square Distance\n- HOG: Cosine Distance\n- Final = 0.4*LBP + 0.6*HOG"]
    Comp --> Rank["6. Candidate Ranking\nSort database images ascending by distance"]
```

### Stage 1: Face Detection (YuNet)
- Model: `face_detection_yunet_2023mar.onnx`, run through `cv2.FaceDetectorYN`.
- Photos larger than 640 px are shrunk for detection, and results are mapped back to the original photo.
- Returns each face's box and five landmarks: both eyes, the nose tip, and both mouth corners.
- Isolates all candidate faces and selects the **largest face** by area ($w \times h$).
- Applies the photo checks below. A failed check returns a message saying how to fix the photo.

| Check | Registration | Detection |
|---|---|---|
| Any other face in the photo, at 0.8 confidence or higher | Rejected | Rejected |
| Minimum face size | 90 px | 70 px |
| Brightness (mean, 0-255) | 60-200 | 45-220 |
| Sharpness (Laplacian variance of the face at 128×128) | At least 60 | At least 40 |
| Left-right brightness difference | Up to 25% | Up to 35% |
| Nose off the middle of the eyes, as a share of eye distance | Up to 15% | Up to 25% |
| Eye tilt | Up to 25° | Up to 25° |
| Face cut off by the photo edge (face box within 5% of an edge, or a landmark outside) | Rejected | Rejected |
| Sunglasses (75th percentile brightness of the brighter eye area under 0.35 of cheek brightness) | Rejected | Rejected |
| Mask (at least 2 of: nose color differs from forehead by over 20, chin detail under 30, lip edges under 18) | Rejected | Rejected |

| Hand (horizontal-to-vertical edge ratio over 1.6 across the lower face, and over 30% skin beside the face box at mouth height) | Rejected | Rejected |

The mask check needs two signals so a beard alone doesn't count: a beard leaves the nose visible and adds texture. The sunglasses and mask limits were set using normal photos and simulated coverings. The hand limits come from only two real photos, so all of these still need testing on more real photos. The hand check works best when fingers lie across the face and a hand or arm comes in from the side. A phone, a paper, vertical fingers, or a hand fully inside the face box may still pass.

Already registered portraits skip these limits when their features are recalculated, but a face must still be found.

### Stage 2: Landmark Alignment
- Uses the five YuNet landmarks and finds the rotation, scale, and shift that best fit them to fixed template spots, using least squares.
- Applies that affine warp (`cv2.warpAffine`) to every photo. This keeps the facial features in the same grid cells.

### Stage 3: Normalization & Illumination Equalization
- The aligned output is a single-channel 8-bit grayscale image of **$128 \times 128$ pixels**.
- Applies **CLAHE (Contrast Limited Adaptive Histogram Equalization)** with `clipLimit=2.0, tileGridSize=(8, 8)` to suppress ambient shadows and lighting imbalances.
- Applies a slight Gaussian blur (sigma 1.0) to reduce noise and small lighting differences.
- Matching uses a slightly wider crop (landmark template scaled by 0.9) to include more of the jaw and face outline. The covering checks keep the original crop, because their face areas are measured on it.

### Stage 4: Feature Extraction
1. **LBP on an $8 \times 8$ Grid**:
   - Uniform LBP with $P=8$ sampling points and radius $R=2$ (`method='uniform'`).
   - Image is split into an $8 \times 8$ grid of cells (each cell is $16 \times 16$ px).
   - Each cell computes a 10-bin histogram normalized with L1 norm.
   - Concatenating all 64 cells produces a **640-dimensional texture descriptor**.
2. **HOG (Edge and Shape Structure)**:
   - 8 orientation bins, $8 \times 8$ pixels per cell, $1 \times 1$ block layout, giving 256 boxes.
   - Outputs a **2048-dimensional edge descriptor** normalized with L2-Hys norm.

### Stage 5: Dual Metric Distance Comparison
Given query descriptors $(P_{\text{LBP}}, \mathbf{u}_{\text{HOG}})$ and reference descriptors $(Q_{\text{LBP}}, \mathbf{v}_{\text{HOG}})$:

Each distance is calculated separately for each grid box. Only the closest 80% of boxes (51 of 64 for LBP, 204 of 256 for HOG) are averaged, so a shadow, hair, or other local difference can't dominate the result.

1. **Chi-Square Distance for LBP**, for each box $b$ with its 10 bins:
   $$\text{dist}_{\text{LBP}}(b) = \frac{1}{2} \sum_{i=1}^{10} \frac{(P_i - Q_i)^2}{P_i + Q_i + \epsilon} \in [0, 1]$$
2. **Cosine Distance for HOG**, for each box $b$ with its 8 orientation values:
   $$\text{dist}_{\text{HOG}}(b) = 1.0 - \frac{\mathbf{u}_b \cdot \mathbf{v}_b}{\|\mathbf{u}_b\|_2 \|\mathbf{v}_b\|_2 + \epsilon} \in [0, 1]$$
3. **Combined Final Distance**, where $\text{dist}_{\text{LBP}}$ and $\text{dist}_{\text{HOG}}$ are the averages of the closest boxes:
   $$\text{Final Distance} = 0.4 \times \text{dist}_{\text{LBP}} + 0.6 \times \text{dist}_{\text{HOG}}$$
   $$\text{Similarity Score} = \max(0.0, 1.0 - \text{Final Distance})$$

### Stage 6: Ranking and Decision
- Compares query against all registered identities, and divides each distance by that person's average distance to the other registered photos, then by this scan's average of those results:
  $$\text{Relative Distance} = \frac{\text{Final Distance}}{\text{Average distance to the other registered photos} \times \text{Scan average}}$$
  A registered photo that looks close to everyone gets a larger result, so it can't win most scans. The scan average means a scan only passes if one person stands out from the rest.
- Sorts results ascending by `relative_distance` (smaller = more alike). The interface calls this the score.
- **Verified**: the closest relative distance is at or below `MATCH_RELATIVE_LIMIT` (default 0.90) and at least `MATCH_CLEAR_WINNER_MARGIN` (default 0.02) below the second-closest.
- **Please retake**: within the limit, but the two closest people are too similar.
- **Unknown person**: above the limit.
- With one person registered, there are no others to compare against. The plain distance must be at or below `MATCH_DISTANCE_CUTOFF` (default 0.12).
- Every scan returns all registered people's scores and distances, closest first, for score analysis. These aren't probabilities. Detection history shows the plain distance, with the limit converted to the same scale.

---

## 3. API Endpoints

- `POST /register_new_person`: Validates a portrait, stores it locally, and caches its LBP + HOG descriptors.
- `POST /detect_person`: Runs the classical LBP + HOG pipeline and returns the decision, distance breakdown, and every registered person's distances.
