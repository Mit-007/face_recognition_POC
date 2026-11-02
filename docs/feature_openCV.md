# OpenCV Image & Photo Validation Features Catalog

This document provides a comprehensive reference of image and photo validation features that can be implemented using the **OpenCV (`cv2`)** library, followed by **tailored recommendations** for this specific Face Recognition project (`face_recognition_POC`).

---

## 1. Executive Summary

In a face recognition pipeline (such as InsightFace + Pinecone), the quality of face embeddings directly depends on the quality of the input image. Running heavy deep-learning models on corrupted, blurry, overexposed, or tilted images leads to poor embeddings and false matches.

**OpenCV acts as the lightweight "first line of defense" (Validation Gate)**:
- **Fast & CPU-efficient**: Executes in milliseconds before costly inference.
- **Actionable User Feedback**: Immediately informs users *why* an image failed (e.g., "Face too dark", "Camera tilted", "Image blurry").
- **Embedding Integrity**: Guarantees that only high-quality, standardized photos enter Pinecone.

---

## 2. Complete Catalog of OpenCV Validation Features

The table below catalogs all major photo validation techniques achievable natively with OpenCV and NumPy:

| Category | Validation Feature | OpenCV Functions / Methods | Metric / Threshold Type | Failure Impact on Face Recognition |
|---|---|---|---|---|
| **1. File & Format** | Image Decoding & Corruption | `cv2.imdecode()`, `cv2.imread()` | Checks for non-null `numpy.ndarray` | Crash or unhandled 500 error |
| | Aspect Ratio & Dimensions | `img.shape[:2]` | Min/Max width, height, aspect ratio | Distorted/stretched faces |
| | Channel Integrity | `img.shape[2]`, `cv2.split()` | 3 channels (BGR/RGB), non-alpha | Unexpected alpha blending artifacts |
| | Bit Depth & Range | `img.dtype`, `np.min()`, `np.max()` | `uint8` with standard `[0, 255]` range | Quantization errors |
| **2. Sharpness & Blur** | Laplacian Variance | `cv2.Laplacian()`, `.var()` | High-frequency variance threshold | Blurry features, lost landmarks |
| | Tenengrad Gradient (Sobel) | `cv2.Sobel()`, `cv2.magnitude()` | Sum of squared directional gradients | Motion and out-of-focus blur |
| | Fast Fourier Transform (FFT) | `np.fft.fft2()`, `cv2.dft()` | High vs. low frequency power spectrum ratio | Defocus blur vs. motion blur |
| | Face ROI Blur | Crop `bbox` $\to$ `cv2.Laplacian()` | Variance restricted only to face box | Eliminates background bokeh false passes/fails |
| **3. Lighting & Exposure** | Overall Luminance | `cv2.cvtColor()` (HSV `V` or LAB `L`) | Mean luminance in `[0, 255]` range | Dark/night photos lower embedding fidelity |
| | Underexposure (Dark) | `cv2.inRange()`, `cv2.countNonZero()` | Percentage of pixels $< 30$ | Loss of facial contours and contrast |
| | Overexposure (Washed Out) | `cv2.inRange()`, `cv2.countNonZero()` | Percentage of pixels $> 245$ | Feature blowout, lost eye/nose details |
| | Dynamic Range (Histogram) | `cv2.calcHist()`, `np.count_nonzero()` | Spread across bin channels | Low-contrast or washed-out images |
| **4. Contrast & Illumination** | RMS / Global Contrast | `np.std(gray)`, Michelson formula | Standard deviation of grayscale | Flattens 3D face structure |
| | Harsh Shadows / Asymmetry | Split face ROI $\to$ Compare left/right mean | $\| \text{Mean}_{\text{Left}} - \text{Mean}_{\text{Right}} \|$ difference | ArcFace feature drift under half-shadow |
| | Glare / Specular Highlight | `cv2.inRange()` on HSV (High V, Low S) | Percentage of glare pixels on face | Blinds detector or keypoint landmarking |
| **5. Color & Noise** | True Color vs Grayscale Check | `cv2.absdiff(b, g)`, `cv2.absdiff(g, r)` | Mean difference across B, G, R channels | ArcFace was trained on color images |
| | Color Cast / White Balance | Gray World Assumption (`mean(B) / mean(R)`) | Ratio deviations from balanced neutral | Extreme tint (fluorescent green, tungsten) |
| | Image Noise / Grain | `cv2.medianBlur()`, `cv2.absdiff()` | Residual standard deviation | Sensor noise from low-light webcams |
| **6. Composition & Geometry** | Face Centering | `bbox` center vs. image center `(w/2, h/2)` | Pixel offset from center | Edge distortion or off-camera faces |
| | Face Margin / Cutoff Check | `x1 > pad`, `y1 > pad`, `x2 < W - pad` | Distance of bounding box to border | Incomplete/clipped chin, forehead, or cheek |
| | Face Area Ratio (Coverage) | `(w * h) / (img_w * img_h)` | Face occupies between 15% – 70% | Person too far away or overly close |
| | Background Uniformity | Invert face mask $\to$ `cv2.Canny()` / std | Background edge density / variance | Busy backgrounds cause false detection |
| **7. Pose & Alignment** | Head Roll / Tilt Angle | Eyes slope angle `atan2(dy, dx)` | Roll angle degree ($< 15^{\circ}$) | Heavy roll rotates face outside alignment model |
| | Frontal Yaw / Pitch Check | Ratio of nose-to-left-eye vs nose-to-right-eye | Symmetry ratio deviation | Profile pictures reduce recognition accuracy |
| **8. Features & Occlusion** | Both Eyes Open & Visible | `cv2.CascadeClassifier('haarcascade_eye')` | Exactly 2 eyes detected in upper face ROI | Blinking or squinting alters eye embedding |
| | Sunglasses Detection | Mean intensity / variance inside eye boxes | Low luminance or low contrast in eye ROI | Eyes are the most discriminatory face feature |
| | Lower Face Occlusion / Mask | Texture analysis / edge check in lower face | Homogeneous texture or high edge density | Masks or hands blocking nose and mouth |
| **9. Anti-Spoofing & Artifacts** | Screen Moire Pattern Detection | 2D Fourier Transform / Frequency Peaks | Periodic frequency spikes | Replay attack (holding a phone/tablet) |
| | JPEG Compression Artifacts | Blockiness grid detection via 8x8 DCT | Variance across 8-pixel grid borders | Heavy compression blur |
| | Paper Edge / Bezel Detection | `cv2.Canny()` + `cv2.findContours()` | Rectangular border around face | Physical printed photo or tablet frame |

---

## 3. Strongly Recommended Validation Features for `face_recognition_POC`

Based on your current codebase (`face_service.py` with InsightFace ArcFace `buffalo_l` and Pinecone vector store), here are the **top recommended features ranked by impact**:

```
                                 INPUT PHOTO
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
                 [HIGH PRIORITY]            [MEDIUM PRIORITY]
              1. Face ROI Blur Check     5. Head Tilt (Roll) Angle
              2. Over/Under Exposure     6. True Color vs Grayscale
              3. Face Boundary Cutoff    7. Glare / Flash Detection
              4. Left-Right Shadow
                         │                         │
                         └────────────┬────────────┘
                                      ▼
                             [ADVANCED PRIORITY]
                           8. Screen Moire Check
                           9. Sunglasses Detection
                                      │
                                      ▼
                        PASS ──► InsightFace + Pinecone
```

### 1. Face-Specific ROI Blur (Highest Priority)
- **Current Issue**: `face_service.py` currently runs `check_blur(img)` across the **entire image**.
  - If a user uploads a portrait with a blurry background (bokeh), OpenCV rejects it even if the face is sharp.
  - If a user uploads a photo with a sharp background (bricks, text) and a blurry face, OpenCV accepts it, yielding a poor embedding.
- **Solution**: Compute Laplacian variance **specifically on the detected face bounding box** (`face.bbox`).

### 2. Lighting & Exposure Validation (Underexposure & Overexposure)
- **Why**: ArcFace feature vectors drift significantly under poor lighting.
  - A face taken in a dark room creates noisy embeddings that match with arbitrary dark photos.
  - Flash blowout on the forehead or cheeks washes out distinctive texture.
- **Solution**: Check mean luminance of the face ROI (HSV `V` channel) to ensure it falls within `[40, 220]`.

### 3. Face Boundary Cutoff / Margin Protection
- **Why**: When a face is too close to the camera, the top of the forehead or the chin gets cut off. InsightFace will still produce a 512-d vector, but it represents an incomplete face, leading to low match scores.
- **Solution**: Ensure `x1, y1 > 10px` and `x2 < width - 10px`, `y2 < height - 10px`.

### 4. Facial Shadow / Illumination Asymmetry Detection
- **Why**: Strong side lighting (e.g., sitting near a bright window) creates a dark shadow on one half of the face. ArcFace embeddings are sensitive to half-lit faces.
- **Solution**: Split the face bounding box in half vertically, compute the average brightness of the left half vs. the right half, and reject if the difference exceeds a threshold (e.g. 50 intensity levels).

### 5. Head Tilt (Roll Angle) Check
- **Why**: While InsightFace performs affine alignment, severe tilt angles ($> 25^{\circ}$) lead to unnatural warp distortion.
- **Solution**: Use the 5 facial keypoints (`kps`) already returned by InsightFace or compute with OpenCV eye detectors:
  $$\theta = \arctan2(y_{\text{right\_eye}} - y_{\text{left\_eye}},\, x_{\text{right\_eye}} - x_{\text{left\_eye}})$$
  Reject if $|\theta| > 20^{\circ}$.

### 6. True Color Verification (Grayscale Rejection)
- **Why**: ArcFace `w600k_r50` is trained on full-color RGB facial imagery. Feeding grayscale images causes subtle vector shifts, reducing confidence scores against color gallery images.
- **Solution**: Measure the mean difference between B, G, and R color channels. If difference $< 5.0$, reject or warn.

### 7. Glare / Flash Specular Reflection Check
- **Why**: Strong direct camera flash creates white "hotspots" (specular highlights) over the forehead, nose, or eyeglasses.
- **Solution**: Measure the percentage of face pixels with saturation $< 20$ and brightness $> 245$.

---

## 4. Ready-to-Use OpenCV Code Implementations

Here are production-ready Python functions implementing the recommended validations:

### A. Face ROI Blur Check
```python
import cv2
import numpy as np

def validate_face_blur(img: np.ndarray, bbox: list[int], threshold: float = 20.0) -> None:
    """Validates sharpness specifically inside the face region."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = img.shape[:2]
    
    # Clip to image boundaries
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    face_roi = img[y1:y2, x1:x2]
    if face_roi.size == 0:
        raise ValueError("Invalid face region for blur detection.")
        
    gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
    
    if variance < threshold:
        raise ValueError(f"Face is too blurry (sharpness score {variance:.1f} < {threshold}).")
```

### B. Exposure & Lighting Validation
```python
def validate_face_lighting(img: np.ndarray, bbox: list[int], min_val: int = 40, max_val: int = 225) -> None:
    """Ensures face is not underexposed (too dark) or overexposed (washed out)."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    face_roi = img[y1:y2, x1:x2]
    
    # Convert to HSV and extract Value (brightness) channel
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    mean_brightness = np.mean(v_channel)
    
    if mean_brightness < min_val:
        raise ValueError("Photo is too dark. Please ensure good lighting on your face.")
    if mean_brightness > max_val:
        raise ValueError("Photo is overexposed/too bright. Avoid harsh direct flash.")
```

### C. Side-Lighting / Shadow Asymmetry
```python
def validate_illumination_symmetry(img: np.ndarray, bbox: list[int], max_diff: float = 45.0) -> None:
    """Detects harsh side lighting where half the face is in deep shadow."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    face_roi = img[y1:y2, x1:x2]
    gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    
    mid_x = gray_roi.shape[1] // 2
    left_mean = np.mean(gray_roi[:, :mid_x])
    right_mean = np.mean(gray_roi[:, mid_x:])
    
    diff = abs(left_mean - right_mean)
    if diff > max_diff:
        raise ValueError("Uneven lighting detected. One side of the face has heavy shadows.")
```

### D. Face Cutoff / Boundary Margin Check
```python
def validate_face_margin(img: np.ndarray, bbox: list[int], margin_px: int = 12) -> None:
    """Ensures face is not clipped at image borders."""
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    
    if x1 <= margin_px or y1 <= margin_px or x2 >= (w - margin_px) or y2 >= (h - margin_px):
        raise ValueError("Face is partially cut off at the edge of the photo. Please center your face.")
```

### E. Head Tilt (Roll Angle) Calculation
```python
def validate_head_roll(landmarks_5: np.ndarray, max_angle_deg: float = 20.0) -> None:
    """Validates that the head is upright (roll angle < max_angle_deg) using InsightFace 5-kps."""
    left_eye, right_eye = landmarks_5[0], landmarks_5[1]
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    
    angle = np.degrees(np.arctan2(dy, dx))
    if abs(angle) > max_angle_deg:
        raise ValueError(f"Head is tilted ({abs(angle):.1f}°). Please keep your head upright.")
```

### F. True Color vs Grayscale Check
```python
def validate_color_image(img: np.ndarray, min_diff: float = 4.0) -> None:
    """Ensures the uploaded image is true color, not black-and-white or monochrome."""
    if len(img.shape) < 3 or img.shape[2] != 3:
        raise ValueError("Image must be a color image.")
        
    b, g, r = cv2.split(img)
    diff_bg = np.mean(cv2.absdiff(b, g))
    diff_gr = np.mean(cv2.absdiff(g, r))
    
    if (diff_bg + diff_gr) < min_diff:
        raise ValueError("Grayscale or black-and-white photos are not supported. Please upload a color photo.")
```

### G. Glare / Specular Highlight Check
```python
def validate_glare(img: np.ndarray, bbox: list[int], max_glare_ratio: float = 0.08) -> None:
    """Detects shiny flash reflections or glare on glasses."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    face_roi = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    
    # Glare appears as high brightness (V > 245) and low saturation (S < 30)
    glare_mask = cv2.inRange(hsv, (0, 0, 245), (180, 30, 255))
    glare_ratio = cv2.countNonZero(glare_mask) / (face_roi.shape[0] * face_roi.shape[1])
    
    if glare_ratio > max_glare_ratio:
        raise ValueError("Severe glare or flash reflection detected on face.")
```

---

## 5. Architectural Recommendation for `face_service.py`

Organize your backend validation into a clean two-stage pipeline:

```
                  ┌──────────────────────────────────────────────┐
                  │ Stage 1: Global OpenCV Checks (Fast, < 2ms)  │
                  │ - Valid image decode                         │
                  │ - Minimum image resolution (e.g. 200x200)    │
                  │ - True color verification                    │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │ Stage 2: Face Detection (InsightFace SCRFD)  │
                  │ - Single face detected                       │
                  │ - Minimum face size (e.g. 40x40 px)          │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │ Stage 3: Face-Specific OpenCV Validations    │
                  │ - Face ROI Blur (sharpness on face only)     │
                  │ - Face Margin / Cutoff check                 │
                  │ - Face Lighting / Exposure (HSV V-channel)   │
                  │ - Illumination symmetry (left vs right)      │
                  │ - Head Roll Tilt (using eye keypoints)       │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │ Stage 4: ArcFace Feature Extraction          │
                  │ - Output 512-d normalized embedding          │
                  │ - Upsert or Query Pinecone                   │
                  └──────────────────────────────────────────────┘
```

### Benefits of this design:
1. **Zero False Blurs**: A soft background no longer causes a rejection when the face is crisp.
2. **Clear UI Messages**: Users receive specific corrections ("Head tilted", "Face too dark", "Face cut off").
3. **Database Consistency**: Pinecone only contains pristine face vectors, directly increasing recognition accuracy and eliminating false positive matches.
