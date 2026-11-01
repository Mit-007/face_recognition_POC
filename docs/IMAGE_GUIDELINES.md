# Image Upload Specification — Face Recognition POC

This document defines what counts as a **valid, clear image** for both the
**Register New Person** and **Detect Person** endpoints. These rules match
what the backend (`face_service.py`) actually enforces via OpenCV +
InsightFace, plus recommended practices for best accuracy.

---

## 1. File Requirements

| Requirement | Rule |
|---|---|
| Format | JPG, JPEG, or PNG |
| File validity | Must be a readable, non-corrupted image (OpenCV must be able to decode it) |
| Max file size | 5 MB (recommended — keep uploads light for fast processing) |
| Color | Color image preferred (grayscale may reduce accuracy) |

**Rejected if:** the file cannot be decoded as an image at all →
`"Uploaded file is not a valid image."`

---

## 2. Face Requirements

| Requirement | Rule |
|---|---|
| Number of faces | Exactly **one** face in the image |
| Face visibility | Face must be clearly visible, unobstructed |
| Minimum face size | At least **40x40 pixels** in the detected bounding box |
| Face angle | Frontal or near-frontal (within ~15-20° of camera) — side profiles reduce accuracy |
| Eyes | Both eyes visible, not closed |
| Accessories | Avoid sunglasses, masks, or anything covering the nose/mouth/eyes |
| Expression | Neutral expression recommended (register); any expression is fine for detect |

**Rejected if:**
- No face detected → `"No face detected in the uploaded image."`
- More than one face detected → `"Multiple faces (N) detected. Please upload an image with a single face."`
- Face bounding box smaller than 40x40 px → `"Detected face is too small. Please upload a closer photo."`

---

## 3. Image Quality Requirements

| Requirement | Rule |
|---|---|
| Sharpness | Must not be blurry — checked via Laplacian variance, threshold = 30.0 |
| Lighting | Even, sufficient lighting on the face — avoid heavy backlighting or shadows across the face |
| Resolution | Recommended minimum **480x480 px** overall image size for reliable detection |
| Orientation | Image should be right-side-up (not rotated/sideways) |

**Rejected if:** the image is too blurry (Laplacian variance below threshold) →
`"Image is too blurry. Please upload a clearer photo."`

---

## 4. Composition Guidelines (Recommended, Not Enforced)

These aren't hard-rejected by the backend yet, but following them significantly
improves match accuracy:

- Face should take up roughly **30-70%** of the frame — not too far, not too close.
- Plain, uncluttered background preferred.
- Avoid extreme angles (looking up/down/sideways).
- Avoid heavy filters, overlays, or watermarks on the face.
- For **registration**, use a recent, well-lit photo — stale/old photos reduce match accuracy over time.

---

## 5. Quick Checklist for Users

Before uploading, confirm:

- [ ] Only one person's face is in the photo
- [ ] Face is facing the camera, not blurry
- [ ] Good lighting — no harsh shadows on the face
- [ ] No sunglasses, mask, or hand covering the face
- [ ] File is JPG/PNG and under 5 MB

---

## 6. Error Messages Reference

| Error Message | Cause |
|---|---|
| `Uploaded file is not a valid image.` | File is corrupted or not an image format OpenCV can decode |
| `Image is too blurry. Please upload a clearer photo.` | Laplacian variance below sharpness threshold |
| `No face detected in the uploaded image.` | InsightFace detector found zero faces |
| `Multiple faces (N) detected. Please upload an image with a single face.` | More than one face found |
| `Detected face is too small. Please upload a closer photo.` | Face bounding box under 40x40 px |
