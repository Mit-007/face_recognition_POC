# Architecture & Workflow Guide — Classical Face Recognition (Non-ML)

This document provides a comprehensive technical breakdown of the **Classical Face Recognition System** (No ML / No Neural Networks / No Cloud Vector DBs). It details the system architecture, directory layout, end-to-end data workflows, hand-crafted feature extractors, and distance ranking mechanics.

---

## 1. High-Level System Architecture

The application is engineered as a lightweight, explainable, and fully self-contained Computer Vision system combining web standards, classical Viola-Jones Haar cascades, illumination-invariant normalization, and hand-crafted texture & edge distance comparisons.

```mermaid
flowchart TD
    subgraph Client["Client Tier (Browser)"]
        UI["React 19 + Vite UI"]
        Cam["HTML5 MediaDevices (Webcam)"]
        Canvas["HTML5 Canvas Frame Grabber"]
        Axios["Axios HTTP Client"]
        Overlay["SVG FaceOverlay (BBox + Eyes)"]
        
        Cam --> Canvas --> UI
        UI --> Overlay
        UI --> Axios
    end

    subgraph Backend["API & Vision Tier (FastAPI Server)"]
        API["FastAPI Routing Gateway (:8000)"]
        Static["FastAPI StaticFiles Mount (/images)"]
        
        subgraph Classical_CV["Pure Classical Computer Vision Pipeline (Zero ML / Zero GPU)"]
            Decode["cv2.imdecode"]
            HaarDetect["1. Largest Face Detection & Photo Checks\nHaar Cascade (haarcascade_frontalface_default.xml)"]
            EyeAlign["2. Eye Alignment\nHaar Eye Cascade -> Both Eyes to Fixed Pixels via cv2.warpAffine()"]
            Normalize["3. Normalization\n128x128 Grayscale -> CLAHE"]
            FeatExtract["4. Hand-Crafted Feature Extraction\n- LBP (8x8 Grid -> 640-dim Texture Vector)\n- HOG (8 Bins, 16px Cell -> 512-dim Edge Vector)"]
            Distance["5. Dual Metric Distance Engine\n- LBP: Chi-Square Distance\n- HOG: Cosine Distance\n- Final Distance = 0.6*LBP + 0.4*HOG"]
            Ranker["6. Database Distance Ranker\nCompare against all registered faces -> Return most similar"]
        end
        
        DiskStore[("Local Disk Store\nbackend/database_images/")]
        JSONStore[("JSON Catalogs\nregistered_users.json\ndetection_history.json")]
        MemCache[("In-Memory Feature Cache\n_feature_cache")]
    end

    Axios -->|POST /register_new_person\nPOST /detect_person| API
    Static -->|GET /images/:id.jpg| UI
    
    API --> Decode --> HaarDetect --> EyeAlign --> Normalize --> FeatExtract
    
    API -->|Save Reference JPG| DiskStore
    API -->|Save Enrolled User Record| JSONStore
    FeatExtract -->|Store Extracted LBP + HOG| MemCache
    
    FeatExtract --> Distance --> Ranker
    Ranker -->|Read Reference Features| MemCache
    Ranker -->|Return Best Match + Rankings| API
```

---

## 2. Project Directory Structure

```
D:\face_recognition_POC\
├── backend\
│   ├── database_images\       # Stored portraits for enrolled users
│   ├── models\                # Official OpenCV Haar Cascades
│   │   ├── haarcascade_frontalface_default.xml  # Viola-Jones frontal face detector
│   │   └── haarcascade_eye.xml                  # Eye pair detector for leveling
│   ├── venv\                  # Python virtual environment
│   ├── .env                   # Distance cutoff and score thresholds
│   ├── .env.example           # Example configuration template
│   ├── classical_face_service.py # Pure Classical CV detection, photo checks, alignment, LBP & HOG
│   ├── evaluate_accuracy.py   # Measures accuracy of each matching approach on test photos
│   ├── history_service.py     # Audit detection history management (JSON backed)
│   ├── main.py                # FastAPI endpoints, CORS, lifespan, and static routes
│   ├── registered_users.json  # Enrolled user identity catalog
│   ├── detection_history.json # Detection audit event log
│   ├── user_service.py        # Enrolled identity CRUD service
│   └── requirements.txt       # Python dependencies (FastAPI, OpenCV, scikit-image)
├── frontend\
│   ├── src\
│   │   ├── components\
│   │   │   ├── CameraCapture.jsx      # Webcam stream, canvas grabber, resolution cap
│   │   │   ├── DetectPerson.jsx       # Detection view, distance chips, and ranking
│   │   │   ├── DetectionHistory.jsx   # Searchable detection audit log table
│   │   │   ├── FaceOverlay.jsx        # SVG visual bounding box & eyes overlay
│   │   │   ├── Navbar.jsx             # Tab navigation (Register, Detect, Users, History)
│   │   │   ├── RegisteredUsers.jsx    # Enrolled user profile cards with search/delete
│   │   │   ├── RegisterPerson.jsx     # Split 2-column registration interface
│   │   │   └── StatusBanner.jsx       # UI alert banners (success / error)
│   │   ├── api.js             # Axios client configuration & endpoint mappings
│   │   ├── App.css            # Component styles, layout, and visual overlays
│   │   ├── App.jsx            # Main tab controller and layout container
│   │   └── main.jsx           # React DOM root entrypoint
│   └── package.json           # Frontend dependencies (React 19, Axios, Vite)
└── docs\
    └── CLASSICAL_FACE_SIMILARITY.md # Complete technical deep-dive on LBP + HOG
```

---

## 3. End-to-End System Workflows

### 3.1 Registration Workflow (`POST /register_new_person`)

When a user is enrolled:
1. Detects the face and locates both eyes using Haar cascades.
2. Applies the strict registration photo checks: one person, face size, brightness, sharpness, even lighting, and a straight-on pose. A failed check returns a message saying how to fix the photo.
3. Warps the face so both eyes land on fixed pixels and normalizes to a standard **$128 \times 128$ CLAHE** grayscale patch.
4. Extracts 640-d LBP texture features and 512-d HOG shape features.
5. Saves the reference JPEG to `backend/database_images/{person_id}.jpg`.
6. Appends identity metadata to `registered_users.json`.
7. Pre-caches LBP and HOG features in memory (`_feature_cache`) for instant comparisons.

### 3.2 Detection Workflow (`POST /detect_person`)

When an identity verification scan is performed:
1. Captures one photo from the webcam.
2. Applies the same photo checks with more lenient detection limits, then extracts eye-aligned 128×128 LBP and HOG descriptors.
3. Compares the query against **every registered face** in the database:
   - Computes Chi-Square distance on LBP: $\text{dist}_{\text{LBP}}$
   - Computes Cosine distance on HOG: $\text{dist}_{\text{HOG}}$
   - Calculates weighted distance: $\mathbf{\text{Distance} = 0.6 \times \text{LBP} + 0.4 \times \text{HOG}}$
   - Resemblance score: $\max(0.0, 1.0 - \text{Distance})$
4. Ranks all candidates ascending by distance (closest resemblance first).
5. If the closest candidate's distance $\le 0.22$ (resemblance $\ge 78\%$), returns **MATCH VERIFIED** with candidate details, side-by-side reference portrait, and database ranking list.
6. Records the audit event in `detection_history.json`.

---

## 4. Key Strengths of the Classical Non-ML Approach

1. **Zero Deep Learning / Zero Weights**: No PyTorch, no ONNX Runtime, no GPU required.
2. **Deterministic & Fully Explainable**: Face resemblance is computed directly from raw skin texture variations (LBP) and gradient edge contours (HOG).
3. **No External Cloud Vector Dependencies**: Runs 100% locally on CPU without Pinecone or third-party cloud database calls.
4. **Measured Memory Use**: About **100 MB of RAM** on Windows with one registered person and photos up to 640 px, with OpenCL off and one server process. Much larger photos sent straight to the API can use far more.
