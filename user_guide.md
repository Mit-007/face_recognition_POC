# User Guide — Face Recognition POC Application

Welcome to the **FaceID Recognition System** (Proof of Concept). This guide walks you through the initial setup, system prerequisites, and the exact step-by-step path to use the application for registering, managing, and recognizing people in real time.

---

## 1. System Overview

This application provides an ultra-lightweight, camera-based facial recognition workflow powered by OpenCV native DNN models (**YuNet** and **SFace int8**):
- **Register New Person**: Captures a live face photo, verifies image quality, detects facial features with YuNet, extracts a 512-dimensional embedding with SFace, and records the identity in Pinecone and local storage.
- **Detect Person**: Captures a face photo from the camera, validates quality, extracts the facial embedding, and searches Pinecone using cosine similarity with a calibrated threshold of `0.363`.
- **Registered Users Catalog**: Dedicated tab displaying all enrolled identities with search filtering and one-click removal.
- **Detection History Log**: Real-time audit log tracking all verification attempts, confidence scores, and statuses.

```
┌─────────────────┐       ┌────────────────────────┐       ┌─────────────────┐
│   Web Browser   │ ────► │    FastAPI Backend     │ ────► │  Pinecone DB    │
│ (React + Camera)│ ◄──── │(OpenCV: YuNet + SFace) │ ◄──── │ (Vector Search) │
└─────────────────┘       └────────────────────────┘       └─────────────────┘
```

---

## 2. Prerequisites & Hardware Requirements

### Minimum Hardware Configuration
- **CPU**: Dual-core x86_64 or ARM processor (Intel Core i3 4th Gen / AMD Ryzen 3 / Apple Silicon).
- **RAM**: **2 GB total system memory** (Application consumes only **~25–35 MB** RAM on backend thanks to YuNet + SFace int8).
- **Storage**: ~500 MB free disk space (Models take < 11 MB total).
- **Camera**: Standard USB webcam or built-in camera (720p or higher).
- **Network**: Internet connection for Pinecone serverless vector database calls.

### Software Prerequisites
1. **Operating System**: Windows 10/11, macOS, or Linux.
2. **Node.js**: Version 18.0 or higher ([Download Node.js](https://nodejs.org/)).
3. **Python**: Version 3.10, 3.11, or 3.12 ([Download Python](https://python.org/)).
4. **Pinecone API Key**: Free API key from [Pinecone.io](https://www.pinecone.io/).

---

## 3. First-Time Setup Instructions

### Step 1: Configure Environment Variables
1. Navigate to the `backend` folder:
   ```powershell
   cd D:\face_recognition_POC\backend
   ```
2. Verify or create the `.env` file:
   ```ini
   PINECONE_API_KEY=your_actual_pinecone_api_key_here
   PINECONE_INDEX_NAME=face-embeddings-poc
   PINECONE_CLOUD=aws
   PINECONE_REGION=us-east-1
   MATCH_SCORE_THRESHOLD=0.363
   ```
   > [!IMPORTANT]
   > Replace `your_actual_pinecone_api_key_here` with your Pinecone API key. The backend automatically creates or connects to the `face-embeddings-poc` index with 512 dimensions and cosine metric.

---

### Step 2: Set Up Backend Virtual Environment
1. Activate the virtual environment in `backend`:
   - **Windows PowerShell**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS**:
     ```bash
     source venv/bin/activate
     ```
2. Install required lightweight Python packages:
   ```powershell
   pip install -r requirements.txt
   ```
3. Models are pre-stored in `backend/models/`:
   - `face_detection_yunet_2023mar.onnx` (~232 KB)
   - `face_recognition_sface_2021dec_int8.onnx` (~9.8 MB)
   *(If downloading fresh models is needed, run `python download_models.py`)*.

---

### Step 3: Set Up Frontend Dependencies
1. Open a second terminal window and navigate to the `frontend` folder:
   ```powershell
   cd D:\face_recognition_POC\frontend
   ```
2. Install npm packages:
   ```powershell
   npm install
   ```

---

## 4. How to Launch the Application

Keep two terminal windows open:

### Terminal 1: Start Backend API Server
```powershell
cd D:\face_recognition_POC\backend
.\venv\Scripts\Activate.ps1
uvicorn main:app --reload --port 8000
```
- The backend API will be available at: `http://localhost:8000`
- Interactive API Documentation (Swagger): `http://localhost:8000/docs`

### Terminal 2: Start Frontend Web App
```powershell
cd D:\face_recognition_POC\frontend
npm run dev
```
- The web application will launch at: `http://localhost:5173`
- Open `http://localhost:5173` in your browser.

---

## 5. Step-by-Step Usage Path

```
[Start App] ──► [Path A: Register Person] ──► [Path B: Detect Person] ──► [Path C: View Users / History]
```

### Path A: Registering a New Person

1. **Select the Register Tab**:
   - In your browser (`http://localhost:5173`), ensure the **"Register"** tab is active in the navigation bar.
2. **Enter Identity Details (Left Column)**:
   - Enter the person's **First Name** and **Last Name** (e.g. `Alex` and `Morgan`).
3. **Capture Face Photo (Right Column)**:
   - Click the blue **"Open Camera"** button and grant camera permissions.
   - Position your face inside the guideline box and click **"Capture Photo"**.
4. **Submit Registration**:
   - Click **"Register Person"**.
   - The backend validates image sharpness, runs YuNet face detection, extracts the SFace feature vector, zero-pads it to 512 dimensions, saves the image locally, logs the metadata, and indexes the vector in Pinecone.
   - A green success alert appears and an SVG face HUD bounding box highlights the face with keypoints.

---

### Path B: Detecting / Recognizing a Person

1. **Select the Detect Tab**:
   - Click the **"Detect"** tab in the navigation bar.
2. **Capture Photo to Check**:
   - Click **"Open Camera"**, align face in view, and click **"Capture Photo"**.
3. **Run Recognition**:
   - Click **"Detect Person"**.
4. **Review Match Result**:
   - **Match Verified (Green HUD + Banner)**:
     - High-contrast green bounding box overlays the face with landmark dots.
     - Displays: `Match verified: Alex Morgan · 78.4%`
     - Displays the reference photo saved during registration side-by-side with your test photo.
   - **Unknown / Rejected (Red HUD + Banner)**:
     - Displays a red bounding box and reports: `Unknown person. Similarity below threshold (0.363).`

---

### Path C: Viewing & Managing Registered Users

1. **Select the Registered Users Tab**:
   - Click the **"Registered Users"** tab in the top navigation bar.
2. **Review Enrolled Identities**:
   - Displays profile cards for all people stored in `registered_users.json` and Pinecone, including their photo, display name, and registration timestamp.
3. **Search & Remove**:
   - Type in the search box to filter enrolled individuals.
   - Click **"✕ Remove"** to deregister any person (deletes the photo from disk, removes the record from `registered_users.json`, and deletes the vector from Pinecone).

---

### Path D: Viewing Detection History Log

1. **Select the Detection History Tab**:
   - Click the **"Detection History"** tab in the top navigation bar.
2. **Review Real-Time Audit Records**:
   - The interactive table displays every recognition attempt with timestamp, status badge (✓ MATCHED or ✕ REJECTED), person avatar and name, and visual confidence bar.
3. **Filter and Search**:
   - Filter by status chips (`All`, `Matched`, `Rejected`) or search by person name.
   - Click **"Refresh"** to pull the latest entries or **"Clear Log"** to reset the audit history.

---

## 6. Photo Guidelines for Reliable Recognition

| Requirement | Best Practice | What to Avoid |
|---|---|---|
| **Lighting** | Even, natural or warm frontal light | Backlighting, window glare, or harsh shadows across one side |
| **Angle & Pose** | Face directly facing camera ($< 15^{\circ}$ tilt) | Extreme side profiles, looking up, or looking down |
| **Expression** | Relaxed, neutral or slight natural smile | Exaggerated facial contortions |
| **Distance** | Face should fill 30%–70% of the camera frame | Standing too far away ($< 40 \text{px}$ face box) |
| **Accessories** | Clear visibility of eyes and nose | Dark sunglasses or masks covering the face |
| **People in Frame** | Exactly **one** person in the frame | Multiple people in the background |

---

## 7. Troubleshooting & Error Solutions

| Error Message / Issue | Root Cause | Solution |
|---|---|---|
| `"Image is too blurry. Please upload a clearer photo."` | Camera moved during capture or lens is out of focus. | Hold steady, clean the camera lens, and ensure adequate room light before retaking. |
| `"No face detected in the uploaded image."` | Face is too dark, face is turned away, or camera is too far. | Move closer to the camera and face the lens directly with proper lighting. |
| `"Multiple faces (N) detected."` | Another person or a face in a poster/photo is in the background. | Ensure only the target individual is visible in the frame. |
| `"Detected face is too small."` | The person is standing too far from the camera lens. | Move closer until your face fits comfortably inside the guide box. |
| `"Could not access camera. Please allow camera permission..."` | Browser blocked webcam access. | Click the camera icon in your browser address bar and select "Always allow". |
| `"PINECONE_API_KEY is not set."` | Missing key in `backend/.env`. | Open `backend/.env`, add your valid API key, and restart the backend server. |

---

## 8. Summary of Useful Links & Files

- [backend/main.py](file:///D:/face_recognition_POC/backend/main.py) — FastAPI endpoints and server initialization.
- [backend/face_service.py](file:///D:/face_recognition_POC/backend/face_service.py) — OpenCV decoding, blur validation, YuNet detection, and SFace int8 feature extraction.
- [backend/pinecone_service.py](file:///D:/face_recognition_POC/backend/pinecone_service.py) — Pinecone vector database connection, upsert, and query handlers.
- [backend/user_service.py](file:///D:/face_recognition_POC/backend/user_service.py) — Registered user JSON management.
- [backend/history_service.py](file:///D:/face_recognition_POC/backend/history_service.py) — Detection history JSON management.
- [frontend/src/App.jsx](file:///D:/face_recognition_POC/frontend/src/App.jsx) — Frontend user interface entrypoint.
- [architecture.md](file:///D:/face_recognition_POC/architecture.md) — Comprehensive technical architecture and data flow document.
