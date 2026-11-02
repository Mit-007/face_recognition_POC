# User Guide — Face Recognition (YuNet + LBP + HOG)

Welcome to the **Face Recognition System** (Proof of Concept). This application finds faces with a small **YuNet** detector, then uses hand-crafted **LBP + HOG** features to register individuals, measure facial feature distances, and return the most similar registered person in real time—with **no embeddings, no large models, and no cloud vector databases**.

---

## 1. System Overview

- **Register New Person**: Captures a live face photo, locates the face and its landmarks with YuNet, checks photo quality, normalizes lighting using CLAHE, and stores the portrait locally in `database_images/` with cached LBP and HOG descriptors.
- **Detect Mode**: Captures a face from the camera, extracts texture (LBP) and edge (HOG) descriptors, compares them against all registered persons using a distance metric ($0.6 \times \text{LBP} + 0.4 \times \text{HOG}$), and returns **Verified**, **Please retake**, or **Unknown person**.
- **Registered Users Catalog**: Dedicated tab displaying all enrolled identities with instant search and one-click removal.
- **Detection History Log**: Real-time audit log tracking every recognition attempt, closest distances, and statuses.

```
┌─────────────────┐       ┌────────────────────────┐       ┌─────────────────┐
│   Web Browser   │ ────► │    FastAPI Backend     │ ────► │  Local Storage  │
│ (React + Camera)│ ◄──── │ (Classical LBP + HOG)  │ ◄──── │(Images + JSONs) │
└─────────────────┘       └────────────────────────┘       └─────────────────┘
```

---

## 2. Prerequisites & Hardware Requirements

### Minimum Hardware Configuration
- **CPU**: Any basic dual-core processor (Intel Core i3 / AMD / ARM / Raspberry Pi).
- **RAM**: The backend uses about **100 MB** on Windows with photos up to 640 px, plus what the operating system needs. Much larger photos sent straight to the API can use far more.
- **Camera**: Standard USB webcam or built-in camera.
- **Network**: **None required** (Runs 100% offline; no cloud API calls).

### Software Prerequisites
1. **Operating System**: Windows 10/11, macOS, or Linux.
2. **Node.js**: Version 18.0 or higher ([Download Node.js](https://nodejs.org/)).
3. **Python**: Version 3.10, 3.11, or 3.12 ([Download Python](https://python.org/)).

---

## 3. How to Launch the Application

Keep two terminal windows open:

### Terminal 1: Start Backend API Server
```powershell
cd D:\face_recognition_POC\backend
.\venv\Scripts\Activate.ps1
python download_models.py   # First time only
uvicorn main:app --reload --port 8000
```
- To use the least memory outside development, run `uvicorn main:app --port 8000 --http h11 --ws none --loop asyncio` instead.
- Backend API: `http://localhost:8000`
- Interactive API Documentation (Swagger): `http://localhost:8000/docs`

### Terminal 2: Start Frontend Web App
```powershell
cd D:\face_recognition_POC\frontend
npm run dev
```
- Open `http://localhost:5173` in your web browser.

---

## 4. Step-by-Step Usage Path

### Path A: Registering a New Person
1. Click the **"Register"** tab in the top navigation bar.
2. Enter the person's **First Name** and **Last Name**.
3. Click **"Open Camera"**, fit your face in the outline with your eyes on the two marks, and click **"Capture Photo"**.
4. Click **"Register Person"**.
5. The system checks the photo, aligns the face using its landmarks, caches the LBP/HOG descriptors, and saves the portrait locally. A green success alert appears. If the photo fails a check, a message explains how to fix it, such as moving closer or facing the light.

---

### Path B: Detecting a Person
1. Click the **"Detect"** tab in the top navigation bar.
2. Click **"Open Camera"**, fit your face in the outline with your eyes on the two marks, and click **"Capture Photo"**. If the photo fails a check, retake it using the message shown.
3. Click **"Detect Person"**.
4. The system compares the captured photo against all enrolled portraits:
   - **Verified (Green Card)**: The closest person is within the distance limit and clearly ahead of the second-closest. Shows their name and reference portrait.
   - **Please Retake (Red Card)**: The closest person is within the limit, but the second-closest is too near to be sure.
   - **Unknown Person (Red Card)**: The closest person is above the distance limit.
   - Every result shows the closest and second-closest distances, where lower means more alike. These are not percentages or probabilities.
   - Every result also lists all registered people, closest first, with their distance, LBP, HOG, and whether they are within the limit. This is for score analysis, and it shows registered people's names and photos to whoever is scanning.
5. To tune the result, change `MATCH_DISTANCE_CUTOFF` and `MATCH_CLEAR_WINNER_MARGIN` in `backend/.env` and restart the backend.

---

### Path C: Managing Registered Users & Audit Logs
- **Registered Users Tab**: Search, browse enrolled portraits, or click **"✕ Remove"** to deregister any identity.
- **Detection History Tab**: View a complete chronological log of all recognition scans with status badges, closest distances, and reference avatars.
