# User Guide — Face Recognition POC Application

Welcome to the **FaceID Recognition System** (Proof of Concept). This guide walks you through the initial setup, system prerequisites, and the exact step-by-step path to use the application for registering and recognizing people in real time.

---

## 1. System Overview

This application provides a seamless, camera-based facial recognition workflow:
- **Register New Person**: Captures a live face photo, verifies image quality, generates a 512-dimensional facial embedding vector, and stores the identity in Pinecone and local storage.
- **Detect Person**: Captures a face photo from the camera, validates quality, extracts the facial embedding, and searches Pinecone using cosine similarity to identify who the person is in real time.

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│   Web Browser   │ ────► │ FastAPI Backend │ ────► │  Pinecone DB    │
│ (React + Camera)│ ◄──── │(OpenCV + Models)│ ◄──── │ (Vector Search) │
└─────────────────┘       └─────────────────┘       └─────────────────┘
```

---

## 2. Prerequisites & Requirements

Before running the application, make sure you have:
1. **Operating System**: Windows 10/11, macOS, or Linux.
2. **Node.js**: Version 18.0 or higher ([Download Node.js](https://nodejs.org/)).
3. **Python**: Version 3.10, 3.11, or 3.12 ([Download Python](https://python.org/)).
4. **Hardware**: A working webcam or built-in camera.
5. **Pinecone API Key**: A free API key from [Pinecone.io](https://www.pinecone.io/).

---

## 3. First-Time Setup Instructions

### Step 1: Configure Environment Variables
1. Navigate to the `backend` folder:
   ```powershell
   cd D:\face_recognition_POC\backend
   ```
2. Verify or create the `.env` file (copy from `.env.example` if not present):
   ```ini
   PINECONE_API_KEY=your_actual_pinecone_api_key_here
   PINECONE_INDEX_NAME=face-embeddings-poc
   PINECONE_CLOUD=aws
   PINECONE_REGION=us-east-1
   ```
   > [!IMPORTANT]
   > Replace `your_actual_pinecone_api_key_here` with your Pinecone API key. The backend automatically creates the `face-embeddings-poc` index with 512 dimensions and cosine metric if it does not already exist.

---

### Step 2: Set Up Backend Virtual Environment
1. In the `backend` folder, check if the virtual environment exists. If not, create it:
   ```powershell
   python -m venv venv
   ```
2. Activate the virtual environment:
   - **Windows PowerShell**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Windows Command Prompt**:
     ```cmd
     venv\Scripts\activate.bat
     ```
   - **Linux / macOS**:
     ```bash
     source venv/bin/activate
     ```
3. Install required Python packages:
   ```powershell
   pip install -r requirements.txt
   ```
   *(On first run, InsightFace will automatically download the `buffalo_l` model pack to `~/.insightface/models/`)*.

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

To run the complete system, keep two terminal windows open:

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
- Open `http://localhost:5173` in your browser (Google Chrome, Microsoft Edge, or Firefox).

---

## 5. Step-by-Step Usage Path

Follow this path to test and operate the application:

```
[Start App] ──► [Grant Camera Permission] ──► [Path A: Register Person] ──► [Path B: Detect Person]
```

### Path A: Registering a New Person

1. **Select the Register Tab**:
   - In your browser (`http://localhost:5173`), ensure the **"Register"** tab is active in the top navigation bar.
2. **Enter Identity Details (Left Column)**:
   - In the **First Name** and **Last Name** input boxes, enter the person's identity credentials (e.g. `Alex` and `Morgan`).
3. **Capture Face Photo (Right Column)**:
   - On the right side, click the blue **"Open Camera"** button.
   - When prompted by the browser, click **"Allow"** to grant camera permissions.
   - Position your face inside the white guideline box and click **"Capture Photo"**.
4. **Submit Registration**:
   - Review your credentials and preview, then click **"Register Person"**.
   - The system validates image sharpness with OpenCV, detects the face, extracts the 512-d feature vector, saves the image locally, and records the identity into Pinecone.
   - You will see a green success alert: `Registered "Alex Morgan" successfully into Pinecone.`

---

### Path B: Detecting / Recognizing a Person

1. **Select the Detect Tab**:
   - Click the **"Detect"** tab in the top navigation bar.
2. **Open the Camera**:
   - Click **"Open Camera"**.
3. **Capture Face to Check**:
   - Align the person's face inside the camera guide outline.
   - Click **"Capture Photo"**.
4. **Run Recognition**:
   - Click **"Detect Person"**.
5. **Review Match Result**:
   - **Match Found (Green Banner)**:
     - Displays: `Match found: Alex Morgan (score: 0.842)`
     - Displays the reference photo saved during registration side-by-side with your test photo.
   - **No Match Found (Red Banner)**:
     - If the person has not been registered in Pinecone, the system reports: `No person found in database.`

---

### Path C: Viewing & Managing Registered Users

1. **Select the Registered Users Tab**:
   - Click the **"Registered Users"** tab in the top navigation bar.
2. **Review Enrolled Identities**:
   - The dedicated gallery shows profile cards for all people stored in `registered_users.json` and Pinecone, including their photo, display name, and registration timestamp.
3. **Search & Remove**:
   - Type in the search box to find a person by name.
   - Click **"✕ Remove"** to deregister any person (deletes the portrait from disk, removes the entry from `registered_users.json`, and deletes their vector from Pinecone).

---

### Path D: Viewing Detection History Log

1. **Select the Detection History Tab**:
   - Click the **"Detection History"** tab in the top navigation bar (or click "View Full Detection History Table" after any scan in Detect mode).
2. **Review Real-Time Audit Records**:
   - The interactive table displays every recognition attempt with its exact timestamp, match status (✓ VERIFIED or ✕ REJECTED), person details (photo avatar and display name), confidence score bar, and execution latency (`⚡ ms`).
3. **Filter and Search**:
   - Filter by status chips (`All`, `Matched`, `Rejected`) or search by person name.
   - Click **"Refresh"** to pull the latest entries or **"Clear Log"** to reset the audit history.

---

## 6. Photo Guidelines for Reliable Recognition

For maximum accuracy and to prevent validation errors:

| Requirement | Best Practice | What to Avoid |
|---|---|---|
| **Lighting** | Even, natural or warm frontal light | Backlighting, window glare, or harsh shadows across one side |
| **Angle & Pose** | Face directly facing camera ($< 15^{\circ}$ tilt) | Extreme side profiles, looking up, or looking down |
| **Expression** | Relaxed, neutral or slight natural smile | Exaggerated facial contortions |
| **Distance** | Face should fill 30%–70% of the camera frame | Standing too far away ($< 40 \text{px}$ face box) |
| **Accessories** | Clear visibility of eyes and nose | Dark sunglasses, COVID masks, or hair covering eyes |
| **People in Frame** | Exactly **one** person in the frame | Multiple people in the background |

---

## 7. Troubleshooting & Error Solutions

| Error Message / Issue | Root Cause | Solution |
|---|---|---|
| `"Image is too blurry. Please upload a clearer photo."` | Camera moved during capture or lens is out of focus. | Hold steady, clean the camera lens, and ensure adequate room light before retaking. |
| `"No face detected in the uploaded image."` | Face is too dark, face is turned away, or camera is too far. | Move closer to the camera and face the lens directly with proper lighting. |
| `"Multiple faces (N) detected."` | Another person or a face in a poster/photo is in the background. | Ensure only the target individual is visible in the frame. |
| `"Detected face is too small."` | The person is standing too far from the camera lens. | Move closer until your face fits comfortably inside the oval guide. |
| `"Could not access camera. Please allow camera permission..."` | Browser blocked webcam access. | Click the camera icon in your browser address bar and select "Always allow". |
| `"PINECONE_API_KEY is not set."` | Missing key in `backend/.env`. | Open `backend/.env`, add your valid API key, and restart the backend server. |
| Network / Failed to fetch error | FastAPI backend is not running. | Make sure `uvicorn main:app --reload --port 8000` is active in Terminal 1. |

---

## 8. Summary of Useful Links & Files

- [backend/main.py](file:///D:/face_recognition_POC/backend/main.py) — FastAPI endpoints and server initialization.
- [backend/face_service.py](file:///D:/face_recognition_POC/backend/face_service.py) — OpenCV decoding, blur validation, and InsightFace model inference.
- [backend/pinecone_service.py](file:///D:/face_recognition_POC/backend/pinecone_service.py) — Pinecone vector database connection and query handlers.
- [frontend/src/App.jsx](file:///D:/face_recognition_POC/frontend/src/App.jsx) — Frontend user interface entrypoint.
- [architecture.md](file:///D:/face_recognition_POC/architecture.md) — Comprehensive technical architecture and data flow document.
- [freature_openCV.md](file:///D:/face_recognition_POC/freature_openCV.md) — Complete catalog of OpenCV validation features.
