import os
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODELS = {
    "yunet": {
        "filename": "face_detection_yunet_2023mar.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    },
    "sface_int8": {
        "filename": "face_recognition_sface_2021dec_int8.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec_int8.onnx",
    },
    "sface_fp32": {
        "filename": "face_recognition_sface_2021dec.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    },
}


def download_file(url: str, dest_path: str):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"[OK] {os.path.basename(dest_path)} already exists ({os.path.getsize(dest_path):,} bytes)")
        return True

    print(f"[Downloading] {url} -> {dest_path}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, "wb") as out:
            data = resp.read()
            out.write(data)
        print(f"[Finished] Saved {os.path.basename(dest_path)} ({os.path.getsize(dest_path):,} bytes)")
        return True
    except Exception as e:
        print(f"[Error] Failed to download {url}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def main():
    print(f"Target directory: {MODELS_DIR}")
    for key in ["yunet", "sface_int8"]:
        item = MODELS[key]
        dest = os.path.join(MODELS_DIR, item["filename"])
        success = download_file(item["url"], dest)
        if not success and key == "sface_int8":
            # Fallback to fp32 if int8 link fails
            fp32 = MODELS["sface_fp32"]
            dest_fp32 = os.path.join(MODELS_DIR, fp32["filename"])
            download_file(fp32["url"], dest_fp32)


if __name__ == "__main__":
    main()
