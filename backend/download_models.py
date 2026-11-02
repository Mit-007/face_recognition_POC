import os
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODELS = {
    "yunet_face_detector": {
        "filename": "face_detection_yunet_2023mar.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    },
}


def download_file(url: str, dest_path: str) -> bool:
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
    for item in MODELS.values():
        dest = os.path.join(MODELS_DIR, item["filename"])
        download_file(item["url"], dest)


if __name__ == "__main__":
    main()
