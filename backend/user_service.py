import json
import os
from datetime import datetime

USERS_FILE = os.path.join(os.path.dirname(__file__), "registered_users.json")
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "database_images")

_users: list[dict] = []


def _load_users():
    global _users
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                _users = json.load(f)
        except Exception:
            _users = []
    else:
        _users = []


_load_users()


def _save_users():
    try:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(_users, f, indent=2)
    except Exception:
        pass


def get_registered_users() -> list[dict]:
    return _users


def add_registered_user(
    person_id: str,
    first_name: str,
    last_name: str,
    name: str,
    image_name: str,
) -> dict:
    record = {
        "person_id": person_id,
        "first_name": first_name,
        "last_name": last_name,
        "name": name,
        "image_name": image_name,
        "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _users.insert(0, record)  # Newest first
    _save_users()
    return record


def delete_registered_user(person_id: str) -> bool:
    global _users
    target = None
    for u in _users:
        if u["person_id"] == person_id:
            target = u
            break

    if target:
        _users = [u for u in _users if u["person_id"] != person_id]
        _save_users()

        # Delete image from disk if exists
        image_name = target.get("image_name")
        if image_name:
            img_path = os.path.join(IMAGES_DIR, image_name)
            if os.path.exists(img_path):
                try:
                    os.remove(img_path)
                except Exception:
                    pass
        return True
    return False
