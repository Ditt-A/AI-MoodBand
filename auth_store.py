import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from dotenv import load_dotenv
from werkzeug.security import check_password_hash, generate_password_hash


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, "API.env")
DATA_DIR = os.path.join(BASE_DIR, "data")
USERS_PATH = os.path.join(DATA_DIR, "users.json")

load_dotenv(ENV_PATH)


class AuthError(ValueError):
    """User-facing authentication validation error."""


def _now() -> str:
    return datetime.now(timezone(timedelta(hours=7))).isoformat()


def normalize_email(email: str) -> str:
    return "".join((email or "").strip().lower().split())


def _read_users() -> dict:
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(USERS_PATH):
        return {"users": []}
    with open(USERS_PATH, "r", encoding="utf-8") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            return {"users": []}
    if not isinstance(data, dict) or not isinstance(data.get("users"), list):
        return {"users": []}
    return data


def _write_users(data: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp_path = USERS_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    os.replace(tmp_path, USERS_PATH)


def _public_user(user: dict) -> dict:
    name = user.get("name") or user.get("email") or "User"
    initial = next((char.upper() for char in name if char.isalnum()), "U")
    return {
        "id": user["id"],
        "name": name,
        "email": user.get("email", ""),
        "initial": initial,
        "created_at": user.get("created_at", ""),
    }


def get_user(user_id: str | None) -> Optional[dict]:
    if not user_id:
        return None
    data = _read_users()
    for user in data["users"]:
        if user.get("id") == user_id:
            return _public_user(user)
    return None


def get_user_by_email(email: str) -> Optional[dict]:
    normalized = normalize_email(email)
    if not normalized:
        return None
    data = _read_users()
    for user in data["users"]:
        if user.get("email") == normalized:
            return user
    return None


def create_user(name: str, email: str, password: str) -> dict:
    clean_name = " ".join((name or "").strip().split())
    normalized_email = normalize_email(email)
    if len(clean_name) < 2:
        raise AuthError("Nama minimal 2 karakter.")
    if "@" not in normalized_email or "." not in normalized_email.rsplit("@", 1)[-1]:
        raise AuthError("Masukkan email yang valid.")
    if len(password or "") < 8:
        raise AuthError("Password minimal 8 karakter.")

    data = _read_users()
    if any(user.get("email") == normalized_email for user in data["users"]):
        raise AuthError("Email ini sudah terdaftar.")

    user = {
        "id": str(uuid.uuid4()),
        "name": clean_name,
        "email": normalized_email,
        "password_hash": generate_password_hash(password),
        "created_at": _now(),
        "last_login_at": None,
    }
    data["users"].append(user)
    _write_users(data)
    return _public_user(user)


def authenticate(email: str, password: str) -> Optional[dict]:
    data = _read_users()
    normalized_email = normalize_email(email)
    for user in data["users"]:
        if user.get("email") != normalized_email:
            continue
        if not check_password_hash(user.get("password_hash", ""), password or ""):
            return None
        user["last_login_at"] = _now()
        _write_users(data)
        return _public_user(user)
    return None
