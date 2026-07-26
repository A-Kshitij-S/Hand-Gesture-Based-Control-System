"""
auth_utils.py — Hand gesture biometric authentication using SQLite.

How it works:
  - Registration: capture N frames → extract MediaPipe landmarks →
    average → store in DB as normalized template
  - Login: capture frame → extract landmarks → cosine similarity
    with stored template → grant if similarity ≥ threshold
"""

import os
import sqlite3
import numpy as np
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "models", "gesture_auth.db")

SIMILARITY_THRESHOLD = 0.90   # cosine similarity required for login
MAX_STORED_SAMPLES   = 10     # samples to average per registration


# ── DB Init ───────────────────────────────────────────────────────────────────
def _get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username    TEXT PRIMARY KEY,
            template    TEXT NOT NULL,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


# ── Helpers ───────────────────────────────────────────────────────────────────
def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / (norm + 1e-8)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_normalize(a), _normalize(b)))


# ── Public API ────────────────────────────────────────────────────────────────
def user_exists(username: str) -> bool:
    with _get_conn() as conn:
        row = conn.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone()
    return row is not None


def register_user(username: str, landmark_samples: list[np.ndarray]) -> bool:
    """
    Register a new user with a list of landmark arrays.
    Averages all samples, normalizes, stores in DB.

    Returns True on success, False if user already exists.
    """
    if user_exists(username):
        return False
    avg  = np.mean(landmark_samples, axis=0)
    norm = _normalize(avg)
    template_json = json.dumps(norm.tolist())
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO users (username, template) VALUES (?, ?)",
            (username, template_json)
        )
        conn.commit()
    return True


def authenticate_user(username: str, landmark: np.ndarray) -> tuple[bool, float]:
    """
    Authenticate user by comparing landmark to stored template.

    Returns:
        (authenticated: bool, similarity_score: float)
    """
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT template FROM users WHERE username=?", (username,)
        ).fetchone()
    if not row:
        return False, 0.0

    template = np.array(json.loads(row[0]), dtype=np.float32)
    sim = _cosine_similarity(landmark, template)
    return sim >= SIMILARITY_THRESHOLD, round(sim, 4)


def delete_user(username: str) -> bool:
    """Delete a registered user. Returns True if deleted."""
    if not user_exists(username):
        return False
    with _get_conn() as conn:
        conn.execute("DELETE FROM users WHERE username=?", (username,))
        conn.commit()
    return True


def list_users() -> list[str]:
    """Return list of all registered usernames."""
    with _get_conn() as conn:
        rows = conn.execute("SELECT username FROM users ORDER BY created_at").fetchall()
    return [r[0] for r in rows]
