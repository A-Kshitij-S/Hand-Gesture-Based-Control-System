"""
word_gesture_utils.py
─────────────────────
Matches LIVE hand gestures against reference images from gesture_to_word_dataset/
using MediaPipe landmark extraction + cosine similarity, exactly like Gesture Auth.

Classes (auto-detected from filenames): Hello, IloveYou, No, Please, Thanks, Yes
"""

import os
import cv2
import numpy as np
import mediapipe as mp

# ── MediaPipe setup ───────────────────────────────────────────────────────────
_mp_hands      = mp.solutions.hands
_static_hands  = None   # for extracting from reference images
_live_hands    = None   # for live camera frames


def _get_static_detector():
    global _static_hands
    if _static_hands is None:
        _static_hands = _mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.3,
        )
    return _static_hands


def _get_live_detector():
    global _live_hands
    if _live_hands is None:
        _live_hands = _mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _live_hands


# ── Landmark extraction ───────────────────────────────────────────────────────
def _extract_landmark_vector(image_bgr, detector) -> np.ndarray | None:
    """
    Run MediaPipe on a BGR image, return wrist-relative scale-normalised
    landmark vector (63,) or None if no hand found.
    """
    rgb    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = detector.process(rgb)

    if not result.multi_hand_landmarks:
        return None

    lm_list = result.multi_hand_landmarks[0].landmark
    raw     = np.array([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float32)

    # Wrist-relative + scale-normalised (same as train_model.py)
    wrist    = raw[0]
    relative = raw - wrist
    scale    = np.max(np.linalg.norm(relative, axis=1)) + 1e-8
    relative /= scale

    return relative.flatten()  # (63,)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── Reference dataset ─────────────────────────────────────────────────────────
# { "Hello": np.ndarray(63,), "Please": np.ndarray(63,), … }
_REFERENCES: dict[str, np.ndarray] = {}
_DATASET_LOADED = False


def _parse_word_from_filename(fname: str) -> str:
    """
    'Hello.8b2540a6-a6d1-11ec-a828-84a93ea18ae6 - Copy.jpg'  →  'Hello'
    'IloveYou.04c7764c-...jpg'                                →  'IloveYou'
    """
    base = os.path.splitext(fname)[0]      # remove .jpg
    # Everything before the first dot (UUID separator)
    word = base.split(".")[0]
    return word


def load_word_gestures(dataset_dir: str) -> dict[str, np.ndarray]:
    """
    Load and extract landmarks from all reference images in dataset_dir.
    Returns dict {word_label: landmark_vector}.
    Skips any image where MediaPipe finds no hand.
    """
    global _REFERENCES, _DATASET_LOADED
    if _DATASET_LOADED:
        return _REFERENCES

    detector = _get_static_detector()
    _REFERENCES = {}

    if not os.path.isdir(dataset_dir):
        _DATASET_LOADED = True
        return _REFERENCES

    for fname in os.listdir(dataset_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        word = _parse_word_from_filename(fname)
        img  = cv2.imread(os.path.join(dataset_dir, fname))
        if img is None:
            continue
        lm = _extract_landmark_vector(img, detector)
        if lm is not None:
            _REFERENCES[word] = lm

    _DATASET_LOADED = True
    return _REFERENCES


# ── Live matching ─────────────────────────────────────────────────────────────
_LM_BUFFER: list[np.ndarray] = []
_LM_BUFFER_SIZE = 5            # smooth over 5 frames like the letter model

MATCH_THRESHOLD = 0.82         # cosine similarity threshold (0–1)


def match_live_frame(frame_bgr: np.ndarray) -> tuple[str | None, float, np.ndarray]:
    """
    Given a live BGR camera frame:
      1. Extract hand landmarks.
      2. Smooth over last N frames.
      3. Compare against all reference gestures via cosine similarity.
      4. Return (best_word, similarity_score, annotated_frame).

    Returns (None, 0.0, annotated_frame) if no hand or no match above threshold.
    """
    global _LM_BUFFER

    mp_drawing = mp.solutions.drawing_utils
    mp_styles  = mp.solutions.drawing_styles
    detector   = _get_live_detector()
    annotated  = frame_bgr.copy()

    # Detect
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = detector.process(rgb)

    if not result.multi_hand_landmarks:
        _LM_BUFFER = []
        return None, 0.0, annotated

    lm_list = result.multi_hand_landmarks[0]

    # Draw colored skeleton (same style as model_utils)
    h, w = annotated.shape[:2]
    connections = _mp_hands.HAND_CONNECTIONS
    for conn in connections:
        s, e = conn[0], conn[1]
        color = (255, 255, 255)
        if s in [1,2,3,4]     or e in [1,2,3,4]:     color = (0,   0,   255)
        elif s in [5,6,7,8]   or e in [5,6,7,8]:     color = (0,   255, 0)
        elif s in [9,10,11,12] or e in [9,10,11,12]: color = (0,   255, 255)
        elif s in [13,14,15,16] or e in [13,14,15,16]: color = (255, 0, 0)
        elif s in [17,18,19,20] or e in [17,18,19,20]: color = (255, 0, 255)
        p1 = (int(lm_list.landmark[s].x * w), int(lm_list.landmark[s].y * h))
        p2 = (int(lm_list.landmark[e].x * w), int(lm_list.landmark[e].y * h))
        cv2.line(annotated, p1, p2, color, 4)
    for lm in lm_list.landmark:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(annotated, (cx, cy), 6, (255, 255, 255), -1)
        cv2.circle(annotated, (cx, cy), 3, (40, 40, 40), -1)

    # Extract features
    raw   = np.array([[lm.x, lm.y, lm.z] for lm in lm_list.landmark], dtype=np.float32)
    wrist = raw[0]
    rel   = raw - wrist
    scale = np.max(np.linalg.norm(rel, axis=1)) + 1e-8
    rel  /= scale
    lm_vec = rel.flatten()

    # Smooth
    _LM_BUFFER.append(lm_vec)
    if len(_LM_BUFFER) > _LM_BUFFER_SIZE:
        _LM_BUFFER.pop(0)
    smooth = np.mean(_LM_BUFFER, axis=0)

    # Match against references
    if not _REFERENCES:
        return None, 0.0, annotated

    best_word  = None
    best_score = 0.0
    for word, ref_vec in _REFERENCES.items():
        score = _cosine_similarity(smooth, ref_vec)
        if score > best_score:
            best_score = score
            best_word  = word

    if best_score < MATCH_THRESHOLD:
        return None, best_score, annotated

    return best_word, best_score, annotated


def get_loaded_words() -> list[str]:
    """Return list of word labels that were successfully loaded."""
    return list(_REFERENCES.keys())
