"""
model_utils.py — Load model, predict gestures from frames.
"""

import os
import cv2
import numpy as np
import pickle
import mediapipe as mp
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(BASE_DIR, "models", "asl_model.h5")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
NORM_MEAN    = os.path.join(BASE_DIR, "models", "norm_mean.npy")
NORM_STD     = os.path.join(BASE_DIR, "models", "norm_std.npy")

# ── MediaPipe ────────────────────────────────────────────────────────────────
mp_hands     = mp.solutions.hands
mp_drawing   = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

_MODEL     = None
_ENCODER   = None
_NORM_MEAN = None
_NORM_STD  = None
_HANDS     = None   # single persistent detector for video frames

# Rolling buffer for landmark smoothing (last N frames averaged)
_LM_BUFFER     = []
_LM_BUFFER_SIZE = 5   # average over 5 frames → much more stable predictions


def _get_hands():
    global _HANDS
    if _HANDS is None:
        _HANDS = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _HANDS


def load_model():
    """Load Keras model + label encoder + normalization params (cached)."""
    global _MODEL, _ENCODER, _NORM_MEAN, _NORM_STD
    if _MODEL is not None:
        return True

    missing = [p for p in [MODEL_PATH, ENCODER_PATH, NORM_MEAN, NORM_STD] if not os.path.exists(p)]
    if missing:
        return False

    _MODEL   = tf.keras.models.load_model(MODEL_PATH)
    with open(ENCODER_PATH, 'rb') as f:
        _ENCODER = pickle.load(f)
    _NORM_MEAN = np.load(NORM_MEAN)
    _NORM_STD  = np.load(NORM_STD)
    return True


def _extract_landmarks_from_frame(frame, detector):
    """Extract 63-feature landmark array from BGR frame. Returns (landmarks, annotated_frame)."""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result  = detector.process(img_rgb)
    annotated = frame.copy()

    if not result.multi_hand_landmarks:
        return None, annotated

    lm_list = result.multi_hand_landmarks[0]
    # Draw landmarks on frame
    mp_drawing.draw_landmarks(
        annotated,
        lm_list,
        mp_hands.HAND_CONNECTIONS,
        mp_draw_styles.get_default_hand_landmarks_style(),
        mp_draw_styles.get_default_hand_connections_style(),
    )

    coords = []
    for lm in lm_list.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    raw = np.array(coords, dtype=np.float32).reshape(21, 3)

    # Wrist is landmark 0
    wrist = raw[0]
    relative = raw - wrist
    scale = np.max(np.linalg.norm(relative, axis=1)) + 1e-8
    relative /= scale

    # Custom Drawing (Colored Lines)
    # Using specific colors for different hand parts
    # Thumb: Red, Index: Green, Middle: Yellow, Ring: Blue, Pinky: Magenta
    connections = mp_hands.HAND_CONNECTIONS
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        
        # Determine color based on finger
        color = (255, 255, 255) # White default
        if start_idx in [1,2,3,4] or end_idx in [1,2,3,4]: color = (0, 0, 255) # Red (BGR)
        elif start_idx in [5,6,7,8] or end_idx in [5,6,7,8]: color = (0, 255, 0) # Green
        elif start_idx in [9,10,11,12] or end_idx in [9,10,11,12]: color = (0, 255, 255) # Yellow
        elif start_idx in [13,14,15,16] or end_idx in [13,14,15,16]: color = (255, 0, 0) # Blue
        elif start_idx in [17,18,19,20] or end_idx in [17,18,19,20]: color = (255, 0, 255) # Magenta

        h, w, _ = annotated.shape
        p1 = (int(lm_list.landmark[start_idx].x * w), int(lm_list.landmark[start_idx].y * h))
        p2 = (int(lm_list.landmark[end_idx].x * w), int(lm_list.landmark[end_idx].y * h))
        cv2.line(annotated, p1, p2, color, 5)  # thicker for vivid visibility

    # Draw landmarks (dots) — larger for visibility
    for lm in lm_list.landmark:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(annotated, (cx, cy), 7, (255, 255, 255), -1)
        cv2.circle(annotated, (cx, cy), 4, (50, 50, 50), -1)  # dark center for contrast

    return relative.flatten(), annotated


def predict_gesture(frame):
    """
    Predict ASL gesture from a BGR webcam frame.
    Uses wrist-relative + scale-normalized landmarks matched to training.
    """
    global _LM_BUFFER

    if not load_model():
        return None, 0.0, frame, None, None

    detector = _get_hands()
    landmarks, annotated = _extract_landmarks_from_frame(frame, detector)

    if landmarks is None:
        _LM_BUFFER = []
        return None, 0.0, annotated, None, None

    _LM_BUFFER.append(landmarks)
    if len(_LM_BUFFER) > _LM_BUFFER_SIZE:
        _LM_BUFFER.pop(0)

    smooth_lm = np.mean(_LM_BUFFER, axis=0)
    norm_lm = (smooth_lm - _NORM_MEAN) / _NORM_STD
    pred    = _MODEL(norm_lm[np.newaxis, :], training=False).numpy()[0]
    idx     = np.argmax(pred)
    label   = str(_ENCODER.classes_[idx])
    conf    = float(pred[idx])

    # Safety guard: filter out digit predictions (0-9) — only letters A-Z allowed
    if not label.isalpha():
        label = None
        conf  = 0.0

    # Return pred array as 5th element so callers can do top-k without a 2nd inference
    return label, conf, annotated, smooth_lm, pred


def top_k_from_pred(pred, k=3):
    """
    Extract top-k letter predictions from an existing softmax pred array.
    Call this after predict_gesture() instead of predict_top_k() to avoid
    a second model.predict() call (which was causing camera lag).
    Returns list of (label, confidence) tuples, letters only, sorted desc.
    """
    if pred is None or _ENCODER is None:
        return []
    top_k_idx = np.argsort(pred)[::-1]
    results = []
    for i in top_k_idx:
        lbl = str(_ENCODER.classes_[i])
        if lbl.isalpha():
            results.append((lbl, float(pred[i])))
        if len(results) >= k:
            break
    return results


# Keep predict_top_k as a thin wrapper for backward compatibility
def predict_top_k(frame, k=3):
    """Legacy wrapper — prefer using top_k_from_pred(pred, k) for zero extra cost."""
    result = predict_gesture(frame)
    if result[0] is None and result[4] is None:
        return []
    return top_k_from_pred(result[4], k)


def get_raw_landmarks(frame):
    """Extract raw landmarks for auth module. Returns (landmarks, annotated_frame)."""
    detector = _get_hands()
    return _extract_landmarks_from_frame(frame, detector)


# ── Separate cached detector for gesture mouse (avoids state conflicts) ───────
_HANDS_MOUSE = None

def _get_hands_mouse():
    global _HANDS_MOUSE
    if _HANDS_MOUSE is None:
        _HANDS_MOUSE = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _HANDS_MOUSE


def get_absolute_landmarks(frame):
    """
    Extract RAW (non-wrist-relative) 21 hand landmarks as a (21, 3) numpy array
    with values in the 0–1 normalized range, plus an annotated frame.
    Used by the Gesture Mouse module.

    Returns:
        (pts_21x3: np.ndarray | None, annotated_frame: np.ndarray)
    """
    img_rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detector  = _get_hands_mouse()
    result    = detector.process(img_rgb)
    annotated = frame.copy()

    if not result.multi_hand_landmarks:
        return None, annotated

    lm_list = result.multi_hand_landmarks[0]

    # Draw coloured hand skeleton
    connections = mp_hands.HAND_CONNECTIONS
    h, w, _    = annotated.shape
    for connection in connections:
        s, e = connection[0], connection[1]
        color = (255, 255, 255)
        if s in [1,2,3,4]   or e in [1,2,3,4]:   color = (0,  0,   255)
        elif s in [5,6,7,8]  or e in [5,6,7,8]:   color = (0,  255, 0)
        elif s in [9,10,11,12] or e in [9,10,11,12]: color = (0,  255, 255)
        elif s in [13,14,15,16] or e in [13,14,15,16]: color = (255, 0, 0)
        elif s in [17,18,19,20] or e in [17,18,19,20]: color = (255, 0, 255)
        p1 = (int(lm_list.landmark[s].x * w), int(lm_list.landmark[s].y * h))
        p2 = (int(lm_list.landmark[e].x * w), int(lm_list.landmark[e].y * h))
        cv2.line(annotated, p1, p2, color, 5)
    for lm in lm_list.landmark:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(annotated, (cx, cy), 7, (255, 255, 255), -1)
        cv2.circle(annotated, (cx, cy), 4, (30, 30, 30), -1)

    pts = np.array([[lm.x, lm.y, lm.z] for lm in lm_list.landmark], dtype=np.float32)
    return pts, annotated


def is_model_ready() -> bool:
    """Check if model files exist without loading."""
    return all(os.path.exists(p) for p in [MODEL_PATH, ENCODER_PATH, NORM_MEAN, NORM_STD])
