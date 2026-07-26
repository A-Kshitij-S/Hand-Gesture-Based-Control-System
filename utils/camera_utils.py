"""
camera_utils.py — Shared camera manager for all SignBridge pages.

All pages use the SAME VideoCapture object (stored in st.session_state["_global_cap"]).
This prevents the "camera already in use" conflict when navigating between pages
that each tried to open their own capture independently.
"""

import cv2

_global_cap = None

def open_camera(cam_index=0) -> cv2.VideoCapture:
    """
    Return the shared VideoCapture, creating it if necessary.
    Uses robust fallback logic as requested by user.
    """
    global _global_cap
    if _global_cap is not None and _global_cap.isOpened():
        return _global_cap

    if _global_cap is not None:
        _global_cap.release()

    cap = None
    try:
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    except Exception:
        cap = cv2.VideoCapture(cam_index)

    if not cap or not cap.isOpened():
        cap = cv2.VideoCapture(cam_index)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS,          30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        
    _global_cap = cap
    return cap


def release_camera() -> None:
    """Release the shared camera and clear the reference."""
    global _global_cap
    if _global_cap and _global_cap.isOpened():
        _global_cap.release()
    _global_cap = None


def read_frame():
    """
    Read one fresh frame from the shared camera.
    Flushes 2 buffered frames first to get the latest image.
    Returns (ret: bool, frame: np.ndarray | None).
    """
    cap = open_camera()
    if not cap.isOpened():
        return False, None
    # Discard buffered frames so we always get the freshest one
    cap.grab()
    cap.grab()
    ret, frame = cap.read()
    return ret, frame
