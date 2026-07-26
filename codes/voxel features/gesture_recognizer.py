"""
gesture_recognizer.py – Detects gestures from hand landmark states with cooldowns.
"""
import time

class GestureRecognizer:
    """Maps finger states + distances to named gestures with cooldown logic."""

    GESTURES = [
        "PINCH", "DRAG", "OPEN_PALM", "ROTATE", "SWIPE",
        "FIST", "PEACE", "THUMB_UP", "THUMB_DOWN", "NONE"
    ]
    COOLDOWN = 0.30  # seconds between gesture changes (snappier switching)

    def __init__(self):
        self.current = "NONE"
        self.confidence = 0.0
        self._last_change = 0
        self._prev_index_x = None
        self._swipe_threshold = 0.22  # raised: must do a big deliberate swipe to erase

    def recognize(self, landmarks, finger_states, pinch_dist, hand_count, landmarks2=None):
        """Detect gesture from hand data. Returns (gesture_name, confidence)."""
        if landmarks is None:
            self.current = "NONE"
            self.confidence = 0.0
            return self.current, self.confidence

        now = time.time()
        thumb, index, middle, ring, pinky = finger_states
        gesture = "NONE"
        conf = 0.0

        # --- Two-hand rotate: only if BOTH hands are open (fingers extended) ---
        # This prevents accidental rotate when second hand drifts into frame
        if hand_count >= 2 and landmarks2 is not None:
            # Require at least 3 fingers on each hand to be extended
            states2 = [landmarks2[t][1] < landmarks2[p][1]
                       for t, p in zip(self.FINGER_TIPS[1:], self.FINGER_PIPS[1:])]
            if sum(finger_states) >= 3 and sum(states2) >= 3:
                gesture = "ROTATE"
                conf = 0.9

        # --- Fist: all fingers closed ---
        elif not any(finger_states):
            gesture = "FIST"
            conf = 0.95

        # --- Pinch: thumb+index close, others don't matter ---
        elif pinch_dist < 0.055:  # slightly looser so pinch is easier to hold
            gesture = "PINCH"
            conf = min(1.0, 1.0 - pinch_dist / 0.055)

        # --- Drag: pinch held (same as pinch but continuous) ---
        elif pinch_dist < 0.075:  # slightly looser
            gesture = "DRAG"
            conf = 0.85

        # --- Open palm: all 5 fingers extended ---
        elif all(finger_states):
            gesture = "OPEN_PALM"
            conf = 0.92

        # --- Peace sign: index+middle up, rest down ---
        elif index and middle and not ring and not pinky:
            gesture = "PEACE"
            conf = 0.9

        # --- Thumb up: only thumb extended ---
        elif thumb and not index and not middle and not ring and not pinky:
            gesture = "THUMB_UP"
            conf = 0.9

        # --- Thumb down: thumb pointing down (tip.y > wrist.y) ---
        elif thumb and not index and not middle and landmarks[4][1] > landmarks[0][1]:
            gesture = "THUMB_DOWN"
            conf = 0.85

        # --- Swipe detection ---
        if self._prev_index_x is not None and index:
            dx = landmarks[8][0] - self._prev_index_x
            if abs(dx) > self._swipe_threshold:
                gesture = "SWIPE"
                conf = min(1.0, abs(dx) / self._swipe_threshold)
        self._prev_index_x = landmarks[8][0] if index else None

        # Apply cooldown
        if gesture != self.current:
            if now - self._last_change > self.COOLDOWN:
                self.current = gesture
                self.confidence = conf
                self._last_change = now
        else:
            self.confidence = conf

        return self.current, self.confidence

    def get_mode_label(self, gesture):
        """Return a user-friendly mode label."""
        return {
            "PEACE":      "Place Voxel",
            "DRAG":       "Draw Voxels",
            "OPEN_PALM":  "Move Structure",
            "ROTATE":     "Rotate View",
            "SWIPE":      "Erase Voxel",
            "FIST":       "Grab & Rotate",
            "PINCH":      "Erase Voxel",
            "THUMB_UP":   "Confirm",
            "THUMB_DOWN": "Undo",
            "NONE":       "Idle",
        }.get(gesture, "Idle")
