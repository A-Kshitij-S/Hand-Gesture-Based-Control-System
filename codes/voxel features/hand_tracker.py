"""
hand_tracker.py – MediaPipe hand tracking with smoothing and finger-state detection.
"""
import cv2
import numpy as np
import mediapipe as mp

class HandTracker:
    """Wraps MediaPipe Hands with EMA smoothing and finger-state helpers."""

    FINGER_TIPS = [4, 8, 12, 16, 20]
    FINGER_PIPS = [3, 6, 10, 14, 18]

    def __init__(self, max_hands=2, detection_conf=0.7, tracking_conf=0.6, smoothing=0.60):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self.smoothing = smoothing
        self._prev = {}  # hand_idx -> smoothed landmarks

    def process(self, frame_bgr):
        """Return (results, annotated_frame). results.multi_hand_landmarks may be None."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        return results

    def draw_landmarks(self, frame, results):
        """Draw hand skeletons with cyan neon style."""
        if not results.multi_hand_landmarks:
            return frame
        for hand_lm in results.multi_hand_landmarks:
            # Draw connections
            self.mp_draw.draw_landmarks(
                frame, hand_lm, self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(0, 200, 200), thickness=2),
            )
            # Bright fingertip circles
            h, w, _ = frame.shape
            for tip_id in self.FINGER_TIPS:
                lm = hand_lm.landmark[tip_id]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 6, (0, 255, 200), -1)
                cv2.circle(frame, (cx, cy), 8, (0, 255, 255), 1)
        return frame

    def get_landmarks(self, results, hand_idx=0):
        """Return numpy array (21,3) of landmarks for given hand, or None."""
        if not results.multi_hand_landmarks:
            return None
        if hand_idx >= len(results.multi_hand_landmarks):
            return None
        hand = results.multi_hand_landmarks[hand_idx]
        raw = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])
        # EMA smoothing
        if hand_idx in self._prev:
            raw = self.smoothing * self._prev[hand_idx] + (1 - self.smoothing) * raw
        self._prev[hand_idx] = raw.copy()
        return raw

    def get_pixel_coords(self, landmarks, frame_shape):
        """Convert normalized landmarks to pixel coordinates (21,2)."""
        h, w = frame_shape[:2]
        return (landmarks[:, :2] * np.array([w, h])).astype(int)

    def finger_states(self, landmarks):
        """Return list of 5 bools [thumb, index, middle, ring, pinky] — True=extended."""
        if landmarks is None:
            return [False] * 5
        states = []
        # Thumb: compare x of tip vs ip (assuming right hand, works approx for left too)
        states.append(landmarks[4][0] < landmarks[3][0])
        # Other fingers: tip.y < pip.y
        for tip, pip in zip(self.FINGER_TIPS[1:], self.FINGER_PIPS[1:]):
            states.append(landmarks[tip][1] < landmarks[pip][1])
        return states

    def pinch_distance(self, landmarks):
        """Distance between thumb tip and index tip (normalized coords)."""
        if landmarks is None:
            return 1.0
        return float(np.linalg.norm(landmarks[4] - landmarks[8]))

    def hand_count(self, results):
        """How many hands detected."""
        if not results.multi_hand_landmarks:
            return 0
        return len(results.multi_hand_landmarks)

    def release(self):
        self.hands.close()
