# """
# gesture_mouse_standalone.py  –  Tkinter-based Gesture Mouse
# ────────────────────────────────────────────────────────────────────────
# A standalone application providing AI-driven Mouse, Volume, and Brightness controls 
# using Mediapipe and the user's specific mathematical gesture logic.
# """

# import sys, os, time, threading
# import tkinter as tk
# from tkinter import ttk
# import cv2
# import numpy as np
# import mediapipe as mp
# from PIL import Image, ImageTk
# from collections import deque

# # Advanced Controls
# import pyautogui
# import screen_brightness_control as sbc
# from math import hypot
# from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
# from ctypes import cast, POINTER
# try:
#     from comtypes import CLSCTX_ALL
# except ImportError:
#     CLSCTX_ALL = None
# from pynput.mouse import Button, Controller

# mouse = Controller()
# pyautogui.FAILSAFE = False
# SCREEN_W, SCREEN_H = pyautogui.size()

# # ── MediaPipe ─────────────────────────────────────────────────────────────
# mp_hands    = mp.solutions.hands
# mp_drawing  = mp.solutions.drawing_utils
# drawing_spec_nodes = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
# drawing_spec_lines = mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)

# # ── Global State ──────────────────────────────────────────────────────────
# _state = {
#     "running": False,
#     "smoothing": 5,
#     "mode": "MOUSE" # Can be MOUSE, VOLUME, or BRIGHTNESS
# }
# _stop_event = threading.Event()
# _lock       = threading.Lock()

# # ── Camera helpers ────────────────────────────────────────────────────────
# def _open_cap(cam_index=0):
#     cap = None
#     try:
#         cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
#     except Exception:
#         cap = cv2.VideoCapture(cam_index)

#     if not cap or not cap.isOpened():
#         cap = cv2.VideoCapture(cam_index)

#     if cap.isOpened():
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#         cap.set(cv2.CAP_PROP_FPS,          30)
#         cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        
#     return cap

# # ── Advanced Math Helpers ─────────────────────────────────────────────────
# def get_distance(p1, p2):
#     """Euclidean distance between two (x,y) points."""
#     return hypot(p2[0] - p1[0], p2[1] - p1[1])

# def get_angle(a, b, c):
#     """Angle at point b given a, b, c."""
#     radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
#     angle = np.abs(np.degrees(radians))
#     return angle

# # ── Background worker ─────────────────────────────────────────────────────
# def _worker(stop: threading.Event):
#     cap   = _open_cap()
#     hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
#                            min_detection_confidence=0.7,
#                            min_tracking_confidence=0.7)
    
#     # Audio Setup – new pycaw API (AudioDevice.EndpointVolume)
#     volume   = None
#     minVol   = -65.25
#     maxVol   = 0.0
#     vol      = 0.0
#     volBar   = 400
#     volPer   = 0
#     try:
#         speaker  = AudioUtilities.GetSpeakers()
#         volume   = speaker.EndpointVolume          # new pycaw API – no Activate() needed
#         volRange = volume.GetVolumeRange()
#         minVol   = volRange[0]
#         maxVol   = volRange[1]
#         print(f"Volume control initialized: range [{minVol:.1f}, {maxVol:.1f}] dB")
#     except Exception as e:
#         print(f"Volume init error: {e}")
#         volume = None

#     plocX, plocY = 0.0, 0.0
#     ptx, pty     = 0, 0
#     cooldown     = 0
#     t_fps, frames = time.time(), 0
    
#     # Stability Buffer
#     gesture_buffer = deque(maxlen=8)

#     while not stop.is_set():
#         if cap is None or not cap.isOpened():
#             time.sleep(0.1)
#             continue
            
#         try:
#             cap.grab()
#             ret, frame = cap.read()
#             if not ret or frame is None:
#                 time.sleep(0.01)
#                 continue
#         except Exception:
#             time.sleep(0.01)
#             continue

#         frame  = cv2.flip(frame, 1)
#         rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         result = hands.process(rgb)
        
#         mode   = _state["mode"]
#         action = f"IDLE ({mode})"
#         sm     = _state["smoothing"]
#         alpha  = 1.0 / max(sm, 1)

#         h, w = frame.shape[:2]

#         if result.multi_hand_landmarks:
#             lm_list = result.multi_hand_landmarks[0]
#             # Convert landmarks to pixel coords
#             lm_px = [(int(pt.x * w), int(pt.y * h)) for pt in lm_list.landmark]
            
#             # Distance based pinches (Robust Industrial Standard)
#             dist_thumb_index  = get_distance(lm_px[4], lm_px[8])
#             dist_thumb_middle = get_distance(lm_px[4], lm_px[12])
            
#             # Draw professional MediaPipe landmarks
#             mp_drawing.draw_landmarks(frame, lm_list, mp_hands.HAND_CONNECTIONS,
#                                       drawing_spec_nodes, drawing_spec_lines)

#             # --- MOUSE MODE ---
#             if mode == "MOUSE":
#                 # Moving Mode (Index straight, Thumb near Index Base)
#                 # Keep distance check for move to ensure stability
#                 if get_distance(lm_px[4], lm_px[5]) < 50 and get_angle(lm_px[5], lm_px[6], lm_px[8]) > 100:
#                     action = "MOVE CURSOR"
#                     ix, iy = lm_list.landmark[8].x, lm_list.landmark[8].y
#                     clocX = (1 - alpha) * plocX + alpha * ix
#                     clocY = (1 - alpha) * plocY + alpha * iy
#                     tx = int(np.interp(clocX, [0.1, 0.9], [0, SCREEN_W]))
#                     ty = int(np.interp(clocY, [0.1, 0.9], [0, SCREEN_H]))
                    
#                     # Movement Dead-zone (reduce jitter)
#                     if abs(tx - ptx) > 4 or abs(ty - pty) > 4:
#                         try: pyautogui.moveTo(tx, ty)
#                         except Exception: pass
#                         ptx, pty = tx, ty
                    
#                     plocX, plocY = clocX, clocY
#                     gesture_buffer.append("NONE")
                
#                 elif cooldown == 0:
#                     # Robust Pinch Detection
#                     if dist_thumb_index < 35: 
#                         gesture_buffer.append("LEFT")
#                     elif dist_thumb_middle < 35: 
#                         gesture_buffer.append("RIGHT")
#                     else:
#                         gesture_buffer.append("NONE")
                    
#                     if gesture_buffer.count("LEFT") > 5:
#                         action = "LEFT CLICK ✓"
#                         mouse.press(Button.left); mouse.release(Button.left)
#                         cooldown = 15; gesture_buffer.clear()
#                     elif gesture_buffer.count("RIGHT") > 5:
#                         action = "RIGHT CLICK ✓"
#                         mouse.press(Button.right); mouse.release(Button.right)
#                         cooldown = 15; gesture_buffer.clear()
                        
#             # --- VOLUME MODE (YouTube reference approach) ---
#             elif mode == "VOLUME":
#                 x1, y1 = lm_px[4]
#                 x2, y2 = lm_px[8]
#                 cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
#                 length = get_distance(lm_px[4], lm_px[8])

#                 # Draw thumb-index connection (YouTube style)
#                 cv2.circle(frame, (x1, y1), 12, (255, 0, 255), cv2.FILLED)
#                 cv2.circle(frame, (x2, y2), 12, (255, 0, 255), cv2.FILLED)
#                 cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
#                 if length < 50:
#                     cv2.circle(frame, (cx, cy), 12, (0, 255, 0), cv2.FILLED)
#                 else:
#                     cv2.circle(frame, (cx, cy), 12, (255, 0, 255), cv2.FILLED)

#                 # Map hand range [50, 300] → volume (YouTube tutorial range)
#                 vol    = np.interp(length, [50, 300], [minVol, maxVol])
#                 volBar = int(np.interp(length, [50, 300], [400, 150]))
#                 volPer = np.interp(length, [50, 300], [0, 100])

#                 # Smooth to nearest 5% to avoid jitter
#                 smoothness = 5
#                 volPer = smoothness * round(volPer / smoothness)
#                 volPer = int(np.clip(volPer, 0, 100))

#                 if volume is not None:
#                     try:
#                         volume.SetMasterVolumeLevel(vol, None)
#                     except Exception:
#                         pass

#                 action = f"VOLUME 🔊 {volPer}%"

#                 # Volume bar overlay (YouTube style)
#                 cv2.rectangle(frame, (50, 150), (85, 400), (255, 0, 0), 3)
#                 cv2.rectangle(frame, (50, volBar), (85, 400), (255, 0, 0), cv2.FILLED)
#                 cv2.putText(frame, f"{volPer}%", (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
#                 try:
#                     cVol = int(volume.GetMasterVolumeLevelScalar() * 100) if volume else volPer
#                 except Exception:
#                     cVol = volPer
#                 cv2.putText(frame, f"Vol Set: {cVol}%", (w - 220, 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
            
#             # --- BRIGHTNESS MODE ---
#             elif mode == "BRIGHTNESS":
#                 dist = dist_thumb_index
#                 cx, cy = (lm_px[8][0] + lm_px[4][0]) // 2, (lm_px[8][1] + lm_px[4][1]) // 2
#                 cv2.circle(frame, (lm_px[4][0], lm_px[4][1]), 8, (250,204,21), -1)
#                 cv2.circle(frame, (lm_px[8][0], lm_px[8][1]), 8, (250,204,21), -1)
#                 cv2.line(frame, lm_px[4], lm_px[8], (250,204,21), 3)
                
#                 action = "BRIGHTNESS ☀️"
#                 b_level = np.interp(dist, [30, 180], [0, 100])
#                 try: sbc.set_brightness(int(b_level))
#                 except Exception: pass
#                 cv2.putText(frame, f"Bright: {int(b_level)}%", (cx - 40, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (250,204,21), 2)
                
#             if cooldown > 0:
#                 cooldown -= 1
        
#         # HUD Overlays for VIVA Impression
#         cv2.putText(frame, f"MODE: {mode}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
#         cv2.putText(frame, f"ACTION: {action}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (130, 140, 248), 2)

#         # FPS calculation
#         frames += 1
#         now = time.time()
#         if now - t_fps >= 1.0:
#             fps = frames / (now - t_fps)
#             t_fps, frames = now, 0
#         else:
#             fps = _state.get("fps", 0)
#         cv2.putText(frame, f"FPS: {int(fps)}", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (52, 211, 153), 1)

#         with _lock:
#             _state["frame"]  = frame
#             _state["fps"]    = fps
#             _state["action"] = action

#     if cap:
#         cap.release()

# # ── Tkinter Application ───────────────────────────────────────────────────
# class GestureMouseApp:
#     FEED_W  = 640
#     FEED_H  = 480
#     WIN_H   = 620
#     BG      = "#040d1a"
#     ACCENT  = "#818cf8"
#     GREEN   = "#34d399"
#     PINK    = "#f472b6"
#     TEXT    = "#f1f5f9"
#     SUBTEXT = "#475569"

#     def __init__(self, root: tk.Tk):
#         self.root   = root
#         self.thread = None
#         self._photo = None   # keep reference to avoid GC

#         root.title("🖱️  SignBridge Advanced Controls")
#         root.configure(bg=self.BG)
#         root.geometry(f"{self.FEED_W + 300}x{self.WIN_H}")
#         root.resizable(False, False)
#         root.protocol("WM_DELETE_WINDOW", self._on_close)

#         self._build_ui()
#         _state["running"] = False
#         self._update_ui()   # start UI refresh loop

#     def _build_ui(self):
#         # Top banner
#         top = tk.Frame(self.root, bg="#0d1b2e", height=60)
#         top.pack(side="top", fill="x")
#         top.pack_propagate(False)

#         ic = tk.Label(top, text="⚡", font=("Segoe UI", 20), fg=self.ACCENT, bg="#0d1b2e")
#         ic.pack(side="left", padx=(20, 10))
#         tk.Label(top, text="Advanced AI Controls", font=("Segoe UI", 16, "bold"),
#                  fg="white", bg="#0d1b2e").pack(side="left")
#         tk.Label(top, text="SignBridge • Real-Time 30fps", font=("Segoe UI", 10),
#                  fg=self.SUBTEXT, bg="#0d1b2e").pack(side="right", padx=20)

#         # Body
#         body = tk.Frame(self.root, bg=self.BG)
#         body.pack(fill="both", expand=True, padx=20, pady=20)

#         # ── LEFT: camera feed ─────────────────────────────────────────────
#         left = tk.Frame(body, bg=self.BG)
#         left.pack(side="left", fill="both", expand=True)

#         self.canvas = tk.Canvas(left, width=self.FEED_W, height=self.FEED_H,
#                                 bg="#060f1e", highlightthickness=1,
#                                 highlightbackground="#1e293b")
#         self.canvas.pack(pady=(self.WIN_H - self.FEED_H - 100) // 2)

#         self.canvas.create_text(self.FEED_W//2, self.FEED_H//2,
#                                 text="Press  ▶  Start  to open camera",
#                                 fill=self.SUBTEXT, font=("Segoe UI", 14),
#                                 tags="placeholder")

#         # ── RIGHT: controls & status ──────────────────────────────────────
#         right = tk.Frame(body, bg=self.BG, width=260)
#         right.pack(side="right", fill="y", padx=(16,4))
#         right.pack_propagate(False)

#         btn_frame = tk.Frame(right, bg=self.BG)
#         btn_frame.pack(fill="x", pady=(0, 16))

#         self.start_btn = tk.Button(btn_frame, text="▶  Start",
#                                    font=("Segoe UI", 11, "bold"),
#                                    fg="white", bg="#6366f1",
#                                    activebackground="#4f46e5",
#                                    relief="flat", cursor="hand2",
#                                    command=self._start, pady=8)
#         self.start_btn.pack(fill="x", pady=(0,6))

#         self.stop_btn = tk.Button(btn_frame, text="⏹  Stop",
#                                   font=("Segoe UI", 11, "bold"),
#                                   fg="white", bg="#334155",
#                                   activebackground="#1e293b",
#                                   relief="flat", cursor="hand2",
#                                   command=self._stop, pady=8,
#                                   state="disabled")
#         self.stop_btn.pack(fill="x")

#         tk.Label(right, text="ACTIVE MODE", font=("Segoe UI", 9, "bold"),
#                  fg=self.SUBTEXT, bg=self.BG).pack(anchor="w", pady=(0,4))
        
#         mode_frame = tk.Frame(right, bg=self.BG)
#         mode_frame.pack(fill="x", pady=(0, 16))
        
#         self.mode_var = tk.StringVar(value="MOUSE")
#         style = ttk.Style()
#         style.configure('Dark.TRadiobutton', background=self.BG, foreground=self.TEXT, font=("Segoe UI", 9))
        
#         ttk.Radiobutton(mode_frame, text="🖱️ Mouse Control", variable=self.mode_var, value="MOUSE", 
#                         command=self._update_mode, style='Dark.TRadiobutton').pack(anchor="w", pady=2)
#         ttk.Radiobutton(mode_frame, text="🔊 Volume Control", variable=self.mode_var, value="VOLUME", 
#                         command=self._update_mode, style='Dark.TRadiobutton').pack(anchor="w", pady=2)
#         ttk.Radiobutton(mode_frame, text="☀️ Brightness Control", variable=self.mode_var, value="BRIGHTNESS", 
#                         command=self._update_mode, style='Dark.TRadiobutton').pack(anchor="w", pady=2)

#         self.status_var = tk.StringVar(value="⬤  IDLE")
#         self.status_lbl = tk.Label(right, textvariable=self.status_var,
#                                    font=("Segoe UI", 12, "bold"),
#                                    fg=self.SUBTEXT, bg="#0d1b2e",
#                                    relief="flat", padx=12, pady=6)
#         self.status_lbl.pack(fill="x", pady=(0,10))

#         mf = tk.Frame(right, bg=self.BG)
#         mf.pack(fill="x", pady=0)
#         self.fps_var    = tk.StringVar(value="0")
#         self.action_var = tk.StringVar(value="—")

#         for label, var, col in [("FPS", self.fps_var, self.GREEN),
#                                  ("ACTION", self.action_var, self.ACCENT)]:
#             box = tk.Frame(mf, bg="#0d1b2e", pady=5)
#             box.pack(fill="x", pady=2)
#             tk.Label(box, textvariable=var, font=("Segoe UI", 11, "bold"),
#                      fg=col, bg="#0d1b2e").pack()

#         tk.Label(right, text="SMOOTHING (Mouse)", font=("Segoe UI", 9, "bold"),
#                  fg=self.SUBTEXT, bg=self.BG).pack(anchor="w", pady=(10,4))
        
#         self.smooth_val = tk.IntVar(value=5)
#         self.slider = ttk.Scale(right, from_=1, to=15, orient="horizontal",
#                                 variable=self.smooth_val, command=self._on_slide)
#         self.slider.pack(fill="x")
#         self.smooth_lbl = tk.Label(right, text="Smoothing: 5", font=("Segoe UI", 8),
#                                    fg=self.ACCENT, bg=self.BG)
#         self.smooth_lbl.pack(anchor="w")

#         tk.Frame(right, bg="#1e293b", height=1).pack(fill="x", pady=10)
#         self.guide_title = tk.Label(right, text="GESTURE GUIDE", font=("Segoe UI", 9, "bold"),
#                                     fg=self.SUBTEXT, bg=self.BG)
#         self.guide_title.pack(anchor="w", pady=(0,4))
        
#         self.guide_frame = tk.Frame(right, bg=self.BG)
#         self.guide_frame.pack(fill="x")
#         self._update_mode()

#     def _update_mode(self):
#         with _lock:
#             _state["mode"] = self.mode_var.get()
#         for widget in self.guide_frame.winfo_children():
#             widget.destroy()
#         if self.mode_var.get() == "MOUSE":
#             guide = [
#                 ("🤏 Pinch Thumb+Idx",  "→ Left Click"),
#                 ("🤏 Pinch Thumb+Mid",  "→ Right Click"),
#                 ("☝️ Idx straight",      "→ Move Cursor")
#             ]
#         else:
#             txt = "Volume" if self.mode_var.get() == "VOLUME" else "Brightness"
#             guide = [
#                 ("🤏 Pinch Thumb+Idx", f"→ Adjust {txt}"),
#                 ("↔️ Move Apart/Closer", "→ Increase/Decrease")
#             ]
#         for gesture, action in guide:
#             row = tk.Frame(self.guide_frame, bg=self.BG)
#             row.pack(fill="x", pady=1)
#             tk.Label(row, text=gesture, font=("Segoe UI", 8),
#                      fg=self.TEXT, bg=self.BG, width=15, anchor="w").pack(side="left")
#             tk.Label(row, text=action, font=("Segoe UI", 8),
#                      fg=self.SUBTEXT, bg=self.BG, anchor="w").pack(side="left")

#     def _start(self):
#         if _state["running"]: return
#         _stop_event.clear()
#         _state["running"] = True
#         self.start_btn.config(state="disabled")
#         self.stop_btn.config(state="normal", bg=self.PINK)
#         self.status_var.set("⬤  STARTING...")
#         self.status_lbl.config(fg=self.ACCENT)
#         self.thread = threading.Thread(target=_worker, args=(_stop_event,), daemon=True)
#         self.thread.start()

#     def _stop(self):
#         if not _state["running"]: return
#         _stop_event.set()
#         if self.thread: self.thread.join(timeout=1.0)
#         _state["running"] = False
#         self.start_btn.config(state="normal")
#         self.stop_btn.config(state="disabled", bg="#334155")
#         self.status_var.set("⬤  IDLE")
#         self.status_lbl.config(fg=self.SUBTEXT)
#         self.canvas.delete("all")
#         self.canvas.create_text(self.FEED_W//2, self.FEED_H//2,
#                                 text="Press  ▶  Start  to open camera",
#                                 fill=self.SUBTEXT, font=("Segoe UI", 14),
#                                 tags="placeholder")

#     def _on_slide(self, val):
#         v = int(float(val))
#         self.smooth_val.set(v)
#         self.smooth_lbl.config(text=f"Smoothing: {v}")
#         with _lock: _state["smoothing"] = v

#     def _update_ui(self):
#         with _lock:
#             frame_bgr   = _state.get("frame", None)
#             fps         = _state.get("fps", 0)
#             action      = _state.get("action", "—")
#             running     = _state["running"]
#         if running and frame_bgr is not None:
#             try:
#                 frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
#                 img  = Image.fromarray(frame_rgb).resize((self.FEED_W, self.FEED_H), Image.NEAREST)
#                 self._photo = ImageTk.PhotoImage(image=img)
#                 self.canvas.delete("all")
#                 self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
#             except Exception: pass
#             self.status_var.set(f"⬤  LIVE  —  {self.mode_var.get()}")
#             self.status_lbl.config(fg=self.GREEN)
#             self.fps_var.set(f"{int(fps)}")
#             self.action_var.set(action)
#         self.root.after(33, self._update_ui)

#     def _on_close(self):
#         _stop_event.set()
#         self.root.destroy()
#         sys.exit(0)

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--mode", type=str, default="MOUSE", choices=["MOUSE", "VOLUME", "BRIGHTNESS"])
#     args = parser.parse_args()
#     root = tk.Tk()
#     app = GestureMouseApp(root)
#     app.mode_var.set(args.mode)
#     app._update_mode()
#     root.mainloop()
"""
gesture_mouse_standalone.py  –  Tkinter-based Gesture Mouse
Now Includes:
- Mouse
- Volume
- Brightness
- Professional Whiteboard Mode
"""

import sys, os, time, threading
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk
from collections import deque

# Advanced Controls
import pyautogui
import screen_brightness_control as sbc
from math import hypot
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
try:
    from comtypes import CLSCTX_ALL
except ImportError:
    CLSCTX_ALL = None
from pynput.mouse import Button, Controller

mouse = Controller()
pyautogui.FAILSAFE = False
SCREEN_W, SCREEN_H = pyautogui.size()

# ── MediaPipe ─────────────────────────────────────────────
mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
drawing_spec_nodes = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
drawing_spec_lines = mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)

# ── Global State ──────────────────────────────────────────
_state = {
    "running": False,
    "smoothing": 5,
    "mode": "MOUSE",
    "draw_color": (255, 0, 255)
}
_stop_event = threading.Event()
_lock       = threading.Lock()

# ── Camera helpers ────────────────────────────────────────
def _open_cap(cam_index=0):
    cap = None
    try:
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    except Exception:
        cap = cv2.VideoCapture(cam_index)

    if not cap or not cap.isOpened():
        cap = cv2.VideoCapture(cam_index)

    if cap.isOpened():
        # Try 1280×720 first, fallback keeps whatever camera supports
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS,          30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        # If camera rejects 1280×720, it silently falls back to its native res
    return cap

# ── Advanced Math Helpers ─────────────────────────────────
def get_distance(p1, p2):
    return hypot(p2[0] - p1[0], p2[1] - p1[1])

def get_angle(a, b, c):
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    return np.abs(np.degrees(radians))
# ── Background worker ─────────────────────────────────────
def _worker(stop: threading.Event):
    cap   = _open_cap()
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.7)
    
    # Whiteboard canvas — lazy init (black, sized to first frame)
    canvas   = None       # initialized on first frame
    prev_x, prev_y = 0, 0

    # Audio Setup
    volume   = None
    minVol   = -65.25
    maxVol   = 0.0
    try:
        speaker  = AudioUtilities.GetSpeakers()
        volume   = speaker.EndpointVolume
        volRange = volume.GetVolumeRange()
        minVol   = volRange[0]
        maxVol   = volRange[1]
    except Exception:
        volume = None

    plocX, plocY = 0.0, 0.0
    ptx, pty     = 0, 0
    cooldown     = 0
    t_fps, frames = time.time(), 0
    
    gesture_buffer = deque(maxlen=8)

    while not stop.is_set():
        if cap is None or not cap.isOpened():
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            continue

        frame  = cv2.flip(frame, 1)
        h, w   = frame.shape[:2]

        # ----- Lazy-init canvas to match actual camera resolution -----
        if canvas is None:
            canvas = np.zeros((h, w, 3), np.uint8)

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        mode   = _state["mode"]
        action = f"IDLE ({mode})"
        sm     = _state["smoothing"]
        alpha  = 1.0 / max(sm, 1)

        if result.multi_hand_landmarks:
            lm_list = result.multi_hand_landmarks[0]
            lm_px = [(int(pt.x * w), int(pt.y * h)) for pt in lm_list.landmark]

            dist_thumb_index  = get_distance(lm_px[4], lm_px[8])
            dist_thumb_middle = get_distance(lm_px[4], lm_px[12])

            # ── Colored per-finger skeleton (matching model_utils.py style) ──
            FINGER_COLORS = [
                ([1, 2, 3, 4],      (0, 0, 255)),    # Thumb  → Red
                ([5, 6, 7, 8],      (0, 255, 0)),    # Index  → Green
                ([9, 10, 11, 12],   (0, 255, 255)),  # Middle → Yellow
                ([13, 14, 15, 16],  (255, 0, 0)),    # Ring   → Blue
                ([17, 18, 19, 20],  (255, 0, 255)),  # Pinky  → Magenta
            ]
            for connection in mp_hands.HAND_CONNECTIONS:
                s, e = connection[0], connection[1]
                col = (255, 255, 255)  # white default (palm connections)
                for idxs, c in FINGER_COLORS:
                    if s in idxs or e in idxs:
                        col = c; break
                cv2.line(frame, lm_px[s], lm_px[e], col, 4)
            # Landmark dots — white with dark centre
            for pt in lm_px:
                cv2.circle(frame, pt, 7, (255, 255, 255), -1)
                cv2.circle(frame, pt, 4, (30, 30, 30), -1)

            # ================= MOUSE MODE =================
            if mode == "MOUSE":

                if get_distance(lm_px[4], lm_px[5]) < 50 and \
                   get_angle(lm_px[5], lm_px[6], lm_px[8]) > 100:

                    action = "MOVE CURSOR"

                    ix, iy = lm_list.landmark[8].x, lm_list.landmark[8].y
                    clocX = (1 - alpha) * plocX + alpha * ix
                    clocY = (1 - alpha) * plocY + alpha * iy

                    tx = int(np.interp(clocX, [0.1, 0.9], [0, SCREEN_W]))
                    ty = int(np.interp(clocY, [0.1, 0.9], [0, SCREEN_H]))

                    if abs(tx - ptx) > 4 or abs(ty - pty) > 4:
                        pyautogui.moveTo(tx, ty)
                        ptx, pty = tx, ty

                    plocX, plocY = clocX, clocY
                    gesture_buffer.append("NONE")

                elif cooldown == 0:

                    if dist_thumb_index < 35:
                        gesture_buffer.append("LEFT")
                    elif dist_thumb_middle < 35:
                        gesture_buffer.append("RIGHT")
                    else:
                        gesture_buffer.append("NONE")

                    if gesture_buffer.count("LEFT") > 5:
                        action = "LEFT CLICK ✓"
                        mouse.click(Button.left)
                        cooldown = 15
                        gesture_buffer.clear()

                    elif gesture_buffer.count("RIGHT") > 5:
                        action = "RIGHT CLICK ✓"
                        mouse.click(Button.right)
                        cooldown = 15
                        gesture_buffer.clear()

            # ================= VOLUME MODE =================
            elif mode == "VOLUME":

                length = dist_thumb_index
                vol    = np.interp(length, [50, 300], [minVol, maxVol])
                volPer = int(np.clip(np.interp(length, [50, 300], [0, 100]), 0, 100))

                if volume is not None:
                    try:
                        volume.SetMasterVolumeLevel(vol, None)
                    except:
                        pass

                action = f"VOLUME 🔊 {volPer}%"

                # ── Visual: thumb–index pinch line (YouTube style) ──
                x1, y1 = lm_px[4]
                x2, y2 = lm_px[8]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(frame, (x1, y1), 14, (255, 0, 255), cv2.FILLED)
                cv2.circle(frame, (x2, y2), 14, (255, 0, 255), cv2.FILLED)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                mid_col = (0, 255, 0) if length < 50 else (255, 0, 255)
                cv2.circle(frame, (cx, cy), 12, mid_col, cv2.FILLED)
                # Volume bar overlay
                bar_y = int(np.interp(volPer, [0, 100], [400, 150]))
                cv2.rectangle(frame, (50, 150), (85, 400), (200, 0, 200), 3)
                cv2.rectangle(frame, (50, bar_y), (85, 400), (200, 0, 200), cv2.FILLED)
                cv2.putText(frame, f"{volPer}%", (35, 430),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 0, 200), 2)
                cv2.putText(frame, f"Vol Set: {volPer}%", (w - 220, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # ================= BRIGHTNESS MODE =================
            elif mode == "BRIGHTNESS":

                dist = dist_thumb_index
                bright = int(np.clip(np.interp(dist, [30, 180], [0, 100]), 0, 100))
                try:
                    sbc.set_brightness(bright)
                except:
                    pass

                action = f"BRIGHTNESS ☀️ {bright}%"

                # ── Visual: yellow thumb–index line ──
                bx1, by1 = lm_px[4]
                bx2, by2 = lm_px[8]
                bcx = (bx1 + bx2) // 2
                bcy = (by1 + by2) // 2
                cv2.circle(frame, (bx1, by1), 12, (0, 200, 255), -1)
                cv2.circle(frame, (bx2, by2), 12, (0, 200, 255), -1)
                cv2.line(frame, (bx1, by1), (bx2, by2), (0, 200, 255), 3)
                cv2.putText(frame, f"Bright: {bright}%", (bcx - 50, bcy - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            # ================= WHITEBOARD MODE =================
            elif mode == "WHITEBOARD":

                # ── Dynamic color palette — drawn first so gestures can hit it ──
                HEADER_H = 90
                WB_PALETTE = [
                    ((255,   0, 255), "Pink"),
                    ((  0,  50, 255), "Red"),
                    ((255,  50,  50), "Blue"),
                    ((  0, 220,   0), "Green"),
                    ((  0, 200, 255), "Yellow"),
                    (( 80,   0, 200), "Purple"),
                    ((  0, 140, 255), "Orange"),
                    ((240, 240, 240), "White"),
                ]
                # Note: black (0,0,0) is intentionally excluded — it's invisible in
                # the Murtaza bitwise overlay (any pixel ≤50 gray = transparent).
                # Eraser is activated by raising only the PINKY finger.
                n_col = len(WB_PALETTE)
                sw    = (w - 20) // n_col   # swatch width

                # Header bar with gradient-feel dark background
                overlay_hdr = frame.copy()
                cv2.rectangle(overlay_hdr, (0, 0), (w, HEADER_H), (20, 20, 30), -1)
                frame = cv2.addWeighted(overlay_hdr, 0.85, frame, 0.15, 0)

                draw_color = _state["draw_color"]
                for i, (col, name) in enumerate(WB_PALETTE):
                    x1s = 10 + i * sw
                    x2s = x1s + sw - 8
                    y1s, y2s = 10, HEADER_H - 10
                    # Draw swatch
                    cv2.rectangle(frame, (x1s, y1s), (x2s, y2s), col, -1)
                    # Selected indicator: bright white border + dot
                    if col == draw_color:
                        cv2.rectangle(frame, (x1s - 4, y1s - 4), (x2s + 4, y2s + 4), (255, 255, 255), 3)
                        cv2.circle(frame, (x1s + (x2s - x1s)//2, y2s + 8), 5, (255, 255, 255), -1)

                # ── Finger state — Murtaza-style (normalize to hand size) ──
                # Index, Middle, Ring, Pinky tip vs their PIP joint
                TIP_IDS = [8, 12, 16, 20]
                PIP_IDS = [6, 10, 14, 18]
                fingers_up = [lm_px[TIP_IDS[i]][1] < lm_px[PIP_IDS[i]][1]
                               for i in range(4)]

                ix, iy = lm_px[8]   # index fingertip
                mx, my = lm_px[12]  # middle fingertip

                # Normalize pinch by hand size (wrist–middle_MCP span)
                hand_span   = max(get_distance(lm_px[0], lm_px[9]), 1)
                pinch_ratio = dist_thumb_index / hand_span
                is_pinch    = pinch_ratio < 0.28     # ~28% of hand = pinch
                all_up      = all(fingers_up)        # all 4 fingers extended

                # ── Gesture logic ──────────────────────────────────────────────────
                # fingers_up = [index, middle, ring, pinky]  (tip above PIP = True)

                pinky_only = (fingers_up[3] and
                              not fingers_up[0] and
                              not fingers_up[1] and
                              not fingers_up[2])
                all_up     = all(fingers_up)   # all 4 fingers open = PALM

                # ── Selection mode: index + middle both up ──
                if fingers_up[0] and fingers_up[1]:
                    action = "COLOR SELECT 🎨"
                    prev_x, prev_y = 0, 0
                    cv2.rectangle(frame, (ix, iy - 25), (mx, my + 25),
                                  draw_color, cv2.FILLED)
                    if iy < HEADER_H:
                        for i, (col, _) in enumerate(WB_PALETTE):
                            x1s = 10 + i * sw
                            x2s = x1s + sw - 8
                            if x1s < ix < x2s:
                                _state["draw_color"] = col
                                break

                # ── Clear board: open palm (all 4 fingers up) ──
                elif all_up:
                    action = "CLEAR BOARD 🧹"
                    canvas[:] = 0
                    prev_x, prev_y = 0, 0

                # ── Erase: PINKY finger only up ──
                elif pinky_only:
                    action = "ERASING 🧽  (raise ☝️ only to draw)"
                    ex, ey = lm_px[20]   # use pinky tip as eraser tip
                    cv2.circle(canvas, (ex, ey), 45, (0, 0, 0), -1)
                    cv2.circle(frame,  (ex, ey), 45, (200, 200, 200), 2)
                    prev_x, prev_y = 0, 0

                # ── Draw: only index finger up ──
                elif fingers_up[0] and not fingers_up[1]:
                    action   = "DRAWING ✏️"
                    cur_col  = _state["draw_color"]
                    # Cursor circle on live frame
                    cv2.circle(frame, (ix, iy), 14, cur_col, cv2.FILLED)

                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = ix, iy

                    # Draw line on canvas with thick brush
                    cv2.line(canvas, (prev_x, prev_y), (ix, iy), cur_col, 20)
                    cv2.line(frame,  (prev_x, prev_y), (ix, iy), cur_col, 10)
                    prev_x, prev_y = ix, iy

                else:
                    prev_x, prev_y = 0, 0

        # ----- Cooldown -----
        if cooldown > 0:
            cooldown -= 1

        # ----- FPS -----
        frames += 1
        now = time.time()
        if now - t_fps >= 1.0:
            fps = frames / (now - t_fps)
            t_fps, frames = now, 0
        else:
            fps = _state.get("fps", 0)

        # ── WHITEBOARD: Murtaza-style overlay (drawings on top of camera) ──
        if mode == "WHITEBOARD" and canvas is not None:
            imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
            imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
            frame  = cv2.bitwise_and(frame, imgInv)
            frame  = cv2.bitwise_or(frame,  canvas)
            # FPS badge
            cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (52, 211, 153), 2)
        else:
            cv2.putText(frame, f"MODE: {mode}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"ACTION: {action}", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (130, 140, 248), 2)
            cv2.putText(frame, f"FPS: {int(fps)}", (15, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (52, 211, 153), 1)

        with _lock:
            _state["frame"]  = frame
            _state["fps"]    = fps
            _state["action"] = action

    if cap:
        cap.release()
        # ── Tkinter Application ───────────────────────────────────────────────────
class GestureMouseApp:
    FEED_W  = 960    # Display size (camera frame is scaled to this)
    FEED_H  = 540
    WIN_H   = 700
    BG      = "#040d1a"
    ACCENT  = "#818cf8"
    GREEN   = "#34d399"
    PINK    = "#f472b6"
    TEXT    = "#f1f5f9"
    SUBTEXT = "#475569"

    def __init__(self, root: tk.Tk):
        self.root   = root
        self.thread = None
        self._photo = None

        root.title("SignBridge — Advanced AI Controls")
        root.configure(bg=self.BG)
        root.geometry(f"{self.FEED_W + 310}x{self.WIN_H + 40}")
        root.resizable(True, True)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        _state["running"] = False
        self._update_ui()

    def _build_ui(self):

        # ── Top banner (premium gradient look) ──────────────────────────────
        top = tk.Frame(self.root, bg="#0d1b2e", height=64)
        top.pack(side="top", fill="x")
        top.pack_propagate(False)

        left_top = tk.Frame(top, bg="#0d1b2e")
        left_top.pack(side="left", fill="y", padx=(16, 0))
        tk.Label(left_top, text="⚡", font=("Segoe UI", 22),
                 fg=self.ACCENT, bg="#0d1b2e").pack(side="left", padx=(0, 8))
        title_col = tk.Frame(left_top, bg="#0d1b2e")
        title_col.pack(side="left")
        tk.Label(title_col, text="Advanced AI Controls",
                 font=("Segoe UI", 15, "bold"),
                 fg="#f1f5f9", bg="#0d1b2e").pack(anchor="w")
        tk.Label(title_col, text="Hand Gesture · Mouse · Volume · Brightness · Whiteboard",
                 font=("Segoe UI", 8),
                 fg=self.SUBTEXT, bg="#0d1b2e").pack(anchor="w")

        tk.Label(top, text="SignBridge  •  Real-Time 30fps",
                 font=("Segoe UI", 10), fg="#334155",
                 bg="#0d1b2e").pack(side="right", padx=20)

        # Body
        body = tk.Frame(self.root, bg=self.BG)
        body.pack(fill="both", expand=True, padx=20, pady=20)

        # LEFT — Camera Feed
        left = tk.Frame(body, bg=self.BG)
        left.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(left,
                                width=self.FEED_W,
                                height=self.FEED_H,
                                bg="#060f1e",
                                highlightthickness=1,
                                highlightbackground="#1e293b")
        self.canvas.pack(pady=(self.WIN_H - self.FEED_H - 100) // 2)

        self.canvas.create_text(self.FEED_W//2,
                                self.FEED_H//2,
                                text="Press  ▶  Start  to open camera",
                                fill=self.SUBTEXT,
                                font=("Segoe UI", 14),
                                tags="placeholder")

        # RIGHT — Controls
        right = tk.Frame(body, bg=self.BG, width=260)
        right.pack(side="right", fill="y", padx=(16,4))
        right.pack_propagate(False)

        btn_frame = tk.Frame(right, bg=self.BG)
        btn_frame.pack(fill="x", pady=(0, 16))

        self.start_btn = tk.Button(btn_frame,
                                   text="▶  Start",
                                   font=("Segoe UI", 11, "bold"),
                                   fg="white",
                                   bg="#6366f1",
                                   relief="flat",
                                   cursor="hand2",
                                   command=self._start,
                                   pady=8)
        self.start_btn.pack(fill="x", pady=(0,6))

        self.stop_btn = tk.Button(btn_frame,
                                  text="⏹  Stop",
                                  font=("Segoe UI", 11, "bold"),
                                  fg="white",
                                  bg="#334155",
                                  relief="flat",
                                  cursor="hand2",
                                  command=self._stop,
                                  pady=8,
                                  state="disabled")
        self.stop_btn.pack(fill="x")

        tk.Label(right,
                 text="ACTIVE MODE",
                 font=("Segoe UI", 9, "bold"),
                 fg=self.SUBTEXT,
                 bg=self.BG).pack(anchor="w", pady=(0,4))

        mode_frame = tk.Frame(right, bg=self.BG)
        mode_frame.pack(fill="x", pady=(0, 16))

        self.mode_var = tk.StringVar(value="MOUSE")

        style = ttk.Style()
        style.configure('Dark.TRadiobutton',
                        background=self.BG,
                        foreground=self.TEXT,
                        font=("Segoe UI", 9))

        ttk.Radiobutton(mode_frame,
                        text="🖱️ Mouse Control",
                        variable=self.mode_var,
                        value="MOUSE",
                        command=self._update_mode,
                        style='Dark.TRadiobutton').pack(anchor="w", pady=2)

        ttk.Radiobutton(mode_frame,
                        text="🔊 Volume Control",
                        variable=self.mode_var,
                        value="VOLUME",
                        command=self._update_mode,
                        style='Dark.TRadiobutton').pack(anchor="w", pady=2)

        ttk.Radiobutton(mode_frame,
                        text="☀️ Brightness Control",
                        variable=self.mode_var,
                        value="BRIGHTNESS",
                        command=self._update_mode,
                        style='Dark.TRadiobutton').pack(anchor="w", pady=2)

        # ✅ Whiteboard Added
        ttk.Radiobutton(mode_frame,
                        text="🖍 Whiteboard",
                        variable=self.mode_var,
                        value="WHITEBOARD",
                        command=self._update_mode,
                        style='Dark.TRadiobutton').pack(anchor="w", pady=2)

        self.status_var = tk.StringVar(value="⬤  IDLE")
        self.status_lbl = tk.Label(right,
                                   textvariable=self.status_var,
                                   font=("Segoe UI", 12, "bold"),
                                   fg=self.SUBTEXT,
                                   bg="#0d1b2e",
                                   padx=12,
                                   pady=6)
        self.status_lbl.pack(fill="x", pady=(0,10))
                # FPS + ACTION display
        mf = tk.Frame(right, bg=self.BG)
        mf.pack(fill="x", pady=0)

        self.fps_var    = tk.StringVar(value="0")
        self.action_var = tk.StringVar(value="—")

        for label, var, col in [
            ("FPS", self.fps_var, self.GREEN),
            ("ACTION", self.action_var, self.ACCENT)
        ]:
            box = tk.Frame(mf, bg="#0d1b2e", pady=5)
            box.pack(fill="x", pady=2)
            tk.Label(box, text=label,
                     font=("Segoe UI", 7, "bold"),
                     fg="#334155", bg="#0d1b2e").pack()
            tk.Label(box,
                     textvariable=var,
                     font=("Segoe UI", 11, "bold"),
                     fg=col,
                     bg="#0d1b2e").pack()

        # Smoothing Slider
        tk.Label(right,
                 text="SMOOTHING (Mouse)",
                 font=("Segoe UI", 9, "bold"),
                 fg=self.SUBTEXT,
                 bg=self.BG).pack(anchor="w", pady=(10,4))

        self.smooth_val = tk.IntVar(value=5)

        self.slider = ttk.Scale(right,
                                from_=1,
                                to=15,
                                orient="horizontal",
                                variable=self.smooth_val,
                                command=self._on_slide)
        self.slider.pack(fill="x")

        self.smooth_lbl = tk.Label(right,
                                   text="Smoothing: 5",
                                   font=("Segoe UI", 8),
                                   fg=self.ACCENT,
                                   bg=self.BG)
        self.smooth_lbl.pack(anchor="w")

        tk.Frame(right, bg="#1e293b", height=1).pack(fill="x", pady=10)

        # Gesture Guide
        self.guide_title = tk.Label(right,
                                    text="GESTURE GUIDE",
                                    font=("Segoe UI", 9, "bold"),
                                    fg=self.SUBTEXT,
                                    bg=self.BG)
        self.guide_title.pack(anchor="w", pady=(0,4))

        self.guide_frame = tk.Frame(right, bg=self.BG)
        self.guide_frame.pack(fill="x")

        self._update_mode()

    # ── Mode Update ─────────────────────────────────────────
    def _update_mode(self):
        with _lock:
            _state["mode"] = self.mode_var.get()

        for widget in self.guide_frame.winfo_children():
            widget.destroy()

        if self.mode_var.get() == "MOUSE":
            guide = [
                ("🤏 Pinch Thumb+Idx", "→ Left Click"),
                ("🤏 Pinch Thumb+Mid", "→ Right Click"),
                ("☝️ Index straight", "→ Move Cursor")
            ]

        elif self.mode_var.get() == "WHITEBOARD":
            guide = [
                ("☝️ Index", "→ Draw"),
                ("✌️ Two Fingers", "→ Select Color"),
                ("🤏 Pinch", "→ Erase"),
                ("✋ Open Palm", "→ Clear Board")
            ]

        else:
            txt = "Volume" if self.mode_var.get() == "VOLUME" else "Brightness"
            guide = [
                ("🤏 Pinch Thumb+Idx", f"→ Adjust {txt}"),
                ("↔️ Move Apart/Closer", "→ Increase/Decrease")
            ]

        for gesture, action in guide:
            row = tk.Frame(self.guide_frame, bg=self.BG)
            row.pack(fill="x", pady=1)

            tk.Label(row,
                     text=gesture,
                     font=("Segoe UI", 8),
                     fg=self.TEXT,
                     bg=self.BG,
                     width=16,
                     anchor="w").pack(side="left")

            tk.Label(row,
                     text=action,
                     font=("Segoe UI", 8),
                     fg=self.SUBTEXT,
                     bg=self.BG,
                     anchor="w").pack(side="left")

    # ── Start ───────────────────────────────────────────────
    def _start(self):
        if _state["running"]:
            return

        _stop_event.clear()
        _state["running"] = True

        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal", bg=self.PINK)

        self.status_var.set("⬤  LIVE")
        self.status_lbl.config(fg=self.GREEN)

        self.thread = threading.Thread(target=_worker,
                                       args=(_stop_event,),
                                       daemon=True)
        self.thread.start()

    # ── Stop ────────────────────────────────────────────────
    def _stop(self):
        if not _state["running"]:
            return

        _stop_event.set()
        if self.thread:
            self.thread.join(timeout=1.0)

        _state["running"] = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled", bg="#334155")

        self.status_var.set("⬤  IDLE")
        self.status_lbl.config(fg=self.SUBTEXT)

        self.canvas.delete("all")
        self.canvas.create_text(self.FEED_W//2,
                                self.FEED_H//2,
                                text="Press  ▶  Start  to open camera",
                                fill=self.SUBTEXT,
                                font=("Segoe UI", 14))

    # ── Slider ──────────────────────────────────────────────
    def _on_slide(self, val):
        v = int(float(val))
        self.smooth_val.set(v)
        self.smooth_lbl.config(text=f"Smoothing: {v}")
        with _lock:
            _state["smoothing"] = v

    # ── UI Update Loop ──────────────────────────────────────
    def _update_ui(self):
        with _lock:
            frame_bgr = _state.get("frame", None)
            fps       = _state.get("fps", 0)
            action    = _state.get("action", "—")
            running   = _state["running"]

        if running and frame_bgr is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb).resize(
                (self.FEED_W, self.FEED_H),
                Image.NEAREST
            )
            self._photo = ImageTk.PhotoImage(image=img)

            self.canvas.delete("all")
            self.canvas.create_image(0, 0,
                                     anchor="nw",
                                     image=self._photo)

            self.fps_var.set(f"{int(fps)}")
            self.action_var.set(action)

        self.root.after(33, self._update_ui)

    def _on_close(self):
        _stop_event.set()
        self.root.destroy()
        sys.exit(0)


# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        type=str,
                        default="MOUSE",
                        choices=["MOUSE", "VOLUME", "BRIGHTNESS", "WHITEBOARD"])
    args = parser.parse_args()

    root = tk.Tk()
    app = GestureMouseApp(root)
    app.mode_var.set(args.mode)
    app._update_mode()
    root.mainloop()