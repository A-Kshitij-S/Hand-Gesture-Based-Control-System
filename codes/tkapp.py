"""
tkapp.py  –  SignBridge ASL Recognition Suite
Full Tkinter desktop app replacing Streamlit entirely.
All pages share a single camera via camera_utils; each page's camera
loop runs in a background thread and updates the UI via root.after().

Run:
    python tkapp.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading, time, subprocess, sys, os
import cv2, numpy as np
from PIL import Image, ImageTk

# ── Path setup ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.model_utils       import predict_gesture, top_k_from_pred, is_model_ready, get_raw_landmarks
from utils.auth_utils        import register_user, authenticate_user, user_exists, list_users, delete_user
from utils.phrase_utils      import speak_text, GestureWordBuilder, PHRASE_DICT
from utils.translation_utils import translate_text, get_language_names, get_gtts_code, SUPPORTED_LANGUAGES
from utils.camera_utils      import open_camera, release_camera, read_frame as cam_read_frame
from utils.word_gesture_utils import load_word_gestures, match_live_frame, get_loaded_words

# ══════════════════════════════════════════════════════════════════════════
# THEME
# ══════════════════════════════════════════════════════════════════════════
T = {
    "bg":       "#040d1a",
    "sidebar":  "#06111f",
    "card":     "#0d1b2e",
    "border":   "#1e293b",
    "accent":   "#818cf8",
    "blue":     "#38bdf8",
    "green":    "#34d399",
    "yellow":   "#fbbf24",
    "pink":     "#f472b6",
    "red":      "#f87171",
    "text":     "#f1f5f9",
    "sub":      "#64748b",
    "muted":    "#334155",
    "font":     "Segoe UI",
}

def themed_frame(parent, bg=None, **kw):
    return tk.Frame(parent, bg=bg or T["bg"], **kw)

def themed_label(parent, text="", font_size=11, bold=False, color=None, **kw):
    weight = "bold" if bold else "normal"
    return tk.Label(parent, text=text,
                    font=(T["font"], font_size, weight),
                    fg=color or T["text"], bg=kw.pop("bg", T["bg"]), **kw)

def card_frame(parent, **kw):
    return tk.Frame(parent, bg=T["card"],
                    highlightbackground=T["border"], highlightthickness=1, **kw)

def accent_button(parent, text, command, color=None, **kw):
    c = color or T["accent"]
    px = kw.pop("padx", 14)
    py = kw.pop("pady", 8)
    b = tk.Button(parent, text=text, command=command,
                  font=(T["font"], 10, "bold"),
                  fg="white", bg=c, activebackground=c,
                  relief="flat", cursor="hand2",
                  padx=px, pady=py, **kw)
    b.bind("<Enter>", lambda e: b.config(bg=_darken(c)))
    b.bind("<Leave>", lambda e: b.config(bg=c))
    return b

def _darken(hex_color):
    r = max(0, int(hex_color[1:3], 16) - 20)
    g = max(0, int(hex_color[3:5], 16) - 20)
    b = max(0, int(hex_color[5:7], 16) - 20)
    return f"#{r:02x}{g:02x}{b:02x}"

def sep(parent, **kw):
    return tk.Frame(parent, bg=T["border"], height=1, **kw)


# ══════════════════════════════════════════════════════════════════════════
# BASE PAGE
# ══════════════════════════════════════════════════════════════════════════
class Page(tk.Frame):
    """Base class — override on_show / on_hide for camera lifecycle."""
    def __init__(self, parent, app):
        super().__init__(parent, bg=T["bg"])
        self.app = app

    def on_show(self): pass
    def on_hide(self): pass


# ══════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════
class HomePage(Page):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        self._build()

    def _build(self):
        canvas = tk.Canvas(self, bg=T["bg"], highlightthickness=0)
        scroll = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = themed_frame(canvas)
        canvas.create_window((0,0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # Hero
        hero = themed_frame(inner); hero.pack(fill="x", padx=40, pady=(40,20))
        themed_label(hero, "🤟  SignBridge", 32, True, T["accent"]).pack(anchor="w")
        themed_label(hero, "ASL Recognition Suite · Real-time Desktop App", 12, color=T["sub"]).pack(anchor="w", pady=(4,0))
        themed_label(hero, "Gesture-based biometric login, live sign-to-text translation in 15+ languages,\n"
                     "phrase detection with TTS, and gesture-controlled mouse — all privacy-first.",
                     10, color=T["muted"]).pack(anchor="w", pady=(8,0))

        sep(inner).pack(fill="x", padx=40, pady=20)

        # Stats strip
        stats_frame = card_frame(inner); stats_frame.pack(fill="x", padx=40, pady=(0,20))
        for val, lbl, sub in [("26","ASL Letters","A–Z real-time"),
                               ("15+","Languages","Live translation"),
                               ("21","Landmarks","63 XYZ coords"),
                               ("30fps","Live Feed","Native speed")]:
            sf = themed_frame(stats_frame, bg=T["card"]); sf.pack(side="left", expand=True, pady=16, padx=10)
            themed_label(sf, val, 22, True, T["accent"], bg=T["card"]).pack()
            themed_label(sf, lbl, 9,  True, T["text"],   bg=T["card"]).pack()
            themed_label(sf, sub, 8,  color=T["sub"],    bg=T["card"]).pack()

        sep(inner).pack(fill="x", padx=40, pady=10)

        # Feature cards
        themed_label(inner, "Suite Features", 14, True, T["text"]).pack(anchor="w", padx=40, pady=(10,8))
        grid = themed_frame(inner); grid.pack(fill="x", padx=40, pady=(0,20))
        features = [
            ("🔐", "Gesture Auth",       T["accent"],  "Biometric login using your hand. No passwords, no images stored."),
            ("🤟", "Sign Translation",   T["blue"],    "Real-time A–Z letter prediction with 5-frame temporal smoothing."),
            ("💬", "Gesture To Word",   T["green"],   "Spell full words letter-by-letter → 30+ phrases auto-detected → TTS."),
            ("🖱️", "Gesture Mouse",     T["pink"],    "Control your cursor, click and scroll with hand gestures at 30fps."),
            ("🔊", "Volume Control",    T["yellow"],  "Adjust system volume intuitively using a two-finger pinch gesture."),
            ("☀️", "Brightness Control", T["red"],     "Change display brightness with an easy two-finger distance metric."),
            ("🎨", "Whiteboard",         "#2dd4bf",   "Draw, erase and color with hand gestures on a virtual canvas at 30fps."),
            ("🧊", "3D Viewer",          "#06b6d4",   "Rotate, zoom & switch between 23 procedural 3D shapes with hand gestures."),
            ("🔷", "Voxel Editor",       "#00ffd5",   "Build 3D voxel structures using hand gestures — Minecraft meets AR."),
            ("🥽", "Headsetless VR",     "#ff2d92",   "Cyberpunk synthwave city — head-tracked 3D parallax via webcam, no headset."),
        ]
        for i, (icon, title, color, desc) in enumerate(features):
            row_idx = i // 3
            col_idx = i % 3
            f = card_frame(grid, padx=14, pady=14)
            f.grid(row=row_idx, column=col_idx, padx=6, pady=6, sticky="nsew")
            grid.columnconfigure(col_idx, weight=1)
            f.grid_propagate(False)
            f.config(height=160)

            themed_label(f, icon, 22, bg=T["card"]).pack(anchor="w")
            themed_label(f, title, 10, True, color, bg=T["card"]).pack(anchor="w", pady=(4,2))
            themed_label(f, desc,  9, color=T["sub"], bg=T["card"], wraplength=170, justify="left").pack(anchor="w")
            btn = tk.Button(f, text="Open →", font=(T["font"],9,"bold"),
                            fg=color, bg=T["card"], activebackground=T["card"],
                            relief="flat", cursor="hand2",
                            command=lambda t=title: self._handle_feature_click(t))
            btn.pack(anchor="w", pady=(8,0))

    def _handle_feature_click(self, title):
        self.app.show_page(title)

        # Auth notice
        if not self.app.authenticated_user:
            notice = card_frame(inner, padx=20, pady=16)
            notice.pack(fill="x", padx=40, pady=(0,20))
            themed_label(notice, "🔒  Authentication Required", 12, True, T["yellow"], bg=T["card"]).pack()
            themed_label(notice, "Go to Gesture Auth → register your hand gesture → login to unlock all features.",
                         10, color=T["sub"], bg=T["card"]).pack(pady=(4,0))
        else:
            notice = card_frame(inner, padx=20, pady=16)
            notice.pack(fill="x", padx=40, pady=(0,20))
            themed_label(notice, f"✅  Welcome back, {self.app.authenticated_user}!", 12, True, T["green"], bg=T["card"]).pack()


# ══════════════════════════════════════════════════════════════════════════
# GESTURE AUTH PAGE
# ══════════════════════════════════════════════════════════════════════════
class AuthPage(Page):
    REQUIRED = 5

    def __init__(self, parent, app):
        super().__init__(parent, app)
        self._cam_running = False
        self._stop        = threading.Event()
        self._lock        = threading.Lock()
        self._frame_rgb   = None
        self._lm          = None
        self._reg_samples = []
        self._build()

    def _build(self):
        themed_label(self, "🔐  Gesture Authentication", 18, True, T["accent"]).pack(anchor="w", padx=30, pady=(20,4))
        themed_label(self, "Biometric login with your hand gesture — no passwords, no images stored.", 10, color=T["sub"]).pack(anchor="w", padx=30)
        sep(self).pack(fill="x", padx=30, pady=12)

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=30, pady=4)

        self._build_login_tab(notebook)
        self._build_register_tab(notebook)
        self._build_manage_tab(notebook)

    # ── LOGIN TAB ──────────────────────────────────────────────────────────
    def _build_login_tab(self, nb):
        tab = themed_frame(nb)
        nb.add(tab, text="🔓  Login")

        body = themed_frame(tab); body.pack(fill="both", expand=True, padx=10, pady=10)
        left = themed_frame(body); left.pack(side="left", fill="both", expand=True)
        right = card_frame(body, padx=16, pady=16); right.pack(side="right", fill="y", padx=(10,0))

        # Camera preview
        self._login_canvas = tk.Canvas(left, width=480, height=360, bg="#060f1e",
                                       highlightbackground=T["border"], highlightthickness=1)
        self._login_canvas.pack()
        self._login_canvas.create_text(240, 180, text="Enable preview to open camera",
                                       fill=T["sub"], font=(T["font"],11), tags="ph")

        preview_f = themed_frame(left); preview_f.pack(fill="x", pady=8)
        self._login_preview_var = tk.BooleanVar(value=False)
        tk.Checkbutton(preview_f, text=" Enable Camera Preview",
                       variable=self._login_preview_var,
                       font=(T["font"],10), fg=T["text"], bg=T["bg"],
                       activebackground=T["bg"], selectcolor=T["bg"],
                       command=self._toggle_preview).pack(side="left")

        # Username + buttons
        themed_label(right, "Username", 10, True, bg=T["card"]).pack(anchor="w")
        self._login_user_var = tk.StringVar()
        tk.Entry(right, textvariable=self._login_user_var,
                 font=(T["font"],11), bg=T["border"], fg=T["text"],
                 insertbackground=T["text"], relief="flat",
                 width=22).pack(fill="x", pady=(2,10))

        accent_button(right, "🔓  Authenticate", self._do_login, T["accent"]).pack(fill="x", pady=2)
        accent_button(right, "🔄  Refresh",      self._refresh_login, T["muted"]).pack(fill="x", pady=2)

        self._login_status = themed_label(right, "", 10, color=T["sub"], bg=T["card"],
                                          wraplength=200, justify="left")
        self._login_status.pack(pady=10, anchor="w")

        # Info
        themed_label(right, "How it works", 9, True, T["sub"], bg=T["card"]).pack(anchor="w", pady=(8,2))
        for line in ["• 21 MediaPipe landmarks → 63 values",
                     "• Wrist-relative normalization",
                     "• Cosine similarity ≥ 90% = access"]:
            themed_label(right, line, 8, color=T["muted"], bg=T["card"]).pack(anchor="w")

    def _toggle_preview(self):
        if self._login_preview_var.get():
            self._start_cam()
        else:
            self._stop_cam()
            self._login_canvas.delete("all")
            self._login_canvas.create_text(240,180, text="Enable preview to open camera",
                                           fill=T["sub"], font=(T["font"],11), tags="ph")

    def _refresh_login(self):
        if self._login_preview_var.get():
            self._stop_cam(); self._start_cam()

    def _do_login(self):
        user = self._login_user_var.get().strip()
        if not user:
            self._login_status.config(text="⚠ Enter a username.", fg=T["yellow"]); return
        if not user_exists(user):
            self._login_status.config(text="❌ User not found.", fg=T["red"]); return
        with self._lock:
            lm = self._lm
        if lm is None:
            self._login_status.config(text="🖐 No hand detected. Show your hand in preview.", fg=T["yellow"]); return
        ok, score = authenticate_user(user, lm)
        if ok:
            self.app.authenticated_user = user
            self._login_status.config(text=f"✅ Access granted!\nSimilarity: {score:.1%}", fg=T["green"])
            self.app.update_sidebar_user()
        else:
            self._login_status.config(text=f"❌ Access denied.\nSimilarity: {score:.1%} (need 90%)", fg=T["red"])

    # ── REGISTER TAB ───────────────────────────────────────────────────────
    def _build_register_tab(self, nb):
        tab = themed_frame(nb)
        nb.add(tab, text="📝  Register")

        body = themed_frame(tab); body.pack(fill="both", expand=True, padx=10, pady=10)
        left = themed_frame(body); left.pack(side="left", fill="both", expand=True)
        right = card_frame(body, padx=16, pady=16); right.pack(side="right", fill="y", padx=(10,0))

        self._reg_canvas = tk.Canvas(left, width=480, height=360, bg="#060f1e",
                                     highlightbackground=T["border"], highlightthickness=1)
        self._reg_canvas.pack()
        self._reg_canvas.create_text(240,180, text="Enable preview to open camera",
                                     fill=T["sub"], font=(T["font"],11), tags="ph")

        preview_f = themed_frame(left); preview_f.pack(fill="x", pady=8)
        self._reg_preview_var = tk.BooleanVar(value=False)
        tk.Checkbutton(preview_f, text=" Enable Camera Preview",
                       variable=self._reg_preview_var,
                       font=(T["font"],10), fg=T["text"], bg=T["bg"],
                       activebackground=T["bg"], selectcolor=T["bg"],
                       command=self._toggle_reg_preview).pack(side="left")

        # Controls
        themed_label(right, "Username", 10, True, bg=T["card"]).pack(anchor="w")
        self._reg_user_var = tk.StringVar()
        tk.Entry(right, textvariable=self._reg_user_var,
                 font=(T["font"],11), bg=T["border"], fg=T["text"],
                 insertbackground=T["text"], relief="flat",
                 width=22).pack(fill="x", pady=(2,10))

        # Progress
        self._reg_progress_lbl = themed_label(right, f"Samples: 0/{self.REQUIRED}", 10, color=T["sub"], bg=T["card"])
        self._reg_progress_lbl.pack(anchor="w")
        self._reg_bar = ttk.Progressbar(right, maximum=self.REQUIRED, value=0, length=200)
        self._reg_bar.pack(fill="x", pady=4)

        accent_button(right, "📸  Capture Sample", self._do_capture, T["blue"]).pack(fill="x", pady=2)
        accent_button(right, "💾  Register",       self._do_register, T["green"]).pack(fill="x", pady=2)
        accent_button(right, "🗑  Reset",          self._do_reset,    T["muted"]).pack(fill="x", pady=2)

        self._reg_status = themed_label(right, "", 10, color=T["sub"], bg=T["card"],
                                        wraplength=200, justify="left")
        self._reg_status.pack(pady=8, anchor="w")

    def _toggle_reg_preview(self):
        if self._reg_preview_var.get():
            self._start_cam()
        else:
            self._stop_cam()
            self._reg_canvas.delete("all")
            self._reg_canvas.create_text(240,180, text="Enable preview to open camera",
                                         fill=T["sub"], font=(T["font"],11), tags="ph")

    def _do_capture(self):
        with self._lock: lm = self._lm
        if lm is None:
            self._reg_status.config(text="⚠ No hand detected.", fg=T["yellow"]); return
        if len(self._reg_samples) < self.REQUIRED:
            self._reg_samples.append(lm)
            n = len(self._reg_samples)
            self._reg_progress_lbl.config(text=f"Samples: {n}/{self.REQUIRED}")
            self._reg_bar["value"] = n
            self._reg_status.config(text=f"✅ Sample {n}/{self.REQUIRED} captured!", fg=T["green"])

    def _do_register(self):
        user = self._reg_user_var.get().strip()
        if not user:
            self._reg_status.config(text="⚠ Enter a username.", fg=T["yellow"]); return
        if len(self._reg_samples) < self.REQUIRED:
            self._reg_status.config(text=f"⚠ Need {self.REQUIRED} samples, have {len(self._reg_samples)}.", fg=T["yellow"]); return
        ok = register_user(user, self._reg_samples)
        if ok:
            self._reg_status.config(text=f"🎉 '{user}' registered! Go to Login.", fg=T["green"])
            self._do_reset()
        else:
            self._reg_status.config(text="❌ Username already exists.", fg=T["red"])

    def _do_reset(self):
        self._reg_samples = []
        self._reg_progress_lbl.config(text=f"Samples: 0/{self.REQUIRED}")
        self._reg_bar["value"] = 0

    # ── MANAGE TAB ─────────────────────────────────────────────────────────
    def _build_manage_tab(self, nb):
        self._manage_tab_inner = themed_frame(nb)
        nb.add(self._manage_tab_inner, text="👥  Users")
        self._refresh_manage()

    def _refresh_manage(self):
        for w in self._manage_tab_inner.winfo_children():
            w.destroy()
        users = list_users()
        themed_label(self._manage_tab_inner, f"Registered Users ({len(users)})", 12, True, T["text"]).pack(anchor="w", padx=20, pady=12)
        if not users:
            themed_label(self._manage_tab_inner, "No users registered yet.", 10, color=T["sub"]).pack(padx=20)
            return
        for u in users:
            row = themed_frame(self._manage_tab_inner); row.pack(fill="x", padx=20, pady=2)
            card_frame(row, padx=12, pady=6).pack(side="left", expand=True, fill="x")
            themed_label(row, f"👤  {u}", 11, bg=T["bg"]).pack(side="left", padx=12)
            def _del(usr=u):
                delete_user(usr)
                if self.app.authenticated_user == usr:
                    self.app.authenticated_user = None
                    self.app.update_sidebar_user()
                self._refresh_manage()
            accent_button(row, "🗑", _del, T["red"], padx=8, pady=4).pack(side="right")

    # ── Camera helpers ─────────────────────────────────────────────────────
    def _start_cam(self):
        if self._cam_running: return
        self._stop.clear()
        self._cam_running = True
        threading.Thread(target=self._cam_loop, daemon=True).start()
        self._refresh_cam_display()

    def _stop_cam(self):
        self._stop.set()
        self._cam_running = False
        release_camera()

    def _cam_loop(self):
        while not self._stop.is_set():
            ret, frame = cam_read_frame()
            if not ret or frame is None: time.sleep(0.02); continue
            frame = cv2.flip(frame, 1)
            lm, annotated = get_raw_landmarks(frame)
            with self._lock:
                self._lm        = lm
                self._frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            time.sleep(0.001)
        self._frame_rgb = None

    def _refresh_cam_display(self):
        if not self._cam_running: return
        with self._lock: fr = self._frame_rgb
        if fr is not None:
            for canvas in [self._login_canvas, self._reg_canvas]:
                try:
                    img = Image.fromarray(fr).resize((480,360), Image.NEAREST)
                    photo = ImageTk.PhotoImage(img)
                    canvas._photo = photo
                    canvas.delete("all")
                    canvas.create_image(0,0, anchor="nw", image=photo)
                except Exception: pass
        self.after(33, self._refresh_cam_display)

    def on_show(self):
        self._refresh_manage()

    def on_hide(self):
        self._stop_cam()
        self._login_preview_var.set(False)
        self._reg_preview_var.set(False)


# ══════════════════════════════════════════════════════════════════════════
# SIGN TRANSLATION PAGE
# ══════════════════════════════════════════════════════════════════════════
class SignTranslationPage(Page):
    HOLD_NEEDED = 5

    def __init__(self, parent, app):
        super().__init__(parent, app)
        self._running    = False
        self._stop       = threading.Event()
        self._lock       = threading.Lock()
        self._frame_rgb  = None
        self._last_label = ""
        self._last_conf  = 0.0
        self._hold_label = None
        self._hold_count = 0
        self._build()

    def _build(self):
        # Header
        hdr = themed_frame(self); hdr.pack(fill="x", padx=24, pady=(16,0))
        themed_label(hdr, "🤟  Real-time Sign Translation", 18, True, T["accent"]).pack(side="left")
        sep(self).pack(fill="x", padx=24, pady=8)

        if not self.app.authenticated_user:
            self._build_auth_required()
            return
        self._build_main_ui()

    def _build_auth_required(self):
        card = card_frame(self, padx=30, pady=30)
        card.pack(padx=40, pady=40)
        themed_label(card, "🔐  Authentication Required", 14, True, T["yellow"], bg=T["card"]).pack()
        themed_label(card, "Login via Gesture Auth to use Sign Translation.", 10, color=T["sub"], bg=T["card"]).pack(pady=8)

    def _build_main_ui(self):
        body = themed_frame(self); body.pack(fill="both", expand=True, padx=24, pady=4)

        # LEFT: camera
        left = themed_frame(body); left.pack(side="left", fill="both", expand=True)
        self._cam_canvas = tk.Canvas(left, width=520, height=390, bg="#060f1e",
                                     highlightbackground=T["border"], highlightthickness=1)
        self._cam_canvas.pack()
        self._placeholder_text = self._cam_canvas.create_text(
            260,195, text="Press  ▶ Start  to begin", fill=T["sub"], font=(T["font"],13))

        btn_row = themed_frame(left); btn_row.pack(fill="x", pady=8)
        self._start_btn = accent_button(btn_row, "▶  Start Session", self._start, T["green"])
        self._start_btn.pack(side="left", padx=(0,6))
        self._stop_btn  = accent_button(btn_row, "⏹  Stop", self._stop_session, T["muted"])
        self._stop_btn.pack(side="left")
        self._stop_btn.config(state="disabled")

        # RIGHT: info panel
        right = themed_frame(body, width=280); right.pack(side="right", fill="y", padx=(12,0))
        right.pack_propagate(False)

        # Current letter
        self._letter_var = tk.StringVar(value="—")
        themed_label(right, "CURRENT GESTURE", 8, color=T["sub"]).pack(anchor="w")
        tk.Label(right, textvariable=self._letter_var,
                 font=(T["font"], 72, "bold"), fg=T["accent"], bg=T["bg"]).pack()

        # Confidence bar
        themed_label(right, "CONFIDENCE", 8, color=T["sub"]).pack(anchor="w", pady=(4,0))
        self._conf_var = tk.DoubleVar(value=0)
        ttk.Progressbar(right, variable=self._conf_var, maximum=100, length=240).pack(fill="x")
        self._conf_lbl = themed_label(right, "0%", 9, color=T["accent"])
        self._conf_lbl.pack(anchor="e")

        sep(right).pack(fill="x", pady=10)

        # Instructions / Info
        themed_label(right, "📝  HOW TO USE", 8, color=T["sub"]).pack(anchor="w")
        themed_label(right, "Hold your hand up to the camera. The system will detect A-Z ASL signs in real-time.", 10, True, T["text"], wraplength=260).pack(anchor="w", pady=4)
        themed_label(right, "Ensure your hand is steady and fully visible for the best accuracy.", 9, color=T["muted"], wraplength=260).pack(anchor="w", pady=4)

    def _start(self):
        if self._running: return
        self._stop.clear()
        self._running = True
        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal", bg=T["red"])
        self._cam_canvas.delete("all")
        threading.Thread(target=self._cam_loop, daemon=True).start()
        self._update_ui()

    def _stop_session(self):
        self._stop.set()
        self._running = False
        release_camera()
        if hasattr(self, '_start_btn'):
            self._start_btn.config(state="normal")
            self._stop_btn.config(state="disabled", bg=T["muted"])
            self._cam_canvas.delete("all")
            self._cam_canvas.create_text(260,195, text="Press ▶ Start to begin",
                                         fill=T["sub"], font=(T["font"],13))

    def _cam_loop(self):
        while not self._stop.is_set():
            ret, frame = cam_read_frame()
            if not ret or frame is None: time.sleep(0.01); continue
            frame = cv2.flip(frame, 1)
            label, conf, annotated, _, _ = predict_gesture(frame)

            with self._lock:
                self._frame_rgb  = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                if label:
                    self._last_label = label
                    self._last_conf  = conf
                    if conf >= 0.75:
                        if label == self._hold_label:
                            self._hold_count += 1
                        else:
                            self._hold_label = label
                            self._hold_count = 1
                    else:
                        self._hold_label = None
                        self._hold_count = 0
                else:
                    self._hold_label = None
                    self._hold_count = 0
            time.sleep(0.001)
        self._frame_rgb = None

    def _update_ui(self):
        if not self._running: return
        with self._lock:
            fr   = self._frame_rgb
            lbl  = self._last_label or "—"
            conf = self._last_conf

        if fr is not None:
            try:
                img   = Image.fromarray(fr).resize((520,390), Image.NEAREST)
                photo = ImageTk.PhotoImage(img)
                self._cam_canvas._photo = photo
                self._cam_canvas.delete("all")
                self._cam_canvas.create_image(0,0, anchor="nw", image=photo)
            except Exception as e:
                open("ui_debug.txt", "a").write(f"SignTrans image err: {e}\n")

        try:
            self._letter_var.set(lbl)
            pct = int(conf * 100)
            self._conf_var.set(pct)
            self._conf_lbl.config(text=f"{pct}%")
        except Exception as e:
            open("ui_debug.txt", "a").write(f"SignTrans label err: {e}\n")

        self.after(33, self._update_ui)

    def on_show(self): pass
    def on_hide(self): self._stop_session()


# ══════════════════════════════════════════════════════════════════════════
# GESTURE TO WORD PAGE
# ══════════════════════════════════════════════════════════════════════════
class GestureToWordPage(Page):
    """
    Matches LIVE hand gestures against reference images in gesture_to_word_dataset/
    using cosine similarity on MediaPipe landmarks.
    One gesture = one whole word (Hello, Please, Yes, No, Thanks, IloveYou).
    """
    HOLD_FRAMES    = 12                   # consecutive frames required before confirming
    DATASET_SUBDIR = "gesture_to_word_dataset"

    def __init__(self, parent, app):
        super().__init__(parent, app)
        self._running      = False
        self._stop         = threading.Event()
        self._lock         = threading.Lock()
        self._frame_rgb    = None
        self._current_word = ""           # best-match word label (live)
        self._current_conf = 0.0          # cosine similarity (live)
        self._hold_label   = None
        self._hold_count   = 0
        self._confirmed    = ""           # word locked in after HOLD_FRAMES
        self._dataset_dir  = os.path.join(BASE_DIR, self.DATASET_SUBDIR)
        # Pre-load reference landmarks (runs once, very fast)
        self._refs = load_word_gestures(self._dataset_dir)
        self._build()

    # ── UI construction ──────────────────────────────────────────────────────
    def _build(self):
        themed_label(self, "💬  Gesture To Word", 18, True, T["green"]).pack(
            anchor="w", padx=24, pady=(16,4))
        themed_label(
            self,
            "Make a whole-word hand gesture → word is detected instantly → Speak or Translate.",
            10, color=T["sub"]
        ).pack(anchor="w", padx=24)
        sep(self).pack(fill="x", padx=24, pady=8)

        if not self.app.authenticated_user:
            card = card_frame(self, padx=30, pady=30)
            card.pack(padx=40, pady=40)
            themed_label(card, "🔐  Login Required", 14, True, T["yellow"], bg=T["card"]).pack()
            themed_label(card, "Go to Gesture Auth to login first.",
                         10, color=T["sub"], bg=T["card"]).pack(pady=4)
            return

        # Dataset status banner
        n = len(self._refs)
        banner = card_frame(self, padx=16, pady=8)
        banner.pack(fill="x", padx=24, pady=(0,8))
        if n == 0:
            themed_label(banner, f"⚠️  No reference gestures loaded from '{self.DATASET_SUBDIR}/'.",
                         10, True, T["yellow"], bg=T["card"]).pack(anchor="w")
            themed_label(banner, "Make sure the folder contains .jpg images.",
                         9, color=T["sub"], bg=T["card"]).pack(anchor="w")
        else:
            words_str = "  •  ".join(get_loaded_words())
            themed_label(banner, f"✅  {n} gestures loaded:   {words_str}",
                         9, color=T["green"], bg=T["card"]).pack(anchor="w")

        body  = themed_frame(self); body.pack(fill="both", expand=True, padx=24, pady=4)
        left  = themed_frame(body); left.pack(side="left", fill="both", expand=True)
        right = themed_frame(body, width=290); right.pack(side="right", fill="y", padx=(12,0))
        right.pack_propagate(False)

        # ── Camera canvas ──────────────────────────────────────────────────
        self._cam_canvas = tk.Canvas(left, width=520, height=370, bg="#060f1e",
                                     highlightbackground=T["border"], highlightthickness=1)
        self._cam_canvas.pack()
        self._cam_canvas.create_text(260, 185, text="Press ▶ Start to begin",
                                     fill=T["sub"], font=(T["font"],13))

        # ── Similarity bar strip ───────────────────────────────────────────
        sim_card = card_frame(left, padx=10, pady=6)
        sim_card.pack(fill="x", pady=(4, 0))
        themed_label(sim_card, "SIMILARITY:", 8, color=T["sub"], bg=T["card"]).pack(
            side="left", padx=(0,8))
        self._sim_var = tk.DoubleVar(value=0)
        ttk.Progressbar(sim_card, variable=self._sim_var, maximum=100,
                        length=280).pack(side="left", fill="x", expand=True)
        self._sim_lbl = themed_label(sim_card, "0%", 8, color=T["green"], bg=T["card"])
        self._sim_lbl.pack(side="left", padx=(6,0))

        # ── Buttons ────────────────────────────────────────────────────────
        btn_row = themed_frame(left); btn_row.pack(fill="x", pady=6)
        self._start_btn = accent_button(btn_row, "▶  Start", self._start, T["green"])
        self._start_btn.pack(side="left", padx=(0,6))
        self._stop_btn = accent_button(btn_row, "⏹  Stop", self._stop_session, T["muted"])
        self._stop_btn.pack(side="left"); self._stop_btn.config(state="disabled")
        accent_button(btn_row, "🗑  Clear", self._clear, T["card"]).pack(side="left", padx=4)

        # ── Right panel ────────────────────────────────────────────────────
        # Live best-match label
        themed_label(right, "LIVE GESTURE", 8, color=T["sub"]).pack(anchor="w")
        self._live_var = tk.StringVar(value="—")
        tk.Label(right, textvariable=self._live_var,
                 font=(T["font"], 14, "bold"), fg="#2dd4bf", bg=T["bg"],
                 wraplength=270).pack(anchor="w", pady=(2,0))

        sep(right).pack(fill="x", pady=8)

        # Confirmed word (big)
        themed_label(right, "✨  CONFIRMED WORD", 8, color=T["sub"]).pack(anchor="w")
        self._word_var = tk.StringVar(value="—")
        tk.Label(right, textvariable=self._word_var,
                 font=(T["font"], 36, "bold"), fg=T["green"], bg=T["bg"],
                 wraplength=270).pack(anchor="w", pady=(2,6))

        self._speak_btn = accent_button(right, "🔊  Speak", self._speak, T["blue"])
        self._speak_btn.pack(fill="x", pady=2)

        sep(right).pack(fill="x", pady=8)

        # Translation
        lang_names = get_language_names()
        self._lang_var = tk.StringVar(value=lang_names[0] if lang_names else "Hindi")
        ttk.Combobox(right, textvariable=self._lang_var, values=lang_names,
                     state="readonly", width=24).pack(fill="x", pady=2)
        accent_button(right, "🌐  Translate", self._do_translate, T["blue"]).pack(fill="x", pady=2)
        self._trans_lbl = themed_label(right, "", 10, color=T["accent"], wraplength=270)
        self._trans_lbl.pack(anchor="w", pady=2)

        sep(right).pack(fill="x", pady=8)

        # Gesture reference list
        themed_label(right, "📸  GESTURE REFERENCE", 8, color=T["sub"]).pack(anchor="w", pady=(0,4))
        guide_frame = tk.Frame(right, bg=T["card"])
        guide_frame.pack(fill="both", expand=True)
        txt = tk.Text(guide_frame, width=28, height=8, bg=T["card"], fg=T["text"],
                      font=(T["font"], 9), bd=0, highlightthickness=0)
        scr = ttk.Scrollbar(guide_frame, command=txt.yview)
        txt.configure(yscrollcommand=scr.set)
        scr.pack(side="right", fill="y")
        txt.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        if self._refs:
            for word in sorted(self._refs.keys()):
                txt.insert("end", f"•  {word}\n")
        else:
            txt.insert("end", "No gestures loaded.\n")
        txt.configure(state="disabled")

    # ── Session control ──────────────────────────────────────────────────────
    def _start(self):
        if self._running: return
        with self._lock:
            self._confirmed    = ""
            self._current_word = ""
            self._hold_label   = None
            self._hold_count   = 0
        self._stop.clear(); self._running = True
        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal", bg=T["red"])
        self._cam_canvas.delete("all")
        threading.Thread(target=self._cam_loop, daemon=True).start()
        self._update_ui()

    def _stop_session(self):
        self._stop.set(); self._running = False; release_camera()
        if hasattr(self, '_start_btn'):
            self._start_btn.config(state="normal")
            self._stop_btn.config(state="disabled", bg=T["muted"])
            self._cam_canvas.delete("all")
            self._cam_canvas.create_text(260, 185, text="Press ▶ Start to begin",
                                         fill=T["sub"], font=(T["font"], 13))

    # ── Camera worker thread ─────────────────────────────────────────────────
    def _cam_loop(self):
        while not self._stop.is_set():
            ret, frame = cam_read_frame()
            if not ret or frame is None:
                time.sleep(0.01); continue

            frame = cv2.flip(frame, 1)
            # Cosine-similarity match against reference word gestures
            word, score, annotated = match_live_frame(frame)

            with self._lock:
                self._frame_rgb    = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                self._current_word = word  or ""
                self._current_conf = score

                if word:
                    if word == self._hold_label:
                        self._hold_count += 1
                    else:
                        self._hold_label = word
                        self._hold_count = 1

                    if self._hold_count >= self.HOLD_FRAMES:
                        if self._confirmed != word:
                            self._confirmed = word
                            try:
                                self._trans_lbl.config(text="")
                            except Exception:
                                pass
                        self._hold_count = 0
                else:
                    self._hold_label = None
                    self._hold_count = 0

            time.sleep(0.001)
        self._frame_rgb = None

    # ── UI refresh (~30fps, main thread) ────────────────────────────────────
    def _update_ui(self):
        if not self._running: return
        with self._lock:
            fr        = self._frame_rgb
            live_word = self._current_word
            score     = self._current_conf
            confirmed = self._confirmed

        if fr is not None:
            try:
                img   = Image.fromarray(fr).resize((520, 370), Image.NEAREST)
                photo = ImageTk.PhotoImage(img)
                self._cam_canvas._photo = photo
                self._cam_canvas.delete("all")
                self._cam_canvas.create_image(0, 0, anchor="nw", image=photo)
            except Exception as e:
                open("ui_debug.txt", "a").write(f"G2W img err: {e}\n")

        try:
            pct = int(score * 100)
            self._sim_var.set(pct)
            self._sim_lbl.config(text=f"{pct}%")
            self._live_var.set(f"▶  {live_word}" if live_word else "No gesture detected")
            self._word_var.set(confirmed if confirmed else "—")
        except Exception as e:
            open("ui_debug.txt", "a").write(f"G2W ui err: {e}\n")

        self.after(33, self._update_ui)

    # ── Button actions ───────────────────────────────────────────────────────
    def _clear(self):
        with self._lock:
            self._confirmed    = ""
            self._current_word = ""
            self._hold_label   = None
            self._hold_count   = 0
        try:
            self._trans_lbl.config(text="")
        except Exception:
            pass

    def _speak(self):
        with self._lock:
            word = self._confirmed
        if not word: return
        lang_name = self._lang_var.get()
        gtts_code = get_gtts_code(lang_name)
        lang_code = SUPPORTED_LANGUAGES.get(lang_name, "en")

        def _do_speak():
            try:
                if lang_name == "English" or lang_code == "en":
                    speak_text(word, "en")
                elif gtts_code:
                    # Always translate fresh to the currently selected language, then speak
                    translated = translate_text(word, lang_code)
                    speak_text(translated, gtts_code)
                else:
                    # Language not supported by gTTS — still translate but speak in English
                    translated = translate_text(word, lang_code)
                    speak_text(translated, "en")
            except Exception:
                speak_text(word, "en")

        threading.Thread(target=_do_speak, daemon=True).start()

    def _do_translate(self):
        with self._lock:
            word = self._confirmed
        if not word:
            self._trans_lbl.config(text="No word confirmed yet.", fg=T["yellow"]); return
        lang = self._lang_var.get()
        code = SUPPORTED_LANGUAGES.get(lang, "en")
        try:
            result = translate_text(word, code)
            self._trans_lbl.config(text=f"[{lang}]  {result}", fg=T["green"])
        except Exception:
            self._trans_lbl.config(text="Translation error.", fg=T["red"])

    def on_hide(self): self._stop_session()



# ══════════════════════════════════════════════════════════════════════════
# GESTURE MOUSE PAGE (launcher)
# ══════════════════════════════════════════════════════════════════════════
class GestureMousePage(Page):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        self._proc = None
        self._build()

    def _build_auth_required(self, feature_name="this feature", accent=None):
        ac = accent or T["yellow"]
        card = card_frame(self, padx=30, pady=30)
        card.pack(padx=40, pady=40)
        themed_label(card, "🔐  Authentication Required", 14, True, T["yellow"], bg=T["card"]).pack()
        themed_label(card, f"Login via Gesture Auth to use {feature_name}.",
                     10, color=T["sub"], bg=T["card"]).pack(pady=8)
        themed_label(card, "Go to Gesture Auth → register your hand → login.",
                     9, color=T["muted"], bg=T["card"]).pack()

    def _build(self):
        themed_label(self, "🖱️  Gesture Controlled Mouse", 18, True, T["pink"]).pack(anchor="w", padx=24, pady=(16,4))
        themed_label(self, "Launches a native window running at 30fps — real-time, zero lag.", 10, color=T["sub"]).pack(anchor="w", padx=24)
        sep(self).pack(fill="x", padx=24, pady=12)

        if not self.app.authenticated_user:
            self._build_auth_required("Gesture Mouse", T["pink"])
            return

        card = card_frame(self, padx=40, pady=40)
        card.pack(padx=60, pady=20, fill="x")

        themed_label(card, "🚀  Native 30fps Desktop Window", 16, True, T["pink"], bg=T["card"]).pack()
        themed_label(card, "Clicking Launch opens a Tkinter window running camera + mouse control at true 30fps,\n"
                     "completely independent of this app.", 10, color=T["sub"], bg=T["card"]).pack(pady=(8,16))

        self._status_lbl = themed_label(card, "○  Not running", 11, color=T["muted"], bg=T["card"])
        self._status_lbl.pack(pady=6)

        bf = themed_frame(card, bg=T["card"]); bf.pack()
        self._launch_btn = accent_button(bf, "🚀  Launch Gesture Mouse", self._launch, T["pink"])
        self._launch_btn.pack(side="left", padx=8)
        self._kill_btn   = accent_button(bf, "⏹  Stop",                 self._kill,   T["muted"])
        self._kill_btn.pack(side="left")

        sep(self).pack(fill="x", padx=24, pady=12)

        # Gesture guide
        themed_label(self, "🤌  Gesture Guide", 13, True, T["text"]).pack(anchor="w", padx=24, pady=(0,8))
        guide_frame = themed_frame(self); guide_frame.pack(anchor="w", padx=24)
        for icon, name, desc in [
            ("☝️","Move Cursor",   "Index fingertip tracks cursor"),
            ("✌️","Left Click",    "Pinch index + middle together"),
            ("👌","Right Click",   "Pinch thumb + index together"),
            ("🖐️","Scroll Up",    "All 4 fingers extended"),
            ("🤙","Scroll Down",  "Only pinky extended"),
        ]:
            row = themed_frame(guide_frame); row.pack(fill="x", pady=2)
            themed_label(row, f"{icon}  {name}", 10, True, T["text"]).pack(side="left", padx=(0,12))
            themed_label(row, desc, 9, color=T["sub"]).pack(side="left")

    def _is_running(self):
        return self._proc is not None and self._proc.poll() is None

    def _launch(self):
        if self._is_running(): return
        script = os.path.join(BASE_DIR, "gesture_mouse_standalone.py")
        flags  = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
        self._proc = subprocess.Popen([sys.executable, script], creationflags=flags)
        self._status_lbl.config(text="⬤  Running (Tkinter window opened)", fg=T["green"])
        self._launch_btn.config(state="disabled")

    def _kill(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None
        self._status_lbl.config(text="○  Not running", fg=T["muted"])
        self._launch_btn.config(state="normal")

    def on_hide(self): pass  # keep mouse running even if you navigate away


class VolumeControlPage(Page):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        self._proc = None
        self._build()

    def _build_auth_required(self):
        card = card_frame(self, padx=30, pady=30)
        card.pack(padx=40, pady=40)
        themed_label(card, "🔐  Authentication Required", 14, True, T["yellow"], bg=T["card"]).pack()
        themed_label(card, "Login via Gesture Auth to use Volume Control.",
                     10, color=T["sub"], bg=T["card"]).pack(pady=8)

    def _build(self):
        themed_label(self, "🔊  Volume Control", 18, True, T["yellow"]).pack(anchor="w", padx=24, pady=(16,4))
        themed_label(self, "Adjust system volume intuitively using a two-finger pinch gesture.", 10, color=T["sub"]).pack(anchor="w", padx=24)
        sep(self).pack(fill="x", padx=24, pady=12)

        if not self.app.authenticated_user:
            self._build_auth_required()
            return

        card = card_frame(self, padx=40, pady=40)
        card.pack(padx=60, pady=20, fill="x")

        themed_label(card, "🚀  Launch Volume Control", 16, True, T["yellow"], bg=T["card"]).pack()
        themed_label(card, "Clicking Launch opens a background Tkinter window running camera control at true 30fps,\n"
                     "completely independent of this main app window.", 10, color=T["sub"], bg=T["card"]).pack(pady=(8,16))

        self._status_lbl = themed_label(card, "○  Not running", 11, color=T["muted"], bg=T["card"])
        self._status_lbl.pack(pady=6)

        bf = themed_frame(card, bg=T["card"]); bf.pack()
        self._launch_btn = accent_button(bf, "▶  Start Volume Control", self._launch, T["yellow"])
        self._launch_btn.pack(side="left", padx=8)
        self._kill_btn   = accent_button(bf, "⏹  Stop",                 self._kill,   T["muted"])
        self._kill_btn.pack(side="left")

        sep(self).pack(fill="x", padx=24, pady=12)

        themed_label(self, "🤌  Gesture Guide", 13, True, T["text"]).pack(anchor="w", padx=24, pady=(0,8))
        guide_frame = themed_frame(self); guide_frame.pack(anchor="w", padx=24)
        for icon, name, desc in [
            ("🤏","Pinch Thumb+Idx", "Adjust Volume"),
            ("↔️","Move Apart/Closer", "Increase/Decrease")
        ]:
            row = themed_frame(guide_frame); row.pack(fill="x", pady=2)
            themed_label(row, f"{icon}  {name}", 10, True, T["text"]).pack(side="left", padx=(0,12))
            themed_label(row, desc, 9, color=T["sub"]).pack(side="left")

    def _is_running(self):
        return self._proc is not None and self._proc.poll() is None

    def _launch(self):
        if self._is_running(): return
        script = os.path.join(BASE_DIR, "gesture_mouse_standalone.py")
        flags  = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
        self._proc = subprocess.Popen([sys.executable, script, "--mode", "VOLUME"], creationflags=flags)
        self._status_lbl.config(text="⬤  Running (Tkinter window opened)", fg=T["green"])
        self._launch_btn.config(state="disabled")

    def _kill(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None
        self._status_lbl.config(text="○  Not running", fg=T["muted"])
        self._launch_btn.config(state="normal")

    def on_hide(self): pass


class BrightnessControlPage(Page):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        self._proc = None
        self._build()

    def _build_auth_required(self):
        card = card_frame(self, padx=30, pady=30)
        card.pack(padx=40, pady=40)
        themed_label(card, "🔐  Authentication Required", 14, True, T["yellow"], bg=T["card"]).pack()
        themed_label(card, "Login via Gesture Auth to use Brightness Control.",
                     10, color=T["sub"], bg=T["card"]).pack(pady=8)

    def _build(self):
        themed_label(self, "☀️  Brightness Control", 18, True, T["red"]).pack(anchor="w", padx=24, pady=(16,4))
        themed_label(self, "Change display brightness with an easy two-finger distance metric.", 10, color=T["sub"]).pack(anchor="w", padx=24)
        sep(self).pack(fill="x", padx=24, pady=12)

        if not self.app.authenticated_user:
            self._build_auth_required()
            return

        card = card_frame(self, padx=40, pady=40)
        card.pack(padx=60, pady=20, fill="x")

        themed_label(card, "🚀  Launch Brightness Control", 16, True, T["red"], bg=T["card"]).pack()
        themed_label(card, "Clicking Launch opens a background Tkinter window running camera control at true 30fps,\n"
                     "completely independent of this main app window.", 10, color=T["sub"], bg=T["card"]).pack(pady=(8,16))

        self._status_lbl = themed_label(card, "○  Not running", 11, color=T["muted"], bg=T["card"])
        self._status_lbl.pack(pady=6)

        bf = themed_frame(card, bg=T["card"]); bf.pack()
        self._launch_btn = accent_button(bf, "▶  Start Brightness Control", self._launch, T["red"])
        self._launch_btn.pack(side="left", padx=8)
        self._kill_btn   = accent_button(bf, "⏹  Stop",                 self._kill,   T["muted"])
        self._kill_btn.pack(side="left")

        sep(self).pack(fill="x", padx=24, pady=12)

        themed_label(self, "🤌  Gesture Guide", 13, True, T["text"]).pack(anchor="w", padx=24, pady=(0,8))
        guide_frame = themed_frame(self); guide_frame.pack(anchor="w", padx=24)
        for icon, name, desc in [
            ("🤏","Pinch Thumb+Idx", "Adjust Brightness"),
            ("↔️","Move Apart/Closer", "Increase/Decrease")
        ]:
            row = themed_frame(guide_frame); row.pack(fill="x", pady=2)
            themed_label(row, f"{icon}  {name}", 10, True, T["text"]).pack(side="left", padx=(0,12))
            themed_label(row, desc, 9, color=T["sub"]).pack(side="left")

    def _is_running(self):
        return self._proc is not None and self._proc.poll() is None

    def _launch(self):
        if self._is_running(): return
        script = os.path.join(BASE_DIR, "gesture_mouse_standalone.py")
        flags  = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
        self._proc = subprocess.Popen([sys.executable, script, "--mode", "BRIGHTNESS"], creationflags=flags)
        self._status_lbl.config(text="⬤  Running (Tkinter window opened)", fg=T["green"])
        self._launch_btn.config(state="disabled")

    def _kill(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None
        self._status_lbl.config(text="○  Not running", fg=T["muted"])
        self._launch_btn.config(state="normal")

    def on_hide(self): pass


# ══════════════════════════════════════════════════════════════════════════
# WHITEBOARD PAGE
# ══════════════════════════════════════════════════════════════════════════
class WhiteboardPage(Page):
    """Launcher for gesture_mouse_standalone.py WHITEBOARD mode."""

    TEAL = "#2dd4bf"

    def __init__(self, parent, app):
        super().__init__(parent, app)
        self._proc = None
        self._build()

    def _build_auth_required(self):
        card = card_frame(self, padx=30, pady=30)
        card.pack(padx=40, pady=40)
        themed_label(card, "🔐  Authentication Required", 14, True, T["yellow"], bg=T["card"]).pack()
        themed_label(card, "Login via Gesture Auth to use the Whiteboard.",
                     10, color=T["sub"], bg=T["card"]).pack(pady=8)

    def _build(self):
        themed_label(self, "🎨  Gesture Controlled Whiteboard", 18, True, self.TEAL).pack(
            anchor="w", padx=24, pady=(16, 4))
        themed_label(self, "Draw, erase and pick colors on a virtual canvas — all hands-free at 30fps.",
                     10, color=T["sub"]).pack(anchor="w", padx=24)
        sep(self).pack(fill="x", padx=24, pady=12)

        if not self.app.authenticated_user:
            self._build_auth_required()
            return

        # ── Launch card ───────────────────────────────────────────────────
        card = card_frame(self, padx=40, pady=30)
        card.pack(padx=60, pady=10, fill="x")

        themed_label(card, "🚀  Native 30fps Drawing Window", 16, True, self.TEAL, bg=T["card"]).pack()
        themed_label(
            card,
            "Click Launch to open a dedicated Tkinter window running your webcam\n"
            "at 30fps. Use hand gestures to draw on the virtual canvas.",
            10, color=T["sub"], bg=T["card"]
        ).pack(pady=(8, 16))

        self._status_lbl = themed_label(card, "○  Not running", 11, color=T["muted"], bg=T["card"])
        self._status_lbl.pack(pady=6)

        bf = themed_frame(card, bg=T["card"]); bf.pack()
        self._launch_btn = accent_button(bf, "🎨  Launch Whiteboard", self._launch, self.TEAL)
        self._launch_btn.pack(side="left", padx=8)
        accent_button(bf, "⏹  Stop", self._kill, T["muted"]).pack(side="left")

        sep(self).pack(fill="x", padx=24, pady=12)

        # ── Gesture guide ─────────────────────────────────────────────────
        themed_label(self, "🤌  Gesture Guide", 13, True, T["text"]).pack(
            anchor="w", padx=24, pady=(0, 8))

        guide_outer = themed_frame(self); guide_outer.pack(anchor="w", padx=24, fill="x")
        gestures = [
            ("☝️",  "Draw",           "Index finger only — traces a line on the canvas",          self.TEAL),
            ("✌️",  "Select Color",   "Index + Middle up — hover over the color palette to pick",  "#818cf8"),
            ("🤏",  "Erase",          "Pinch Thumb + Index — erases under your fingertip",          "#f87171"),
            ("✋",  "Clear Board",    "All 4 fingers extended — wipes the entire canvas clean",     "#fbbf24"),
        ]
        for icon, name, desc, color in gestures:
            row = card_frame(guide_outer, padx=14, pady=10)
            row.pack(fill="x", pady=4)
            # Icon badge
            tk.Label(row, text=icon, font=(T["font"], 22),
                     bg=T["card"], fg=color).pack(side="left", padx=(0, 12))
            # Text block
            col = themed_frame(row, bg=T["card"])
            col.pack(side="left", fill="x", expand=True)
            themed_label(col, name, 11, True, color, bg=T["card"]).pack(anchor="w")
            themed_label(col, desc, 9, color=T["sub"], bg=T["card"],
                         wraplength=600, justify="left").pack(anchor="w")

        # ── Color palette preview ─────────────────────────────────────────
        sep(self).pack(fill="x", padx=24, pady=12)
        themed_label(self, "🎨  Color Palette (select with ✌️ hover)", 11, True, T["text"]).pack(
            anchor="w", padx=24, pady=(0, 8))
        pal_row = themed_frame(self); pal_row.pack(anchor="w", padx=24)
        palette = [
            ("#ff00ff", "Magenta/Pink"),
            ("#0000ff", "Blue"),
            ("#00ff00", "Green"),
            ("#00ffff", "Cyan"),
        ]
        for hex_col, name in palette:
            box = tk.Frame(pal_row, bg=hex_col, width=60, height=40,
                           highlightbackground=T["border"], highlightthickness=1)
            box.pack(side="left", padx=4)
            box.pack_propagate(False)
            themed_label(box, name, 7, color="#000000" if hex_col == "#00ffff" else "white",
                         bg=hex_col).pack(expand=True)

    def _is_running(self):
        return self._proc is not None and self._proc.poll() is None

    def _launch(self):
        if self._is_running(): return
        script = os.path.join(BASE_DIR, "gesture_mouse_standalone.py")
        flags  = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
        self._proc = subprocess.Popen(
            [sys.executable, script, "--mode", "WHITEBOARD"], creationflags=flags)
        self._status_lbl.config(text="⬤  Running — Whiteboard window opened", fg=self.TEAL)
        self._launch_btn.config(state="disabled")

    def _kill(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None
        self._status_lbl.config(text="○  Not running", fg=T["muted"])
        self._launch_btn.config(state="normal")

    def on_hide(self): pass   # keep drawing even when navigating elsewhere


# ══════════════════════════════════════════════════════════════════════════
# GESTURE 3D VIEWER PAGE
# ══════════════════════════════════════════════════════════════════════════
class Gesture3DViewerPage(Page):
    """Launcher for gesture_3d_viewer_standalone.py — 23 procedural shapes."""

    CYAN = "#06b6d4"

    def __init__(self, parent, app):
        super().__init__(parent, app)
        self._proc = None
        self._build()

    def _build_auth_required(self):
        card = card_frame(self, padx=30, pady=30)
        card.pack(padx=40, pady=40)
        themed_label(card, "🔐  Authentication Required", 14, True, T["yellow"], bg=T["card"]).pack()
        themed_label(card, "Login via Gesture Auth to use the 3D Viewer.",
                     10, color=T["sub"], bg=T["card"]).pack(pady=8)

    def _build(self):
        themed_label(self, "🧊  Gesture 3D Shape Viewer", 18, True, self.CYAN).pack(
            anchor="w", padx=24, pady=(16, 4))
        themed_label(self, "Rotate, zoom & browse 23 procedural 3D shapes using hand gestures — powered by OpenGL.",
                     10, color=T["sub"]).pack(anchor="w", padx=24)
        sep(self).pack(fill="x", padx=24, pady=12)

        if not self.app.authenticated_user:
            self._build_auth_required()
            return

        # Launch card
        card = card_frame(self, padx=40, pady=30)
        card.pack(padx=60, pady=10, fill="x")

        themed_label(card, "🚀  OpenGL 3D Viewer · 23 Shapes", 16, True, self.CYAN, bg=T["card"]).pack()
        themed_label(
            card,
            "Click Launch to open a dedicated OpenGL window with a live camera overlay.\n"
            "Use hand gestures to rotate, zoom, and switch between 23 different 3D shapes.",
            10, color=T["sub"], bg=T["card"]
        ).pack(pady=(8, 16))

        self._status_lbl = themed_label(card, "○  Not running", 11, color=T["muted"], bg=T["card"])
        self._status_lbl.pack(pady=6)

        bf = themed_frame(card, bg=T["card"]); bf.pack()
        self._launch_btn = accent_button(bf, "🧊  Launch 3D Viewer", self._launch, self.CYAN)
        self._launch_btn.pack(side="left", padx=8)
        accent_button(bf, "⏹  Stop", self._kill, T["muted"]).pack(side="left")

        sep(self).pack(fill="x", padx=24, pady=12)

        # Gesture guide
        themed_label(self, "🤌  Gesture Guide", 13, True, T["text"]).pack(
            anchor="w", padx=24, pady=(0, 8))
        guide_outer = themed_frame(self); guide_outer.pack(anchor="w", padx=24, fill="x")
        gestures = [
            ("☝️",  "Rotate",        "Move index finger to control yaw & pitch",        self.CYAN),
            ("🤏",  "Zoom",          "Pinch thumb + index to zoom in/out",              "#818cf8"),
            ("👍",  "Next Shape",    "Thumbs up (thumb only extended) — hold ~0.5s",    "#34d399"),
            ("🤙",  "Prev Shape",    "Pinky only extended — hold ~0.5s",                "#f87171"),
            ("✊",  "Freeze",         "Make a fist to freeze current rotation",          "#fbbf24"),
            ("✋",  "Reset View",     "Hold open palm for 1 sec to reset rotation & zoom", "#f472b6"),
            ("⌨️",  "Keyboard",      "n=next  p=prev  r=reset  q=quit (in camera window)", "#94a3b8"),
        ]
        for icon, name, desc, color in gestures:
            row = card_frame(guide_outer, padx=14, pady=8)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=icon, font=(T["font"], 18),
                     bg=T["card"], fg=color).pack(side="left", padx=(0, 12))
            col = themed_frame(row, bg=T["card"])
            col.pack(side="left", fill="x", expand=True)
            themed_label(col, name, 10, True, color, bg=T["card"]).pack(anchor="w")
            themed_label(col, desc, 9, color=T["sub"], bg=T["card"],
                         wraplength=600, justify="left").pack(anchor="w")

        sep(self).pack(fill="x", padx=24, pady=12)

        # Shape list
        themed_label(self, "🧊  Available Shapes (23)", 11, True, T["text"]).pack(
            anchor="w", padx=24, pady=(0, 6))
        shape_row = themed_frame(self); shape_row.pack(anchor="w", padx=24, fill="x")
        shapes = ["Cube", "Sphere", "Cylinder", "Cone", "Torus", "Pyramid",
                  "Tetrahedron", "Octahedron", "Diamond", "Star", "HexPrism",
                  "Arrow", "Cross", "Hemisphere", "Capsule", "Wedge",
                  "LowPolyHuman", "Heart", "Gear", "Spring", "PentaPrism",
                  "TriPrism", "RectPrism"]
        row_f = themed_frame(shape_row)
        row_f.pack(fill="x")
        for i, s in enumerate(shapes):
            if i > 0 and i % 8 == 0:
                row_f = themed_frame(shape_row)
                row_f.pack(fill="x", pady=2)
            lbl = tk.Label(row_f, text=f" {s} ", font=(T["font"], 8, "bold"),
                           fg=self.CYAN, bg=T["card"],
                           highlightbackground=T["border"], highlightthickness=1)
            lbl.pack(side="left", padx=2, pady=2)

    def _is_running(self):
        return self._proc is not None and self._proc.poll() is None

    def _launch(self):
        if self._is_running(): return
        script = os.path.join(BASE_DIR, "gesture_3d_viewer_standalone.py")
        flags  = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
        self._proc = subprocess.Popen([sys.executable, script], creationflags=flags)
        self._status_lbl.config(text="⬤  Running — OpenGL viewer opened", fg=self.CYAN)
        self._launch_btn.config(state="disabled")

    def _kill(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None
        self._status_lbl.config(text="○  Not running", fg=T["muted"])
        self._launch_btn.config(state="normal")

    def on_hide(self): pass  # keep viewer running


# ══════════════════════════════════════════════════════════════════════════
# ABOUT PAGE
# ══════════════════════════════════════════════════════════════════════════
class AboutPage(Page):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        self._build()

    def _build(self):
        canvas = tk.Canvas(self, bg=T["bg"], highlightthickness=0)
        scroll = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = themed_frame(canvas)
        canvas.create_window((0,0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        themed_label(inner, "📖  About SignBridge", 20, True, T["accent"]).pack(anchor="w", padx=40, pady=(24,4))
        sep(inner).pack(fill="x", padx=40, pady=8)

        sections = [
            ("🎯  Project Overview",
             "SignBridge is a real-time American Sign Language (ASL) recognition system developed as a BTP project.\n"
             "It enables gesture-based biometric login, live sign-to-text translation, phrase detection with TTS,\n"
             "and gesture-controlled mouse — all running locally with zero cloud dependency."),
            ("🔬  Tech Stack",
             "• Python 3.11\n• MediaPipe  (21-landmark hand detection)\n• TensorFlow / Keras  (1D-CNN classification)\n"
             "• Tkinter  (native desktop UI, 30fps real-time)\n• OpenCV  (camera capture)\n"
             "• PyAutoGUI  (mouse/keyboard control)\n• googletrans  (15+ language translation)\n"
             "• pyttsx3  (text-to-speech)"),
            ("💡  Novelty & Advantages",
             "• Privacy-first: no images stored, only 63 numbers per user\n"
             "• Real-time at 30fps via native Tkinter (not web)\n"
             "• Wrist-relative landmark normalization for position-invariant recognition\n"
             "• 5-frame temporal smoothing eliminates false positives\n"
             "• Combined system: auth + translation + phrase detection + mouse in one app"),
            ("🚀  Future Extensions",
             "• Dynamic gesture recognition (motion-based signs)\n• Sentence-level language models\n"
             "• Mobile deployment  • Two-hand support\n• Sign-to-Speech (direct TTS without spelling)"),
        ]
        for title, content in sections:
            card = card_frame(inner, padx=20, pady=16)
            card.pack(fill="x", padx=40, pady=6)
            themed_label(card, title, 12, True, T["accent"], bg=T["card"]).pack(anchor="w")
            themed_label(card, content, 9, color=T["sub"], bg=T["card"],
                         justify="left", wraplength=780).pack(anchor="w", pady=(6,0))

        themed_label(inner, "SignBridge · NSUT BTP 2025–26 · ECE Department",
                     9, color=T["muted"]).pack(pady=24)


# ══════════════════════════════════════════════════════════════════════════
# VOXEL EDITOR PAGE (launcher)
# ══════════════════════════════════════════════════════════════════════════
class VoxelEditorPage(Page):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        self._proc = None
        self._build()

    def _build(self):
        themed_label(self, "🧊  Gesture Voxel Editor", 18, True, "#00ffd5").pack(anchor="w", padx=24, pady=(16,4))
        themed_label(self, "Build 3D voxel structures using hand gestures — Minecraft meets AR.", 10, color=T["sub"]).pack(anchor="w", padx=24)
        sep(self).pack(fill="x", padx=24, pady=12)

        if not self.app.authenticated_user:
            card = card_frame(self, padx=30, pady=30)
            card.pack(padx=40, pady=40)
            themed_label(card, "🔐  Authentication Required", 14, True, T["yellow"], bg=T["card"]).pack()
            themed_label(card, "Login via Gesture Auth to use Voxel Editor.",
                         10, color=T["sub"], bg=T["card"]).pack(pady=8)
            return

        card = card_frame(self, padx=40, pady=40)
        card.pack(padx=60, pady=20, fill="x")
        themed_label(card, "🚀  Native 3D Voxel Window", 16, True, "#00ffd5", bg=T["card"]).pack()
        themed_label(card, "Opens a Tkinter window with live webcam + 3D voxel viewport.\n"
                     "Use hand gestures to place, erase, rotate, and build.",
                     10, color=T["sub"], bg=T["card"]).pack(pady=(8,16))

        self._status_lbl = themed_label(card, "○  Not running", 11, color=T["muted"], bg=T["card"])
        self._status_lbl.pack(pady=6)

        bf = themed_frame(card, bg=T["card"]); bf.pack()
        self._launch_btn = accent_button(bf, "🚀  Launch Voxel Editor", self._launch, "#00ffd5")
        self._launch_btn.pack(side="left", padx=8)
        self._kill_btn = accent_button(bf, "⏹  Stop", self._kill, T["muted"])
        self._kill_btn.pack(side="left")

        sep(self).pack(fill="x", padx=24, pady=12)

        themed_label(self, "🤌  Gesture Guide", 13, True, T["text"]).pack(anchor="w", padx=24, pady=(0,8))
        guide = themed_frame(self); guide.pack(anchor="w", padx=24)
        for icon, name, desc in [
            ("👌","Pinch",       "Place voxel at cursor"),
            ("✊","Fist",        "Orbit camera around scene"),
            ("🖐","Open Palm",   "Pan camera view"),
            ("✌️","Peace Sign",  "Cycle material / color"),
            ("👆","Swipe",       "Erase voxels"),
            ("👍","Thumb Up",    "Save structure to file"),
            ("👎","Thumb Down",  "Undo last action"),
            ("🤲","Two Hands",   "Rotate 3D view"),
        ]:
            row = themed_frame(guide); row.pack(fill="x", pady=2)
            themed_label(row, f"{icon}  {name}", 10, True, T["text"]).pack(side="left", padx=(0,12))
            themed_label(row, desc, 9, color=T["sub"]).pack(side="left")

        sep(self).pack(fill="x", padx=24, pady=12)
        themed_label(self, "⌨️  Keyboard Shortcuts (in voxel window)", 11, True, T["text"]).pack(anchor="w", padx=24, pady=(0,6))
        keys_frame = themed_frame(self); keys_frame.pack(anchor="w", padx=24)
        for key, action in [("R","Reset camera"),("C","Clear all"),("M","Cycle material"),
                            ("+/-","Change depth layer"),("S","Save"),("L","Load"),("Z","Undo"),("Q","Quit")]:
            row = themed_frame(keys_frame); row.pack(fill="x", pady=1)
            themed_label(row, key, 9, True, "#00ffd5", width=6).pack(side="left")
            themed_label(row, action, 9, color=T["sub"]).pack(side="left")

    def _is_running(self):
        return self._proc is not None and self._proc.poll() is None

    def _launch(self):
        if self._is_running(): return
        script = os.path.join(BASE_DIR, "voxel_editor_standalone.py")
        flags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
        self._proc = subprocess.Popen([sys.executable, script], creationflags=flags)
        self._status_lbl.config(text="⬤  Running (Voxel Editor window opened)", fg=T["green"])
        self._launch_btn.config(state="disabled")

    def _kill(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None
        self._status_lbl.config(text="○  Not running", fg=T["muted"])
        self._launch_btn.config(state="normal")

    def on_hide(self): pass


# ══════════════════════════════════════════════════════════════════════════
# HEADSETLESS VR PAGE (launcher)
# ══════════════════════════════════════════════════════════════════════════
class HeadlessVRPage(Page):
    """Launches the NXS-2087 cyberpunk-city headsetless VR experience.
    Starts headsetless_vr/serve.py as a local HTTP server on port 8000
    and opens the browser — webcam face-tracking drives off-axis parallax.
    """
    VR_COLOR  = "#ff2d92"
    VR_DIR    = os.path.join(BASE_DIR, "headsetless_vr")
    SERVE_PY  = os.path.join(BASE_DIR, "headsetless_vr", "serve.py")
    PORT      = 8000

    def __init__(self, parent, app):
        super().__init__(parent, app)
        self._proc = None
        self._build()

    def _build(self):
        # ── Header ──────────────────────────────────────────────────────────
        themed_label(self, "🥽  Headsetless VR", 18, True, self.VR_COLOR).pack(
            anchor="w", padx=24, pady=(16, 4))
        themed_label(
            self,
            "NXS-2087 · Cyberpunk synthwave city — head-tracked parallax via webcam, no headset needed.",
            10, color=T["sub"]
        ).pack(anchor="w", padx=24)
        sep(self).pack(fill="x", padx=24, pady=12)

        # ── Launch card ─────────────────────────────────────────────────────
        card = card_frame(self, padx=40, pady=32)
        card.pack(padx=60, pady=20, fill="x")

        themed_label(card, "🚀  Launch VR Experience", 16, True, self.VR_COLOR,
                     bg=T["card"]).pack()
        themed_label(
            card,
            "Starts a local web server (localhost:8000) and opens your browser.\n"
            "Sit ~50 cm from the camera — tilt/shift your head to look around the city.",
            10, color=T["sub"], bg=T["card"]
        ).pack(pady=(8, 16))

        self._status_lbl = themed_label(card, "○  Server not running", 11,
                                        color=T["muted"], bg=T["card"])
        self._status_lbl.pack(pady=6)

        bf = themed_frame(card, bg=T["card"]); bf.pack(pady=(4, 0))
        self._launch_btn = accent_button(bf, "🚀  Launch VR", self._launch, self.VR_COLOR)
        self._launch_btn.pack(side="left", padx=8)
        self._open_btn = accent_button(bf, "🌐  Open Browser", self._open_browser, T["blue"])
        self._open_btn.pack(side="left", padx=8)
        self._kill_btn = accent_button(bf, "⏹  Stop Server", self._kill, T["muted"])
        self._kill_btn.pack(side="left")

        sep(self).pack(fill="x", padx=24, pady=12)

        # ── How it works ────────────────────────────────────────────────────
        themed_label(self, "🧠  How It Works", 13, True, T["text"]).pack(
            anchor="w", padx=24, pady=(0, 8))
        guide = themed_frame(self); guide.pack(anchor="w", padx=24)
        for icon, step, desc in [
            ("📡", "Webcam Face Tracking",
             "MediaPipe detects your eye midpoint every frame — runs in the browser"),
            ("📐", "Off-Axis Projection",
             "Three.js recalculates frustum based on head position for true parallax"),
            ("🏙️", "Synthwave City Scene",
             "Procedural cyberpunk city with neon grid, mountains, stars & streaks"),
            ("🖱️", "Mouse Fallback",
             "Move cursor around the screen to look — works without any webcam"),
            ("🔁", "Recalibrate",
             "Press R or click RECALIBRATE to reset head position to neutral"),
        ]:
            row = themed_frame(guide); row.pack(fill="x", pady=3)
            themed_label(row, f"{icon}  {step}", 10, True, T["text"]).pack(
                side="left", padx=(0, 12))
            themed_label(row, desc, 9, color=T["sub"]).pack(side="left")

        sep(self).pack(fill="x", padx=24, pady=12)

        # ── Controls reference ───────────────────────────────────────────────
        themed_label(self, "⌨️  Browser Controls", 11, True, T["text"]).pack(
            anchor="w", padx=24, pady=(0, 6))
        keys_frame = themed_frame(self); keys_frame.pack(anchor="w", padx=24)
        for key, action in [
            ("R",         "Recalibrate head position"),
            ("M",         "Toggle webcam / mouse mode"),
            ("P",         "Toggle webcam preview"),
            ("LOOK slider","Adjust head-tilt sensitivity"),
            ("SPEED slider","Change drive / scroll speed"),
        ]:
            row = themed_frame(keys_frame); row.pack(fill="x", pady=1)
            themed_label(row, key, 9, True, self.VR_COLOR, width=14).pack(side="left")
            themed_label(row, action, 9, color=T["sub"]).pack(side="left")

        sep(self).pack(fill="x", padx=24, pady=8)
        themed_label(
            self,
            "⚠  Requires Google Chrome / Firefox and a working webcam for full experience.",
            9, color=T["yellow"]
        ).pack(anchor="w", padx=24)

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _is_running(self):
        return self._proc is not None and self._proc.poll() is None

    def _launch(self):
        """Start serve.py (opens browser automatically via webbrowser.open)."""
        if self._is_running():
            # Server already up — just open browser
            self._open_browser()
            return
        flags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
        self._proc = subprocess.Popen(
            [sys.executable, self.SERVE_PY, str(self.PORT)],
            cwd=self.VR_DIR,
            creationflags=flags,
        )
        self._status_lbl.config(
            text=f"⬤  Server running on http://localhost:{self.PORT}/",
            fg=self.VR_COLOR
        )
        self._launch_btn.config(state="disabled")

    def _open_browser(self):
        import webbrowser
        webbrowser.open(f"http://localhost:{self.PORT}/")

    def _kill(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None
        self._status_lbl.config(text="○  Server not running", fg=T["muted"])
        self._launch_btn.config(state="normal")

    def on_hide(self): pass   # keep server running in background


# ══════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════
class SignBridgeApp(tk.Tk):
    # (label, page_name, accent_color)
    PAGE_ORDER = [
        ("🏠  Home",               "Home",               "#f1f5f9"),
        ("🔐  Gesture Auth",       "Gesture Auth",       "#818cf8"),
        ("🤟  Sign Translation",   "Sign Translation",   "#38bdf8"),
        ("💬  Gesture To Word",    "Gesture To Word",    "#34d399"),
        ("🖱️  Mouse Control",     "Gesture Mouse",      "#f472b6"),
        ("🔊  Volume Control",    "Volume Control",     "#fbbf24"),
        ("☀️  Brightness Control", "Brightness Control", "#f87171"),
        ("🎨  Whiteboard",         "Whiteboard",         "#2dd4bf"),
        ("🧊  3D Viewer",          "3D Viewer",          "#06b6d4"),
        ("🔷  Voxel Editor",       "Voxel Editor",       "#00ffd5"),
        ("🥽  Headsetless VR",     "Headsetless VR",     "#ff2d92"),
        ("📖  About",              "About",              "#94a3b8"),
    ]

    def __init__(self):
        super().__init__()
        self.authenticated_user = None
        self.title("SignBridge — ASL Recognition Suite")
        self.geometry("1280x760")
        self.minsize(1100, 680)
        self.configure(bg=T["bg"])

        # Try to set a nice icon
        try: self.iconbitmap(default="")
        except Exception: pass

        self._setup_styles()
        self._build_layout()
        self._init_pages()
        self.show_page("Home")

    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook",       background=T["bg"],    borderwidth=0)
        style.configure("TNotebook.Tab",   background=T["card"],  foreground=T["sub"],
                        padding=[12,6],    font=(T["font"],10))
        style.map("TNotebook.Tab",         background=[("selected", T["sidebar"])],
                                           foreground=[("selected", T["accent"])])
        style.configure("TProgressbar",    background=T["accent"], troughcolor=T["border"],
                        borderwidth=0,     thickness=8)
        style.configure("TCombobox",       fieldbackground=T["border"], background=T["border"],
                        foreground=T["text"], arrowcolor=T["text"])
        style.configure("TScrollbar",      background=T["border"], troughcolor=T["bg"],
                        arrowcolor=T["sub"])
        style.configure("TScale",          background=T["bg"],    troughcolor=T["border"])

    def _build_layout(self):
        # Sidebar
        self.sidebar = tk.Frame(self, bg=T["sidebar"], width=220)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Logo
        logo_f = tk.Frame(self.sidebar, bg=T["sidebar"])
        logo_f.pack(fill="x", padx=16, pady=20)
        tk.Label(logo_f, text="🤟", font=(T["font"],28), bg=T["sidebar"]).pack()
        tk.Label(logo_f, text="SignBridge", font=(T["font"],14,"bold"),
                 fg=T["accent"], bg=T["sidebar"]).pack()
        tk.Label(logo_f, text="ASL Recognition Suite", font=(T["font"],8),
                 fg=T["muted"], bg=T["sidebar"]).pack()

        tk.Frame(self.sidebar, bg=T["border"], height=1).pack(fill="x", padx=12, pady=8)

        # Nav buttons — store accent color per page for active highlight
        self._nav_btns   = {}
        self._nav_colors = {}
        for label, page_name, page_color in self.PAGE_ORDER:
            btn = tk.Button(self.sidebar, text=label,
                            font=(T["font"],10), anchor="w",
                            fg=T["sub"], bg=T["sidebar"],
                            activeforeground=page_color,
                            activebackground=T["bg"],
                            relief="flat", cursor="hand2",
                            padx=16, pady=10,
                            command=lambda p=page_name: self.show_page(p))
            btn.pack(fill="x")
            self._nav_btns[page_name]   = btn
            self._nav_colors[page_name] = page_color

        tk.Frame(self.sidebar, bg=T["border"], height=1).pack(fill="x", padx=12, pady=8)

        # User / model status
        self._user_lbl = tk.Label(self.sidebar, text="Not logged in",
                                  font=(T["font"],9), fg=T["muted"], bg=T["sidebar"],
                                  wraplength=190)
        self._user_lbl.pack(padx=12, pady=4, anchor="w")

        model_text = "● Model Ready" if is_model_ready() else "● Model not found"
        model_color = T["green"] if is_model_ready() else T["red"]
        tk.Label(self.sidebar, text=model_text, font=(T["font"],8,"bold"),
                 fg=model_color, bg=T["sidebar"]).pack(padx=12, pady=2, anchor="w")

        # Content area
        self.content = tk.Frame(self, bg=T["bg"])
        self.content.pack(side="right", fill="both", expand=True)

    def _init_pages(self):
        self._pages = {
            "Home":               HomePage(self.content, self),
            "Gesture Auth":       AuthPage(self.content, self),
            "Sign Translation":   SignTranslationPage(self.content, self),
            "Gesture To Word":    GestureToWordPage(self.content, self),
            "Gesture Mouse":      GestureMousePage(self.content, self),
            "Volume Control":     VolumeControlPage(self.content, self),
            "Brightness Control": BrightnessControlPage(self.content, self),
            "Whiteboard":         WhiteboardPage(self.content, self),
            "3D Viewer":          Gesture3DViewerPage(self.content, self),
            "Voxel Editor":       VoxelEditorPage(self.content, self),
            "Headsetless VR":     HeadlessVRPage(self.content, self),
            "About":              AboutPage(self.content, self),
        }
        self._current_page = None

    def show_page(self, name):
        if self._current_page:
            self._pages[self._current_page].on_hide()
            self._pages[self._current_page].pack_forget()
            # Reset nav button to dim style
            btn = self._nav_btns.get(self._current_page)
            if btn: btn.config(fg=T["sub"], bg=T["sidebar"], font=(T["font"], 10))

        self._current_page = name
        page = self._pages[name]
        page.pack(fill="both", expand=True)
        page.on_show()

        # Highlight active nav button with the page's accent color
        btn = self._nav_btns.get(name)
        col = self._nav_colors.get(name, T["accent"])
        if btn: btn.config(fg=col, bg=T["bg"], font=(T["font"], 10, "bold"))

    def update_sidebar_user(self):
        _AUTH_PAGES = {
            "Sign Translation":   SignTranslationPage,
            "Gesture To Word":    GestureToWordPage,
            "Gesture Mouse":      GestureMousePage,
            "Volume Control":     VolumeControlPage,
            "Brightness Control": BrightnessControlPage,
            "Whiteboard":         WhiteboardPage,
            "3D Viewer":          Gesture3DViewerPage,
            "Voxel Editor":       VoxelEditorPage,
        }
        if self.authenticated_user:
            self._user_lbl.config(text=f"✅  {self.authenticated_user}", fg=T["green"])
        else:
            self._user_lbl.config(text="Not logged in", fg=T["muted"])

        # Rebuild all auth-gated pages so they show/hide correctly
        for key, cls in _AUTH_PAGES.items():
            old = self._pages.get(key)
            if old:
                old.destroy()
            self._pages[key] = cls(self.content, self)
        # Re-show current page if it was one of the rebuilt ones
        if self._current_page in _AUTH_PAGES:
            self._pages[self._current_page].pack(fill="both", expand=True)


# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = SignBridgeApp()
    app.mainloop()
