"""
voxel_editor_standalone.py  – Gesture Voxel Editor (compact rewrite)
Gestures:  PINCH=Place(one-shot)  FIST+drag=Rotate  PEACE=Erase  THUMB_DOWN=Undo
Keys:  R=Reset  C=Clear  Z=Undo  S=Save  +/-=Depth  Q=Quit
"""
import tkinter as tk
from tkinter import filedialog
import threading, time, os, sys, cv2, numpy as np
from PIL import Image, ImageTk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from voxel_editor.hand_tracker    import HandTracker
from voxel_editor.gesture_recognizer import GestureRecognizer
from voxel_editor.voxel_engine    import VoxelEngine
from voxel_editor.renderer        import VoxelRenderer

# ── theme ─────────────────────────────────────────────────────────────────
BG,CARD,BORDER = "#040d1a","#0d1b2e","#1e293b"
CYAN,ACC,TEXT,SUB,MUTED = "#00ffd5","#818cf8","#f1f5f9","#64748b","#334155"
FONT, MONO = "Segoe UI", "Consolas"
SAVE_DIR = os.path.join(BASE_DIR, "saves")

# Color palette (name → RGB tuple for voxels)
COLORS = {
    "Cyan":  (0, 255, 220), "Blue":  (0, 150, 255),
    "Pink":  (255, 0, 200), "Green": (0, 255, 100),
    "Gold":  (255, 215, 0), "White": (230, 230, 255),
    "Red":   (255, 80, 80), "Orange":(255, 160, 0),
}
COLOR_HEX = {
    "Cyan":"#00ffdc","Blue":"#0096ff","Pink":"#ff00c8",
    "Green":"#00ff64","Gold":"#ffd700","White":"#e6e6ff",
    "Red":"#ff5050","Orange":"#ffa000",
}

# View presets (yaw, pitch, dist)
VIEWS = {
    "Iso":   (0.6,  0.35, 24),
    "Front": (0.0,  0.0,  24),
    "Top":   (0.0,  1.55, 24),
    "Side":  (1.57, 0.0,  24),
    "Back":  (3.14, 0.0,  24),
}


class VoxelEditorApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🧊 Voxel Editor")
        self.root.geometry("1360x780")
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        self.tracker  = HandTracker(max_hands=2)
        self.gesture  = GestureRecognizer()
        self.engine   = VoxelEngine()
        self.renderer = VoxelRenderer()

        # State
        self._stop    = threading.Event()
        self._running = False
        self._lock    = threading.Lock()
        self._frame   = None
        self._gname   = "NONE"
        self._gconf   = 0.0
        self._fps     = 0.0
        self._depth   = 4
        self._cur_color = "Cyan"

        # Placement one-shot control
        self._prev_gesture   = "NONE"
        self._last_place_t   = 0.0
        self._place_cooldown = 0.3   # seconds (controlled by slider)

        # Gesture rotation inertia
        self._prev_px = None
        self._prev_py = None
        self._rot_vx  = 0.0
        self._rot_vy  = 0.0
        # Mouse orbit inertia
        self._orb_vx  = 0.0
        self._orb_vy  = 0.0

        self._build_ui()
        self.root.bind("<Key>", self._on_key)
        self._place_starter()
        self._render_loop()

    # ── starter ───────────────────────────────────────────────────────────
    def _place_starter(self):
        mid = 8
        for dx in range(-2, 3):
            self.engine.add_voxel(mid+dx, 0, mid, record_undo=False)
        for dz in range(-2, 3):
            self.engine.add_voxel(mid, 0, mid+dz, record_undo=False)
        for dy in range(1, 4):
            self.engine.add_voxel(mid, dy, mid, record_undo=False)

    # ── UI ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # LEFT panel
        left = tk.Frame(self.root, bg=BG, width=390)
        left.pack(side="left", fill="y", padx=(10,0), pady=10)
        left.pack_propagate(False)

        tk.Label(left, text="🧊 Voxel Editor", font=(FONT,15,"bold"),
                 fg=CYAN, bg=BG).pack(anchor="w")
        tk.Label(left, text="Gesture-Controlled 3D Builder", font=(FONT,8),
                 fg=SUB, bg=BG).pack(anchor="w")

        # Camera feed
        self.cam_cv = tk.Canvas(left, width=375, height=280, bg="#060f1e",
                                highlightbackground=BORDER, highlightthickness=1)
        self.cam_cv.pack(pady=(6,4))
        self.cam_cv.create_text(187,140, text="Press START", fill=SUB, font=(FONT,10))

        # Start/Stop
        bf = tk.Frame(left, bg=BG); bf.pack(fill="x", pady=2)
        self.btn_start = tk.Button(bf, text="▶ START", font=(FONT,9,"bold"),
            fg="white", bg="#10b981", relief="flat", cursor="hand2",
            command=self._start_cam)
        self.btn_start.pack(side="left", padx=(0,6))
        self.btn_stop  = tk.Button(bf, text="⏹ STOP", font=(FONT,9,"bold"),
            fg="white", bg=MUTED, relief="flat", cursor="hand2",
            command=self._stop_cam, state="disabled")
        self.btn_stop.pack(side="left")

        # HUD
        hud = tk.Frame(left, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        hud.pack(fill="x", pady=(6,4))
        self._hud = {}
        for k, lbl in [("gesture","Gesture"),("mode","Mode"),
                        ("conf","Confidence"),("voxels","Voxels"),("fps","FPS")]:
            row = tk.Frame(hud, bg=CARD); row.pack(fill="x", padx=8, pady=1)
            tk.Label(row, text=f"{lbl}:", font=(MONO,8), fg=SUB, bg=CARD,
                     width=11, anchor="w").pack(side="left")
            v = tk.Label(row, text="—", font=(MONO,8,"bold"), fg=CYAN, bg=CARD, anchor="w")
            v.pack(side="left"); self._hud[k] = v

        # Sensitivity slider
        sf = tk.Frame(left, bg=BG); sf.pack(fill="x", pady=(4,2))
        tk.Label(sf, text="Place Cooldown (s):", font=(FONT,8), fg=SUB, bg=BG).pack(side="left")
        self._sens_var = tk.DoubleVar(value=0.3)
        tk.Scale(sf, from_=0.1, to=1.5, resolution=0.05, orient="horizontal",
                 variable=self._sens_var, bg=CARD, fg=CYAN, troughcolor=BORDER,
                 highlightthickness=0, length=160,
                 command=lambda v: setattr(self, "_place_cooldown", float(v))
                 ).pack(side="left", padx=4)

        # Gesture guide (compact)
        tk.Frame(left, bg=BORDER, height=1).pack(fill="x", pady=4)
        tk.Label(left, text="GESTURES", font=(FONT,8,"bold"), fg=SUB, bg=BG).pack(anchor="w")
        for g,d in [("👌 Pinch","Place voxel (one shot)"),
                    ("✊ Fist+drag","Rotate structure"),
                    ("✌ Peace","Erase at cursor"),
                    ("👎 Thumb Down","Undo")]:
            r = tk.Frame(left, bg=BG); r.pack(fill="x")
            tk.Label(r, text=g, font=(FONT,8), fg=TEXT, bg=BG, width=16, anchor="w").pack(side="left")
            tk.Label(r, text=d, font=(FONT,8), fg=MUTED, bg=BG).pack(side="left")

        # RIGHT panel
        right = tk.Frame(self.root, bg=BG)
        right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        tk.Label(right, text="3D VIEWPORT", font=(FONT,10,"bold"),
                 fg=ACC, bg=BG).pack(anchor="w")

        # Toolbar row 1: actions
        tb1 = tk.Frame(right, bg=BG); tb1.pack(fill="x", pady=(2,1))
        for txt,cmd,c in [("↩ Undo",self._undo,ACC),
                           ("💾 Save",self._save,"#10b981"),
                           ("📂 Load",self._load,"#38bdf8"),
                           ("🗑 Clear",self._clear,"#f87171"),
                           ("🔄 Reset Cam",self._reset_cam,MUTED)]:
            tk.Button(tb1, text=txt, font=(FONT,8,"bold"), fg="white", bg=c,
                      relief="flat", cursor="hand2", command=cmd,
                      padx=5, pady=2).pack(side="left", padx=2)

        # Toolbar row 2: view presets + depth
        tb2 = tk.Frame(right, bg=BG); tb2.pack(fill="x", pady=(0,2))
        tk.Label(tb2, text="View:", font=(FONT,8), fg=SUB, bg=BG).pack(side="left", padx=(0,4))
        for name,(y,p,d) in VIEWS.items():
            tk.Button(tb2, text=name, font=(FONT,8), fg=TEXT, bg=CARD,
                      relief="flat", cursor="hand2", padx=4, pady=1,
                      command=lambda yy=y,pp=p,dd=d: self.renderer.set_view(yy,pp,dd)
                      ).pack(side="left", padx=1)

        tk.Label(tb2, text="  Depth:", font=(FONT,8), fg=SUB, bg=BG).pack(side="left", padx=(8,2))
        tk.Button(tb2, text="-", font=(FONT,9,"bold"), fg=TEXT, bg=CARD,
                  relief="flat", command=lambda: self._chg_depth(-1)).pack(side="left")
        self._dlbl = tk.Label(tb2, text=f"Z={self._depth}", font=(MONO,8,"bold"),
                               fg=CYAN, bg=BG)
        self._dlbl.pack(side="left", padx=4)
        tk.Button(tb2, text="+", font=(FONT,9,"bold"), fg=TEXT, bg=CARD,
                  relief="flat", command=lambda: self._chg_depth(1)).pack(side="left")

        # Color palette row
        tb3 = tk.Frame(right, bg=BG); tb3.pack(fill="x", pady=(0,3))
        tk.Label(tb3, text="Color:", font=(FONT,8), fg=SUB, bg=BG).pack(side="left", padx=(0,4))
        self._col_btns = {}
        for name, hx in COLOR_HEX.items():
            b = tk.Button(tb3, text=name, font=(FONT,7,"bold"), fg="white",
                          bg=hx, relief="flat", cursor="hand2", padx=3, pady=1,
                          command=lambda n=name: self._set_color(n))
            b.pack(side="left", padx=1)
            self._col_btns[name] = b
        self._update_color_btn()

        # 3D canvas
        self.viewport = tk.Canvas(right, bg="#020810",
                                  highlightbackground=BORDER, highlightthickness=1)
        self.viewport.pack(fill="both", expand=True)
        self.viewport.bind("<B1-Motion>",    self._m_orbit)
        self.viewport.bind("<ButtonPress-1>",self._m_press)
        self.viewport.bind("<B3-Motion>",    self._m_rotate_struct)
        self.viewport.bind("<ButtonPress-3>",self._m_press3)
        self.viewport.bind("<MouseWheel>",   self._m_zoom)
        self._mlast  = None
        self._mlast3 = None

    # ── color ─────────────────────────────────────────────────────────────
    def _set_color(self, name):
        self._cur_color = name
        self.engine.current_material = name
        # patch MATERIALS so engine uses this color
        self.engine.MATERIALS[name] = COLORS[name]
        self._update_color_btn()

    def _update_color_btn(self):
        for n, b in self._col_btns.items():
            b.config(relief="sunken" if n == self._cur_color else "flat")

    # ── camera / start-stop ───────────────────────────────────────────────
    def _start_cam(self):
        if self._running: return
        self._stop.clear(); self._running = True
        self.tracker = HandTracker(max_hands=2)   # always fresh instance
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal", bg="#f87171")
        threading.Thread(target=self._cam_loop, daemon=True).start()
        self._cam_ui()

    def _stop_cam(self):
        self._stop.set(); self._running = False
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled", bg=MUTED)

    def _cam_loop(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        t0 = time.time(); frames = 0; fails = 0

        try:
            while not self._stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    fails += 1
                    if fails > 30:   # camera truly gone — stop gracefully
                        break
                    time.sleep(0.03); continue
                fails = 0
                frame = cv2.flip(frame, 1)

                try:
                    res   = self.tracker.process(frame)
                    frame = self.tracker.draw_landmarks(frame, res)
                    lm1   = self.tracker.get_landmarks(res, 0)
                    lm2   = self.tracker.get_landmarks(res, 1)
                    nH    = self.tracker.hand_count(res)
                    fst   = self.tracker.finger_states(lm1)
                    pd    = self.tracker.pinch_distance(lm1)
                    g, c  = self.gesture.recognize(lm1, fst, pd, nH, lm2)
                except Exception:
                    g, c, lm1 = "NONE", 0.0, None

                cur = None
                if lm1 is not None:
                    nx, ny = lm1[8][0], lm1[8][1]
                    cur = self.engine.get_grid_pos_from_cursor(nx, ny, self._depth)

                self._process(g, lm1, cur)

                frames += 1
                if time.time() - t0 >= 1.0:
                    self._fps = frames / (time.time() - t0)
                    frames = 0; t0 = time.time()

                with self._lock:
                    self._frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self._gname, self._gconf = g, c
        finally:
            cap.release()
            try: self.tracker.release()
            except Exception: pass
            self._running = False
            self.root.after(0, lambda: (
                self.btn_start.config(state="normal"),
                self.btn_stop.config(state="disabled", bg=MUTED)
            ))

    def _process(self, g, lm, cur):
        """Act on gesture — PEACE=Place  PINCH=Erase  FIST=Rotate  THUMB_DOWN=Undo."""
        now = time.time()

        # ── Place: ✌ Victory/Peace sign ──────────────────────────────────
        if g == "PEACE":
            if cur and now - self._last_place_t >= self._place_cooldown:
                self.engine.add_voxel(*cur)
                self._last_place_t = now

        # ── Erase: Pinch gesture ──────────────────────────────────────────
        elif g == "PINCH" and cur:
            self.engine.erase_radius(*cur, radius=1)

        # ── Undo ──────────────────────────────────────────────────────────
        elif g == "THUMB_DOWN" and self._prev_gesture != "THUMB_DOWN":
            self.engine.undo()

        # ── Rotate structure with FIST drag ───────────────────────────────
        elif g == "FIST" and lm is not None:
            px, py = lm[0][0], lm[0][1]
            if self._prev_px is not None:
                dx = px - self._prev_px
                dy = py - self._prev_py
                DEAD = 0.004
                vx_target = (-dx * 4.0) if abs(dx) > DEAD else 0
                vy_target = ( dy * 3.0) if abs(dy) > DEAD else 0
                self._rot_vx = self._rot_vx * 0.35 + vx_target * 0.65
                self._rot_vy = self._rot_vy * 0.35 + vy_target * 0.65
                self.renderer.rotate_struct(self._rot_vx, self._rot_vy)
            self._prev_px, self._prev_py = px, py
        else:
            # Smooth inertia coast after releasing fist
            if abs(self._rot_vx) > 0.0005 or abs(self._rot_vy) > 0.0005:
                self._rot_vx *= 0.88
                self._rot_vy *= 0.88
                self.renderer.rotate_struct(self._rot_vx, self._rot_vy)
            else:
                self._rot_vx = self._rot_vy = 0.0
            self._prev_px = self._prev_py = None

        self.engine.set_ghost(cur)
        self._prev_gesture = g

    def _cam_ui(self):
        if not self._running: return
        with self._lock:
            fr = self._frame; g = self._gname; c = self._gconf

        if fr is not None:
            try:
                img   = Image.fromarray(fr).resize((375, 280), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.cam_cv._photo = photo
                self.cam_cv.delete("all")
                self.cam_cv.create_image(0, 0, anchor="nw", image=photo)
            except Exception:
                pass

        self._hud["gesture"].config(text=g)
        self._hud["mode"].config(text=self.gesture.get_mode_label(g))
        self._hud["conf"].config(text=f"{c:.0%}")
        self._hud["voxels"].config(text=str(self.engine.count()))
        self._hud["fps"].config(text=f"{self._fps:.1f}")
        self.root.after(33, self._cam_ui)

    # ── 3D render loop ─────────────────────────────────────────────────────
    def _render_loop(self):
        w = self.viewport.winfo_width()
        h = self.viewport.winfo_height()
        if w > 50 and h > 50:
            self.renderer.cw, self.renderer.ch = w, h
            self.renderer.render(self.viewport, self.engine)
            self.viewport.delete("ov")
            self.viewport.create_text(
                10, 10, anchor="nw",
                text=f"Voxels:{self.engine.count()}  Color:{self._cur_color}  Z={self._depth}  "
                     f"[LMB=orbit cam  RMB=rotate struct  Scroll=zoom]",
                fill=CYAN, font=(MONO, 8), tags="ov")
        self.root.after(16, self._render_loop)

    # ── mouse controls ────────────────────────────────────────────────────
    def _m_press(self, e):  self._mlast  = (e.x, e.y)
    def _m_press3(self, e): self._mlast3 = (e.x, e.y)

    def _m_orbit(self, e):
        """Left-drag → orbit camera with inertia."""
        if self._mlast:
            dx = (e.x - self._mlast[0]) * 0.006
            dy = (e.y - self._mlast[1]) * 0.006
            self._orb_vx = self._orb_vx * 0.3 + dx * 0.7
            self._orb_vy = self._orb_vy * 0.3 + dy * 0.7
            self.renderer.cam_yaw   += self._orb_vx
            self.renderer.cam_pitch  = max(-1.4, min(1.4,
                                      self.renderer.cam_pitch + self._orb_vy))
        self._mlast = (e.x, e.y)

    def _m_rotate_struct(self, e):
        """Right-drag → rotate structure."""
        if self._mlast3:
            dx = (e.x - self._mlast3[0]) * 0.010
            dy = (e.y - self._mlast3[1]) * 0.010
            self.renderer.rotate_struct(dx, dy)
        self._mlast3 = (e.x, e.y)

    def _m_zoom(self, e):
        self.renderer.cam_dist = max(8, min(40,
            self.renderer.cam_dist - e.delta/120.0))

    # ── keyboard ──────────────────────────────────────────────────────────
    def _on_key(self, e):
        k = e.keysym.lower()
        if k == "r":   self._reset_cam()
        elif k == "c": self._clear()
        elif k == "z": self._undo()
        elif k == "s": self._save()
        elif k in ("plus","equal"): self._chg_depth(1)
        elif k == "minus":          self._chg_depth(-1)
        elif k in ("q","escape"):   self._close()

    # ── actions ───────────────────────────────────────────────────────────
    def _chg_depth(self, d):
        self._depth = max(0, min(15, self._depth + d))
        self._dlbl.config(text=f"Z={self._depth}")

    def _clear(self):   self.engine.clear_all()
    def _undo(self):    self.engine.undo()
    def _reset_cam(self):
        self.renderer.cam_yaw, self.renderer.cam_pitch, self.renderer.cam_dist = 0.6, 0.35, 24.0
        self.renderer.struct_yaw = self.renderer.struct_pitch = 0.0

    def _save(self):
        os.makedirs(SAVE_DIR, exist_ok=True)
        self.engine.save(os.path.join(SAVE_DIR, "voxel_save.json"))

    def _load(self):
        p = filedialog.askopenfilename(initialdir=SAVE_DIR,
            filetypes=[("JSON","*.json")])
        if p: self.engine.load(p)

    def _close(self):
        self._stop.set(); self._running = False
        time.sleep(0.1); self.root.destroy()

    def run(self): self.root.mainloop()


if __name__ == "__main__":
    VoxelEditorApp().run()
