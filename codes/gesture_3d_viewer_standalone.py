"""
gesture_3d_viewer_standalone.py — Gesture-Controlled 3D Shape Viewer
Standalone OpenGL + MediaPipe viewer with 20+ procedural shapes.
Gestures:
  - Index finger  → rotate (yaw/pitch)
  - Pinch         → zoom
  - Thumbs up 👍  → next shape
  - Pinky only 🤙  → previous shape
  - Fist          → freeze
  - Open palm     → reset view
Keyboard (in camera window):
  - n = next, p = prev, r = reset, q = quit
"""
import threading, time, math, sys, os, ctypes
import cv2, numpy as np
import mediapipe as mp
from collections import deque

from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from OpenGL.GLUT import *
from pyrr import Matrix44, Vector3

# Add project root so we can import shapes_3d
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.shapes_3d import ALL_SHAPES

W, H = 800, 600

# ── Shaders ───────────────────────────────────────────────────────────────
VS = """
#version 330 core
layout(location=0) in vec3 pos;
layout(location=1) in vec3 norm;
uniform mat4 model, view, proj;
uniform float pointSize;
out vec3 fPos, fNorm;
void main(){
    fPos  = vec3(model * vec4(pos,1));
    fNorm = mat3(transpose(inverse(model))) * norm;
    gl_Position = proj * view * vec4(fPos,1);
    gl_PointSize = pointSize;
}
"""
FS = """
#version 330 core
in vec3 fPos, fNorm;
out vec4 color;
uniform vec3 lightPos, viewPos, objColor, lightColor;
uniform int drawMode;  // 0=solid, 1=wireframe, 2=points
void main(){
    if(drawMode == 2){
        // Glowing dot effect
        vec2 c = gl_PointCoord - 0.5;
        float dist = length(c);
        if(dist > 0.5) discard;
        float glow = 1.0 - smoothstep(0.0, 0.5, dist);
        color = vec4(objColor * glow * 1.5, glow);
        return;
    }
    if(drawMode == 1){
        color = vec4(objColor, 1.0);
        return;
    }
    // Phong lighting
    vec3 amb = 0.15 * lightColor;
    vec3 n   = normalize(fNorm);
    vec3 ld  = normalize(lightPos - fPos);
    float diff = max(dot(n, ld), 0.0);
    vec3 vd  = normalize(viewPos - fPos);
    vec3 rf  = reflect(-ld, n);
    float spec = pow(max(dot(vd, rf), 0.0), 64.0);
    vec3 result = (amb + diff * 0.7 + spec * 0.5) * objColor * lightColor;
    color = vec4(result, 1.0);
}
"""

# ── Hand tracker ──────────────────────────────────────────────────────────
class HandTracker:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=1,
            model_complexity=1, min_detection_confidence=0.6,
            min_tracking_confidence=0.6)
        self.hist = deque(maxlen=5)
        self.fsz = (640, 480)

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        h, w = frame.shape[:2]
        self.fsz = (w, h)
        lm = None
        if res.multi_hand_landmarks:
            pts = []
            for l in res.multi_hand_landmarks[0].landmark:
                pts.append([l.x*w, l.y*h, l.z*w])
            lm = np.array(pts, dtype=np.float32)
        self.hist.append(lm)
        valid = [x for x in self.hist if x is not None]
        return np.mean(np.stack(valid), axis=0) if valid else None

# ── Gesture interpreter ──────────────────────────────────────────────────────────
class GestureInterp:
    def __init__(self, fsz=(640,480)):
        self.fsz = fsz
        self.open_since = None
        self.fist_since = None
        self.yaw = self.pitch = 0.0
        self.zoom = 1.0
        self.alpha = 0.25
        self.last_switch_time = 0
        self.switch_cooldown = 1.0
        self.switch_gesture = None
        self.switch_frames = 0
        self.switch_needed = 8

    def dist(self, a, b):
        return np.linalg.norm(a - b)

    def _is_finger_up(self, lm, tip, pip):
        return lm[tip][1] < lm[pip][1] - 12

    def _is_finger_down(self, lm, tip, pip):
        return lm[tip][1] > lm[pip][1] - 5

    def interpret(self, lm):
        gesture = 'NONE'
        pinch_d = 0.0
        switch_dir = None
        w, h = self.fsz
        now = time.time()

        if lm is not None:
            # pinch
            d = self.dist(lm[4][:2], lm[8][:2])
            pinch_d = d
            if d < max(30, w*0.045):
                gesture = 'PINCH'

            # Detect specific poses for switching
            thumb_up = self.dist(lm[4][:2], lm[0][:2]) > self.dist(lm[3][:2], lm[0][:2]) + 15
            idx_down = self._is_finger_down(lm, 8, 6)
            mid_down = self._is_finger_down(lm, 12, 10)
            ring_down = self._is_finger_down(lm, 16, 14)
            pinky_down = self._is_finger_down(lm, 20, 18)
            pinky_up = self._is_finger_up(lm, 20, 18)

            # THUMBS UP: thumb extended + all 4 fingers closed = NEXT
            is_thumbs_up = thumb_up and idx_down and mid_down and ring_down and pinky_down
            # PINKY ONLY: only pinky extended + all others closed = PREV
            is_pinky_only = pinky_up and idx_down and mid_down and ring_down and not thumb_up

            can_switch = (now - self.last_switch_time) > self.switch_cooldown

            if is_thumbs_up and can_switch and gesture != 'PINCH':
                gesture = 'THUMBS_UP'
                if self.switch_gesture == 'NEXT':
                    self.switch_frames += 1
                else:
                    self.switch_gesture = 'NEXT'
                    self.switch_frames = 1
                if self.switch_frames >= self.switch_needed:
                    switch_dir = 'NEXT'
                    self.last_switch_time = now
                    self.switch_frames = 0
                    self.switch_gesture = None
            elif is_pinky_only and can_switch and gesture != 'PINCH':
                gesture = 'PINKY_ONLY'
                if self.switch_gesture == 'PREV':
                    self.switch_frames += 1
                else:
                    self.switch_gesture = 'PREV'
                    self.switch_frames = 1
                if self.switch_frames >= self.switch_needed:
                    switch_dir = 'PREV'
                    self.last_switch_time = now
                    self.switch_frames = 0
                    self.switch_gesture = None
            else:
                self.switch_gesture = None
                self.switch_frames = 0

            # fist
            tips = [4,8,12,16,20]
            close = sum(1 for ti in tips if self.dist(lm[ti][:2], lm[0][:2]) < max(30, w*0.06))
            if close >= 4:
                if self.fist_since is None: self.fist_since = now
                elif now - self.fist_since >= 0.3: gesture = 'FREEZE'
            else:
                self.fist_since = None

            # open palm
            ext = sum(1 for ti in [8,12,16,20] if lm[ti][1] < lm[ti-2][1] - 8)
            if ext >= 4 and thumb_up:
                if self.open_since is None: self.open_since = now
                elif now - self.open_since >= 1.0: gesture = 'RESET'
            else:
                self.open_since = None

            # rotation
            idx = lm[8][:2]
            tgt_yaw = -(idx[0] - w/2) / w * 180
            tgt_pitch = -(idx[1] - h/2) / h * 180
        else:
            tgt_yaw = self.yaw
            tgt_pitch = self.pitch

        self.yaw   = self.alpha * tgt_yaw   + (1-self.alpha) * self.yaw
        self.pitch = self.alpha * tgt_pitch  + (1-self.alpha) * self.pitch
        zt = np.interp(pinch_d, [20, w*0.5], [2.0, 0.4]) if gesture == 'PINCH' else 1.0
        self.zoom = self.alpha * zt + (1-self.alpha) * self.zoom

        return {'gesture': gesture, 'yaw': self.yaw, 'pitch': self.pitch,
                'zoom': self.zoom, 'switch': switch_dir, 'pinch_dist': pinch_d}

# ── Renderer ─────────────────────────────────────────────────────────────
class Renderer:
    def __init__(self):
        self.models = []
        self.cur = 0
        self.yaw = self.pitch = 0.0
        self.zoom = 1.0
        self.frozen = False
        self.program = None
        self.auto_rotate = False
        self.auto_angle = 0.0
        self.show_wireframe = True
        self.show_corners = True

    def init_gl(self):
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_MULTISAMPLE)
        glutInitWindowSize(W, H)
        self.win = glutCreateWindow(b"Gesture 3D Viewer - 23 Shapes")
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.program = compileProgram(
            compileShader(VS, GL_VERTEX_SHADER),
            compileShader(FS, GL_FRAGMENT_SHADER))

    def _extract_corners(self, data):
        """Extract unique corner vertices from interleaved data for point rendering."""
        verts = data.reshape(-1, 6)[:, :3]  # extract xyz only
        # Round to find unique corners
        rounded = np.round(verts, decimals=2)
        unique = np.unique(rounded, axis=0)
        # Build point VBO (pos + dummy normal)
        point_data = []
        for v in unique:
            point_data.extend(v.tolist() + [0.0, 1.0, 0.0])
        return np.array(point_data, dtype=np.float32), len(unique)

    def add(self, name, data, color):
        if data.size == 0:
            return
        # Main shape VAO
        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glBindVertexArray(0)

        # Corner points VAO
        pt_data, pt_count = self._extract_corners(data)
        pt_vao = glGenVertexArrays(1)
        pt_vbo = glGenBuffers(1)
        glBindVertexArray(pt_vao)
        glBindBuffer(GL_ARRAY_BUFFER, pt_vbo)
        glBufferData(GL_ARRAY_BUFFER, pt_data.nbytes, pt_data, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glBindVertexArray(0)

        self.models.append({
            'name': name, 'vao': vao, 'cnt': len(data)//6, 'color': color,
            'pt_vao': pt_vao, 'pt_cnt': pt_count
        })

    def set_transform(self, y, p, z):
        if not self.frozen:
            self.yaw, self.pitch, self.zoom = y, p, np.clip(z, 0.2, 3.0)

    def next(self): self.cur = (self.cur + 1) % len(self.models)
    def prev(self): self.cur = (self.cur - 1) % len(self.models)
    def reset(self): self.yaw = self.pitch = 0.0; self.zoom = 1.0

    def display(self):
        # Gradient background
        glDisable(GL_DEPTH_TEST)
        glUseProgram(0)
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        glBegin(GL_QUADS)
        glColor3f(0.05, 0.05, 0.12); glVertex2f(-1, -1); glVertex2f(1, -1)
        glColor3f(0.12, 0.08, 0.18); glVertex2f(1, 1); glVertex2f(-1, 1)
        glEnd()
        glEnable(GL_DEPTH_TEST)
        glClear(GL_DEPTH_BUFFER_BIT)

        glUseProgram(self.program)

        # Auto-rotate
        yaw = self.yaw
        if self.auto_rotate:
            self.auto_angle += 0.5
            yaw += self.auto_angle

        view = Matrix44.look_at(Vector3([0,0,4*self.zoom]), Vector3([0,0,0]), Vector3([0,1,0]))
        proj = Matrix44.perspective_projection(45, W/H, 0.1, 100)
        mdl  = Matrix44.from_x_rotation(math.radians(self.pitch)) * Matrix44.from_y_rotation(math.radians(yaw)) * Matrix44.from_scale([0.8]*3)

        for loc, val in [('view', view), ('proj', proj), ('model', mdl)]:
            glUniformMatrix4fv(glGetUniformLocation(self.program, loc), 1, GL_FALSE, val.astype('f4').flatten())

        glUniform3f(glGetUniformLocation(self.program,'lightPos'), 3,4,3)
        glUniform3f(glGetUniformLocation(self.program,'viewPos'), 0,0,4)
        glUniform3fv(glGetUniformLocation(self.program,'lightColor'), 1, np.array([1,1,1],'f4'))
        glUniform1f(glGetUniformLocation(self.program,'pointSize'), 1.0)

        m = self.models[self.cur]

        # Pass 1: Solid fill
        glUniform1i(glGetUniformLocation(self.program,'drawMode'), 0)
        glUniform3fv(glGetUniformLocation(self.program,'objColor'), 1, np.array(m['color'],'f4'))
        glBindVertexArray(m['vao'])
        glDrawArrays(GL_TRIANGLES, 0, m['cnt'])

        # Pass 2: Wireframe edges
        if self.show_wireframe:
            glUniform1i(glGetUniformLocation(self.program,'drawMode'), 1)
            edge_color = [min(1.0, c + 0.3) for c in m['color']]
            glUniform3fv(glGetUniformLocation(self.program,'objColor'), 1, np.array(edge_color,'f4'))
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glLineWidth(1.5)
            glDrawArrays(GL_TRIANGLES, 0, m['cnt'])
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glBindVertexArray(0)

        # Pass 3: Corner dots (glowing)
        if self.show_corners:
            glUniform1i(glGetUniformLocation(self.program,'drawMode'), 2)
            glUniform1f(glGetUniformLocation(self.program,'pointSize'), 14.0)
            glUniform3fv(glGetUniformLocation(self.program,'objColor'), 1, np.array([1.0, 0.95, 0.7],'f4'))
            glDepthMask(GL_FALSE)
            glBindVertexArray(m['pt_vao'])
            glDrawArrays(GL_POINTS, 0, m['pt_cnt'])
            glBindVertexArray(0)
            glDepthMask(GL_TRUE)

        glutSwapBuffers()

# ── Main ─────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(0.3)
    if not cap.isOpened():
        print("Cannot open camera"); return

    ht = HandTracker()
    gi = GestureInterp()
    rn = Renderer()
    rn.init_gl()

    # Load all 23 shapes
    print(f"Loading {len(ALL_SHAPES)} shapes...")
    for name, gen_fn, color in ALL_SHAPES:
        try:
            data = gen_fn()
            rn.add(name, data, color)
            print(f"  ✓ {name} ({len(data)//6} verts)")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    print(f"Loaded {len(rn.models)} shapes.")
    print("Gestures: Thumbs-up=next, Pinky-only=prev, Fist=freeze, Open-palm=reset")
    print("Keyboard: n/p=next/prev  r=reset  a=auto-rotate  w=wireframe  c=corners  q=quit")

    running = True

    def cam_loop():
        nonlocal running
        while running:
            ret, frame = cap.read()
            if not ret: time.sleep(0.02); continue
            frame = cv2.flip(frame, 1)
            lm = ht.process(frame)
            gi.fsz = ht.fsz
            info = gi.interpret(lm)

            g = info['gesture']
            if info['switch'] == 'NEXT': rn.next()
            elif info['switch'] == 'PREV': rn.prev()
            if g == 'RESET':
                rn.reset(); rn.frozen = False
            if g == 'FREEZE':
                rn.frozen = True
            elif g != 'FREEZE':
                rn.frozen = False

            if not rn.frozen:
                rn.set_transform(info['yaw'], info['pitch'], info['zoom'])

            # Styled overlay
            ov = frame.copy()
            overlay = ov.copy()
            cv2.rectangle(overlay, (0, 0), (640, 160), (10, 10, 30), -1)
            cv2.addWeighted(overlay, 0.65, ov, 0.35, 0, ov)
            m = rn.models[rn.cur]
            texts = [
                (f"  Shape: {m['name']}  ({rn.cur+1}/{len(rn.models)})", (100, 220, 255)),
                (f"  Gesture: {g}", (120, 255, 160)),
                (f"  Zoom:{rn.zoom:.1f}  Yaw:{rn.yaw:.0f}  Pitch:{rn.pitch:.0f}", (200, 200, 200)),
                (f"  n/p=next/prev  r=reset  a=autoRot  w=wire  c=corners  q=quit", (170, 170, 170)),
                (f"  AutoRot:{'ON' if rn.auto_rotate else 'off'}  Wire:{'ON' if rn.show_wireframe else 'off'}  Corners:{'ON' if rn.show_corners else 'off'}", (200, 200, 100)),
            ]
            for i, (ln, col) in enumerate(texts):
                cv2.putText(ov, ln, (0, 27+i*28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 2)
            if lm is not None:
                for x, y, z in lm:
                    cv2.circle(ov, (int(x), int(y)), 5, (0, 255, 180), -1)
                    cv2.circle(ov, (int(x), int(y)), 7, (0, 200, 255), 1)
            cv2.imshow('Camera - 3D Viewer', ov)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): running = False; break
            elif key == ord('n'): rn.next()
            elif key == ord('p'): rn.prev()
            elif key == ord('r'): rn.reset(); rn.frozen = False
            elif key == ord('a'): rn.auto_rotate = not rn.auto_rotate
            elif key == ord('w'): rn.show_wireframe = not rn.show_wireframe
            elif key == ord('c'): rn.show_corners = not rn.show_corners
        cap.release()
        cv2.destroyAllWindows()

    t = threading.Thread(target=cam_loop, daemon=True)
    t.start()

    glutDisplayFunc(rn.display)
    glutIdleFunc(lambda: glutPostRedisplay())
    try:
        glutMainLoop()
    except Exception:
        pass
    running = False

if __name__ == '__main__':
    main()
