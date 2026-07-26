"""renderer.py – Fixed look-at projection, structure rotation, fixed axes."""
import math, numpy as np

GRID = 16
_C = np.array([8.0, 4.0, 8.0])          # orbit center
_VERTS = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                   [0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=float)
_FACES = [[0,1,2,3],[5,4,7,6],[4,0,3,7],[1,5,6,2],[3,2,6,7],[4,5,1,0]]
_SHADE = [0.85, 0.70, 0.75, 0.80, 1.00, 0.60]

def _hex(rgb, s):
    return "#{:02x}{:02x}{:02x}".format(*[max(0,min(255,int(c*s))) for c in rgb])

def _Ry(a):
    c,s = math.cos(a), math.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def _Rx(a):
    c,s = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])


class VoxelRenderer:
    def __init__(self, w=800, h=600):
        self.cw, self.ch = w, h
        self.fov = 550.0
        self.cam_dist = 24.0
        self.cam_yaw   = 0.6
        self.cam_pitch = 0.35
        self.struct_yaw   = 0.0
        self.struct_pitch = 0.0

    def set_view(self, yaw, pitch, dist=24.0):
        self.cam_yaw, self.cam_pitch, self.cam_dist = yaw, pitch, dist

    def rotate_struct(self, dyaw, dpitch=0.0):
        self.struct_yaw += dyaw
        self.struct_pitch = max(-1.2, min(1.2, self.struct_pitch + dpitch))

    # ── projection ──────────────────────────────────────────────────────
    def _eye(self):
        cy,sy = math.cos(self.cam_yaw), math.sin(self.cam_yaw)
        cp,sp = math.cos(self.cam_pitch), math.sin(self.cam_pitch)
        return _C + np.array([self.cam_dist*cp*sy,
                               self.cam_dist*sp,
                               self.cam_dist*cp*cy])

    def _project(self, pts):
        """Correct look-at perspective. pts: (N,3) array."""
        eye = self._eye()
        fwd = _C - eye;  fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0,1,0]); right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        rel = np.asarray(pts, dtype=float) - eye
        cx = rel @ right
        cy2= rel @ up
        cz = rel @ fwd
        cz = np.where(cz < 0.1, 0.1, cz)
        sx  = (self.cw/2 + cx * self.fov / cz).astype(int)
        sy2 = (self.ch/2 - cy2* self.fov / cz).astype(int)
        return sx, sy2, cz

    def _rot_struct(self, pts):
        R = _Ry(self.struct_yaw) @ _Rx(self.struct_pitch)
        c = np.array([8.0, 0.0, 8.0])
        return (R @ (np.asarray(pts, dtype=float) - c).T).T + c

    # ── public ───────────────────────────────────────────────────────────
    def render(self, canvas, engine):
        canvas.delete("voxel")
        faces = []

        for vox in engine.voxels.values():
            raw = _VERTS + [vox.x, vox.y, vox.z]
            pts = self._rot_struct(raw)
            sx, sy, sz = self._project(pts)
            for fi, idx in enumerate(_FACES):
                p2 = [(sx[i], sy[i]) for i in idx]
                d  = sum(sz[i] for i in idx) / 4
                faces.append((d, p2, vox.color, _SHADE[fi], False))

        if engine.ghost_pos:
            raw = _VERTS + np.array(engine.ghost_pos, dtype=float)
            pts = self._rot_struct(raw)
            sx, sy, sz = self._project(pts)
            for fi, idx in enumerate(_FACES):
                p2 = [(sx[i], sy[i]) for i in idx]
                d  = sum(sz[i] for i in idx) / 4
                faces.append((d, p2, (80, 160, 255), 0.4, True))

        faces.sort(key=lambda x: -x[0])
        for _, p2, col, sh, ghost in faces:
            flat = [v for p in p2 for v in p]
            fill = _hex(col, sh)
            outl = _hex(col, min(sh*1.4, 1.0)) if not ghost else "#336688"
            canvas.create_polygon(flat, fill=fill, outline=outl, width=1, tags="voxel")

        self._draw_grid(canvas)
        self._draw_axes(canvas)

    def _draw_grid(self, canvas):
        for i in range(0, GRID+1, 2):
            for a,b in [([0,0,i],[GRID,0,i]), ([i,0,0],[i,0,GRID])]:
                sx, sy, _ = self._project([a, b])
                canvas.create_line(sx[0],sy[0],sx[1],sy[1],
                                   fill="#1a3a4a", width=1, tags="voxel")

    def _draw_axes(self, canvas):
        """2-D corner widget — always fixed, never rotates with structure."""
        ox, oy = self.cw - 80, 80
        # Only reflect camera rotation, NOT structure rotation
        R = _Ry(self.cam_yaw) @ _Rx(-self.cam_pitch)
        sc = 48
        for axis, col, lbl in [(np.array([1,0,0]),"#ff4444","X"),
                                (np.array([0,1,0]),"#44ff44","Y"),
                                (np.array([0,0,1]),"#4488ff","Z")]:
            d  = R @ axis
            ex = int(ox + d[0]*sc)
            ey = int(oy - d[1]*sc)
            canvas.create_line(ox,oy,ex,ey, fill=col, width=2, tags="voxel")
            canvas.create_text(ex,ey, text=lbl, fill=col,
                               font=("Consolas",9,"bold"), tags="voxel")
        canvas.create_oval(ox-4,oy-4,ox+4,oy+4, fill="white",
                           outline="", tags="voxel")
