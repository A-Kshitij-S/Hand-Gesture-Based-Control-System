"""
shapes_3d.py — Procedural 3D shape generators for the Gesture 3D Viewer.
Each function returns a flat np.float32 array of interleaved [x,y,z, nx,ny,nz] per vertex.
"""
import numpy as np, math

def _tri_normal(a, b, c):
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n)
    return n / ln if ln > 0 else np.array([0., 0., 1.])

def _emit_tri(out, a, b, c):
    n = _tri_normal(a, b, c)
    for v in (a, b, c):
        out.extend(v.tolist() + n.tolist())

def _box(cx, cy, cz, sx, sy, sz):
    """Box centered at (cx,cy,cz) with half-sizes sx,sy,sz."""
    corners = []
    for dx in (-sx, sx):
        for dy in (-sy, sy):
            for dz in (-sz, sz):
                corners.append(np.array([cx+dx, cy+dy, cz+dz]))
    c = corners
    faces = [
        (0,2,3,1),(4,5,7,6),(0,1,5,4),(2,6,7,3),(0,4,6,2),(1,3,7,5)
    ]
    out = []
    for f in faces:
        _emit_tri(out, c[f[0]], c[f[1]], c[f[2]])
        _emit_tri(out, c[f[0]], c[f[2]], c[f[3]])
    return out

def create_cube():
    return np.array(_box(0,0,0, 1,1,1), dtype=np.float32)

def create_rectangular_prism():
    return np.array(_box(0,0,0, 1.5,0.6,0.4), dtype=np.float32)

def create_sphere(stacks=12, slices=16, r=1.0):
    out = []
    for i in range(stacks):
        t0 = math.pi * i / stacks
        t1 = math.pi * (i+1) / stacks
        for j in range(slices):
            p0 = 2*math.pi * j / slices
            p1 = 2*math.pi * (j+1) / slices
            def pt(t, p):
                return np.array([r*math.sin(t)*math.cos(p), r*math.cos(t), r*math.sin(t)*math.sin(p)])
            a, b, c, d = pt(t0,p0), pt(t0,p1), pt(t1,p1), pt(t1,p0)
            _emit_tri(out, a, d, c)
            _emit_tri(out, a, c, b)
    return np.array(out, dtype=np.float32)

def create_cylinder(segs=20, r=0.6, h=2.0):
    out = []
    top_c = np.array([0, h/2, 0])
    bot_c = np.array([0, -h/2, 0])
    for i in range(segs):
        a0 = 2*math.pi*i/segs
        a1 = 2*math.pi*(i+1)/segs
        t0 = np.array([r*math.cos(a0), h/2, r*math.sin(a0)])
        t1 = np.array([r*math.cos(a1), h/2, r*math.sin(a1)])
        b0 = np.array([r*math.cos(a0), -h/2, r*math.sin(a0)])
        b1 = np.array([r*math.cos(a1), -h/2, r*math.sin(a1)])
        _emit_tri(out, t0, b0, b1)
        _emit_tri(out, t0, b1, t1)
        _emit_tri(out, top_c, t1, t0)
        _emit_tri(out, bot_c, b0, b1)
    return np.array(out, dtype=np.float32)

def create_cone(segs=20, r=0.8, h=2.0):
    out = []
    tip = np.array([0, h/2, 0])
    base_c = np.array([0, -h/2, 0])
    for i in range(segs):
        a0 = 2*math.pi*i/segs
        a1 = 2*math.pi*(i+1)/segs
        b0 = np.array([r*math.cos(a0), -h/2, r*math.sin(a0)])
        b1 = np.array([r*math.cos(a1), -h/2, r*math.sin(a1)])
        _emit_tri(out, tip, b0, b1)
        _emit_tri(out, base_c, b1, b0)
    return np.array(out, dtype=np.float32)

def create_torus(R=0.8, r_=0.3, segs=20, rings=20):
    out = []
    for i in range(rings):
        t0 = 2*math.pi*i/rings
        t1 = 2*math.pi*(i+1)/rings
        for j in range(segs):
            p0 = 2*math.pi*j/segs
            p1 = 2*math.pi*(j+1)/segs
            def pt(t, p):
                x = (R + r_*math.cos(p))*math.cos(t)
                y = r_*math.sin(p)
                z = (R + r_*math.cos(p))*math.sin(t)
                return np.array([x, y, z])
            a, b, c, d = pt(t0,p0), pt(t1,p0), pt(t1,p1), pt(t0,p1)
            _emit_tri(out, a, b, c)
            _emit_tri(out, a, c, d)
    return np.array(out, dtype=np.float32)

def create_pyramid():
    tip = np.array([0, 1.2, 0])
    b = [np.array([-1,-0.5,-1]), np.array([1,-0.5,-1]),
         np.array([1,-0.5,1]), np.array([-1,-0.5,1])]
    bc = np.array([0,-0.5,0])
    out = []
    for i in range(4):
        _emit_tri(out, tip, b[i], b[(i+1)%4])
    _emit_tri(out, bc, b[1], b[0])
    _emit_tri(out, bc, b[3], b[2])
    _emit_tri(out, bc, b[2], b[1])
    _emit_tri(out, bc, b[0], b[3])
    return np.array(out, dtype=np.float32)

def create_tetrahedron():
    s = 1.2
    v = [np.array([s,s,s]), np.array([s,-s,-s]),
         np.array([-s,s,-s]), np.array([-s,-s,s])]
    out = []
    faces = [(0,1,2),(0,2,3),(0,3,1),(1,3,2)]
    for f in faces:
        _emit_tri(out, v[f[0]], v[f[1]], v[f[2]])
    return np.array(out, dtype=np.float32)

def create_octahedron():
    s = 1.2
    v = [np.array([0,s,0]), np.array([0,-s,0]),
         np.array([s,0,0]), np.array([-s,0,0]),
         np.array([0,0,s]), np.array([0,0,-s])]
    out = []
    faces = [(0,4,2),(0,2,5),(0,5,3),(0,3,4),
             (1,2,4),(1,5,2),(1,3,5),(1,4,3)]
    for f in faces:
        _emit_tri(out, v[f[0]], v[f[1]], v[f[2]])
    return np.array(out, dtype=np.float32)

def create_diamond():
    out = []
    segs = 8
    top = np.array([0, 1.5, 0])
    bot = np.array([0, -1.0, 0])
    r = 0.7
    ring = [np.array([r*math.cos(2*math.pi*i/segs), 0.2, r*math.sin(2*math.pi*i/segs)]) for i in range(segs)]
    for i in range(segs):
        _emit_tri(out, top, ring[i], ring[(i+1)%segs])
        _emit_tri(out, bot, ring[(i+1)%segs], ring[i])
    return np.array(out, dtype=np.float32)

def create_star():
    out = []
    n = 5
    top = np.array([0, 0.3, 0])
    bot = np.array([0, -0.3, 0])
    pts = []
    for i in range(n*2):
        a = math.pi*i/n - math.pi/2
        r = 1.2 if i%2==0 else 0.5
        pts.append(np.array([r*math.cos(a), 0, r*math.sin(a)]))
    for i in range(n*2):
        _emit_tri(out, top, pts[i], pts[(i+1)%(n*2)])
        _emit_tri(out, bot, pts[(i+1)%(n*2)], pts[i])
    return np.array(out, dtype=np.float32)

def create_hexagonal_prism():
    out = []
    n, r, h = 6, 0.8, 1.5
    top_c = np.array([0, h/2, 0])
    bot_c = np.array([0, -h/2, 0])
    for i in range(n):
        a0 = 2*math.pi*i/n
        a1 = 2*math.pi*(i+1)/n
        t0 = np.array([r*math.cos(a0), h/2, r*math.sin(a0)])
        t1 = np.array([r*math.cos(a1), h/2, r*math.sin(a1)])
        b0 = np.array([r*math.cos(a0), -h/2, r*math.sin(a0)])
        b1 = np.array([r*math.cos(a1), -h/2, r*math.sin(a1)])
        _emit_tri(out, t0, b0, b1); _emit_tri(out, t0, b1, t1)
        _emit_tri(out, top_c, t1, t0); _emit_tri(out, bot_c, b0, b1)
    return np.array(out, dtype=np.float32)

def create_arrow():
    out = []
    # shaft
    out.extend(_box(0, -0.3, 0, 0.15, 0.6, 0.15))
    # arrowhead (pyramid on top)
    tip = np.array([0, 1.0, 0])
    b = [np.array([-0.4, 0.3, -0.4]), np.array([0.4, 0.3, -0.4]),
         np.array([0.4, 0.3, 0.4]), np.array([-0.4, 0.3, 0.4])]
    for i in range(4):
        _emit_tri(out, tip, b[i], b[(i+1)%4])
    return np.array(out, dtype=np.float32)

def create_cross():
    out = []
    out.extend(_box(0, 0, 0, 0.2, 1.0, 0.2))
    out.extend(_box(0, 0.4, 0, 0.7, 0.2, 0.2))
    return np.array(out, dtype=np.float32)

def create_hemisphere(stacks=10, slices=16, r=1.0):
    out = []
    for i in range(stacks):
        t0 = math.pi/2 * i / stacks
        t1 = math.pi/2 * (i+1) / stacks
        for j in range(slices):
            p0 = 2*math.pi*j/slices
            p1 = 2*math.pi*(j+1)/slices
            def pt(t, p):
                return np.array([r*math.sin(t)*math.cos(p), r*math.cos(t), r*math.sin(t)*math.sin(p)])
            a, b, c, d = pt(t0,p0), pt(t0,p1), pt(t1,p1), pt(t1,p0)
            _emit_tri(out, a, d, c); _emit_tri(out, a, c, b)
    # bottom cap
    bc = np.array([0, 0, 0])
    for j in range(slices):
        p0 = 2*math.pi*j/slices
        p1 = 2*math.pi*(j+1)/slices
        a = np.array([r*math.cos(p0), 0, r*math.sin(p0)])
        b = np.array([r*math.cos(p1), 0, r*math.sin(p1)])
        _emit_tri(out, bc, b, a)
    return np.array(out, dtype=np.float32)

def create_capsule(segs=16, r=0.4, h=1.2):
    out = []
    # cylinder body
    for i in range(segs):
        a0 = 2*math.pi*i/segs; a1 = 2*math.pi*(i+1)/segs
        t0 = np.array([r*math.cos(a0), h/2, r*math.sin(a0)])
        t1 = np.array([r*math.cos(a1), h/2, r*math.sin(a1)])
        b0 = np.array([r*math.cos(a0), -h/2, r*math.sin(a0)])
        b1 = np.array([r*math.cos(a1), -h/2, r*math.sin(a1)])
        _emit_tri(out, t0, b0, b1); _emit_tri(out, t0, b1, t1)
    # top half-sphere
    stk = 6
    for i in range(stk):
        t0_ = math.pi/2 * i / stk; t1_ = math.pi/2 * (i+1) / stk
        for j in range(segs):
            p0 = 2*math.pi*j/segs; p1 = 2*math.pi*(j+1)/segs
            def ptt(t, p):
                return np.array([r*math.sin(t)*math.cos(p), h/2+r*math.cos(t), r*math.sin(t)*math.sin(p)])
            a, b, c, d = ptt(t0_,p0), ptt(t0_,p1), ptt(t1_,p1), ptt(t1_,p0)
            _emit_tri(out, a, d, c); _emit_tri(out, a, c, b)
    # bottom half-sphere
    for i in range(stk):
        t0_ = math.pi/2 + math.pi/2*i/stk; t1_ = math.pi/2 + math.pi/2*(i+1)/stk
        for j in range(segs):
            p0 = 2*math.pi*j/segs; p1 = 2*math.pi*(j+1)/segs
            def ptb(t, p):
                return np.array([r*math.sin(t)*math.cos(p), -h/2+r*math.cos(t), r*math.sin(t)*math.sin(p)])
            a, b, c, d = ptb(t0_,p0), ptb(t0_,p1), ptb(t1_,p1), ptb(t1_,p0)
            _emit_tri(out, a, d, c); _emit_tri(out, a, c, b)
    return np.array(out, dtype=np.float32)

def create_wedge():
    v = [np.array([-1,-0.5,-0.5]), np.array([1,-0.5,-0.5]),
         np.array([1,-0.5,0.5]), np.array([-1,-0.5,0.5]),
         np.array([-1,0.5,-0.5]), np.array([1,0.5,-0.5])]
    out = []
    _emit_tri(out, v[0],v[1],v[2]); _emit_tri(out, v[0],v[2],v[3])
    _emit_tri(out, v[0],v[4],v[5]); _emit_tri(out, v[0],v[5],v[1])
    _emit_tri(out, v[0],v[3],v[4]); _emit_tri(out, v[3],v[4],v[4])
    _emit_tri(out, v[1],v[5],v[2]); _emit_tri(out, v[4],v[3],v[2])
    _emit_tri(out, v[4],v[2],v[5])
    return np.array(out, dtype=np.float32)

def create_low_poly_human():
    out = []
    out.extend(_box(0, 0, 0, 0.3, 0.5, 0.15))
    out.extend(_box(0, 0.9, 0, 0.2, 0.2, 0.2))
    out.extend(_box(-0.2, -1.0, 0, 0.12, 0.3, 0.12))
    out.extend(_box(0.2, -1.0, 0, 0.12, 0.3, 0.12))
    out.extend(_box(-0.5, 0.1, 0, 0.1, 0.4, 0.1))
    out.extend(_box(0.5, 0.1, 0, 0.1, 0.4, 0.1))
    return np.array(out, dtype=np.float32)

def create_heart():
    out = []
    segs = 20
    pts = []
    for i in range(segs):
        t = 2*math.pi*i/segs
        x = 16*math.sin(t)**3
        y = 13*math.cos(t)-5*math.cos(2*t)-2*math.cos(3*t)-math.cos(4*t)
        pts.append(np.array([x/16, y/16, 0]))
    center = np.array([0, 0, 0])
    fz = np.array([0, 0, 0.15])
    bz = np.array([0, 0, -0.15])
    for i in range(segs):
        a = pts[i]; b = pts[(i+1)%segs]
        _emit_tri(out, center+fz, a+fz, b+fz)
        _emit_tri(out, center+bz, b+bz, a+bz)
        _emit_tri(out, a+fz, a+bz, b+bz)
        _emit_tri(out, a+fz, b+bz, b+fz)
    return np.array(out, dtype=np.float32)

def create_gear():
    out = []
    teeth = 10
    inner_r, outer_r = 0.5, 0.9
    h = 0.3
    top_c = np.array([0, h/2, 0])
    bot_c = np.array([0, -h/2, 0])
    pts = []
    for i in range(teeth*2):
        a = 2*math.pi*i/(teeth*2)
        r = outer_r if i%2==0 else inner_r
        pts.append(np.array([r*math.cos(a), 0, r*math.sin(a)]))
    n = len(pts)
    for i in range(n):
        t0 = pts[i].copy(); t0[1] = h/2
        t1 = pts[(i+1)%n].copy(); t1[1] = h/2
        b0 = pts[i].copy(); b0[1] = -h/2
        b1 = pts[(i+1)%n].copy(); b1[1] = -h/2
        _emit_tri(out, top_c, t1, t0)
        _emit_tri(out, bot_c, b0, b1)
        _emit_tri(out, t0, b0, b1); _emit_tri(out, t0, b1, t1)
    return np.array(out, dtype=np.float32)

def create_spiral_spring(coils=3, segs_per_coil=16, r=0.6, h=2.0, wire_r=0.08):
    out = []
    total_segs = coils * segs_per_coil
    for i in range(total_segs):
        t0 = 2*math.pi*i/segs_per_coil
        t1 = 2*math.pi*(i+1)/segs_per_coil
        y0 = -h/2 + h*i/total_segs
        y1 = -h/2 + h*(i+1)/total_segs
        c0 = np.array([r*math.cos(t0), y0, r*math.sin(t0)])
        c1 = np.array([r*math.cos(t1), y1, r*math.sin(t1)])
        # simple thick line as two triangles
        up = np.array([0, wire_r, 0])
        side = np.array([wire_r, 0, 0])
        _emit_tri(out, c0-up, c1-up, c1+up)
        _emit_tri(out, c0-up, c1+up, c0+up)
        _emit_tri(out, c0-side, c1-side, c1+side)
        _emit_tri(out, c0-side, c1+side, c0+side)
    return np.array(out, dtype=np.float32)

def create_pentagonal_prism():
    out = []
    n, r, h = 5, 0.9, 1.5
    top_c = np.array([0, h/2, 0])
    bot_c = np.array([0, -h/2, 0])
    for i in range(n):
        a0 = 2*math.pi*i/n; a1 = 2*math.pi*(i+1)/n
        t0 = np.array([r*math.cos(a0), h/2, r*math.sin(a0)])
        t1 = np.array([r*math.cos(a1), h/2, r*math.sin(a1)])
        b0 = np.array([r*math.cos(a0), -h/2, r*math.sin(a0)])
        b1 = np.array([r*math.cos(a1), -h/2, r*math.sin(a1)])
        _emit_tri(out, t0, b0, b1); _emit_tri(out, t0, b1, t1)
        _emit_tri(out, top_c, t1, t0); _emit_tri(out, bot_c, b0, b1)
    return np.array(out, dtype=np.float32)

def create_triangular_prism():
    out = []
    n, r, h = 3, 0.9, 1.8
    top_c = np.array([0, h/2, 0])
    bot_c = np.array([0, -h/2, 0])
    for i in range(n):
        a0 = 2*math.pi*i/n; a1 = 2*math.pi*(i+1)/n
        t0 = np.array([r*math.cos(a0), h/2, r*math.sin(a0)])
        t1 = np.array([r*math.cos(a1), h/2, r*math.sin(a1)])
        b0 = np.array([r*math.cos(a0), -h/2, r*math.sin(a0)])
        b1 = np.array([r*math.cos(a1), -h/2, r*math.sin(a1)])
        _emit_tri(out, t0, b0, b1); _emit_tri(out, t0, b1, t1)
        _emit_tri(out, top_c, t1, t0); _emit_tri(out, bot_c, b0, b1)
    return np.array(out, dtype=np.float32)

# ── Master list of all shapes ──────────────────────────────────────────
ALL_SHAPES = [
    ("Cube",              create_cube,              (0.2, 0.6, 0.9)),
    ("Sphere",            create_sphere,            (0.9, 0.3, 0.3)),
    ("Cylinder",          create_cylinder,          (0.3, 0.8, 0.4)),
    ("Cone",              create_cone,              (0.9, 0.7, 0.2)),
    ("Torus",             create_torus,             (0.6, 0.2, 0.8)),
    ("Pyramid",           create_pyramid,           (0.9, 0.5, 0.1)),
    ("Tetrahedron",       create_tetrahedron,       (0.1, 0.7, 0.7)),
    ("Octahedron",        create_octahedron,        (0.8, 0.4, 0.6)),
    ("Diamond",           create_diamond,           (0.3, 0.9, 0.9)),
    ("Star",              create_star,              (0.9, 0.9, 0.2)),
    ("HexPrism",          create_hexagonal_prism,   (0.5, 0.5, 0.9)),
    ("Arrow",             create_arrow,             (0.2, 0.9, 0.4)),
    ("Cross",             create_cross,             (0.9, 0.2, 0.2)),
    ("Hemisphere",        create_hemisphere,        (0.4, 0.6, 0.9)),
    ("Capsule",           create_capsule,           (0.7, 0.3, 0.7)),
    ("Wedge",             create_wedge,             (0.8, 0.6, 0.3)),
    ("LowPolyHuman",      create_low_poly_human,    (0.9, 0.6, 0.4)),
    ("Heart",             create_heart,             (0.9, 0.1, 0.3)),
    ("Gear",              create_gear,              (0.6, 0.6, 0.6)),
    ("Spring",            create_spiral_spring,     (0.3, 0.8, 0.2)),
    ("PentaPrism",        create_pentagonal_prism,  (0.7, 0.4, 0.8)),
    ("TriPrism",          create_triangular_prism,  (0.4, 0.8, 0.6)),
    ("RectPrism",         create_rectangular_prism, (0.5, 0.7, 0.3)),
]
