"""
voxel_engine.py – 3D voxel grid data structure with save/load and undo.
"""
import json
import os
import numpy as np

class Voxel:
    """Single voxel block."""
    __slots__ = ('x', 'y', 'z', 'color', 'alpha', 'material')

    def __init__(self, x, y, z, color=(0, 255, 220), alpha=1.0, material="hologram"):
        self.x = x
        self.y = y
        self.z = z
        self.color = color
        self.alpha = alpha
        self.material = material

    def pos(self):
        return (self.x, self.y, self.z)

    def to_dict(self):
        return {"x": self.x, "y": self.y, "z": self.z,
                "color": list(self.color), "alpha": self.alpha, "material": self.material}

    @staticmethod
    def from_dict(d):
        return Voxel(d["x"], d["y"], d["z"], tuple(d["color"]), d["alpha"], d["material"])


class VoxelEngine:
    """Manages a grid of voxels with add/remove/undo/save/load."""

    GRID_SIZE = 16  # 16x16x16 grid
    MATERIALS = {
        "hologram":  (0, 255, 220),
        "neon_blue":  (0, 150, 255),
        "neon_pink":  (255, 0, 200),
        "neon_green": (0, 255, 100),
        "gold":       (255, 215, 0),
        "white":      (230, 230, 255),
        # palette aliases
        "Cyan":  (0, 255, 220), "Blue":  (0, 150, 255),
        "Pink":  (255, 0, 200), "Green": (0, 255, 100),
        "Gold":  (255, 215, 0), "White": (230, 230, 255),
        "Red":   (255, 80, 80), "Orange":(255, 160, 0),
    }

    def __init__(self):
        self.voxels = {}  # (x,y,z) -> Voxel
        self._undo_stack = []  # list of (action, voxel_data)
        self.current_material = "hologram"
        self.ghost_pos = None  # preview position

        # World transform
        self.offset = np.array([0.0, 0.0, 0.0])
        self.rotation_y = 0.3  # radians
        self.rotation_x = -0.4
        self.scale = 1.0

    def add_voxel(self, x, y, z, color=None, record_undo=True):
        key = (x, y, z)
        if 0 <= x < self.GRID_SIZE and 0 <= y < self.GRID_SIZE and 0 <= z < self.GRID_SIZE:
            if key not in self.voxels:
                c = color or self.MATERIALS.get(self.current_material, (0, 255, 220))
                v = Voxel(x, y, z, c)
                self.voxels[key] = v
                if record_undo:
                    self._undo_stack.append(("add", v.to_dict()))
                return True
        return False

    def remove_voxel(self, x, y, z, record_undo=True):
        key = (x, y, z)
        if key in self.voxels:
            if record_undo:
                self._undo_stack.append(("remove", self.voxels[key].to_dict()))
            del self.voxels[key]
            return True
        return False

    def undo(self):
        if not self._undo_stack:
            return
        action, data = self._undo_stack.pop()
        if action == "add":
            self.voxels.pop((data["x"], data["y"], data["z"]), None)
        elif action == "remove":
            v = Voxel.from_dict(data)
            self.voxels[v.pos()] = v

    def clear_all(self):
        self.voxels.clear()
        self._undo_stack.clear()

    def set_ghost(self, pos):
        self.ghost_pos = pos

    def cycle_material(self):
        mats = list(self.MATERIALS.keys())
        idx = mats.index(self.current_material)
        self.current_material = mats[(idx + 1) % len(mats)]
        return self.current_material

    def count(self):
        return len(self.voxels)

    def save(self, filepath):
        data = {"voxels": [v.to_dict() for v in self.voxels.values()],
                "material": self.current_material}
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath):
        if not os.path.exists(filepath):
            return False
        with open(filepath, "r") as f:
            data = json.load(f)
        self.voxels.clear()
        for vd in data.get("voxels", []):
            v = Voxel.from_dict(vd)
            self.voxels[v.pos()] = v
        self.current_material = data.get("material", "hologram")
        return True

    def get_grid_pos_from_cursor(self, nx, ny, grid_depth=0):
        """Map normalized cursor (0-1) to grid coords at a given depth layer."""
        gx = int(nx * self.GRID_SIZE)
        gy = int((1.0 - ny) * self.GRID_SIZE)   # invert: finger up → voxel up
        gz = grid_depth
        gx = max(0, min(self.GRID_SIZE - 1, gx))
        gy = max(0, min(self.GRID_SIZE - 1, gy))
        gz = max(0, min(self.GRID_SIZE - 1, gz))
        return gx, gy, gz

    def erase_radius(self, x, y, z, radius=1):
        """Remove voxels in a radius around (x,y,z)."""
        removed = 0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if self.remove_voxel(x + dx, y + dy, z + dz):
                        removed += 1
        return removed
