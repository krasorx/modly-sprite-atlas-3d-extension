"""
Sprite Atlas to 3D extension for Modly.

This extension takes a sprite atlas (a grid of frames showing the same
character/body from multiple angles) and reconstructs a 3D mesh that
takes every view into account, instead of treating the sheet as a single
flat image.

Modly core nodes only accept a single image input, so all of the atlas
parsing happens inside this generator's generate().
"""

from __future__ import annotations

import math
import os
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Callable, Optional

# NOTE: the runner imports `services.generators.base` (NOT `api.services...`).
from services.generators.base import BaseGenerator, GenerationCancelled


# ---------------------------------------------------------------------------
# 1) Atlas slicing
# ---------------------------------------------------------------------------

def slice_atlas(image, cols: int, rows: int) -> list:
    """Cut a sprite sheet into its grid frames, left-to-right, top-to-bottom."""
    w, h = image.size
    cell_w = max(1, w // cols)
    cell_h = max(1, h // rows)
    frames = []
    for r in range(rows):
        for c in range(cols):
            left = c * cell_w
            top = r * cell_h
            box = (left, top, min(left + cell_w, w), min(top + cell_h, h))
            frames.append(image.crop(box).convert("RGBA"))
    return frames


def recover_view_angles(count: int, order: str) -> list[float]:
    """Assign an azimuth angle (degrees) to each frame around the full 360."""
    angles = []
    for i in range(count):
        step = 360.0 / max(1, count)
        angles.append(i * step)
    return angles


def normalize_frame(frame, size: int, background: str):
    """Center a frame on a square canvas with the requested background."""
    from PIL import Image

    bg = background if background != "alpha" else (0, 0, 0, 0)
    canvas = Image.new("RGBA", (size, size), bg)
    frame = frame.convert("RGBA")
    frame.thumbnail((size, size), Image.LANCZOS)
    ox = (size - frame.width) // 2
    oy = (size - frame.height) // 2
    canvas.paste(frame, (ox, oy), frame)
    return canvas


# ---------------------------------------------------------------------------
# 2) Reconstruction backend (pluggable)
# ---------------------------------------------------------------------------

class Reconstructor:
    """Base class. Subclasses implement reconstruct() for a concrete model."""

    def load(self) -> None:
        raise NotImplementedError

    def reconstruct(self, frames: list, angles: list[float], out_path: Path, params: dict) -> Path:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Visual-hull (shape-from-silhouette) reconstruction
#
# This is a real volumetric fusion that takes EVERY atlas view into account:
#   1. Each frame is reduced to a binary silhouette (foreground mask).
#   2. Every voxel in a 3D grid is projected to each view (the character is
#      treated as a turntable: camera keeps fixed height, object rotates by
#      the azimuth angle recovered from the atlas grid).
#   3. A voxel is part of the model iff it projects inside the silhouette of
#      ALL views (visual hull = intersection of the silhouette cones).
#   4. Occupied-voxel faces are extracted into a closed mesh and per-vertex
#      color is baked by averaging the visible foreground color over views.
#
# All dependencies (numpy, cv2, trimesh) are already in the extension venv.
# ---------------------------------------------------------------------------

class VisualHullReconstructor(Reconstructor):
    """Full-angle volumetric reconstruction from sprite silhouettes."""

    def load(self) -> None:
        pass

    def reconstruct(self, frames, angles, out_path: Path, params: dict):
        import cv2
        import numpy as np
        import trimesh

        bg = params.get("background", "alpha")
        res = int(params.get("resolution", 96))
        colorize = bool(params.get("colorize", True))

        # -- 1) silhouettes ------------------------------------------------
        views = []               # dicts: mask, cx, cy, px_per_world, rgb
        for i, frame in enumerate(frames):
            mask = _extract_silhouette(frame, bg)
            # dilate a touch to bridge thin sprite edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0
            ys, xs = np.where(mask)
            if xs.size == 0:
                continue
            cx = float(xs.mean())
            cy = float(ys.mean())
            bw = int(xs.max() - xs.min()) + 1
            bh = int(ys.max() - ys.min()) + 1
            px_per_world = max(bw, bh) / 2.0
            theta = float(angles[i % len(angles)]) * np.pi / 180.0
            views.append({
                "mask": mask,
                "cx": cx,
                "cy": cy,
                "px_per_world": px_per_world,
                "sin": np.sin(theta),
                "cos": np.cos(theta),
                "rgb": _foreground_rgb(frame, mask),
            })
        if not views:
            raise RuntimeError("No foreground silhouettes could be extracted from the atlas.")

        # -- 2) choose strategy + build occupancy ---------------------------
        # Visual-hull needs a *consistent turntable* (same body rotating). A
        # sprite sheet of animation frames has very different silhouettes per
        # frame (idle/run/jump), where intersecting the cones yields an empty
        # or garbage volume. Detect that case and fall back to inflating the
        # most descriptive silhouette so we never fail or emit a blob.
        mode = _pick_mode(views)

        if mode == "inflate":
            occ = _inflate_volume(views, res)
        else:
            occ = _visual_hull(views, res)
            # safety net: an empty/tiny hull → inflate the largest view
            if occ.sum() < res * res * res * 0.004:
                occ = _inflate_volume(views, res)

        # -- 3) surface extraction (voxel-face quads) ----------------------
        vertices, triangles, colors = _surface_from_volume(occ, views, colorize)

        # -- 4) export -----------------------------------------------------
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, vertex_colors=colors)
        mesh.export(str(out_path))
        return out_path


def _extract_silhouette(frame: np.ndarray, bg: str) -> np.ndarray:
    """frame: (H, W, 4) RGBA uint8 → bool mask of the foreground shape."""
    import numpy as np
    alpha = frame[..., 3]
    if bg == "alpha":
        return alpha > 128
    rgb = frame[..., :3].astype(np.int16)
    bg_rgb = 255 if bg == "white" else 0
    diff = np.abs(rgb - bg_rgb).sum(axis=2)
    return (diff > 40) & (alpha > 128)


def _foreground_rgb(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import numpy as np
    fg = frame[..., :3][mask]
    if fg.size == 0:
        return np.array([160, 160, 160], dtype=np.uint8)
    return fg.mean(axis=0).astype(np.uint8)


def _pick_mode(views) -> str:
    """Return 'hull' for a consistent turntable, 'inflate' for animation frames."""
    import numpy as np
    if len(views) < 4:
        return "inflate"
    widths = []
    heights = []
    for v in views:
        ys, xs = np.where(v["mask"])
        if xs.size == 0:
            continue
        widths.append(float(xs.max() - xs.min()) + 1.0)
        heights.append(float(ys.max() - ys.min()) + 1.0)
    cv = lambda a: (np.std(a) / np.mean(a)) if np.mean(a) > 0 else 1.0  # noqa: E731
    # animation frames swing wildly in size/shape; turntables stay ~constant
    if cv(widths) > 0.22 or cv(heights) > 0.22:
        return "inflate"
    return "hull"


def _inflate_volume(views, res: int) -> np.ndarray:
    """Build a solid from the best (most pixel-dense) silhouette via an
    ellipsoidal depth profile centred on the sprite's horizontal axis.
    occ[y, x, z]; world x in [-1, 1], y up."""
    import numpy as np
    best = max(views, key=lambda v: int(v["mask"].sum()))
    mask = best["mask"]
    h = mask.shape[0]
    axis = np.linspace(-1.0, 1.0, res)

    occ = np.zeros((res, res, res), dtype=bool)
    for iy, wy in enumerate(axis):
        v_px = int(best["cy"] - wy * best["px_per_world"])
        if not (0 <= v_px < h):
            continue
        for ix, wx in enumerate(axis):
            u_px = int(best["cx"] + wx * best["px_per_world"])
            if not (0 <= u_px < mask.shape[1]) or not mask[v_px, u_px]:
                continue
            # ellipsoid cross-section: thick where the sprite is wide/central,
            # thin at the horizontal edges (a believable standing-character form)
            depth = 0.55 * math.sqrt(max(0.0, 1.0 - wx * wx))
            for iz, wz in enumerate(axis):
                if -depth <= wz <= depth:
                    occ[iy, ix, iz] = True
    return occ


def _visual_hull(views, res: int) -> np.ndarray:
    """bool (res, res, res): voxel occupied iff its projection into EVERY view
    falls inside that view's silhouette.

    occ[y, x, z] ↔ world coords (x, y, z) with y up.
    """
    import numpy as np
    axis = np.linspace(-1.0, 1.0, res)
    grid_x = axis[None, :, None]      # (1, N, 1) → index 1
    grid_y = axis[:, None, None]      # (N, 1, 1) → index 0
    grid_z = axis[None, None, :]      # (1, 1, N) → index 2

    occ = np.ones((res, res, res), dtype=bool)
    for v in views:
        mask = v["mask"]
        h, w = mask.shape
        # camera looks horizontally at a turntable (object rotated about Y):
        # u = x·cosθ + z·sinθ, v = y (Y up)
        u = np.broadcast_to(grid_x * v["cos"] + grid_z * v["sin"], occ.shape)
        vv = np.broadcast_to(grid_y, occ.shape)
        u_px = v["cx"] + u * v["px_per_world"]
        v_px = v["cy"] - vv * v["px_per_world"]  # image Y grows downward
        ui = np.floor(u_px).astype(int)
        vi = np.floor(v_px).astype(int)
        inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        # out-of-frame voxels are NOT in this silhouette cone
        hit = np.zeros(occ.shape, dtype=bool)
        hit[inside] = mask[vi[inside], ui[inside]]
        occ &= hit
    return occ


def _surface_from_volume(occ: np.ndarray, views, colorize: bool):
    """Emit a quad (2 triangles) per occupied-voxel face adjacent to empty space."""
    import numpy as np
    n = occ.shape[0]
    # pad so grid edges have an "empty" layer
    padded = np.zeros((n + 2, n + 2, n + 2), dtype=bool)
    padded[1:-1, 1:-1, 1:-1] = occ
    occ_indices = np.argwhere(occ)

    half = 1.0 / n  # voxel world size in [-1,1] grid

    verts: list[np.ndarray] = []
    faces: list[list[int]] = []
    cols: list[list[int]] = []
    for (iy, ix, iz) in occ_indices:
        for (dx, dy, dz) in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
            nx, ny, nz = ix + dx, iy + dy, iz + dz
            if padded[nx + 1, ny + 1, nz + 1]:
                continue  # neighbor occupied → internal face
            wx = -1.0 + (2.0 / n) * (0.5 + ix)
            wy = -1.0 + (2.0 / n) * (0.5 + iy)
            wz = -1.0 + (2.0 / n) * (0.5 + iz)
            base = np.array([wx, wy, wz])
            corners = _face_corners(dx, dy, dz, base, half)
            face_col = _face_color(ix, iy, iz, views, n, colorize) if colorize else None
            start = len(verts)
            for corner in corners:
                verts.append(corner)
                cols.append(face_col if face_col is not None else [255, 255, 255])
            faces.append([start, start + 1, start + 2])
            faces.append([start, start + 2, start + 3])
    if not verts:
        raise RuntimeError("Empty occupancy volume — cannot build a mesh.")

    vertices = np.array(verts, dtype=np.float32)
    triangles = np.array(faces, dtype=np.int64)
    colors = np.array(cols, dtype=np.uint8)
    if colors.ndim == 2 and colors.shape[1] == 3:
        colors = np.hstack([colors, np.full((len(colors), 1), 255, np.uint8)])
    else:
        colors = np.full((len(vertices), 4), 255, dtype=np.uint8)
    return vertices, triangles, colors


def _face_corners(dx, dy, dz, base, half):
    """4 corners of the face normal to (dx,dy,dz), CCW when seen from outside."""
    import numpy as np
    # tangents t1, t2 with cross(t1, t2) == (dx, dy, dz)
    if dx != 0:
        t1 = np.array([0.0, 1.0, 0.0])
        t2 = np.array([0.0, 0.0, float(dx)])
    elif dy != 0:
        t1 = np.array([1.0, 0.0, 0.0])
        t2 = np.array([0.0, 0.0, float(-dy)])
    else:
        t1 = np.array([1.0, 0.0, 0.0])
        t2 = np.array([0.0, float(dz), 0.0])
    s = half
    c00 = base - s * t1 - s * t2
    c01 = base + s * t1 - s * t2
    c11 = base + s * t1 + s * t2
    c10 = base - s * t1 + s * t2
    return [c00, c01, c11, c10]


def _face_color(ix, iy, iz, views, n, colorize):
    """Average foreground color of the voxel over the views that see it."""
    import numpy as np
    if not colorize:
        return None
    x = -1.0 + (2.0 / n) * (0.5 + ix)
    y = -1.0 + (2.0 / n) * (0.5 + iy)
    z = -1.0 + (2.0 / n) * (0.5 + iz)
    accumulate = np.zeros(3, dtype=np.float64)
    count = 0.0
    for v in views:
        mask = v["mask"]
        hh, ww = mask.shape
        u = x * v["cos"] + z * v["sin"]
        u_px = v["cx"] + u * v["px_per_world"]
        v_px = v["cy"] - y * v["px_per_world"]
        ui, vi = int(np.floor(u_px)), int(np.floor(v_px))
        if 0 <= ui < ww and 0 <= vi < hh and mask[vi, ui]:
            accumulate += v["rgb"].astype(np.float64)
            count += 1.0
    if count == 0:
        return None
    return (accumulate / count).astype(np.uint8)


# ---------------------------------------------------------------------------
# 3) Modly generator
# ---------------------------------------------------------------------------

class SpriteAtlas3DGenerator(BaseGenerator):
    MODEL_ID = "sprite-atlas-3d"
    DISPLAY_NAME = "Sprite Atlas to 3D"
    VRAM_GB = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._backbone: Reconstructor | None = None

    # -- Lifecycle ---------------------------------------------------------

    def is_downloaded(self) -> bool:
        return True

    def load(self) -> None:
        if self._backbone is None:
            self._backbone = self._make_reconstructor()
            self._backbone.load()

    def unload(self) -> None:
        self._backbone = None

    def is_loaded(self) -> bool:
        return self._backbone is not None

    # -- Inference ---------------------------------------------------------

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[object] = None,
    ) -> Path:
        from PIL import Image
        import numpy as np

        cols = int(params.get("cols", 4))
        rows = int(params.get("rows", 4))
        bg = params.get("background", "alpha")
        order = params.get("view_order", "row-major")
        size = int(params.get("sample_size", 512))

        self._report(progress_cb, 5, "Slicing atlas…")
        atlas = Image.open(BytesIO(image_bytes)).convert("RGBA")
        frames = slice_atlas(atlas, cols, rows)
        frames = [normalize_frame(f, size, bg) for f in frames]
        self._check_cancelled(cancel_event)

        angles = recover_view_angles(len(frames), order)

        self._report(progress_cb, 40, "Reconstructing mesh from views…")

        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.outputs_dir / f"{int(time.time())}_{uuid.uuid4().hex[:8]}.glb"

        self._backbone.reconstruct([np.asarray(f) for f in frames], angles, out_path, params)
        self._check_cancelled(cancel_event)

        self._report(progress_cb, 100, "Done")
        return out_path

    # -- Internal ----------------------------------------------------------

    @staticmethod
    def _make_reconstructor() -> Reconstructor:
        return VisualHullReconstructor()

    def _report(self, cb, pct, msg):
        if cb:
            cb(pct, msg)

    def _check_cancelled(self, cancel_event):
        if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
            raise GenerationCancelled("Generation cancelled")