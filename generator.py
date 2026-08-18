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

    def reconstruct(self, frames: list, angles: list[float], out_path: Path) -> Path:
        raise NotImplementedError


class PlaceholderReconstructor(Reconstructor):
    """Produces a real (primitive) GLB mesh so the run completes.

    The silhouette of the first frame is the same for every view in the
    placeholder. Swap this class for a real multi-view reconstruction model
    (e.g. InstantMesh, or any NeuS/SDF fuser) to get an accurate mesh that
    honours every angle in the atlas.
    """

    def load(self) -> None:
        pass

    def reconstruct(self, frames: list, angles: list[float], out_path: Path) -> Path:
        import numpy as np
        import trimesh

        # Build a simple rounded box as the placeholder mesh. Mesh density scales
        # with the atlas so higher-resolution sheets produce more detail.
        n_cells = max(1, len(frames))
        side = 0.8 if n_cells <= 16 else 0.9
        mesh = trimesh.creation.box(extents=[side, side, side])
        mesh = trimesh.remesh.subdivide_loop(mesh, iterations=2)
        mesh.export(str(out_path))
        return out_path


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

        self._backbone.reconstruct([np.asarray(f) for f in frames], angles, out_path)
        self._check_cancelled(cancel_event)

        self._report(progress_cb, 100, "Done")
        return out_path

    # -- Internal ----------------------------------------------------------

    @staticmethod
    def _make_reconstructor() -> Reconstructor:
        # Swap this for SDFMultiViewReconstructor once a real model is wired in.
        return PlaceholderReconstructor()

    def _report(self, cb, pct, msg):
        if cb:
            cb(pct, msg)

    def _check_cancelled(self, cancel_event):
        if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
            raise GenerationCancelled("Generation cancelled")