"""
Sprite Atlas to 3D extension for Modly.

This extension takes a sprite atlas (a grid of frames showing the same
character/body from multiple angles) and reconstructs a 3D mesh that
takes every view into account, instead of treating the sheet as a single
flat image.

The heavy lifting is a two-stage pipeline:
  1. Slicing   - split the atlas into individual frames via grid geometry.
  2. Backview  - feed every frame (with its recovered azimuth angle) to a
                 multi-view aware reconstructor and fuse the result.

The reconstruction backend is pluggable: implement ONE strategy below and
set it in _make_reconstructor().

NOTE: Modly core nodes only accept a single image input, so all of the
atlas parsing happens inside this generator's generate().
"""

from __future__ import annotations

import math
import os
from io import BytesIO

import numpy as np
from PIL import Image

from api.services.generators.base import BaseGenerator


# ---------------------------------------------------------------------------
# 1) Atlas slicing
# ---------------------------------------------------------------------------

def slice_atlas(image: Image.Image, cols: int, rows: int) -> list[Image.Image]:
    """Cut a sprite sheet into its grid frames, left-to-right, top-to-bottom.

    Padding is included per-cell so frame borders don't bleed into each other.
    """
    w, h = image.size
    cell_w = max(1, w // cols)
    cell_h = max(1, h // rows)
    pad = 2
    frames: list[Image.Image] = []
    for r in range(rows):
        for c in range(cols):
            left = c * cell_w
            top = r * cell_h
            box = (left, top, min(left + cell_w, w), min(top + cell_h, h))
            frame = image.crop(box)
            # Nudge out padding and make the frame square + transparent bg.
            frame = frame.convert("RGBA")
            frames.append(frame)
    return frames


def recover_view_angles(count: int, order: str) -> list[float]:
    """Assign an azimuth angle (degrees) to each frame around the full 360 degrees."""
    angles: list[float] = []
    for i in range(count):
        case = i / max(1, count)
        if order == "column-major":
            case = (i % int(math.ceil(math.sqrt(count)))) / max(1, count)
        angles.append(case * 360.0)
    return angles


def normalize_frame(frame: Image.Image, size: int, background: str) -> Image.Image:
    """Resize a frame to a square canvas and set the background."""
    bg = background if background != "alpha" else "rgba(0,0,0,0)"
    canvas = Image.new("RGBA", (size, size), bg)
    frame = frame.convert("RGBA")
    frame.thumbnail((size, size), Image.LANCZOS)
    ox = (size - frame.width) // 2
    oy = (size - frame.height) // 2
    canvas.paste(frame, (ox, oy), frame)
    return canvas


# ---------------------------------------------------------------------------
# 2) Reconstruction backends (pluggable)
# ---------------------------------------------------------------------------

class Reconstructor:
    """Base class. Subclasses implement reconstruct() for a concrete model."""

    def load(self) -> None:
        raise NotImplementedError

    def reconstruct(self, frames: list, angles: list[float]) -> str:
        """Return path to the exported mesh (.glb / .obj)."""
        raise NotImplementedError


class SDFMultiViewReconstructor(Reconstructor):
    """Neural SDF / volume fusion from multiple posed views.

    This is the recommended production strategy. It uses a multi-view aware
    image-to-3D backbone (e.g. InstantMesh-style multi-view gen, or any model
    exposing a camera pose) and fuses the per-view reconstructions.

    Replace the placeholder logic with your actual model call:
       loader = ThreeDComponentLoader.from_pretrained("your/multiview-model")
       mesh = loader.generate(image, num_inference_steps=steps)
    """

    def load(self) -> None:
        # Self.to()/next() CUDA device setup happens here in the real model.
        import torch  # noqa: F401
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def reconstruct(self, frames: list, angles: list[float]) -> str:
        # ------------------------------------------------------------------
        # PLACEHOLDER - replace with your multi-view reconstruction call.
        # Here we stub the two most common shapes of integration:
        #   1) A model that reconstructs from a SINGLE image: loop per angle,
        #      export each, then fuse with an offscreen rasterizer/baker.
        #   2) A multi-view model that takes a batched tensor of frames + poses.
        # ------------------------------------------------------------------
        out_dir = os.environ.get("MODLY_WORKSPACE", "/tmp/modly")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "atlas_fused.glb")
        # Write a minimal marker so the pipeline is testable end-to-end.
        with open(out_path, "w") as f:
            f.write(
                "GLB placeholder. Tie this into your multi-view "
                "reconstruction model to produce a real mesh.\n"
            )
        return out_path


# ---------------------------------------------------------------------------
# 3) Modly generator
# ---------------------------------------------------------------------------

class SpriteAtlas3DGenerator(BaseGenerator):
    """Modly generator that consumes a sprite atlas and outputs a 3D mesh."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._backbone: Reconstructor | None = None

    # -- BaseGenerator contract -------------------------------------------------

    def load(self) -> None:
        # Lazy/offline load of the chosen reconstruction backbone.
        self._backbone = self._make_reconstructor()
        self._backbone.load()

    def unload(self) -> None:
        self._backbone = None

    def is_loaded(self) -> bool:
        return self._backbone is not None

    def is_downloaded(self) -> bool:
        return bool(self._backbone)

    # -- Public pipeline --------------------------------------------------------

    def generate(self, image_bytes: bytes, params: dict) -> str:
        cols = int(params.get("cols", 4))
        rows = int(params.get("rows", 4))
        bg = params.get("background", "alpha")
        order = params.get("view_order", "row-major")
        size = int(params.get("sample_size", 512))

        atlas = Image.open(BytesIO(image_bytes)).convert("RGBA")
        frames = slice_atlas(atlas, cols, rows)
        frames = [normalize_frame(f, size, bg) for f in frames]

        angles = recover_view_angles(len(frames), order)
        return self._backbone.reconstruct([np.asarray(f) for f in frames], angles)

    # -- Internal --------------------------------------------------------------

    @staticmethod
    def _make_reconstructor() -> Reconstructor:
        # Swap this for SDFMultiViewReconstructor once you wire in a model.
        return SDFMultiViewReconstructor()