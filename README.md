# Modly Sprite Atlas to 3D Extension

Loads a **sprite atlas** (a grid of frames showing the same character/body
from multiple angles) as input and reconstructs a **3D mesh** that takes
every view angle into account, instead of treating the sheet as one flat image.

> Built for [Modly](https://github.com/lightningpixel/modly) — local,
> open-source, AI image-to-3D generation. Based on the extension format of
> [modly-hunyuan3d-mini-extension](https://github.com/lightningpixel/modly-hunyuan3d-mini-extension).

## How it works

Because Modly core nodes only accept a **single image**, all the atlas logic
lives inside `generator.py`:

1. **Slice** — `slice_atlas()` cuts the sprite sheet into its grid frames.
2. **Align** — `recover_view_angles()` assigns an azimuth angle (0–360°) to
   each frame based on your grid layout (`row-major` or `column-major`).
3. **Normalize** — each frame is resized to a square canvas with the selected
   background (transparent/white/black).
4. **Reconstruct** — a **visual-hull (shape-from-silhouette)** volumetric fusion
   reconstructs the model from every posed frame:
   - each frame is reduced to a binary foreground silhouette,
   - every voxel of a 3D grid is projected to each view (the character is
     treated as a turntable rotated about the recovered azimuth angle),
   - a voxel is part of the model iff it projects inside the silhouette of
     **all** views (intersection of the silhouette cones),
   - occupied-voxel faces are extracted into a closed mesh and **per-vertex
     colour** is baked by averaging the visible foreground colour over views.

## Files

| File | Purpose |
| --- | --- |
| `manifest.json` | Extension metadata + node/params schema shown in Modly UI |
| `generator.py` | Atlas slicing + reconstruction pipeline (Modly `BaseGenerator`) |
| `setup.py` | Generates `pyproject.toml` for the isolated extension venv |
| `LICENSE` | MIT |

## Install in Modly

1. In the **Models** page click **Install from GitHub**.
2. Enter: `https://github.com/krasorx/modly-sprite-atlas-3d-extension`
3. Run **Repair** if the install fails so the extension venv is recreated.

## Input parameters (node)

| Param | Type | Default | Meaning |
| --- | --- | --- | --- |
| `cols` | int | 4 | Columns of the atlas grid |
| `rows` | int | 4 | Rows of the atlas grid |
| `background` | select | `alpha` | Frame background handling |
| `view_order` | select | `row-major` | How frames map to azimuth angles |
| `sample_size` | int | 512 | Per-frame resolution sent to the model |
| `resolution` | select | 96 | Voxel grid resolution (64/96/128) of the reconstruction |
| `colorize` | select | on | Bake per-vertex colour from the atlas frames |
| `seed` | int | -1 | Reproducibility; -1 = random |

## Reconstruction backend

By default the extension uses `VisualHullReconstructor`, a self-contained,
CPU-only volumetric fusion (numpy + opencv + trimesh) that uses **every**
atlas angle. It is deterministic and needs no model weights to download.

To push quality further, swap it for a neural SDF / NeuS fuser (e.g.
InstantMesh) by replacing `_make_reconstructor()` in `generator.py`.

## License

Unlicense (public domain). See `LICENSE`.