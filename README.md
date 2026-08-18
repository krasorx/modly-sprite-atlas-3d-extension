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
4. **Reconstruct** — every posed frame is fed to a multi-view reconstruction
   backend and fused into a single mesh.

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
| `seed` | int | -1 | Reproducibility; -1 = random |

## Putting real reconstruction behind it

`generator.py` ships with a **placeholder** `SDFMultiViewReconstructor`. To get
a real mesh, wire in your multi-view backbone. Recommended model families:

- InstantMesh (multi-view gen + NeuS fusion)
- Any model exposing camera pose + NeuS/SDF fusion from multiple views
- A per-angle reconstruction loop fused with an offscreen baker

Two integration shapes are documented in `SDFMultiViewReconstructor.reconstruct()`:
single-image-backbone looping per angle, or a batched multi-view call.

## License

Unlicense (public domain). See `LICENSE`.