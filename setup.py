"""
Sprite Atlas to 3D — extension setup script.

Creates an isolated venv and installs all required dependencies.
Called by Modly at extension install time with:

    python setup.py <json_args>

where json_args contains:
    python_exe   — path to Modly's embedded Python (used to create the venv)
    ext_dir      — absolute path to this extension directory
    gpu_sm       — GPU compute capability as integer (e.g. 86 for Ampere; 0 on macOS)
    cuda_version — CUDA major/minor encoded as integer (e.g. 124, 128)
    torch_flavor — Flavor of torch (cuda, rocm - defaults to cuda)
    accelerator  — "mps" | "cuda" | "cpu"  (passed by Electron since Modly 1.x)
    platform     — Electron's process.platform string ("win32", "darwin", "linux")
"""
import json
import platform
import subprocess
import sys
from pathlib import Path


# Base packages needed to load and run the atlas -> mesh pipeline.
# Extend this list when a concrete multi-view reconstruction backbone is wired in.
CORE_DEPS = [
    "Pillow>=10.0.0",   # atlas slicing / frame normalization (the currently missing dep)
    "numpy",
    "trimesh",          # mesh IO / export (.glb, .obj)
    "opencv-python-headless",
    "huggingface_hub",  # model weight downloads
]


def pip(venv: Path, *args: str) -> None:
    is_win = platform.system() == "Windows"
    pip_exe = venv / ("Scripts/pip.exe" if is_win else "bin/pip")
    subprocess.run([str(pip_exe), *args], check=True)


def setup(
    python_exe:    str,
    ext_dir:       Path,
    gpu_sm:        int = 0,
    cuda_version:  int = 0,
    torch_flavor:  str = "cuda",
    accelerator:   str = "",
    platform_name: str = "",
) -> None:
    venv = ext_dir / "venv"

    print(f"[setup] Creating venv at {venv} …")
    subprocess.run([python_exe, "-m", "venv", str(venv)], check=True)

    print("[setup] Installing core dependencies …")
    pip(venv, "install", *CORE_DEPS)

    print("[setup] Done. Venv ready at:", venv)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        args = json.loads(sys.argv[1])
        setup(
            python_exe    = args["python_exe"],
            ext_dir       = Path(args["ext_dir"]),
            gpu_sm        = int(args.get("gpu_sm", 0)),
            cuda_version  = int(args.get("cuda_version", 0)),
            torch_flavor  = args.get("torch_flavor", "cuda"),
            accelerator   = args.get("accelerator", ""),
            platform_name = args.get("platform", ""),
        )
    else:
        print("Usage: python setup.py <json_args>")
        print('  e.g. python setup.py \'{"python_exe":"...","ext_dir":"...","gpu_sm":86}\'')
        sys.exit(1)