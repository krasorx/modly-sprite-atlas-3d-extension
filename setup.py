"""Shared build metadata to keep dependency versions consistent."""
import os
from pathlib import Path

MODLY_EXT_ID = "sprite-atlas-3d"

TORCH_INDEX = "https://download.pytorch.org/whl/cu121"


def pinned_requirements(platform_python: dict | None = None) -> list[str]:
    """Return the base dependency list for the extension venv.

    Extend this set with the packages required by your chosen
    multi-view reconstruction backbone (torch, numpy, PIL, trimesh, etc.).
    """
    return [
        "numpy",
        "pillow",
        "trimesh",
        "torch",         # resize/tense helpers
        "torchvision",
    ]


def extra_index_urls() -> list[str]:
    """PyTorch wheel index; helps pip resolve CUDA wheels on Linux/Windows."""
    urls = [TORCH_INDEX]
    if os.getenv("MODLY_CUDA_VERSION", "").startswith("12."):
        urls.append("https://download.pytorch.org/whl/cu128")
    return urls


def resolved_pyproject() -> str:
    requires = pinned_requirements()
    index_marker = "\n".join(
        "      --extra-index-url " + u for u in extra_index_urls()
    )
    return f"""\
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "modly-{MODLY_EXT_ID}"
version = "0.1.0"
description = "Modly extension: sprite atlas -> 3D mesh with multi-view reconstruction."
requires-python = ">=3.10"
dependencies = {requires!r}

[tool.pip]
{index_marker}
"""


def write_pyproject(path: Path) -> Path:
    target = path / "pyproject.toml"
    target.write_text(resolved_pyproject())
    return target


if __name__ == "__main__":
    out = Path(os.getenv("MODLY_INSTALL_DIR", "."))
    print(f"Written: {write_pyproject(out)}")