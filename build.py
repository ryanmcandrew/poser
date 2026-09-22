"""Package the fpv_pose_tracker addon into a Blender-installable zip.

Usage:
    python build.py

Output:
    dist/fpv_pose_tracker-<version>.zip

Drop the produced zip on Blender (Edit > Preferences > Add-ons > Install)
or drag it onto a 3D Viewport (Blender 4.2+).
"""

import ast
import os
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "fpv_pose_tracker"
DIST = ROOT / "dist"


def read_version():
    text = (SRC / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r'"version"\s*:\s*\(([^)]+)\)', text)
    if not m:
        return "0.0.0"
    parts = [p.strip() for p in m.group(1).split(",") if p.strip()]
    return ".".join(parts)


def should_include(path: Path) -> bool:
    parts = set(path.parts)
    if "__pycache__" in parts:
        return False
    if path.suffix in (".pyc", ".pyo"):
        return False
    # Editor and tooling scratch output that has no business in a release.
    if any(p.startswith(".") and p not in (".", "..") for p in parts):
        return False
    return True


def main():
    if not SRC.exists():
        raise SystemExit("addon source not found: %s" % SRC)
    DIST.mkdir(exist_ok=True)
    version = read_version()
    out = DIST / ("fpv_pose_tracker-%s.zip" % version)
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in SRC.rglob("*"):
            if not path.is_file() or not should_include(path):
                continue
            rel = path.relative_to(SRC.parent)  # keep top-level pkg folder
            zf.write(path, rel.as_posix())
    print("Built:", out)


if __name__ == "__main__":
    main()
