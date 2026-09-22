"""Package the fpv_pose_tracker addon into a Blender-installable zip.

Usage:
    python build.py

Output:
    dist/fpv_pose_tracker-<version>.zip  (packaged from src/)

Drop the produced zip on Blender (Edit > Preferences > Add-ons > Install)
or drag it onto a 3D Viewport (Blender 4.2+).
"""

import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
DIST = ROOT / "dist"
# The folder name inside the zip. Blender imports the addon as a Python
# package named after this directory, and `bl_idname = __package__` in the
# preferences depends on it, so it stays fixed whatever the source folder
# on disk is called.
PKG = "fpv_pose_tracker"


def read_version():
    text = (SRC / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r'"version"\s*:\s*\(([^)]+)\)', text)
    if not m:
        return "0.0.0"
    parts = [p.strip() for p in m.group(1).split(",") if p.strip()]
    return ".".join(parts)


def should_include(rel: Path) -> bool:
    """Decide on a path relative to SRC, not an absolute one.

    An absolute path drags in every parent directory name, so a checkout
    living under a dotted folder would silently exclude the whole addon.
    """
    parts = set(rel.parts)
    if "__pycache__" in parts:
        return False
    # The default clip library lives inside the addon folder, so a
    # developer who has recorded anything would otherwise ship gigabytes
    # of their own webcam footage in the release zip.
    if "clips" in parts:
        return False
    if rel.suffix in (".pyc", ".pyo"):
        return False
    # Editor and tooling scratch output that has no business in a release.
    if any(p.startswith(".") for p in parts):
        return False
    return True


def main():
    if not SRC.exists():
        raise SystemExit("addon source not found: %s" % SRC)
    DIST.mkdir(exist_ok=True)
    version = read_version()
    out = DIST / ("%s-%s.zip" % (PKG, version))
    if out.exists():
        out.unlink()
    count = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(SRC.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(SRC)
            if not should_include(rel):
                continue
            # Rewrite the source folder's name to the package name Blender
            # will import it under.
            zf.write(path, (Path(PKG) / rel).as_posix())
            count += 1
    print("Built: %s (%d files)" % (out, count))


if __name__ == "__main__":
    main()
