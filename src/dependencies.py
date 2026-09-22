"""Handles installing MediaPipe / OpenCV into Blender's bundled Python."""

import importlib
import os
import subprocess
import sys

REQUIRED = (
    ("cv2", "opencv-python"),
    ("mediapipe", "mediapipe"),
    ("numpy", "numpy"),
)


def missing_modules():
    missing = []
    for import_name, pip_name in REQUIRED:
        try:
            importlib.import_module(import_name)
        except Exception:
            missing.append((import_name, pip_name))
    return missing


def dependencies_ok():
    return len(missing_modules()) == 0


def _python_executable():
    # Blender >= 2.92 exposes sys.executable pointing at bundled python
    return sys.executable


def install_dependencies():
    """Install missing wheels into Blender's bundled Python.

    Returns (ok, log_text).
    """
    log_lines = []
    py = _python_executable()

    def run(cmd):
        log_lines.append("$ " + " ".join(cmd))
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.stdout:
                log_lines.append(proc.stdout.strip())
            if proc.stderr:
                log_lines.append(proc.stderr.strip())
            return proc.returncode == 0
        except Exception as e:
            log_lines.append("error: " + str(e))
            return False

    # Ensure pip is available.
    run([py, "-m", "ensurepip", "--upgrade"])
    run([py, "-m", "pip", "install", "--upgrade", "pip"])

    ok_all = True
    for _, pip_name in missing_modules():
        ok = run([py, "-m", "pip", "install", "--user", pip_name])
        if not ok:
            # fall back to non-user site-packages
            ok = run([py, "-m", "pip", "install", pip_name])
        ok_all = ok_all and ok

    # Blender caches sys.path; refresh user site
    try:
        import site
        importlib.reload(site)
    except Exception:
        pass

    # Also add the per-user site path in case pip installed there.
    try:
        import site as _s
        user_site = _s.getusersitepackages()
        if user_site and user_site not in sys.path:
            sys.path.append(user_site)
    except Exception:
        pass

    return ok_all and dependencies_ok(), "\n".join(log_lines)


def register():
    pass


def unregister():
    pass
