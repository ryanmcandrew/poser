"""Installs each pose engine's wheels where Blender's Python will find them.

Which packages are needed depends on the engine the user picked, and the
engines do not overlap: MediaPipe ships its own runtime, while
RTMPose/ViTPose want `rtmlib` over ONNX Runtime. Nothing here installs an
engine the user is not running - the ONNX stack alone is a few hundred
megabytes, and demanding it from someone on MediaPipe would be rude.

Where the wheels go matters as much as which ones. Blender embeds Python
with `ENABLE_USER_SITE` turned off (confirmed on 5.1), so
`pip install --user` lands in a directory Blender never puts on
`sys.path`. The install then looks like it worked, the session that ran it
can even use the packages if something appends the path by hand, and every
later Blender start reports the module missing again. That is the bug this
module exists to not have: wheels go into an addon-managed directory under
Blender's user scripts folder, and `ensure_paths()` -- called from
`register()`, before any backend tries to import cv2 -- puts that
directory, and the invisible user site, on `sys.path` for good.

The user site is included rather than abandoned: earlier versions of this
addon installed there, so for an existing install those wheels are already
on disk and re-downloading them would be a few hundred megabytes of
pointless traffic.
"""

import importlib
import os
import site
import subprocess
import sys

from . import pose_engines

# The historical default set: what the addon needed before the engine
# selector existed, kept so callers that pass no engine behave as before.
REQUIRED = (
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("mediapipe", "mediapipe"),
)

# onnxruntime-gpu replaces onnxruntime rather than adding to it, and pulls
# a CUDA stack behind it, so it is never installed as a requirement - it is
# offered as its own button for users who have the hardware.
GPU_PIP = {
    "RTMPOSE": "onnxruntime-gpu",
    "VITPOSE": "onnxruntime-gpu",
}

# Directory name for the installed wheels. It sits outside the addon's own
# folder on purpose: reinstalling the addon zip replaces that folder, and a
# few hundred megabytes of wheels should not evaporate with it. Under
# Blender's versioned scripts directory is right, because these are
# compiled wheels tied to one Python ABI.
_LIB_DIRNAME = "fpv_pose_tracker_wheels"

_paths_added = []


def lib_dir(create=False):
    """The directory this addon installs wheels into.

    Falls back to a folder beside the addon package when Blender's user
    paths are unavailable (importing these modules outside Blender, as the
    tests do), so nothing here needs `bpy`.
    """
    base = None
    try:
        import bpy
        base = bpy.utils.user_resource("SCRIPTS", create=create)
    except Exception:
        base = None
    if not base:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, _LIB_DIRNAME)
    if create:
        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            pass
    return path


def _user_site():
    """The per-user site-packages directory, visible to Blender or not."""
    try:
        path = site.getusersitepackages()
    except Exception:
        return None
    if isinstance(path, (list, tuple)):
        path = path[0] if path else None
    return path or None


def search_paths():
    """Extra import directories, in the order they are searched."""
    out = [lib_dir()]
    user = _user_site()
    if user:
        out.append(user)
    return out


def ensure_paths():
    """Put this addon's wheel directories on `sys.path`.

    Appends rather than prepends: Blender ships its own numpy, pip will
    happily drag a second copy in as a dependency of something else, and
    the copy Blender loaded at startup has to keep winning.

    Returns the directories this call added.
    """
    added = []
    for path in search_paths():
        if not path or not os.path.isdir(path):
            continue
        if path in sys.path:
            continue
        # addsitedir appends and also processes any .pth files, which is
        # how a few distributions register themselves.
        try:
            site.addsitedir(path)
        except Exception:
            sys.path.append(path)
        added.append(path)
        if path not in _paths_added:
            _paths_added.append(path)
    if added:
        # A directory that did not exist when an earlier import failed is
        # remembered as "nothing here"; without this, a fresh install stays
        # invisible until Blender restarts.
        importlib.invalidate_caches()
    return added


def path_report():
    """Human-readable account of where imports are looked for."""
    lines = []
    lines.append("python: %s" % sys.executable)
    lines.append("user site enabled in this Python: %s"
                 % getattr(site, "ENABLE_USER_SITE", "?"))
    for path in search_paths():
        state = "on sys.path" if path in sys.path else "NOT on sys.path"
        exists = "exists" if os.path.isdir(path) else "missing"
        lines.append("  %s  [%s, %s]" % (path, exists, state))
    return "\n".join(lines)


def _missing(modules):
    missing = []
    for import_name, pip_name in modules:
        try:
            importlib.import_module(import_name)
        except Exception:
            missing.append((import_name, pip_name))
    return missing


def missing_modules(engine=None):
    """Modules missing for `engine`, or for the historical default set."""
    if engine is None:
        return _missing(REQUIRED)
    return _missing(pose_engines.required_modules(engine))


def dependencies_ok(engine=None):
    return len(missing_modules(engine)) == 0


def missing_base_modules():
    """Modules no engine can do without: OpenCV and numpy.

    These gate the whole addon. Which estimator is installed does not,
    because a user running RTMPose has no reason to fetch MediaPipe too.
    """
    return _missing(pose_engines.BASE_MODULES)


def module_origin(import_name):
    """Where `import_name` actually resolves, or None. For diagnostics."""
    try:
        mod = importlib.import_module(import_name)
    except Exception:
        return None
    path = getattr(mod, "__file__", None)
    if path:
        return os.path.dirname(path)
    return "(namespace or builtin)"


def engine_status(engine):
    """(ok, message) describing whether `engine` can run right now.

    "Right now" is the literal claim: every module is imported, not merely
    looked for on disk, because a wheel in a directory Blender cannot see
    is exactly as useful as no wheel at all.
    """
    missing = missing_modules(engine)
    if not missing:
        return True, "ready"
    return False, "missing: " + ", ".join(name for name, _ in missing)


def _python_executable():
    # Blender >= 2.92 exposes sys.executable pointing at bundled python
    return sys.executable


def install_dependencies(engine=None, packages=None):
    """Install missing wheels so Blender's Python can import them.

    With `engine`, only that engine's requirements are considered; with
    `packages`, exactly those pip names are installed whatever imports
    already succeed. Returns (ok, log_text).
    """
    log_lines = []
    py = _python_executable()
    target = lib_dir(create=True)

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

    wanted = list(packages) if packages else \
        [pip_name for _, pip_name in missing_modules(engine)]
    if not wanted:
        log_lines.append("nothing to install")
        ensure_paths()
        return True, "\n".join(log_lines)

    log_lines.append("target: " + target)
    ok_all = True
    for pip_name in wanted:
        # --target puts the wheels somewhere Blender is guaranteed to look
        # (see ensure_paths) and somewhere writable, which Blender's own
        # site-packages under Program Files is not. --user is not an option
        # at all: Blender disables the user site, and pip refuses the flag
        # when it is disabled. --upgrade is what lets --target replace
        # files already in place.
        ok = run([py, "-m", "pip", "install", "--upgrade",
                  "--target", target,
                  "--no-warn-script-location",
                  "--disable-pip-version-check",
                  pip_name])
        if not ok:
            log_lines.append("! pip failed for " + pip_name)
        ok_all = ok_all and ok

    added = ensure_paths()
    if added:
        log_lines.append("added to sys.path: " + ", ".join(added))

    # An explicit package list has no import name to verify against
    # (onnxruntime-gpu imports as `onnxruntime`), so pip's own verdict is
    # all there is to go on.
    if packages:
        installed = ok_all
    else:
        still_missing = missing_modules(engine)
        installed = ok_all and not still_missing
        if still_missing:
            log_lines.append("still not importable: "
                             + ", ".join(n for n, _ in still_missing))
            log_lines.append(path_report())
    return installed, "\n".join(log_lines)


def register():
    # Before any engine backend tries to import anything: an addon-managed
    # install is only real once its directory is on sys.path, and doing it
    # here is what makes it survive a Blender restart.
    ensure_paths()


def unregister():
    pass
