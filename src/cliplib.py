"""Clip library: recorded video organised as category / subgroup / clip.

Why a library and not a file path
---------------------------------
Offline analysis only earns its cost if the same performance can be run
through several engines and the results compared side by side, and that
means the footage has to still be there tomorrow, under a name that says
what it is. So a clip is a file inside a two-level folder tree the user
owns, and every derived artefact sits beside it:

    <root>/<category>/<subgroup>/<name>.mp4             the footage
    <root>/<category>/<subgroup>/<name>.timing.json     capture timestamps
    <root>/<category>/<subgroup>/<name>.RTMPOSE.pose.json.gz   one analysis

Categories and subgroups are plain directories rather than an index file,
so the user can rearrange takes in Explorer and the addon simply sees the
new arrangement on the next scan. The subgroup level is optional: a clip
directly under a category is a clip with an empty subgroup.

Timing file
-----------
A webcam does not deliver frames at the rate it claims, and a video
container can only store one nominal rate. Baking an animation from the
nominal rate stretches or squeezes the performance, so the recorder writes
the wall-clock stamp of every frame it wrote next to the video, and the
analyser prefers those stamps over CAP_PROP_FPS.

Calibration profiles
--------------------
`<root>/_calibration/<name>.json` holds one saved calibration: the
performer's torso basis and each palm basis, as measured while they held
the rig's rest pose. Live tracking can calibrate because the user is
there to be asked to hold still; a recorded clip usually opens mid
performance, and calibrating off its first frames declares whatever the
hands happened to be doing to be the rest pose -- which is why offline
takes come out with the wrists rotated. A profile recorded once from a
calibration clip fixes every clip shot in that setup.

Analysis sidecar
----------------
One gzipped JSON per (clip, engine): a header describing how it was
produced and what it scored, plus the raw per-frame landmarks exactly as
the backend returned them. Raw landmarks rather than rig rotations,
because retargeting settings (limb lengths, smoothing, FOV, which
armature) change far more often than the estimation does, and re-solving
them is free while re-running ViTPose over a minute of video is not.
"""

import glob
import gzip
import json
import os
import re

import bpy


VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

_SIDECAR_SUFFIX = ".pose.json.gz"

# Folder inside the library holding calibration profiles. Named with a
# leading underscore so the category scan skips it.
CALIB_DIRNAME = "_calibration"

# Enough precision to place a landmark inside a pixel on an 8K frame, and
# about a third of the file size of full repr() floats.
_ROUND = 5


# ----------------------------------------------------------------------
# paths
# ----------------------------------------------------------------------

def default_root():
    """Where clips go when the user has not chosen anywhere.

    The addon's own folder: it is writable in every install layout Blender
    supports (the same place _models already caches downloads), and it
    keeps a fresh install working with no setup at all.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "clips")


def resolve_root(path_str):
    """Absolute library root for a (possibly blank or //relative) setting."""
    p = (path_str or "").strip()
    if not p:
        return default_root()
    return os.path.abspath(bpy.path.abspath(p))


def safe_name(name, fallback="clip"):
    """A file name that survives every filesystem we might land on."""
    cleaned = re.sub(r"[^A-Za-z0-9 _.-]+", "_", (name or "").strip())
    cleaned = cleaned.strip(" .")
    return cleaned or fallback


def clip_dir(root, category, subgroup):
    parts = [root]
    cat = safe_name(category, "")
    if cat:
        parts.append(cat)
        grp = safe_name(subgroup, "")
        if grp:
            parts.append(grp)
    return os.path.join(*parts)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def unique_path(directory, name, ext=".mp4"):
    """directory/name+ext, with a numeric suffix if that is taken.

    Recording over an existing clip would destroy footage that cannot be
    re-shot, so a name collision is resolved rather than reported.
    """
    base = safe_name(name, "take")
    candidate = os.path.join(directory, base + ext)
    n = 2
    while os.path.exists(candidate):
        candidate = os.path.join(directory, "%s_%03d%s" % (base, n, ext))
        n += 1
    return candidate


def stem(video_path):
    return os.path.splitext(video_path)[0]


def timing_path(video_path):
    return stem(video_path) + ".timing.json"


def sidecar_path(video_path, engine):
    return "%s.%s%s" % (stem(video_path), (engine or "").upper(),
                        _SIDECAR_SUFFIX)


def relative_label(root, path):
    """category/subgroup/name for display, or the bare name if off-tree."""
    try:
        rel = os.path.relpath(path, root)
    except ValueError:
        return os.path.basename(path)
    if rel.startswith(".."):
        return os.path.basename(path)
    return rel.replace("\\", "/")


# ----------------------------------------------------------------------
# scanning
# ----------------------------------------------------------------------

def list_categories(root):
    if not os.path.isdir(root):
        return []
    # A leading dot or underscore marks a folder the addon owns (the
    # calibration profiles live in one), so it is not offered as a place
    # to shoot into.
    return sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d))
                  and not d.startswith((".", "_")))


def list_subgroups(root, category):
    if not category:
        return []
    d = os.path.join(root, safe_name(category, ""))
    if not os.path.isdir(d):
        return []
    return sorted(x for x in os.listdir(d)
                  if os.path.isdir(os.path.join(d, x)) and not x.startswith("."))


def list_clips(root, category=None, subgroup=None, recursive=True):
    """Video files under the library, newest first.

    `category`/`subgroup` narrow the search; blank means "every one". The
    default is recursive because the usual question is "what have I shot",
    not "what is in this exact folder".
    """
    base = root
    if category:
        base = os.path.join(root, safe_name(category, ""))
        if subgroup:
            base = os.path.join(base, safe_name(subgroup, ""))
    if not os.path.isdir(base):
        return []
    pattern = "**/*" if recursive else "*"
    out = []
    skip = os.path.join(root, CALIB_DIRNAME).lower()
    for path in glob.iglob(os.path.join(glob.escape(base), pattern),
                           recursive=recursive):
        if os.path.splitext(path)[1].lower() not in VIDEO_EXTS:
            continue
        if path.lower().startswith(skip) or not os.path.isfile(path):
            continue
        out.append(os.path.abspath(path))
    out.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return out


def split_location(root, video_path):
    """(category, subgroup) a clip sits in, either possibly empty."""
    try:
        rel = os.path.relpath(os.path.dirname(video_path), root)
    except ValueError:
        return "", ""
    if rel in (".", ""):
        return "", ""
    parts = [p for p in rel.replace("\\", "/").split("/") if p not in (".", "")]
    if not parts or parts[0] == "..":
        return "", ""
    return parts[0], (parts[1] if len(parts) > 1 else "")


def list_analyses(video_path):
    """[(engine, sidecar_path)] for one clip, in engine-name order."""
    base = stem(video_path)
    out = []
    for path in glob.iglob(glob.escape(base) + ".*" + _SIDECAR_SUFFIX):
        engine = os.path.basename(path)[len(os.path.basename(base)) + 1:]
        engine = engine[:-len(_SIDECAR_SUFFIX)]
        if engine:
            out.append((engine.upper(), os.path.abspath(path)))
    out.sort()
    return out


def delete_clip(video_path):
    """Remove a clip and everything derived from it. Returns files removed."""
    removed = 0
    targets = [video_path, timing_path(video_path)]
    targets += [p for _, p in list_analyses(video_path)]
    for path in targets:
        try:
            if os.path.exists(path):
                os.remove(path)
                removed += 1
        except OSError as e:
            print("[FPV Pose Tracker] cannot delete %s: %s" % (path, e))
    return removed


# ----------------------------------------------------------------------
# video / timing metadata
# ----------------------------------------------------------------------

def video_info(path):
    """{'frames','fps','width','height','duration'} for a video, or {}."""
    try:
        import cv2
    except Exception:
        return {}
    cap = None
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return {}
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    except Exception:
        return {}
    finally:
        if cap is not None:
            cap.release()
    return {"frames": frames, "fps": fps, "width": w, "height": h,
            "duration": (frames / fps) if fps > 1e-6 else 0.0}


def write_timing(video_path, stamps, meta=None):
    """Store per-frame capture times (seconds from the first frame)."""
    data = dict(meta or {})
    data["stamps"] = [round(float(t), 4) for t in stamps]
    try:
        with open(timing_path(video_path), "w", encoding="utf-8") as f:
            json.dump(data, f)
    except OSError as e:
        print("[FPV Pose Tracker] cannot write timing file:", e)


def read_timing(video_path):
    path = timing_path(video_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        print("[FPV Pose Tracker] cannot read timing file:", e)
        return None


def frame_times(video_path, frame_count, fallback_fps):
    """Seconds from clip start for each frame, measured where possible.

    Recorded stamps are truncated or extended to `frame_count`, so a timing
    file that disagrees with the container (a dropped last frame, say)
    degrades into the nominal rate for the tail rather than misaligning
    everything after it.
    """
    frame_count = max(0, int(frame_count))
    timing = read_timing(video_path)
    stamps = [float(t) for t in (timing.get("stamps") or [])] if timing else []
    if len(stamps) >= frame_count > 0:
        return stamps[:frame_count]
    fps = float(fallback_fps or 0.0)
    if fps <= 1e-6 and timing:
        fps = float(timing.get("fps") or 0.0)
    if fps <= 1e-6:
        fps = 30.0
    if not stamps:
        return [i / fps for i in range(frame_count)]
    last = stamps[-1]
    return stamps + [last + (i + 1) / fps
                     for i in range(frame_count - len(stamps))]


# ----------------------------------------------------------------------
# analysis sidecars
# ----------------------------------------------------------------------

def write_analysis(path, header, frames):
    """Write one analysis, via a temporary file.

    An analysis can be twenty minutes of GPU time; a half-written file left
    by a crash mid-write would be indistinguishable from a good one until
    it failed to load, so the rename is the commit.
    """
    tmp = path + ".part"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump({"version": 1, "header": header, "frames": frames}, f)
    os.replace(tmp, path)
    return path


def read_analysis(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def read_header(path):
    """Just the header of an analysis. Cheap enough to call while drawing
    the clip list, which is why an analysis carries its scores in the
    header rather than making the panel reduce 2000 frames to show one
    number."""
    try:
        return read_analysis(path).get("header", {})
    except (OSError, ValueError, EOFError) as e:
        print("[FPV Pose Tracker] unreadable analysis %s: %s" % (path, e))
        return {}


# ----------------------------------------------------------------------
# calibration profiles
# ----------------------------------------------------------------------

def calib_dir(root):
    """Where calibration profiles live: one folder beside the categories.

    They sit in the library rather than in the .blend because a
    calibration describes the *performer and camera*, not the scene: the
    same one primes every rig and every project shot in that setup, which
    is the whole reason for recording a calibration pose once.
    """
    return os.path.join(root, CALIB_DIRNAME)


def profile_path(root, name):
    return os.path.join(calib_dir(root), safe_name(name, "default") + ".json")


def list_profiles(root):
    d = calib_dir(root)
    if not os.path.isdir(d):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(d)
                  if f.lower().endswith(".json"))


def read_profile(root, name):
    path = profile_path(root, name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        print("[FPV Pose Tracker] unreadable calibration profile %s: %s"
              % (path, e))
        return None


def write_profile(root, name, data):
    path = profile_path(root, name)
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    return path


def delete_profile(root, name):
    path = profile_path(root, name)
    try:
        if os.path.exists(path):
            os.remove(path)
            return True
    except OSError as e:
        print("[FPV Pose Tracker] cannot delete %s: %s" % (path, e))
    return False


def pack_landmarks(seq):
    """Landmark list -> plain rounded lists, JSON's smallest honest form."""
    return [[round(float(v), _ROUND) for v in pt] for pt in (seq or [])]


def register():
    pass


def unregister():
    pass
