"""Record tracked poses into an editable "take", then bake it to an Action.

Why a take instead of keyframing live
-------------------------------------
Webcam tracking produces bad frames. MediaPipe loses the body for a moment
and hallucinates a pose somewhere else, so the rig snaps to an unintended
position and snaps back. Keyframing straight into an Action bakes those
snaps in, and cleaning them up afterwards in the Graph Editor means hunting
through 160 quaternion curves.

So a recording is held in memory as a list of samples first. Each sample
keeps every mapped bone's rotation plus a handful of armature-space points
along the arm chain. The points are what make cleanup tractable: a bad
frame is one whose wrist/elbow/palm sits far off the path its own
neighbours describe, which is a single scalar per sample rather than a
quaternion per bone. The user trims the ends, drops the flagged samples,
and only then do we build the Action.

Everything here is scale-free: deviations are divided by the rig's shoulder
width and reported as a percentage of it, so the same thresholds work on a
0.4 m rig and an 8 m one.

Blender 4.4+ / 5.x note
-----------------------
``Action.fcurves`` no longer exists; actions are layered and slotted. Curves
are created through ``action.layers[0].strips[0].channelbag(slot).fcurves``
and the slot has to be assigned to the object's animation data alongside the
action, or nothing evaluates. :func:`_channelbag` hides that, with a
fallback to the legacy API for Blender 3.6/4.0-4.3.
"""

import math
import time

import bpy


# A take longer than this is almost certainly a forgotten "stop". At 30 fps
# this is ~20 minutes, about 60 MB of samples.
MAX_SAMPLES = 36000

PREVIEW_ACTION_NAME = "FPV_Take_Preview"

# Reasons a sample can be flagged. Stored as a bitmask on the sample.
FLAG_OUTLIER = 1 << 0    # sits off the path its neighbours describe
FLAG_JUMP = 1 << 1       # large step away from the previous sample
FLAG_LOST = 1 << 2       # tracker reported nothing detected

_FLAG_LABELS = (
    (FLAG_OUTLIER, "off path"),
    (FLAG_JUMP, "jump"),
    (FLAG_LOST, "no tracking"),
)


def flag_text(flags):
    parts = [label for bit, label in _FLAG_LABELS if flags & bit]
    return ", ".join(parts)


class Sample:
    """One captured frame of the take."""

    __slots__ = ("t", "quats", "points", "det", "flags", "score", "step")

    def __init__(self, t, quats, points, det):
        self.t = t
        # Flat float list, 4 per bone, aligned with Take.bone_names.
        self.quats = quats
        # Flat float list, 3 per point, aligned with Take.point_names.
        self.points = points
        # (pose, left hand, right hand) detection flags.
        self.det = det
        self.flags = 0
        self.score = 0.0
        self.step = 0.0


class Take:
    """A finished (or in-progress) recording."""

    def __init__(self, armature_name, bone_names, point_names, scale):
        self.armature_name = armature_name
        self.bone_names = list(bone_names)
        self.point_names = list(point_names)
        # Rig shoulder width, used to make deviation scores scale-free.
        self.scale = float(scale) if scale and scale > 1e-6 else 1.0
        self.samples = []
        self.started = time.time()
        self.finished = None
        self.capture_fps = 0.0

    def duration(self):
        if not self.samples:
            return 0.0
        return self.samples[-1].t - self.samples[0].t

    def __len__(self):
        return len(self.samples)


# The live take. Kept module-level rather than in the scene because the
# quaternion arrays are far too heavy for RNA properties; the scene only
# mirrors the light per-sample row data the UI list needs.
_STATE = {
    "take": None,          # finished take being edited
    "capturing": None,     # take currently being filled
    "prev_action": None,   # action displaced by the preview, to restore
    "prev_slot": "",
    "dirty": True,         # preview action needs a rebake
    "live": False,         # a tracker is driving the rig right now
}


def set_live(value):
    """Tell the recorder whether a tracker owns the pose bones. While it
    does, an assigned preview action would be evaluated on top of every
    tick and the two would fight."""
    _STATE["live"] = bool(value)


def is_live():
    return _STATE["live"]


# ----------------------------------------------------------------------
# capture
# ----------------------------------------------------------------------

def is_recording():
    return _STATE["capturing"] is not None


def has_take():
    return _STATE["take"] is not None


def take():
    return _STATE["take"]


def capturing():
    """The take currently being filled, if any."""
    return _STATE["capturing"]


def start(retargeter, armature):
    """Begin a take. Returns the new Take."""
    bone_names = sorted({n for n in retargeter.bone_map.values() if n
                         and armature.pose.bones.get(n) is not None})
    point_names = retargeter.capture_point_names()
    scale = getattr(retargeter, "_rig_shoulder_half", None)
    scale = (scale * 2.0) if scale else 1.0
    tk = Take(armature.name, bone_names, point_names, scale)
    _STATE["capturing"] = tk
    return tk


def capture(retargeter, frame_time=None):
    """Append one sample. Called from the tracking timer, after the frame
    has been retargeted. Returns False once the take is full.

    `frame_time` is the tracker's own capture time for this frame. It must
    be used rather than the wall clock at call time: the Blender timer polls
    faster than the camera delivers, so wall-clock stamps would space
    samples by the tick rather than by when the motion actually happened,
    and the baked action's timing would not match the performance.
    """
    tk = _STATE["capturing"]
    if tk is None:
        return True
    if len(tk.samples) >= MAX_SAMPLES:
        return False
    quats, points = retargeter.capture_state(tk.bone_names, tk.point_names)
    det = retargeter.last_detection()
    now = time.time() if frame_time is None else float(frame_time)
    tk.samples.append(Sample(
        now - tk.started, quats, points,
        (bool(det["pose"]), bool(det["left"]), bool(det["right"])),
    ))
    return True


def finish():
    """End the take being captured and promote it to the editable take."""
    tk = _STATE["capturing"]
    _STATE["capturing"] = None
    if tk is None:
        return None
    tk.finished = time.time()
    d = tk.duration()
    tk.capture_fps = (len(tk.samples) - 1) / d if d > 1e-6 else 0.0
    if not tk.samples:
        return None
    _STATE["take"] = tk
    mark_dirty()
    return tk


def discard():
    _STATE["take"] = None
    _STATE["capturing"] = None
    mark_dirty()


def mark_dirty():
    _STATE["dirty"] = True


# ----------------------------------------------------------------------
# outlier scoring
# ----------------------------------------------------------------------

def _median(values):
    if not values:
        return 0.0
    v = sorted(values)
    n = len(v)
    mid = n // 2
    return v[mid] if n % 2 else 0.5 * (v[mid - 1] + v[mid])


def compute_scores(tk):
    """Score every sample by how far it sits off the path its neighbours
    describe.

    For each tracked point, the neighbours' midpoint is where the point
    would be if it were travelling smoothly (at constant velocity the
    midpoint is exact, so honest motion scores ~0 no matter how fast it is).
    The deviation from that midpoint is what a snap-and-return looks like.
    The score is the worst point on the body, as a percentage of the rig's
    shoulder width.

    ``step`` is the plain distance from the previous sample. It catches the
    other failure: tracking that latches onto a wrong position and *stays*
    there, where only the transition frame is anomalous.
    """
    n = len(tk.samples)
    npts = len(tk.point_names)
    inv = 100.0 / tk.scale
    for i, sm in enumerate(tk.samples):
        sm.score = 0.0
        sm.step = 0.0
    if npts == 0:
        return
    for i in range(n):
        sm = tk.samples[i]
        prev = tk.samples[i - 1] if i > 0 else None
        nxt = tk.samples[i + 1] if i + 1 < n else None
        worst_dev = 0.0
        worst_step = 0.0
        for p in range(npts):
            o = p * 3
            cx, cy, cz = sm.points[o], sm.points[o + 1], sm.points[o + 2]
            if prev is not None:
                px, py, pz = prev.points[o], prev.points[o + 1], prev.points[o + 2]
                d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2 + (cz - pz) ** 2)
                if d > worst_step:
                    worst_step = d
                if nxt is not None:
                    nx, ny, nz = nxt.points[o], nxt.points[o + 1], nxt.points[o + 2]
                    mx, my, mz = (px + nx) * 0.5, (py + ny) * 0.5, (pz + nz) * 0.5
                    d = math.sqrt((cx - mx) ** 2 + (cy - my) ** 2 + (cz - mz) ** 2)
                    if d > worst_dev:
                        worst_dev = d
        sm.score = worst_dev * inv
        sm.step = worst_step * inv


def detect_outliers(tk, sensitivity=0.5, min_jump=5.0, flag_lost=True,
                    lo=None, hi=None):
    """Flag anomalous samples. Returns (threshold, flagged_count).

    The threshold is robust (median + k x scaled MAD) so that a take with a
    few violent snaps does not raise its own bar until the snaps look
    normal. ``min_jump`` is a floor, as a percentage of shoulder width, so a
    take of someone holding still does not flag its own sensor noise.
    """
    compute_scores(tk)
    n = len(tk.samples)
    lo = 0 if lo is None else max(0, lo)
    hi = n - 1 if hi is None else min(n - 1, hi)
    for sm in tk.samples:
        sm.flags = 0
    if n < 3 or lo >= hi:
        return 0.0, 0

    window = tk.samples[lo:hi + 1]
    scores = [sm.score for sm in window]
    med = _median(scores)
    mad = _median([abs(s - med) for s in scores])
    # 1.4826 x MAD estimates sigma for normally distributed data.
    sigma = 1.4826 * mad
    k = 8.0 - 7.0 * max(0.0, min(1.0, float(sensitivity)))
    thr = max(float(min_jump), med + k * sigma)

    # A step is suspicious at a higher bar than a deviation: honest fast
    # motion produces big steps but small deviations.
    step_scores = [sm.step for sm in window]
    smed = _median(step_scores)
    smad = _median([abs(s - smed) for s in step_scores])
    step_thr = max(float(min_jump) * 2.0, smed + (k + 2.0) * 1.4826 * smad)

    flagged = 0
    for sm in window:
        f = 0
        if sm.score > thr:
            f |= FLAG_OUTLIER
        if sm.step > step_thr:
            f |= FLAG_JUMP
        if flag_lost and not (sm.det[0] or sm.det[1] or sm.det[2]):
            f |= FLAG_LOST
        sm.flags = f
        if f:
            flagged += 1
    return thr, flagged


# ----------------------------------------------------------------------
# baking
# ----------------------------------------------------------------------

def frame_of(tk, index, scene_fps, start_frame, origin_index=0):
    """Timeline frame a sample lands on. The take is anchored so that
    `origin_index` (the first kept sample) sits on `start_frame`, which is
    why trimming the head of a take does not leave a gap in front of it."""
    if not tk.samples:
        return start_frame
    t0 = tk.samples[origin_index].t
    return start_frame + int(round((tk.samples[index].t - t0) * scene_fps))


def _channelbag(action, armature):
    """(fcurve container, slot) for `armature` on a fresh action.

    Blender 4.4 replaced ``action.fcurves`` with layers/slots/channelbags.
    """
    slot = None
    if hasattr(action, "slots"):
        slot = action.slots.new(id_type="OBJECT", name=armature.name)
        layer = action.layers.new("Layer")
        strip = layer.strips.new(type="KEYFRAME")
        return strip.channelbag(slot, ensure=True), slot
    return action, None   # legacy: action.fcurves / action.groups


def assign_action(armature, action, slot):
    if armature.animation_data is None:
        armature.animation_data_create()
    armature.animation_data.action = action
    if slot is not None:
        try:
            armature.animation_data.action_slot = slot
        except Exception:
            pass


def bake(tk, armature, name, indices, scene_fps, start_frame,
         interpolation="BEZIER"):
    """Build an Action from the given sample indices.

    Samples that were dropped simply get no key, so Blender interpolates
    across the hole — which is exactly the desired repair for a one-frame
    snap, and why removal needs no separate "fill gap" step.
    """
    if not indices:
        return None, None

    origin = indices[0]
    # Several captured samples can land on one timeline frame when the
    # tracker runs faster than the scene fps; the last one wins.
    by_frame = {}
    for i in indices:
        by_frame[frame_of(tk, i, scene_fps, start_frame, origin)] = i
    frames = sorted(by_frame)
    if not frames:
        return None, None

    action = bpy.data.actions.new(name)
    cbag, slot = _channelbag(action, armature)
    interp = 2 if interpolation == "BEZIER" else 1

    nkeys = len(frames)
    for b, bone_name in enumerate(tk.bone_names):
        if armature.pose.bones.get(bone_name) is None:
            continue
        base = b * 4
        # Pull the bone's quaternion track out first so neighbouring keys
        # can be put on the same hemisphere. Without this, q and -q (the
        # same orientation) read as opposite ends of the sphere and the
        # interpolated rotation swings the long way round.
        track = []
        prev = None
        for f in frames:
            sm = tk.samples[by_frame[f]]
            q = sm.quats[base:base + 4]
            if prev is not None:
                if (q[0] * prev[0] + q[1] * prev[1]
                        + q[2] * prev[2] + q[3] * prev[3]) < 0.0:
                    q = [-c for c in q]
            prev = q
            track.append(q)

        path = 'pose.bones["%s"].rotation_quaternion' % bpy.utils.escape_identifier(bone_name)
        group = None
        try:
            group = cbag.groups.new(bone_name)
        except Exception:
            group = None
        for ci in range(4):
            fc = cbag.fcurves.new(path, index=ci)
            if group is not None:
                fc.group = group
            kp = fc.keyframe_points
            kp.add(nkeys)
            co = []
            for j, f in enumerate(frames):
                co.append(float(f))
                co.append(track[j][ci])
            kp.foreach_set("co", co)
            kp.foreach_set("interpolation", [interp] * nkeys)
            fc.update()

    return action, slot


def _restore_previous(armature):
    prev = _STATE["prev_action"]
    slot_id = _STATE["prev_slot"]
    _STATE["prev_action"] = None
    _STATE["prev_slot"] = ""
    if armature is None or armature.animation_data is None:
        return
    try:
        armature.animation_data.action = prev
    except Exception:
        # `prev` can be a dangling pointer if the action was deleted while
        # the preview was up.
        armature.animation_data.action = None
        return
    if prev is not None and slot_id:
        for s in getattr(prev, "slots", []):
            if s.identifier == slot_id:
                try:
                    armature.animation_data.action_slot = s
                except Exception:
                    pass
                break


def clear_preview(armature=None):
    """Drop the preview action and put back whatever the rig had before.

    Does nothing when no preview is up: otherwise every call would clear the
    rig's own action, since "what it had before" was never recorded.
    """
    act = bpy.data.actions.get(PREVIEW_ACTION_NAME)
    if act is None:
        _STATE["prev_action"] = None
        _STATE["prev_slot"] = ""
        return
    if armature is not None:
        _restore_previous(armature)
    try:
        bpy.data.actions.remove(act, do_unlink=True)
    except Exception:
        pass
    mark_dirty()


def update_preview(scene, armature, indices, force=False):
    """Rebuild the preview action so the viewport shows exactly what would
    be saved. Cheap enough to redo on every edit (one foreach_set per
    curve), so the edits stay honest rather than approximate."""
    if not _STATE["dirty"] and not force:
        return None
    tk = _STATE["take"]
    if tk is None or armature is None:
        return None

    old = bpy.data.actions.get(PREVIEW_ACTION_NAME)
    had_preview = old is not None
    if not had_preview and armature.animation_data is not None:
        # First preview: remember what we are displacing.
        _STATE["prev_action"] = armature.animation_data.action
        slot = getattr(armature.animation_data, "action_slot", None)
        _STATE["prev_slot"] = slot.identifier if slot else ""
    if old is not None:
        try:
            bpy.data.actions.remove(old, do_unlink=True)
        except Exception:
            pass

    s = scene.fpv_pose
    action, slot = bake(
        tk, armature, PREVIEW_ACTION_NAME, indices,
        scene_fps=scene_fps(scene), start_frame=int(s.take_start_frame),
        interpolation=s.take_interpolation,
    )
    if action is None:
        _STATE["dirty"] = False
        return None
    assign_action(armature, action, slot)
    _STATE["dirty"] = False
    return action


def push_nla(armature, action, slot, track_name, start_frame):
    """Drop the action onto its own NLA track. The action is unlinked from
    the active slot first, which is what the NLA expects."""
    if armature.animation_data is None:
        armature.animation_data_create()
    ad = armature.animation_data
    if ad.action is action:
        ad.action = None
    track = None
    for t in ad.nla_tracks:
        if t.name == track_name:
            track = t
    if track is None:
        track = ad.nla_tracks.new()
        track.name = track_name
    strip = track.strips.new(action.name, int(start_frame), action)
    if slot is not None:
        try:
            strip.action_slot = slot
        except Exception:
            pass
    return strip


def scene_fps(scene):
    r = scene.render
    return float(r.fps) / float(r.fps_base or 1.0)


# ----------------------------------------------------------------------
# scene mirror
#
# The heavy arrays stay in this module; the scene only carries one light
# row per sample so the UI list has something to draw and so the keep/drop
# decision survives an undo.
# ----------------------------------------------------------------------

def sync_rows(scene):
    """Rebuild scene.fpv_take_frames from the current take."""
    rows = scene.fpv_take_frames
    rows.clear()
    tk = _STATE["take"]
    if tk is None:
        return
    s = scene.fpv_pose
    fps = scene_fps(scene)
    start = int(s.take_start_frame)
    for i, sm in enumerate(tk.samples):
        r = rows.add()
        r.index = i
        r.frame = frame_of(tk, i, fps, start)
        r.score = sm.score
        r.step = sm.step
        r.flags = sm.flags
        r.set_enabled_quiet(True)
    mark_dirty()


def resync_frames(scene):
    """Recompute each row's timeline frame after the start frame or the
    scene's frame rate changed. Keep/drop decisions and flags are left
    alone — only the mapping from sample to frame moved."""
    tk = _STATE["take"]
    if tk is None:
        return
    rows = scene.fpv_take_frames
    if len(rows) != len(tk.samples):
        return
    fps = scene_fps(scene)
    start = int(scene.fpv_pose.take_start_frame)
    for i, r in enumerate(rows):
        r.frame = frame_of(tk, i, fps, start)
    mark_dirty()


def apply_flags(scene, auto_exclude):
    """Copy freshly computed flags onto the rows, optionally dropping the
    flagged samples. Only flagged rows are touched, so a sample the user
    already judged by hand keeps that judgement."""
    tk = _STATE["take"]
    if tk is None:
        return 0
    rows = scene.fpv_take_frames
    dropped = 0
    for r in rows:
        if r.index >= len(tk.samples):
            continue
        sm = tk.samples[r.index]
        r.flags = sm.flags
        r.score = sm.score
        r.step = sm.step
        if sm.flags and auto_exclude:
            r.set_enabled_quiet(False)
        if sm.flags and not r.enabled:
            dropped += 1
    mark_dirty()
    return dropped


def included_indices(scene):
    """Sample indices that would be keyed, in order: inside the trim range
    and not dropped."""
    tk = _STATE["take"]
    if tk is None:
        return []
    s = scene.fpv_pose
    rows = scene.fpv_take_frames
    n = len(tk.samples)
    lo = max(0, min(int(s.take_start), n - 1))
    hi = max(lo, min(int(s.take_end), n - 1))
    if len(rows) != n:
        return list(range(lo, hi + 1))
    return [i for i in range(lo, hi + 1) if rows[i].enabled]


def refresh_preview(context, force=False):
    """Rebake the preview action for the current edits, if previewing is on
    and nothing is live-driving the rig."""
    scene = context.scene
    s = scene.fpv_pose
    tk = _STATE["take"]
    if tk is None or not s.take_preview:
        return None
    armature = s.armature
    if armature is None or armature.name != tk.armature_name:
        armature = bpy.data.objects.get(tk.armature_name)
    if armature is None:
        return None
    # A live tracker writes pose bones every tick; an assigned action would
    # be evaluated on top of that and the two would fight.
    if _STATE["live"]:
        return None
    return update_preview(scene, armature, included_indices(scene), force=force)


def stats(scene):
    """(kept, total, dropped, flagged, frame_lo, frame_hi) for the UI."""
    tk = _STATE["take"]
    if tk is None:
        return (0, 0, 0, 0, 0, 0)
    total = len(tk.samples)
    kept = included_indices(scene)
    flagged = sum(1 for sm in tk.samples if sm.flags)
    if not kept:
        return (0, total, total, flagged, 0, 0)
    fps = scene_fps(scene)
    start = int(scene.fpv_pose.take_start_frame)
    lo = frame_of(tk, kept[0], fps, start, kept[0])
    hi = frame_of(tk, kept[-1], fps, start, kept[0])
    return (len(kept), total, total - len(kept), flagged, lo, hi)


def register():
    pass


def unregister():
    _STATE["take"] = None
    _STATE["capturing"] = None
    _STATE["prev_action"] = None
