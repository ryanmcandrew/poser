"""Joint angles from canonical pose landmarks.

Landmark accuracy is not the same question as body form. "Is this
shoulder in the right place" is answered in pixels or metres and depends
on where the camera happens to be; "is this arm raised, and is the other
one matching it" is answered in degrees and does not. The angles are also
what a rig actually consumes, so they are the honest readout of whether a
change of estimator improved anything — a model can win on landmark error
and still produce a jittery elbow.

Angles are computed from the 3D lift when it is available and from image
coordinates otherwise, and which one was used is reported, because a 2D
angle is a projection: an arm swinging toward the camera changes a 3D
elbow angle and barely moves the 2D one.
"""

import math

from . import pose_schema as ps


def _angle(a, b, c):
    """Angle at `b` in the triangle a-b-c, in degrees, or None."""
    if a is None or b is None or c is None:
        return None
    u = tuple(a[i] - b[i] for i in range(len(b)))
    v = tuple(c[i] - b[i] for i in range(len(b)))
    nu = math.sqrt(sum(t * t for t in u))
    nv = math.sqrt(sum(t * t for t in v))
    if nu < 1e-9 or nv < 1e-9:
        return None
    dot = sum(u[i] * v[i] for i in range(len(u))) / (nu * nv)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def _mid(a, b):
    if a is None or b is None:
        return None
    return tuple((a[i] + b[i]) * 0.5 for i in range(len(a)))


class JointAngles:
    """One frame's body-form readout. Every field may be None."""

    __slots__ = ("elbow_L", "elbow_R", "shoulder_L", "shoulder_R",
                 "torso_lean", "elbow_symmetry", "shoulder_symmetry", "space")

    def __init__(self, space="none"):
        self.elbow_L = None
        self.elbow_R = None
        self.shoulder_L = None
        self.shoulder_R = None
        self.torso_lean = None
        self.elbow_symmetry = None
        self.shoulder_symmetry = None
        # "3d" (from the depth lift), "2d" (image plane) or "none".
        self.space = space

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}

    def summary(self):
        """One short line for the panel, skipping what was not measured."""
        if self.space == "none":
            return ""
        parts = []
        if self.elbow_L is not None or self.elbow_R is not None:
            parts.append("elbow %s/%s" % (_fmt(self.elbow_L), _fmt(self.elbow_R)))
        if self.shoulder_L is not None or self.shoulder_R is not None:
            parts.append("shoulder %s/%s" % (_fmt(self.shoulder_L),
                                             _fmt(self.shoulder_R)))
        if self.torso_lean is not None:
            parts.append("lean %s" % _fmt(self.torso_lean))
        if self.elbow_symmetry is not None:
            parts.append("asym %s" % _fmt(self.elbow_symmetry))
        if not parts:
            return ""
        return "%s [%s]" % ("  ".join(parts), self.space)


def _fmt(v):
    return "-" if v is None else "%.0f" % v


def compute(points, visibility=None, thr=0.0, space="3d"):
    """Joint angles from `points`, a mapping or sequence of canonical slots.

    `points` may be the lifted 3D dict (`PoseFrame.pose_lifted`), the world
    dict, or the 33-entry image landmark list; anything indexable by the
    MediaPipe slot numbers works. Landmarks below `thr` visibility are
    treated as absent.
    """
    def get(i):
        if visibility is not None:
            try:
                if float(visibility[i]) < thr:
                    return None
            except (IndexError, KeyError, TypeError):
                return None
        try:
            p = points[i]
        except (IndexError, KeyError, TypeError):
            return None
        return None if p is None else tuple(float(c) for c in p)

    out = JointAngles(space=space)
    sh_l, sh_r = get(ps.MP_LSHOULDER), get(ps.MP_RSHOULDER)
    el_l, el_r = get(ps.MP_LELBOW), get(ps.MP_RELBOW)
    wr_l, wr_r = get(ps.MP_LWRIST), get(ps.MP_RWRIST)
    hip_l, hip_r = get(ps.MP_LHIP), get(ps.MP_RHIP)

    out.elbow_L = _angle(sh_l, el_l, wr_l)
    out.elbow_R = _angle(sh_r, el_r, wr_r)
    # Arm elevation: measured against the torso, not against the world, so
    # it means the same thing whether the user is upright or leaning.
    out.shoulder_L = _angle(el_l, sh_l, hip_l)
    out.shoulder_R = _angle(el_r, sh_r, hip_r)

    if out.elbow_L is not None and out.elbow_R is not None:
        out.elbow_symmetry = abs(out.elbow_L - out.elbow_R)
    if out.shoulder_L is not None and out.shoulder_R is not None:
        out.shoulder_symmetry = abs(out.shoulder_L - out.shoulder_R)

    sh_mid, hip_mid = _mid(sh_l, sh_r), _mid(hip_l, hip_r)
    if sh_mid is not None and hip_mid is not None:
        # Lean away from vertical. In the body frame z is up; in image
        # space y is down, and the spine runs from hips up to shoulders in
        # both, so the reference axis is whichever one that is.
        if len(sh_mid) >= 3 and space == "3d":
            up = (hip_mid[0], hip_mid[1], hip_mid[2] + 1.0)
        else:
            up = (hip_mid[0], hip_mid[1] - 1.0) + tuple(hip_mid[2:])
        # Zero when the spine is parallel to that axis, i.e. upright.
        out.torso_lean = _angle(sh_mid, hip_mid, up)
    return out


def from_frame(pf, thr=0.0):
    """Best available angles for a `PoseFrame`.

    Prefers the depth lift, falls back to the engine's world pose, then to
    the raw image landmarks — in that order, because each is one step
    further from a real 3D measurement.
    """
    if pf is None or not getattr(pf, "pose_present", False):
        return JointAngles()
    vis = getattr(pf, "pose_visibility", None)
    lifted = getattr(pf, "pose_lifted", None)
    if lifted:
        return compute(lifted, vis, thr, space="3d")
    world = getattr(pf, "pose_world", None)
    if world:
        return compute(world, vis, thr, space="3d")
    image = getattr(pf, "pose_image", None)
    if image:
        return compute(image, vis, thr, space="2d")
    return JointAngles()


def register():
    pass


def unregister():
    pass
