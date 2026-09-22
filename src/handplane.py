"""Recover palm depth for pose engines that return 2D hands.

The problem
-----------
MediaPipe's hand model returns a depth for every landmark. RTMPose and
ViTPose do not: their hand keypoints are pixels, and `pose_schema` fills
the third component with a literal 0.0. A hand whose landmarks are all at
the same depth is a flat sheet facing the lens, and the palm basis the
retargeter builds from it (fingers x across-palm) then has a normal that
is *always* the camera axis. The wrist can only roll around the view
direction; it cannot tilt or turn, and the sign of the normal flips as
soon as the 2D triangle changes orientation, which snaps the hand through
180 degrees. That is what "the palm direction is getting worse" looks like
after switching engine, and no amount of calibration fixes it, because the
input carries no depth to calibrate against.

What this does
--------------
The palm -- wrist and the four finger MCP knuckles -- is close enough to a
rigid planar shape that its image is a projection of a known template.
Under weak perspective (orthographic + scale, which a hand a metre from
the lens is well inside), fitting that template to the observed 2D points
recovers the plane's orientation up to one sign: the classic two-fold
ambiguity of a planar target, where tilting the plane towards or away from
the camera produces the same picture. The sign is carried over from the
previous frame, so it stays consistent through a motion instead of
flipping frame to frame.

Every landmark is then given the depth it would have if it lay in that
plane. For the palm points that is nearly true; for a curled finger it is
not, so finger curl in depth is still missing. What it buys is a palm
basis that actually rotates, which is what wrist and hand orientation are
driven from -- and fingers are driven from angles, not depth.

Units: landmarks are image-normalised (x by width, y by height), so `y`
is scaled by the frame aspect before fitting and the depth that comes out
is in x's units, which is the convention the rest of the tracker uses.
"""

import math


# Palm template in palm coordinates: +x towards the pinky side, +y from
# the wrist towards the middle knuckle, in units of palm length. Rough
# adult proportions; only the shape matters, since scale is solved for.
_TEMPLATE = {
    0: (0.00, 0.00),    # wrist
    5: (-0.40, 0.95),   # index MCP
    9: (-0.06, 1.00),   # middle MCP
    13: (0.22, 0.95),   # ring MCP
    17: (0.46, 0.84),   # pinky MCP
}

# Below this the palm is so close to edge-on that the plane says nothing
# about depth and dividing by the normal's z explodes. Clamping rather
# than refusing keeps the hand from jumping when it passes through
# edge-on; the depth is wrong there either way, but it stays bounded.
_MIN_NZ = 0.25

# A hand whose landmark depths never differ by more than this is taken to
# carry no depth at all. Real MediaPipe output is orders of magnitude
# above it.
_FLAT_EPS = 1e-9

# How far the previous frame's answer outweighs the forearm's vote. Below
# 1 so that a confident forearm can pull the plane back after a pass
# through face-on, where continuity is certain and wrong.
_CONTINUITY_WEIGHT = 0.7


def is_flat(landmarks):
    """True when these landmarks carry no usable depth."""
    if not landmarks or len(landmarks) < 18:
        return False
    zs = [float(p[2]) for p in landmarks if len(p) > 2]
    if not zs:
        return True
    return (max(zs) - min(zs)) < _FLAT_EPS


def _fit_plane(landmarks, aspect, prev_normal, arm_hint=0.0):
    """(normal, note) of the palm plane, or (None, why not).

    The normal is in image units: x right, y down (image convention, as
    the landmarks are), z away from the camera.

    `arm_hint` is the depth component of the forearm's unit direction,
    elbow to wrist. It votes on which of the two candidate planes is
    real: a hand roughly continues the forearm, so the palm's wrist ->
    middle-knuckle axis should lean the same way in depth. The vote
    weighs itself -- a forearm across the view has a hint near zero and
    no say -- and it is the only thing that can recover the right branch
    after the palm passes through face-on, where continuity alone locks
    onto whichever tilt it had before.
    """
    pts = []
    for i, (tx, ty) in _TEMPLATE.items():
        if i >= len(landmarks):
            continue
        lm = landmarks[i]
        pts.append((tx, ty, float(lm[0]), float(lm[1]) * aspect))
    if len(pts) < 4:
        return None, "palm landmarks missing"

    # Least squares for u = A11*X + A12*Y + tx, v = A21*X + A22*Y + ty.
    n = float(len(pts))
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    mu = sum(p[2] for p in pts) / n
    mv = sum(p[3] for p in pts) / n
    sxx = syy = sxy = sxu = syu = sxv = syv = 0.0
    for tx, ty, u, v in pts:
        dx, dy = tx - mx, ty - my
        du, dv = u - mu, v - mv
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
        sxu += dx * du
        syu += dy * du
        sxv += dx * dv
        syv += dy * dv
    det = sxx * syy - sxy * sxy
    if abs(det) < 1e-12:
        return None, "degenerate palm"
    a11 = (syy * sxu - sxy * syu) / det
    a12 = (sxx * syu - sxy * sxu) / det
    a21 = (syy * sxv - sxy * syv) / det
    a22 = (sxx * syv - sxy * sxv) / det

    # The two template axes map to camera-space vectors c1, c2 whose first
    # two components we just solved for. Requiring them to be orthogonal
    # and equal in length -- which they are, being a rotation times a
    # scale -- pins down the missing depth components up to one sign.
    a = a11 * a11 + a21 * a21
    b = a12 * a12 + a22 * a22
    d = a11 * a12 + a21 * a22
    # z1, z2 are the depth components the projection threw away, from
    # |c1| = |c2| and c1.c2 = 0. Substituting one into the other gives a
    # quadratic in z1**2, whose non-negative root is this:
    disc = (b - a) * (b - a) + 4.0 * d * d
    p = 0.5 * ((b - a) + math.sqrt(max(0.0, disc)))
    if p > 1e-12:
        z1 = math.sqrt(p)
        z2 = -d / z1
    else:
        # The root collapses when the two axes stay perpendicular in the
        # image (d = 0) and it is the *second* that is foreshortened, so
        # all the missing depth sits in z2. Falling through the general
        # branch here would report a flat palm for every rotation about
        # the image x axis.
        z1 = 0.0
        z2 = math.sqrt(max(0.0, a - b))

    # Two plane orientations produce the same picture -- tilting towards
    # or away from the lens -- and they differ by negating both depths.
    # That is a real ambiguity in the data, not a bug to be solved, so
    # both are built and the one continuous with the last frame wins.
    best_n = None
    best_score = None
    for sign in (1.0, -1.0):
        c1 = (a11, a21, sign * z1)
        c2 = (a12, a22, sign * z2)
        nx = c1[1] * c2[2] - c1[2] * c2[1]
        ny = c1[2] * c2[0] - c1[0] * c2[2]
        nz = c1[0] * c2[1] - c1[1] * c2[0]
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if length < 1e-12:
            continue
        n = (nx / length, ny / length, nz / length)
        # A plane has no front, and negating the normal changes no depth
        # this produces, so fix the facing towards the camera and the two
        # candidates stay genuinely distinct instead of being each
        # other's negation.
        if n[2] > 0.0:
            n = (-n[0], -n[1], -n[2])
        c2_len = math.sqrt(b + z2 * z2)
        score = 0.0
        if c2_len > 1e-9:
            score += arm_hint * (sign * z2) / c2_len
        if prev_normal is not None:
            score += _CONTINUITY_WEIGHT * (n[0] * prev_normal[0]
                                           + n[1] * prev_normal[1]
                                           + n[2] * prev_normal[2])
        if best_score is None or score > best_score:
            best_n, best_score = n, score
    if best_n is None:
        return None, "degenerate palm fit"
    return best_n, ""


class PalmPlanes:
    """Per-side plane state, so the sign stays consistent over time."""

    def __init__(self):
        self._normal = {}
        self.note = ""

    def reset(self):
        self._normal.clear()

    def normal(self, side):
        return self._normal.get(side)

    def lift(self, side, landmarks, aspect, arm_hint=0.0):
        """Landmarks with a plausible depth, or the originals unchanged.

        Only touches hands that arrived with no depth at all, so a
        MediaPipe result passes through untouched.
        """
        if not is_flat(landmarks):
            return landmarks
        normal, note = _fit_plane(landmarks, aspect, self._normal.get(side),
                                  arm_hint)
        if normal is None:
            self.note = note
            return landmarks
        self._normal[side] = normal
        nz = normal[2]
        nz = math.copysign(max(abs(nz), _MIN_NZ), nz if nz else 1.0)
        wx = float(landmarks[0][0])
        wy = float(landmarks[0][1]) * aspect
        out = []
        for lm in landmarks:
            dx = float(lm[0]) - wx
            dy = float(lm[1]) * aspect - wy
            out.append((float(lm[0]), float(lm[1]),
                        -(normal[0] * dx + normal[1] * dy) / nz))
        self.note = "planar"
        return out


def register():
    pass


def unregister():
    pass
