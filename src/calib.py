"""Where the cameras actually stand, recovered from the user's own body.

Triangulating a wrist between two cameras needs more than the Yaw slider.
Yaw alone says which way a camera faces; it says nothing about where it is,
and a ray needs both. Worse, a hand-typed yaw is wrong by however much the
user's guess is wrong, and a few degrees of extrinsic error moves a
triangulated wrist further than the depth error we are trying to remove.

So the rig measures itself. Each camera already produces a metric estimate
of the body in its own frame (``PoseFrame.pose_lifted_cam``: the depth lift,
in metres, in that camera's physical frame). Two cameras looking at the same
person therefore hold two copies of one rigid point set, and the transform
between them *is* the extrinsic. Recovering it is Kabsch, in closed form,
with no scale ambiguity to resolve afterwards - which is the main reason
this is done from the lift rather than from an essential matrix, whose
result would be up to scale and far more sensitive to landmark noise.

That closed-form answer is only as good as the monocular lift that fed it,
so it is the starting point, not the result. It is then refined against the
2D, which is the part that was actually measured: triangulate the landmarks
with the current extrinsics, re-solve each non-reference camera's pose from
those points by minimising reprojection error, repeat. Averaging the lift's
depth error over a calibration window makes the init unbiased enough for
that refinement to land in the right basin.

One gauge has to be pinned by hand. Scaling every point and every baseline
together leaves all reprojection errors unchanged, so the refinement cannot
see scale at all; after each triangulation the points are rescaled so the
shoulder width matches the width the user was measured to have. Everything
downstream is metric because of that one line.
"""

import math
import time

from . import tracker as T

# Landmarks the calibration is built from. Shoulders and hips are the most
# stable points on the body and the arms supply the motion that makes the
# pose observable; fingers and face are too small and too fast.
CALIB_LANDMARKS = (
    T.POSE_LSHOULDER, T.POSE_RSHOULDER,
    T.POSE_LELBOW, T.POSE_RELBOW,
    T.POSE_LWRIST, T.POSE_RWRIST,
    T.POSE_LHIP, T.POSE_RHIP,
)

# A landmark below this visibility is not worth calibrating from.
_CALIB_VIS = 0.6
# Samples needed before an estimate is attempted, and the most kept.
_MIN_SAMPLES = 40
_MAX_SAMPLES = 900
# Frames closer together than this add nothing: the body has not moved, and
# a thousand copies of one pose cannot constrain anything a single copy
# cannot. Calibration wants *variety*, not volume.
_SAMPLE_INTERVAL = 0.08
# A sample must differ from the ones already kept by at least this much
# (metres, summed over the tracked points) to earn its place.
_NOVELTY = 0.05
# Refinement passes of (triangulate -> re-solve each camera's pose).
_REFINE_PASSES = 6
# Reprojection error (pixels, median) above which the result is not trusted
# and the rig falls back to the Yaw-derived guess.
_MAX_REPROJ_PX = 12.0
# A camera whose recovered position is further than this from the reference
# has almost certainly been fitted to noise.
_MAX_BASELINE = 6.0


def _lazy_np():
    import numpy
    return numpy


class CameraModel:
    """A pinhole camera with a pose, in the reference camera's frame.

    ``R`` and ``t`` take a world point to this camera: ``x_cam = R @ X + t``.
    The principal point is the image centre, which is what the rest of the
    add-on already assumes when it turns a FOV into a focal length.
    """

    __slots__ = ("f_px", "W", "H", "R", "t", "source")

    def __init__(self, f_px, W, H, R=None, t=None, source="identity"):
        np = _lazy_np()
        self.f_px = float(f_px)
        self.W = int(W)
        self.H = int(H)
        self.R = np.eye(3) if R is None else np.asarray(R, dtype=np.float64)
        self.t = np.zeros(3) if t is None else np.asarray(t, dtype=np.float64)
        self.source = source

    # --- geometry ---------------------------------------------------------

    def center(self):
        """Where this camera sits in world coordinates."""
        return -self.R.T @ self.t

    def ray(self, u, v):
        """Unit direction in *world* coordinates for an image-normalised
        point, pointing away from the lens."""
        np = _lazy_np()
        d = np.array([(u - 0.5) * self.W,
                      (v - 0.5) * self.H,
                      self.f_px], dtype=np.float64)
        d = self.R.T @ d
        n = float(np.linalg.norm(d))
        return d / n if n > 1e-12 else d

    def project(self, X):
        """World point -> (u, v) image-normalised, or None behind the lens."""
        np = _lazy_np()
        x = self.R @ np.asarray(X, dtype=np.float64) + self.t
        if x[2] <= 1e-4:
            return None
        return (0.5 + (self.f_px * x[0] / x[2]) / self.W,
                0.5 + (self.f_px * x[1] / x[2]) / self.H)

    def project_px(self, X):
        """World point -> (x, y) in pixels from the image centre, or None."""
        np = _lazy_np()
        x = self.R @ np.asarray(X, dtype=np.float64) + self.t
        if x[2] <= 1e-4:
            return None
        return (self.f_px * x[0] / x[2], self.f_px * x[1] / x[2])

    def to_px(self, u, v):
        """Image-normalised -> pixels from the image centre."""
        return ((u - 0.5) * self.W, (v - 0.5) * self.H)

    # --- serialisation ----------------------------------------------------

    def as_list(self):
        return [self.f_px, self.W, self.H] + \
            [float(x) for x in self.R.ravel()] + [float(x) for x in self.t]

    @classmethod
    def from_list(cls, vals, source="stored"):
        np = _lazy_np()
        vals = list(vals)
        if len(vals) < 15:
            return None
        R = np.array(vals[3:12], dtype=np.float64).reshape(3, 3)
        t = np.array(vals[12:15], dtype=np.float64)
        return cls(vals[0], int(vals[1]), int(vals[2]), R, t, source)


def yaw_guess(f_px, W, H, yaw_deg, standoff=1.2, ref_standoff=1.2):
    """The camera a Yaw slider describes, as a fallback when nothing has
    been measured yet.

    The reference camera sits at the origin looking down +Z at a user
    ``ref_standoff`` metres away; this camera stands at ``yaw_deg`` around
    that user, at its own distance, looking back at them. It is a guess in
    exactly the way the user's typed angle is a guess, and it is used only
    until the rig has measured itself.
    """
    np = _lazy_np()
    a = math.radians(float(yaw_deg))
    # The user is on the reference camera's axis at ref_standoff.
    user = np.array([0.0, 0.0, ref_standoff])
    # Positive yaw is to the user's right, which is the reference image's
    # left, i.e. -x in the reference camera's physical frame.
    pos = user + np.array([-math.sin(a), 0.0, -math.cos(a)]) * standoff
    fwd = user - pos
    n = float(np.linalg.norm(fwd))
    if n < 1e-6:
        return CameraModel(f_px, W, H, source="yaw")
    fwd = fwd / n
    # The world frame is the reference camera's, so world +y is already
    # "down". Build image-right from it rather than from an "up" vector:
    # getting that sign wrong rolls the camera 180 degrees about its own
    # optical axis, which still projects onto the sensor and so fails
    # silently as a plausible-looking but upside-down rig.
    down_world = np.array([0.0, 1.0, 0.0])
    right = np.cross(down_world, fwd)
    rn = float(np.linalg.norm(right))
    if rn < 1e-6:
        return CameraModel(f_px, W, H, source="yaw")
    right = right / rn
    down = np.cross(fwd, right)
    R = np.vstack([right, down, fwd])        # world -> camera
    return CameraModel(f_px, W, H, R, -R @ pos, source="yaw")


# --------------------------------------------------------------------------
# triangulation
# --------------------------------------------------------------------------

def triangulate(cams, obs, weights=None):
    """Least-squares intersection of the rays from several cameras.

    ``obs`` is {slot: (u, v)}; at least two are needed. Solved as the point
    minimising the squared perpendicular distance to every ray, which is
    linear and needs no initial guess.
    """
    np = _lazy_np()
    if len(obs) < 2:
        return None
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for slot, (u, v) in obs.items():
        cam = cams[slot]
        d = cam.ray(u, v)
        c = cam.center()
        w = 1.0 if weights is None else max(0.0, float(weights.get(slot, 1.0)))
        if w <= 0.0:
            continue
        # I - d d^T projects onto the plane perpendicular to the ray.
        P = np.eye(3) - np.outer(d, d)
        A += w * P
        b += w * (P @ c)
    if abs(np.linalg.det(A)) < 1e-12:
        return None
    try:
        return np.linalg.solve(A, b)
    except Exception:
        return None


def _kabsch(np, P, Q):
    """Rigid transform taking P onto Q: returns (R, t) with Q ~ R @ P + t."""
    pc = P.mean(axis=0)
    qc = Q.mean(axis=0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = 1.0 if np.linalg.det(Vt.T @ U.T) > 0 else -1.0
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, qc - R @ pc


def _kabsch_robust(np, P, Q, trim=0.25):
    """Kabsch, then re-fit on the best-agreeing points.

    The lift is occasionally badly wrong on one joint - it has a front/back
    ambiguity it resolves with a learned sign - and one such joint pulls a
    plain least-squares fit off by tens of degrees.
    """
    R, t = _kabsch(np, P, Q)
    if len(P) >= 8:
        err = np.linalg.norm((P @ R.T + t) - Q, axis=1)
        keep = err <= np.quantile(err, 1.0 - trim)
        if keep.sum() >= 6:
            R, t = _kabsch(np, P[keep], Q[keep])
    return R, t


class RigCalibrator:
    """Collects paired observations and solves for where the cameras are."""

    def __init__(self, n_cams, min_samples=_MIN_SAMPLES):
        self.n_cams = int(n_cams)
        self.min_samples = int(min_samples)
        self._samples = []        # list of {slot: {idx: (uv, X_cam, vis)}}
        self._last_t = 0.0
        self._last_key = None
        self.status = "waiting for both cameras"
        self.reproj_px = None
        self.done = False

    # --- collection -------------------------------------------------------

    def sample_count(self):
        return len(self._samples)

    def progress(self):
        return min(1.0, len(self._samples) / float(max(1, self.min_samples)))

    def _novel(self, np, key):
        """Reject a sample that says the same thing as the last one. A user
        holding still would otherwise fill the window with one pose."""
        if self._last_key is None:
            return True
        d = float(np.abs(key - self._last_key).sum())
        return d >= _NOVELTY

    def add(self, frames, now=None):
        """``frames`` is {slot: PoseFrame}. Returns True if it was kept."""
        np = _lazy_np()
        now = time.time() if now is None else now
        if now - self._last_t < _SAMPLE_INTERVAL:
            return False
        usable = {}
        for slot, pf in frames.items():
            if pf is None or not pf.pose_present or not pf.pose_image:
                continue
            per = {}
            for idx in CALIB_LANDMARKS:
                uv = pf.pose_image.get(idx)
                X = pf.pose_lifted_cam.get(idx)
                vis = float(pf.pose_visibility.get(idx, 0.0))
                if uv is None or vis < _CALIB_VIS:
                    continue
                per[idx] = (uv, X, vis)
            if len(per) >= 4:
                usable[slot] = per
        if len(usable) < 2:
            self.status = "needs %d cameras seeing you at once" % max(2, self.n_cams)
            return False
        shared = set.intersection(*(set(v) for v in usable.values()))
        if len(shared) < 4:
            self.status = "cameras are not seeing the same joints"
            return False
        key = np.array(sorted(
            [usable[s][i][0][0] for s in usable for i in shared] +
            [usable[s][i][0][1] for s in usable for i in shared]))
        if self._last_key is not None and key.shape == self._last_key.shape \
                and not self._novel(np, key):
            return False
        self._last_key = key
        self._last_t = now
        self._samples.append(usable)
        if len(self._samples) > _MAX_SAMPLES:
            # Drop from the middle: the earliest samples anchor the fit and
            # the newest carry whatever the user is doing now.
            del self._samples[len(self._samples) // 2]
        self.status = "collecting (%d/%d) - keep moving your arms" % (
            len(self._samples), self.min_samples)
        return True

    # --- solving ----------------------------------------------------------

    def _pairs(self, np, ref, slot):
        """Point pairs (X in ref's frame, X in slot's frame) over the window."""
        P, Q = [], []
        for s in self._samples:
            if ref not in s or slot not in s:
                continue
            for idx, (_, Xr, _) in s[ref].items():
                other = s[slot].get(idx)
                if other is None or Xr is None or other[1] is None:
                    continue
                P.append(Xr)
                Q.append(other[1])
        if not P:
            return None, None
        return np.array(P, dtype=np.float64), np.array(Q, dtype=np.float64)

    def _solve_pose(self, np, cam, points, uvs):
        """Re-solve one camera's (R, t) from world points and their pixels.

        Gauss-Newton on the reprojection error, with the rotation carried as
        an incremental rotation vector so it stays a rotation.
        """
        R = cam.R.copy()
        t = cam.t.copy()
        for _ in range(12):
            J = np.zeros((2 * len(points), 6))
            r = np.zeros(2 * len(points))
            rows = 0
            for X, (px, py) in zip(points, uvs):
                x = R @ X + t
                if x[2] <= 1e-3:
                    continue
                inv = 1.0 / x[2]
                u = cam.f_px * x[0] * inv
                v = cam.f_px * x[1] * inv
                # d(proj)/d(x_cam)
                D = cam.f_px * np.array([[inv, 0.0, -x[0] * inv * inv],
                                         [0.0, inv, -x[1] * inv * inv]])
                # d(x_cam)/d(omega) for x_cam -> (I + [w]x) x_cam
                S = np.array([[0.0, x[2], -x[1]],
                              [-x[2], 0.0, x[0]],
                              [x[1], -x[0], 0.0]])
                J[rows:rows + 2, 0:3] = D @ S
                J[rows:rows + 2, 3:6] = D
                r[rows] = u - px
                r[rows + 1] = v - py
                rows += 2
            if rows < 12:
                return R, t
            J = J[:rows]
            r = r[:rows]
            try:
                # Small damping: a near-degenerate view (everything on one
                # line) otherwise throws the step to infinity.
                step = np.linalg.solve(J.T @ J + 1e-6 * np.eye(6), -J.T @ r)
            except Exception:
                break
            w = step[0:3]
            th = float(np.linalg.norm(w))
            if th > 1e-12:
                K = np.array([[0.0, -w[2], w[1]],
                              [w[2], 0.0, -w[0]],
                              [-w[1], w[0], 0.0]]) / th
                dR = np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)
            else:
                dR = np.eye(3)
            R = dR @ R
            t = dR @ t + step[3:6]
            if th < 1e-7 and float(np.linalg.norm(step[3:6])) < 1e-7:
                break
        return R, t

    def _triangulate_window(self, np, cams, shoulder_width):
        """Every sample's landmarks, triangulated with the current cameras,
        paired with the pixels each camera saw them at.

        Returns (world points, {slot: [(px, py), ...]}, median reprojection).
        """
        pts = []
        pix = {slot: [] for slot in range(self.n_cams)}
        owners = []
        for s in self._samples:
            shared = set.intersection(*(set(v) for v in s.values()))
            obs_all = {}
            for idx in shared:
                obs = {slot: s[slot][idx][0] for slot in s}
                w = {slot: s[slot][idx][2] for slot in s}
                X = triangulate(cams, obs, w)
                if X is None:
                    continue
                obs_all[idx] = X
            if len(obs_all) < 4:
                continue
            # Pin the scale gauge: reprojection cannot see a uniform scaling
            # of the whole rig, so the shoulder width has to declare it.
            if (T.POSE_LSHOULDER in obs_all and T.POSE_RSHOULDER in obs_all
                    and shoulder_width):
                d = float(np.linalg.norm(obs_all[T.POSE_LSHOULDER]
                                         - obs_all[T.POSE_RSHOULDER]))
                if d > 1e-3:
                    k = shoulder_width / d
                    if 0.2 < k < 5.0:
                        obs_all = {i: X * k for i, X in obs_all.items()}
            for idx, X in obs_all.items():
                pts.append(X)
                owners.append({slot: cams[slot].to_px(*s[slot][idx][0])
                               for slot in s})
        if not pts:
            return None, None, None
        errs = []
        for slot in range(self.n_cams):
            pix[slot] = [o.get(slot) for o in owners]
        for X, o in zip(pts, owners):
            for slot, target in o.items():
                got = cams[slot].project_px(X)
                if got is not None:
                    errs.append(math.hypot(got[0] - target[0], got[1] - target[1]))
        med = float(np.median(errs)) if errs else None
        return np.array(pts), pix, med

    def solve(self, cams, shoulder_width=None):
        """Refine ``cams`` in place-ish: returns a new list of CameraModel.

        ``cams`` supplies the intrinsics and the starting extrinsics (the
        Yaw guess). Camera 0 is the reference and never moves - it *is* the
        world frame, so there is nothing for it to be measured against.
        """
        np = _lazy_np()
        if len(self._samples) < self.min_samples:
            self.status = "collecting (%d/%d) - keep moving your arms" % (
                len(self._samples), self.min_samples)
            return None
        out = [CameraModel(c.f_px, c.W, c.H, c.R, c.t, c.source) for c in cams]

        # --- closed-form start from the two metric point clouds ----------
        for slot in range(1, len(out)):
            P, Q = self._pairs(np, 0, slot)
            if P is None or len(P) < 12:
                self.status = "cam%d: not enough shared joints" % slot
                continue
            R, t = _kabsch_robust(np, P, Q)
            out[slot] = CameraModel(out[slot].f_px, out[slot].W, out[slot].H,
                                    R, t, source="measured")

        # --- refine against the pixels -----------------------------------
        best = None
        for _ in range(_REFINE_PASSES):
            pts, pix, med = self._triangulate_window(np, out, shoulder_width)
            if pts is None:
                break
            if best is None or (med is not None and med < best[0]):
                best = (med, [CameraModel(c.f_px, c.W, c.H, c.R, c.t, c.source)
                              for c in out])
            for slot in range(1, len(out)):
                keep = [(X, p) for X, p in zip(pts, pix[slot]) if p is not None]
                if len(keep) < 12:
                    continue
                R, t = self._solve_pose(np, out[slot],
                                        [k[0] for k in keep], [k[1] for k in keep])
                out[slot] = CameraModel(out[slot].f_px, out[slot].W,
                                        out[slot].H, R, t, "measured")
        if best is not None and best[0] is not None:
            med, cand = best
            self.reproj_px = med
            baselines = [float(np.linalg.norm(c.center())) for c in cand[1:]]
            if med > _MAX_REPROJ_PX:
                self.status = ("fit too loose (%.1f px) - falling back to Yaw. "
                               "Check each camera's FOV and Yaw." % med)
                return None
            if baselines and max(baselines) > _MAX_BASELINE:
                self.status = ("fit put a camera %.1f m away - falling back to "
                               "Yaw" % max(baselines))
                return None
            self.done = True
            self.status = "calibrated: %.1f px error, baseline %s" % (
                med, ", ".join("%.2f m" % b for b in baselines) or "n/a")
            return cand
        self.status = "could not fit the cameras - falling back to Yaw"
        return None


def describe(cams):
    """One short line per camera for the panel.

    Reported as distance from the reference camera and the angle between
    the two cameras' optical axes. Both are properties of the recovered
    geometry alone. Restating it as a Yaw would be friendlier but it would
    be a guess: yaw is an angle *around the user*, and where the user is
    standing is not part of what was solved for.
    """
    np = _lazy_np()
    out = []
    if not cams:
        return out
    ref_fwd = cams[0].R[2]
    for i, c in enumerate(cams):
        if i == 0:
            out.append("cam0: reference")
            continue
        sep = math.degrees(math.acos(
            max(-1.0, min(1.0, float(np.dot(c.R[2], ref_fwd))))))
        out.append("cam%d: %.2f m from cam0, %.0f deg apart (%s)" % (
            i, float(np.linalg.norm(c.center())), sep, c.source))
    return out


def register():
    pass


def unregister():
    pass
