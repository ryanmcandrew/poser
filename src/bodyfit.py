"""One body, fitted to every camera's pixels at once.

The previous multi-camera path averaged each camera's finished 3D guess.
That is the weakest thing two cameras can do together. MediaPipe's world
landmarks and the depth lift are both *inferences* about depth, drawn from
similar images by similar means, so their errors are correlated: averaging
them cancels a little noise and leaves the systematic error in place. The
information a second camera genuinely adds lives in its 2D landmarks, which
are measurements, and in the baseline between the two lenses, which turns
two rays into an intersection.

So nothing here averages 3D. A single articulated body is fitted directly
to the 2D observations of every camera, by Gauss-Newton on three kinds of
residual:

* **Reprojection.** Where each camera says the joint is on its sensor,
  weighted by that landmark's visibility, the camera's Trust, and how old
  that camera's frame is. This is the term that carries the real
  information.

* **Limb length.** The upper arm and forearm the user was measured to have,
  and their shoulder width. A joint the cameras disagree about cannot go
  anywhere the skeleton cannot reach. This is also what makes a landmark
  only one camera can see solvable at all: one ray plus a known bone length
  from a parent that *is* triangulated pins it down.

* **Continuity.** A constant-velocity prediction from the previous solve.
  Besides smoothing, this is what resolves the front/back ambiguity a
  single ray leaves - the arm cannot teleport through the shoulder between
  frames - which the monocular path had to guess at with a learned sign.

The continuity term is also why the cameras no longer have to be
synchronised. Each camera's residual is weighted by the age of its frame
rather than being thrown away for being stale, so a webcam running at 10 fps
next to one at 30 contributes on every solve instead of on a third of them.
"""

import math

from . import tracker as T

# Joints the articulated fit covers. This is exactly the set the retargeter
# reads, and that is the point rather than an economy: the solved landmarks
# are written back over the averaged ones, so any landmark it reads but the
# fit skips would be compared against solved neighbours in a different
# frame. The eyes are here for that reason - they carry no bone constraint,
# but the first-person camera offset is measured as eyes minus shoulder
# centre, which would be a mix of two estimates otherwise.
FIT_LANDMARKS = (
    T.POSE_NOSE, T.POSE_LEYE, T.POSE_REYE,
    T.POSE_LSHOULDER, T.POSE_RSHOULDER,
    T.POSE_LELBOW, T.POSE_RELBOW,
    T.POSE_LWRIST, T.POSE_RWRIST,
    T.POSE_LHIP, T.POSE_RHIP,
)

# (a, b, which measured length) - the bones the fit is allowed to rely on,
# which is exactly the set the tracker measures per user.
_BONES = (
    (T.POSE_LSHOULDER, T.POSE_RSHOULDER, "shoulder"),
    (T.POSE_LSHOULDER, T.POSE_LELBOW, "upper"),
    (T.POSE_LELBOW, T.POSE_LWRIST, "fore"),
    (T.POSE_RSHOULDER, T.POSE_RELBOW, "upper"),
    (T.POSE_RELBOW, T.POSE_RWRIST, "fore"),
)

# Residual scales. Every residual is divided by the error we expect of it,
# which is what lets pixels, metres and metres-per-second be summed in one
# least-squares problem.
_SIGMA_PX = 3.5        # landmark scatter on a good frame
_SIGMA_BONE = 0.025    # how far a limb length may be off, metres
_SIGMA_PRED = 0.12     # how far a joint may move from its prediction
# The prior is deliberately loose. Tightening it buys smoothness at the
# price of lag, and the retargeter already has a One-Euro filter for that.

# A landmark below this visibility contributes nothing from that camera.
_MIN_VIS = 0.25
# Frame age at which a camera's say is halved, seconds. This can be
# generous because staleness is *corrected* before it is discounted: a late
# observation is compared against where the joint was when that camera
# looked, not against where it is now. What is left to discount is only the
# error in that correction, which grows slowly.
_AGE_TAU = 0.25
# Gauss-Newton iterations. The problem is small and starts from last
# frame's answer, so it is converged well before this.
_ITERS = 6
_DAMPING = 1e-4
# A step larger than this is a divergence, not a solution.
_MAX_STEP = 0.5
# Velocity is smoothed this hard before being used to predict.
_VEL_SMOOTH = 0.5
# Beyond this gap the previous state says nothing useful about now.
_MAX_DT = 0.5


def _lazy_np():
    import numpy
    return numpy


class Observation:
    """What one camera saw on one frame."""

    __slots__ = ("slot", "points", "conf", "timestamp", "trust")

    def __init__(self, slot, points, conf, timestamp, trust=1.0):
        self.slot = int(slot)
        self.points = points          # {idx: (u, v)} physical image-normalised
        self.conf = conf              # {idx: visibility}
        self.timestamp = float(timestamp)
        self.trust = max(0.0, float(trust))


class BodySolver:
    """Fits FIT_LANDMARKS to several cameras' 2D, frame after frame."""

    def __init__(self, cams, bone_sigma=_SIGMA_BONE, pred_sigma=_SIGMA_PRED):
        self.cams = list(cams)
        self.bone_sigma = float(bone_sigma)
        self.pred_sigma = float(pred_sigma)
        self._X = {}              # idx -> world point, last solve
        self._V = {}              # idx -> velocity, world units/s
        self._t = None
        self.last_observed = {}   # idx -> how many cameras measured it
        self.last_residual = None  # median reprojection error, pixels
        self.last_iters = 0

    # --- state ------------------------------------------------------------

    def reset(self):
        self._X = {}
        self._V = {}
        self._t = None

    def has_state(self):
        return bool(self._X)

    def _predict(self, np, now):
        """Where each joint should be, from the last solve and its velocity."""
        if self._t is None:
            return {}
        dt = now - self._t
        if dt <= 0.0 or dt > _MAX_DT:
            return dict(self._X)
        return {i: X + self._V.get(i, np.zeros(3)) * dt
                for i, X in self._X.items()}

    # --- the fit ----------------------------------------------------------

    def _seed(self, np, keys, obs_by_idx, pred, lifted_world):
        """A starting point for every joint being solved.

        Prediction first (it is last frame's fitted answer, the best thing
        available), then a fresh triangulation, then whichever camera's own
        monocular lift is available. The fit is tolerant of a poor seed but
        not of a missing one.
        """
        from . import calib
        X0 = {}
        for idx in keys:
            if idx in pred:
                X0[idx] = pred[idx]
                continue
            seen = obs_by_idx.get(idx, {})
            if len(seen) >= 2:
                pt = calib.triangulate(
                    self.cams, {s: v[0] for s, v in seen.items()},
                    {s: v[1] for s, v in seen.items()})
                if pt is not None:
                    X0[idx] = pt
                    continue
            if idx in lifted_world:
                X0[idx] = lifted_world[idx]
        return X0

    def solve(self, observations, lengths, now, lifted_world=None):
        """Fit the body to ``observations``.

        ``lengths`` is {"shoulder": m, "upper": m, "fore": m}; ``lifted_world``
        is an optional {idx: world point} fallback seed built from a camera's
        own monocular lift. Returns ({idx: (x, y, z)}, {idx: n_cameras}).
        """
        np = _lazy_np()
        # --- gather, per landmark, which cameras measured it --------------
        obs_by_idx = {}
        for o in observations:
            if o.trust <= 0.0:
                continue
            age = max(0.0, now - o.timestamp)
            fresh = math.exp(-(age / _AGE_TAU) ** 2)
            if fresh < 0.02:
                continue
            for idx, uv in o.points.items():
                if idx not in FIT_LANDMARKS:
                    continue
                c = float(o.conf.get(idx, 0.0))
                if c < _MIN_VIS:
                    continue
                obs_by_idx.setdefault(idx, {})[o.slot] = (
                    uv, c * o.trust * fresh, age)

        keys = [i for i in FIT_LANDMARKS if i in obs_by_idx]
        if not keys:
            self.last_observed = {}
            return {}, {}
        pred = self._predict(np, now)
        X0 = self._seed(np, keys, obs_by_idx, pred, lifted_world or {})
        keys = [i for i in keys if i in X0]
        if not keys:
            self.last_observed = {}
            return {}, {}
        slot_of = {idx: n for n, idx in enumerate(keys)}
        x = np.concatenate([np.asarray(X0[i], dtype=np.float64) for i in keys])

        w_px = 1.0 / _SIGMA_PX
        w_bone = 1.0 / max(1e-4, self.bone_sigma)
        w_pred = 1.0 / max(1e-4, self.pred_sigma)
        bones = [(a, b, float(lengths.get(k, 0.0)))
                 for a, b, k in _BONES
                 if a in slot_of and b in slot_of and lengths.get(k)]

        n = len(keys) * 3
        iters = 0
        for iters in range(1, _ITERS + 1):
            rows = []
            res = []
            # ---- reprojection ------------------------------------------
            for idx, seen in obs_by_idx.items():
                if idx not in slot_of:
                    continue
                col = slot_of[idx] * 3
                X = x[col:col + 3]
                vel = self._V.get(idx)
                for s, (uv, w, age) in seen.items():
                    cam = self.cams[s]
                    # Where this joint was when that camera looked. Without
                    # this a late frame pulls the joint backwards along its
                    # own path, so a slow camera actively drags the fit -
                    # which is why the old code could only afford to throw
                    # stale frames away.
                    Xs = X if (vel is None or age <= 0.0) else X - vel * age
                    xc = cam.R @ Xs + cam.t
                    if xc[2] <= 1e-3:
                        continue
                    inv = 1.0 / xc[2]
                    px, py = cam.to_px(*uv)
                    D = cam.f_px * np.array(
                        [[inv, 0.0, -xc[0] * inv * inv],
                         [0.0, inv, -xc[1] * inv * inv]])
                    J = (D @ cam.R) * (w * w_px)
                    for k in range(2):
                        row = np.zeros(n)
                        row[col:col + 3] = J[k]
                        rows.append(row)
                    res.append((cam.f_px * xc[0] * inv - px) * w * w_px)
                    res.append((cam.f_px * xc[1] * inv - py) * w * w_px)
            # ---- limb lengths -------------------------------------------
            for a, b, L in bones:
                ca, cb = slot_of[a] * 3, slot_of[b] * 3
                d = x[ca:ca + 3] - x[cb:cb + 3]
                dist = float(np.linalg.norm(d))
                if dist < 1e-5:
                    continue
                u = d / dist
                row = np.zeros(n)
                row[ca:ca + 3] = u * w_bone
                row[cb:cb + 3] = -u * w_bone
                rows.append(row)
                res.append((dist - L) * w_bone)
            # ---- continuity ---------------------------------------------
            for idx in keys:
                p = pred.get(idx)
                if p is None:
                    continue
                col = slot_of[idx] * 3
                for k in range(3):
                    row = np.zeros(n)
                    row[col + k] = w_pred
                    rows.append(row)
                    res.append((x[col + k] - p[k]) * w_pred)

            if len(rows) < n:
                break
            J = np.array(rows)
            r = np.array(res)
            try:
                step = np.linalg.solve(J.T @ J + _DAMPING * np.eye(n), -J.T @ r)
            except Exception:
                break
            mag = float(np.abs(step).max())
            if not np.isfinite(mag):
                break
            if mag > _MAX_STEP:
                step *= _MAX_STEP / mag
            x = x + step
            if mag < 1e-5:
                break

        out = {i: tuple(float(v) for v in x[slot_of[i] * 3:slot_of[i] * 3 + 3])
               for i in keys}
        observed = {i: len(obs_by_idx.get(i, {})) for i in keys}

        # ---- residual, for the HUD ---------------------------------------
        errs = []
        for idx, seen in obs_by_idx.items():
            if idx not in out:
                continue
            vel = self._V.get(idx)
            for s, (uv, _, age) in seen.items():
                X = np.asarray(out[idx])
                if vel is not None and age > 0.0:
                    X = X - vel * age
                got = self.cams[s].project_px(X)
                if got is None:
                    continue
                px, py = self.cams[s].to_px(*uv)
                errs.append(math.hypot(got[0] - px, got[1] - py))
        self.last_residual = float(np.median(errs)) if errs else None
        self.last_iters = iters
        self.last_observed = observed

        # ---- advance the constant-velocity state -------------------------
        if self._t is not None and 0.0 < (now - self._t) <= _MAX_DT:
            dt = now - self._t
            for i, X in out.items():
                Xn = np.asarray(X)
                prev = self._X.get(i)
                if prev is None:
                    continue
                v = (Xn - prev) / dt
                old = self._V.get(i)
                self._V[i] = v if old is None else \
                    old * _VEL_SMOOTH + v * (1.0 - _VEL_SMOOTH)
        else:
            self._V = {}
        self._X = {i: np.asarray(X, dtype=np.float64) for i, X in out.items()}
        self._t = now
        return out, observed


def register():
    pass


def unregister():
    pass
