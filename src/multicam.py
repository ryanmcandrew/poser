"""Several webcams, one pose.

A single webcam has to guess depth. The pose lifter in `tracker` recovers
it from foreshortening, but the error is always worst along the camera's
own optical axis: a hand moving straight at the lens barely moves in the
image, so its distance is the least observable thing in the frame. What
that camera *is* good at is everything across its view, which it measures
directly in pixels.

Two cameras looking at the same person from different angles therefore have
complementary blind spots: what is depth to one is lateral to the other.
This module runs one full `WebcamTracker` per device and fuses their
`PoseFrame`s into a single frame the retargeter consumes unchanged.

Fusion is a weighted least squares per landmark. Each camera contributes
its estimate `p_i` with an information matrix

    W_i = c_i * R_i @ diag(1, depth_trust, 1) @ R_i.T

where `R_i` rotates that camera's body frame into the reference camera's
(the cameras' Yaw setting), `c_i` is the landmark's confidence in that
camera (visibility for pose, apparent hand size for fingers), and
`depth_trust` < 1 says how much less the camera's forward axis is worth
than its lateral ones. The fused point is the minimiser

    x = (sum W_i)^-1 @ sum W_i @ R_i @ p_i

which collapses to "just use that camera" when only one sees the landmark,
and to "take each axis from whichever camera measures it across its view"
when two cameras sit at a right angle.

The frames the cameras produce are only in a common body frame if each one
resolves the user's left and right consistently, which is what Selfie view
mode plus Mirror does; a body cam's frame is tied to the user's own heading
and cannot be fused this way.

That weighted average is now the fallback rather than the goal. It has a
ceiling that no weighting can lift: both cameras' 3D is *inferred* from a
single image each, by the same network and the same lift, so their depth
errors point the same way and averaging them cancels noise but not bias.
Note also that the anisotropy above only does anything when the cameras
have different Yaws - with two cameras at the same yaw every ``W_i`` is the
same matrix, ``depth_trust`` divides out of the solve, and what is left is a
plain confidence-weighted mean.

The default path is now `bodyfit`, which throws away neither camera's
pixels and fits one skeleton to all of them (see that module). It needs to
know where the cameras stand, which `calib` measures from the user's own
body rather than trusting the Yaw slider.
"""

import math
import os
import threading
import time
from collections import deque

from . import bodyfit
from . import calib
from . import tracker

_np = None


def _lazy_np():
    global _np
    if _np is None:
        import numpy
        _np = numpy
    return _np


# Tile labels are drawn at this height in the combined preview.
_LABEL_H = 18


# Seconds between one camera's open and the next.
_OPEN_STAGGER = 1.5


class CameraSpec:
    """One device in a multi-camera rig."""

    __slots__ = ("index", "fov", "yaw", "trust", "label",
                 "exposure_mode", "exposure")

    def __init__(self, index, fov=65.0, yaw=0.0, trust=1.0, label=None,
                 exposure_mode=None, exposure=None):
        self.index = int(index)
        self.fov = float(fov)
        self.yaw = float(yaw)
        self.trust = max(0.0, float(trust))
        self.label = label or ("Webcam %d" % self.index)
        # None means "whatever the shared setting says".
        self.exposure_mode = exposure_mode
        self.exposure = exposure


def _yaw_matrix(np, yaw_deg):
    """Rotate a vector from a camera's own body frame into the reference
    frame. Yaw is where the camera stands around the user, positive to the
    user's right; the camera's own "forward" points back at itself, so the
    rotation is by -yaw about the vertical axis."""
    a = math.radians(-float(yaw_deg))
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0],
                     [s, c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _record_path_for(base, index):
    if not base:
        return None
    root, ext = os.path.splitext(base)
    return "%s-cam%d%s" % (root, index, ext or ".jsonl")


class MultiCamTracker:
    """Runs one WebcamTracker per device and publishes fused PoseFrames.

    Exposes the same surface as WebcamTracker so `operators` can drive
    either without caring which it holds.
    """

    def __init__(self, specs, depth_trust=0.35, sync_window=0.12,
                 show_window=False, preview_width=320, preview_fps=12.0,
                 record_path=None, solver="REPROJECT", auto_calibrate=True,
                 bone_stiffness=0.025, extrinsics=None, **tracker_kwargs):
        self.specs = [s for s in specs]
        if not self.specs:
            raise ValueError("MultiCamTracker needs at least one camera")
        self.depth_trust = max(0.05, min(1.0, float(depth_trust)))
        self.sync_window = max(0.01, float(sync_window))
        self.show_window = bool(show_window)
        self.preview_width = int(preview_width)
        self.preview_fps = float(preview_fps)
        self.frame_size = (0, 0)
        self.solver_mode = (solver or "REPROJECT").upper()
        self.auto_calibrate = bool(auto_calibrate)
        self.bone_stiffness = float(bone_stiffness)
        self.view_mode = (tracker_kwargs.get("view_mode") or "BODY_CAM").upper()
        self._cams = None            # list of calib.CameraModel
        self._solver = None
        self._calibrator = None
        self._extrinsics_in = extrinsics
        self._calib_state = "not started"
        self._solved = 0             # frames the reprojection fit produced
        self._fallbacks = 0          # frames that fell back to averaging

        want_preview = self.show_window or self.preview_fps > 0.0
        self._units = []
        for spec in self.specs:
            kw = dict(tracker_kwargs)
            if spec.exposure_mode:
                kw["exposure_mode"] = spec.exposure_mode
                if spec.exposure is not None:
                    kw["exposure"] = float(spec.exposure)
            kw.update(
                camera_index=spec.index,
                camera_fov=spec.fov,
                # Sub-trackers never open their own window: one combined
                # preview is the point of this mode.
                show_window=False,
                preview_width=self.preview_width,
                preview_fps=self.preview_fps if want_preview else 0.0,
                record_path=_record_path_for(record_path, spec.index),
            )
            self._units.append(tracker.WebcamTracker(**kw))

        self._rot = None             # filled once numpy is importable
        self._info = None            # per-camera information matrices
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = None
        self._error = None
        self._state = "idle"
        self._fps_samples = deque(maxlen=30)
        self._last_fused_t = 0.0
        self._last_preview_t = 0.0
        self._tiles = [None] * len(self._units)
        self._live = 0
        self._fused_count = 0
        self._used = [0] * len(self._units)     # frames each camera contributed
        self._stale = [0] * len(self._units)    # frames each camera was too old for

    # --- lifecycle --------------------------------------------------------

    def start(self):
        if self._thread and self._thread.is_alive():
            return True
        self._stop.clear()
        self._error = None
        self._state = "starting %d cameras" % len(self._units)
        # Stagger the opens. Two DirectShow devices opening at the same
        # instant contend, and the loser comes up in a degraded mode: the
        # second camera measured 3.3 fps and a 51% share of fused frames
        # opening simultaneously, against 3.9 fps and 69% when its open was
        # held back. The delay is paid once, at Start.
        for i, unit in enumerate(self._units):
            if i:
                time.sleep(_OPEN_STAGGER)
            unit.start()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._thread = None
        for unit in self._units:
            try:
                unit.stop()
            except Exception as e:
                print("[FPV Pose Tracker] camera stop failed:", e)
        self._state = "stopped"

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def latest_frame(self):
        with self._lock:
            return self._latest

    def error(self):
        return self._error

    def state(self):
        return self._state

    def fps(self):
        if len(self._fps_samples) < 2:
            return 0.0
        span = self._fps_samples[-1] - self._fps_samples[0]
        if span <= 0:
            return 0.0
        return (len(self._fps_samples) - 1) / span

    def hand_scale(self):
        # The hand-size estimate is per camera; report the primary's.
        return self._units[0].hand_scale()

    def timings(self):
        """Worst stage time across the cameras, since they run in parallel
        and the fused frame waits for the slowest."""
        out = {}
        for unit in self._units:
            for k, v in unit.timings().items():
                if k == "dropped":
                    out[k] = out.get(k, 0.0) + v
                else:
                    out[k] = max(out.get(k, 0.0), v)
        return out

    def engine_info(self):
        """The models one unit loaded.

        Every camera runs the same engine with the same settings, so
        listing it once per camera would only repeat itself; what differs
        between units is the capture, and `cap_info` already covers that.
        """
        for unit in self._units:
            info = unit.engine_info()
            if info:
                return "%s (x%d cameras)" % (info, len(self._units))
        return ""

    def cap_info(self):
        parts = []
        for spec, unit in zip(self.specs, self._units):
            info = unit.cap_info()
            if info:
                parts.append("cam%d %s" % (spec.index, info))
        return " | ".join(parts)

    def cap_warning(self):
        parts = []
        for spec, unit in zip(self.specs, self._units):
            w = unit.cap_warning()
            if w:
                parts.append("cam%d: %s" % (spec.index, w))
        return " | ".join(parts)

    def brightness(self):
        """Darkest camera: the black-frame banner should fire if any feed
        has gone dark, not only when all of them have."""
        vals = [u.brightness() for u in self._units]
        vals = [v for v in vals if v is not None]
        return min(vals) if vals else None

    def is_black(self):
        return any(u.is_black() for u in self._units)

    def live_cameras(self):
        return self._live

    # --- fusion -----------------------------------------------------------

    def _build_matrices(self, np):
        """Per-camera rotation and information matrix, built once."""
        self._rot = [_yaw_matrix(np, spec.yaw) for spec in self.specs]
        d = np.diag([1.0, self.depth_trust, 1.0])
        self._info = [spec.trust * (R @ d @ R.T)
                      for spec, R in zip(self.specs, self._rot)]

    def _fuse_points(self, np, contribs, keys):
        """contribs: list of (camera slot, {key: (x,y,z)}, {key: weight}).

        Returns {key: (x, y, z)} in the reference frame.
        """
        if not keys:
            return {}
        keys = sorted(keys)
        n = len(keys)
        A = np.zeros((n, 3, 3))
        b = np.zeros((n, 3))
        for slot, points, weights in contribs:
            R = self._rot[slot]
            W = self._info[slot]
            w = np.array([max(0.0, float(weights.get(k, 0.0)))
                          if k in points else 0.0 for k in keys])
            if not w.any():
                continue
            P = np.array([points.get(k, (0.0, 0.0, 0.0)) for k in keys])
            P = P @ R.T                       # into the reference frame
            A += w[:, None, None] * W
            b += w[:, None] * (P @ W.T)
        # A landmark no camera saw leaves a singular block; drop those
        # rather than letting the solve raise for the whole batch.
        good = np.abs(np.linalg.det(A)) > 1e-12
        if not good.any():
            return {}
        try:
            # b must be a stack of column vectors: numpy 2 reads a trailing
            # (N, 3) as one matrix, not N right-hand sides.
            x = np.linalg.solve(A[good], b[good][:, :, None])[:, :, 0]
        except Exception as e:
            print("[FPV Pose Tracker] fusion solve failed:", e)
            return {}
        out = {}
        gi = 0
        for i, k in enumerate(keys):
            if good[i]:
                out[k] = (float(x[gi][0]), float(x[gi][1]), float(x[gi][2]))
                gi += 1
        return out

    def _fuse(self, np, pairs, t):
        """pairs: list of (slot, PoseFrame) fresh enough to combine."""
        pf = tracker.PoseFrame()
        pf.timestamp = t

        # ---- pose ---------------------------------------------------------
        pose_pairs = [(slot, f) for slot, f in pairs if f.pose_present]
        if pose_pairs:
            vis = {}
            for _, f in pose_pairs:
                for i, v in f.pose_visibility.items():
                    # A landmark one camera sees clearly is trustworthy even
                    # if another camera has it behind the user's torso, so
                    # the fused visibility is the best view of it, not a mean
                    # that the blocked camera would drag under the threshold.
                    vis[i] = max(vis.get(i, 0.0), float(v))
            pf.pose_visibility = vis
            pf.pose_present = True

            world = [(slot, f.pose_world, f.pose_visibility) for slot, f in pose_pairs]
            pf.pose_world = self._fuse_points(
                np, world, {i for _, p, _ in world for i in p})

            lifted = [(slot, f.pose_lifted, f.pose_visibility)
                      for slot, f in pose_pairs if f.pose_lifted]
            if lifted:
                pf.pose_lifted = self._fuse_points(
                    np, lifted, {i for _, p, _ in lifted for i in p})

        # ---- hands --------------------------------------------------------
        for side, attr, present_attr in (("L", "left_hand", "left_present"),
                                         ("R", "right_hand", "right_present")):
            contribs = []
            best = (0.0, None)
            for slot, f in pairs:
                hand = getattr(f, attr)
                if not hand or not getattr(f, present_attr):
                    continue
                # Fingers are wrist-relative vectors, so the same weighting
                # applies: a hand that fills more of the frame was measured
                # by more pixels.
                q = float(f.hand_quality.get(side, 0.0)) or 1e-3
                contribs.append((slot, hand, {k: q for k in hand}))
                if q > best[0]:
                    best = (q, f)
            if not contribs:
                continue
            fused = self._fuse_points(
                np, contribs, {k for _, h, _ in contribs for k in h})
            setattr(pf, attr, fused)
            setattr(pf, present_attr, True)
            pf.hand_quality[side] = best[0]
            # Wrist distance is measured from one camera's lens, so it means
            # nothing averaged across cameras; take the best view's.
            if best[1] is not None and side in best[1].hand_depth:
                pf.hand_depth[side] = best[1].hand_depth[side]

        pf.sides_info = "fused %d/%d cams" % (len(pairs), len(self._units))
        pf.depth_info = " ".join(
            "cam%d:%s" % (self.specs[slot].index, f.depth_info)
            for slot, f in pairs if f.depth_info)
        return pf

    # --- reprojection solver ---------------------------------------------

    def _body_of(self, X):
        """World point (reference camera's physical frame) -> user body frame.

        The reference camera is rarely square-on to the user, so this leaves
        a constant rotation in the result. That is deliberate: the
        retargeter measures its torso frame from the data during
        calibration, so a fixed rotation is absorbed there, and pretending
        to know the camera's absolute heading would only add an error.
        """
        selfie = self.view_mode == "SELFIE"
        sx = -1.0 if selfie else 1.0
        sdepth = -1.0 if selfie else 1.0
        return (sx * float(X[0]), sdepth * float(X[2]), -float(X[1]))

    def _build_cameras(self, frames):
        """Intrinsics from the frames, extrinsics from whatever we have.

        The intrinsics have to come from a real frame because they depend on
        the capture resolution, which is not known until a camera has
        delivered something.
        """
        models = []
        for slot, spec in enumerate(self.specs):
            pf = frames[slot] if slot < len(frames) else None
            if pf is None or not pf.cam_model:
                return None
            f_px, W, H = pf.cam_model
            if slot == 0:
                models.append(calib.CameraModel(f_px, W, H, source="reference"))
            else:
                models.append(calib.yaw_guess(f_px, W, H, spec.yaw))
        # Stored extrinsics win: they were measured, the Yaw was typed. But
        # only if they describe *these* cameras - a saved placement re-applied
        # after the device rows were reordered would be confidently wrong,
        # which is worse than having none.
        stored = self._extrinsics_in or {}
        want = [spec.index for spec in self.specs]
        mismatched = bool(stored) and list(stored.get("indices", ())) != want
        if mismatched:
            stored = {}
        if stored:
            for slot, vals in enumerate(stored.get("cams", ())):
                if slot == 0 or slot >= len(models) or not vals:
                    continue
                m = calib.CameraModel.from_list(vals, source="stored")
                if m is not None:
                    # Keep this session's real intrinsics; only the pose is
                    # reusable, since the resolution may have changed.
                    models[slot] = calib.CameraModel(
                        models[slot].f_px, models[slot].W, models[slot].H,
                        m.R, m.t, "stored")
            self._calib_state = "using stored placement"
        elif mismatched:
            self._calib_state = ("saved placement was for other cameras - "
                                 "re-run Calibrate Placement")
        else:
            self._calib_state = "using Yaw (not calibrated)"
        return models

    def _solver_window(self, slot):
        # Deliberately generous. This is only a bound on how far back it is
        # worth looking; which observations matter is settled by the age
        # weighting and the motion compensation inside the solver, not by a
        # hard cutoff here. A camera's own cadence still widens it, for a
        # camera slower than the floor.
        fps = self._units[slot].fps()
        floor = max(self.sync_window, 0.35)
        return max(floor, 3.0 / fps) if fps > 0.5 else floor

    def _solver_observations(self, frames, newest):
        """Every camera's 2D, including cameras a strict sync window would
        drop. The solver weights each by its frame's age and lets the
        continuity term carry the rest, so a slow camera contributes
        continuously rather than only on the frames it happens to be fresh
        for."""
        obs = []
        for slot, pf in enumerate(frames):
            if pf is None or not pf.pose_present or not pf.pose_image:
                continue
            if (newest - pf.timestamp) > self._solver_window(slot):
                continue
            obs.append(bodyfit.Observation(
                slot, pf.pose_image, pf.pose_visibility, pf.timestamp,
                trust=self.specs[slot].trust))
        return obs

    def _lengths(self, frames):
        """The user's measured dimensions, from whichever cameras have seen
        them. These are what stop the fit drifting."""
        out = {}
        for pf in frames:
            if pf is None or not pf.body_lengths:
                continue
            for k, v in pf.body_lengths.items():
                if v and v > 0.0:
                    out.setdefault(k, []).append(float(v))
        return {k: sorted(v)[len(v) // 2] for k, v in out.items() if v}

    def _lifted_world(self, frames):
        """Seed points, from each camera's own monocular lift mapped into the
        world frame. Only used for a joint with no history and no
        triangulation."""
        np = _lazy_np()
        out = {}
        if self._cams is None:
            return out
        for slot, pf in enumerate(frames):
            if pf is None or not pf.pose_lifted_cam:
                continue
            cam = self._cams[slot]
            for idx, X in pf.pose_lifted_cam.items():
                if idx in out:
                    continue
                out[idx] = cam.R.T @ (np.asarray(X, dtype=np.float64) - cam.t)
        return out

    def _placement_measured(self):
        """Whether every camera's pose was measured rather than guessed.

        Triangulation needs a baseline, and the Yaw slider does not supply
        one: it says which way a camera faces, not where it stands. Measured
        end to end, fitting on the Yaw guess alone lands at 174 mm rms
        against 75 mm for the averaging it would be replacing - a baseline
        wrong by 20 cm moves a triangulated wrist further than the monocular
        depth error being removed. So the fit does not run until the rig has
        measured itself.
        """
        if not self._cams:
            return False
        return all(c.source in ("measured", "stored", "reference")
                   for c in self._cams)

    def _apply_solver(self, np, pf, frames, newest):
        """Replace the averaged arm joints with a fit to every camera's 2D.

        Hands, visibility and the preview stay as `_fuse` built them; this
        only touches the joints the fit covers.
        """
        if self.solver_mode != "REPROJECT" or len(self._units) < 2:
            return False
        if self._cams is None:
            self._cams = self._build_cameras(frames)
            if self._cams is None:
                return False
            self._solver = bodyfit.BodySolver(
                self._cams, bone_sigma=self.bone_stiffness)
        if not self._placement_measured():
            # Averaging is in charge until then, which is the honest answer
            # rather than a worse one dressed up as an upgrade.
            if self.auto_calibrate and self._calibrator is None:
                self.request_calibration()
                self._calib_state = ("measuring camera placement - move your "
                                     "arms around")
            self._fallbacks += 1
            return False
        obs = self._solver_observations(frames, newest)
        if not obs:
            return False
        lengths = self._lengths(frames)
        solved, observed = self._solver.solve(
            obs, lengths, newest, self._lifted_world(frames))
        if not solved:
            self._fallbacks += 1
            return False
        multi = sum(1 for n in observed.values() if n >= 2)
        if multi < 2:
            # One camera's worth of joints is what the monocular path
            # already does better, with its hand-depth cues.
            self._fallbacks += 1
            return False
        self._solved += 1
        body = {i: self._body_of(X) for i, X in solved.items()}
        pf.pose_lifted.update(body)
        pf.pose_observed = dict(observed)
        # pose_world is root-relative metres, the way MediaPipe's is, so the
        # retargeter's torso calibration reads it unchanged: it measures the
        # axes it is given rather than assuming them.
        if tracker.POSE_LHIP in body and tracker.POSE_RHIP in body:
            a, b = body[tracker.POSE_LHIP], body[tracker.POSE_RHIP]
        elif tracker.POSE_LSHOULDER in body and tracker.POSE_RSHOULDER in body:
            a, b = body[tracker.POSE_LSHOULDER], body[tracker.POSE_RSHOULDER]
        else:
            a = b = None
        if a is not None:
            root = ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5)
            for i, X in body.items():
                pf.pose_world[i] = (X[0] - root[0], X[1] - root[1], X[2] - root[2])
        res = self._solver.last_residual
        pf.solver_info = "fit %d/%d joints on 2+ cams%s" % (
            multi, len(observed), "" if res is None else ", %.1f px" % res)
        return True

    # --- calibration ------------------------------------------------------

    def request_calibration(self, min_samples=None):
        """Start measuring where the cameras stand. Cheap and idempotent."""
        kw = {} if min_samples is None else {"min_samples": int(min_samples)}
        self._calibrator = calib.RigCalibrator(len(self._units), **kw)
        self._calib_state = "calibrating"
        return True

    def cancel_calibration(self):
        self._calibrator = None
        self._calib_state = "cancelled"

    def is_calibrating(self):
        return self._calibrator is not None

    def calib_status(self):
        if self._calibrator is not None:
            return self._calibrator.status
        return self._calib_state

    def calib_progress(self):
        return self._calibrator.progress() if self._calibrator else 0.0

    def extrinsics(self):
        """The measured placement, in a form the scene can store."""
        if not self._cams:
            return None
        return [c.as_list() for c in self._cams]

    def camera_lines(self):
        return calib.describe(self._cams) if self._cams else []

    def solver_info(self):
        if self.solver_mode != "REPROJECT":
            return "weighted average"
        if not self._placement_measured():
            return "averaging until the placement is measured | %s" % (
                self.calib_status())
        total = self._solved + self._fallbacks
        if not total:
            return "reprojection fit: no frames yet"
        res = self._solver.last_residual if self._solver else None
        return "reprojection fit on %.0f%% of frames%s | %s" % (
            100.0 * self._solved / total,
            "" if res is None else ", %.1f px residual" % res,
            self.calib_status())

    def _run_calibration(self, frames, newest):
        c = self._calibrator
        if c is None:
            return
        c.add({slot: pf for slot, pf in enumerate(frames) if pf is not None},
              newest)
        if c.sample_count() < c.min_samples:
            return
        if self._cams is None:
            self._cams = self._build_cameras(frames)
            if self._cams is None:
                return
        lengths = self._lengths(frames)
        got = c.solve(self._cams, lengths.get("shoulder"))
        if got is not None:
            self._cams = got
            self._solver = bodyfit.BodySolver(
                self._cams, bone_sigma=self.bone_stiffness)
            self._calib_state = c.status
            self._calibrator = None
        elif c.sample_count() >= 3 * c.min_samples:
            # Plenty of data and still no fit: stop burning CPU on it and
            # leave the Yaw guess in place.
            self._calib_state = c.status
            self._calibrator = None

    def _fresh(self, frames, newest):
        """Cameras whose newest frame is recent enough to fuse.

        A camera whose frame predates the window is lagging badly, and fusing
        it would smear a stale limb into the current pose. But the window
        cannot be one fixed number: an old webcam running at 11 fps delivers a
        frame every 90 ms, so a flat 120 ms window drops it from most fused
        frames and the rig quietly falls back to single-camera accuracy while
        the panel still says two cameras are running. Each camera is therefore
        judged against its own measured cadence, and the fixed window is only
        a floor.
        """
        out = []
        for slot, f in enumerate(frames):
            if f is None:
                continue
            fps = self._units[slot].fps()
            limit = self.sync_window
            if fps > 0.5:
                limit = max(limit, 1.5 / fps)
            if (newest - f.timestamp) <= limit:
                out.append((slot, f))
                self._used[slot] += 1
            else:
                self._stale[slot] += 1
        return out

    def unit_fps(self):
        return [u.fps() for u in self._units]

    def per_camera_info(self):
        """One line per camera: its own rate and how much of the fused pose it
        is actually contributing. A camera at 2% is present but useless, which
        is invisible from the fused frame rate alone."""
        total = max(1, self._fused_count)
        lines = []
        for slot, (spec, unit) in enumerate(zip(self.specs, self._units)):
            share = 100.0 * self._used[slot] / total
            err = unit.error()
            if err:
                lines.append("cam%d: FAILED - %s" % (spec.index, err))
            else:
                exp = "" if not spec.exposure_mode else \
                    (" exp %.0f" % spec.exposure if spec.exposure_mode == "MANUAL"
                     else " auto exp")
                revived = 0
                if hasattr(unit, "grab_reopens"):
                    revived = unit.grab_reopens()
                if revived:
                    exp += ", revived %dx" % revived
                lines.append("cam%d: %.1f fps, in %.0f%% of fused frames%s" % (
                    spec.index, unit.fps(), share, exp))
        return lines

    # --- combined preview -------------------------------------------------

    def _tile(self, cv2, np):
        """Lay the cameras' annotated frames out in a grid.

        Each tile keeps whatever that camera last produced, so a camera
        between preview frames leaves its previous image up instead of
        collapsing the layout (and churning the Blender Image datablock
        the panel draws from).
        """
        tiles = [(spec, img) for spec, img in zip(self.specs, self._tiles)
                 if img is not None]
        if not tiles:
            return None
        h = max(img.shape[0] for _, img in tiles)
        w = max(img.shape[1] for _, img in tiles)
        cols = 1 if len(tiles) == 1 else 2
        rows = int(math.ceil(len(tiles) / float(cols)))
        cell_h, cell_w = h + _LABEL_H, w
        out = np.zeros((rows * cell_h, cols * cell_w, 4), dtype=np.uint8)
        out[:, :, 3] = 255
        for i, (spec, img) in enumerate(tiles):
            if img.shape[0] != h or img.shape[1] != w:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            r, c = divmod(i, cols)
            y0 = r * cell_h + _LABEL_H
            x0 = c * cell_w
            out[y0:y0 + h, x0:x0 + w] = img
            label = "%s  (idx %d, yaw %+.0f)" % (spec.label, spec.index, spec.yaw)
            cv2.putText(out, label, (x0 + 6, r * cell_h + _LABEL_H - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255, 255), 1,
                        cv2.LINE_AA)
        return out

    # --- thread -----------------------------------------------------------

    def _collect_errors(self):
        """All cameras down is fatal; some down is a warning the fused frame
        can still be built around."""
        errs = []
        live = 0
        for spec, unit in zip(self.specs, self._units):
            e = unit.error()
            if e:
                errs.append("cam%d: %s" % (spec.index, e))
            elif unit.is_running():
                live += 1
        self._live = live
        return errs, live

    def _run(self):
        try:
            np = _lazy_np()
            import cv2
        except Exception as e:
            self._error = "numpy/OpenCV import failed: %s" % e
            self._state = "import failed"
            return
        self._build_matrices(np)

        window_name = "FPV Pose Tracker - combined (press Q here to close)"
        window_opened = False
        try:
            while not self._stop.is_set():
                errs, live = self._collect_errors()
                if live == 0:
                    if errs:
                        self._error = "; ".join(errs)
                        self._state = "all cameras failed"
                        return
                    # Sub-trackers are still opening their devices.
                    self._state = self._units[0].state()
                    time.sleep(0.02)
                    continue

                frames = [u.latest_frame() for u in self._units]
                stamps = [f.timestamp for f in frames if f is not None]
                if not stamps:
                    self._state = self._units[0].state()
                    time.sleep(0.005)
                    continue
                newest = max(stamps)
                if newest <= self._last_fused_t:
                    time.sleep(0.002)
                    continue
                self._last_fused_t = newest

                pairs = self._fresh(frames, newest)
                pf = self._fuse(np, pairs, newest)
                self._fused_count += 1
                if self._calibrator is not None:
                    self._run_calibration(frames, newest)
                try:
                    self._apply_solver(np, pf, frames, newest)
                except Exception as e:
                    # A solver failure must not take tracking down: the
                    # averaged frame in hand is still usable.
                    self._fallbacks += 1
                    if self._fallbacks < 4:
                        print("[FPV Pose Tracker] body fit failed:", e)

                for slot, f in pairs:
                    if f.annotated_rgba is not None:
                        self._tiles[slot] = f.annotated_rgba

                want_preview = self.show_window or self.preview_fps > 0.0
                interval = 1.0 / self.preview_fps if self.preview_fps > 0 else 0.0
                if want_preview and (newest - self._last_preview_t) >= interval:
                    self._last_preview_t = newest
                    combined = self._tile(cv2, np)
                    if combined is not None:
                        pf.annotated_rgba = combined
                        self.frame_size = (combined.shape[1], combined.shape[0])
                        if self.show_window:
                            try:
                                bgr = cv2.cvtColor(combined, cv2.COLOR_RGBA2BGR)
                                cv2.imshow(window_name, bgr)
                                window_opened = True
                                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                                    self._stop.set()
                            except Exception as e:
                                self.show_window = False
                                print("[FPV] combined imshow disabled:", e)

                with self._lock:
                    self._latest = pf
                self._fps_samples.append(newest)
                note = "" if not errs else "  [%s]" % "; ".join(errs)
                self._state = "tracking %d/%d cams (%d fused)%s" % (
                    live, len(self._units), len(pairs), note)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._error = "multi-camera error: %s" % e
            self._state = "error"
        finally:
            self._stop.set()
            if window_opened:
                try:
                    import cv2
                    cv2.destroyWindow(window_name)
                    cv2.waitKey(1)
                except Exception:
                    pass


def register():
    pass


def unregister():
    pass
