"""Webcam capture + MediaPipe processing on a background thread.

Uses the mp_backend abstraction so it works on both the legacy
`mp.solutions.hands` API and the newer `mp.tasks.vision.HandLandmarker` API.

Besides raw landmarks, each PoseFrame carries two derived products:

* Side assignment. MediaPipe's pose labels are anatomical *as they appear in
  the image*, while the hand landmarker's handedness assumes a mirrored
  selfie feed, and real builds don't always match their docs. Instead of
  trusting either label we decide sides geometrically: the shoulder on the
  image's left is the user's left when (selfie and mirrored) or (body cam and
  not mirrored), and each detected hand is attached to the pose wrist it sits
  on in the image. Label conventions are learned from that and used only as a
  fallback when the pose is missing.

* Depth lifting (`pose_lifted`). MediaPipe's 3D depth is a learned guess and
  compresses badly when a limb points at the camera. Given the camera's
  horizontal FOV and the user's limb lengths (estimated from the metric
  world landmarks), we lift the 2D landmarks along their camera rays so each
  segment has its true length: foreshortening in the image then becomes real
  depth. Output is in the user body frame (x right, y forward, z up), metres.

* Hand depth (`hand_depth`). Each ray solve above has two answers (joint
  toward the camera or away) and the image alone cannot tell them apart when
  a limb points at the lens. MediaPipe's metric hand model plus the hand's
  image landmarks give an absolute hand position through a perspective-n-
  point solve; its depth picks the elbow/wrist roots. The image z ordering
  is only a fallback when no hand is tracked.
"""

import math
import threading
import time
from collections import deque

from . import handplane
from . import mp_backend
from . import pose_engines
from . import pose_metrics
from .one_euro import OneEuroScalar

_cv2 = None
_np = None


def _lazy_cv():
    """cv2/numpy are optional at import time so the addon loads (and can
    offer to install them) when they are missing."""
    global _cv2, _np
    if _cv2 is None:
        import cv2
        import numpy
        _cv2, _np = cv2, numpy
    return _cv2, _np


POSE_NOSE = 0
POSE_LEYE = 2
POSE_REYE = 5
POSE_LSHOULDER = 11
POSE_RSHOULDER = 12
POSE_LELBOW = 13
POSE_RELBOW = 14
POSE_LWRIST = 15
POSE_RWRIST = 16
POSE_LHIP = 23
POSE_RHIP = 24

_POSE_SWAP_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10)] + \
                   [(i, i + 1) for i in range(11, 33, 2)]


def _swap_pose_sides(seq):
    out = list(seq)
    for a, b in _POSE_SWAP_PAIRS:
        if a < len(out) and b < len(out):
            out[a], out[b] = out[b], out[a]
    return out


HAND_WRIST = 0
HAND_THUMB = (1, 2, 3, 4)
HAND_INDEX = (5, 6, 7, 8)
HAND_MIDDLE = (9, 10, 11, 12)
HAND_RING = (13, 14, 15, 16)
HAND_PINKY = (17, 18, 19, 20)

# How often to re-measure the feed's brightness, in seconds.
_BRIGHT_CHECK_INTERVAL = 1.0

# Anthropometric fallbacks (metres) until running estimates exist.
_DEFAULT_SHOULDER_WIDTH = 0.38
_DEFAULT_UPPER_ARM = 0.30
_DEFAULT_FOREARM = 0.26


def _percentile(values, frac):
    s = sorted(values)
    if not s:
        return None
    i = int(round((len(s) - 1) * frac))
    return s[max(0, min(len(s) - 1, i))]


# Some webcams refuse to open roughly half the time, at random: measured on
# an older Logitech, a reopen succeeded 3/3 with no delay and 1/3 with a
# 0.4 s delay, so this is driver flakiness rather than a release race and
# waiting longer does not help. Only retrying does, so retry often and
# briefly: six attempts leaves under a 2 % chance of a spurious failure and
# costs at most ~1.5 s, against a fall-through to another backend that can
# block for over 90 seconds or hand back a black 1 fps feed.
# A read that takes at least this long did not return a frame, it returned
# DirectShow's internal timeout. Measured at a consistent ~1001 ms.
_STALL_READ_MS = 900.0
# How long a camera may deliver nothing before it is presumed wedged. Long
# enough that a slow camera (an old webcam can sit at 8-13 fps) is never
# reopened for merely being slow.
_STALL_TIMEOUT = 2.5
# Rest between the release and the reopen; the device needs about two
# seconds before it will open again.
_STALL_REOPEN_REST = 2.5
# Ceiling for the backoff between repeated revival attempts.
_STALL_REOPEN_REST_MAX = 10.0
# Attempts on the known-good backend before considering another one.
_STALL_REOPEN_RETRIES = 4
# Full open sweeps (each its own set of per-backend retries) before giving up
# on a camera at start-up.
_OPEN_SWEEPS = 5

OPEN_RETRIES = 6
OPEN_RETRY_DELAY = 0.25


def _capture_is_sane(cv2, cap, frame):
    """True if this capture is really delivering pictures.

    `read()` returning True is not enough. A device that has been wedged and
    reopened has been measured coming back claiming a -1x-1 frame size with a
    garbage FOURCC, handing back buffers that `read()` reports as successful
    but that are not images. Accepting that is worse than failing the open,
    because it ends the search with a camera that will never produce a frame.
    """
    if frame is None or getattr(frame, "ndim", 0) != 3:
        return False
    h, w = frame.shape[:2]
    if w < 32 or h < 32:
        return False
    try:
        # A driver that cannot state its own frame size is not in a real mode.
        if int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) <= 0:
            return False
        if int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) <= 0:
            return False
    except Exception:
        return False
    return True


def open_camera(cv2, index, retries=OPEN_RETRIES, delay=OPEN_RETRY_DELAY,
                backend=None, configure=None, want_frame=True):
    """Open a device, retrying the same backend, and release every failure.

    A handle that never opened still has to be released: a half-built capture
    graph keeps the device claimed, which is what makes the *next* attempt
    fail for the same reason the first one did.

    Returns (cap, attempts) with cap None if every attempt failed.
    """
    if backend is None:
        backend = getattr(cv2, "CAP_DSHOW", 0)
    attempts = []
    for attempt in range(max(1, retries)):
        cap = cv2.VideoCapture(index, backend) if backend else \
              cv2.VideoCapture(index)
        if cap.isOpened():
            if configure is not None:
                try:
                    configure(cap)
                except Exception as e:
                    attempts.append("configure failed: %s" % e)
            if not want_frame:
                return cap, attempts
            ok, frame = cap.read()
            if ok and _capture_is_sane(cv2, cap, frame):
                return cap, attempts
            attempts.append("#%d opened, %s" % (
                attempt + 1, "no frame" if not ok else "bogus mode"))
        else:
            attempts.append("#%d will not open" % (attempt + 1))
        cap.release()
        if attempt + 1 < retries:
            time.sleep(delay)
    return None, attempts


def _user_left_is_image_left(view_mode, mirror):
    """Selfie camera faces the user: their left is on the image's right
    unless the feed is mirrored. A body cam looks the way the user looks,
    so their left is on the image's left unless mirrored."""
    return ((view_mode or "").upper() == "SELFIE") == bool(mirror)


class PoseFrame:
    __slots__ = ("timestamp", "pose_world", "pose_visibility",
                 "pose_lifted", "left_hand", "right_hand",
                 "pose_present", "left_present", "right_present",
                 "annotated_rgba", "sides_info", "depth_info", "hand_depth",
                 "hand_quality", "pose_image", "cam_model", "body_lengths",
                 "pose_observed", "solver_info", "pose_lifted_cam",
                 "joint_angles")

    def __init__(self):
        self.timestamp = 0.0
        self.pose_world = {}
        self.pose_visibility = {}
        self.pose_lifted = {}      # idx -> (x, y, z) body frame, metres; empty if unavailable
        self.left_hand = {}
        self.right_hand = {}
        self.pose_present = False
        self.left_present = False
        self.right_present = False
        self.annotated_rgba = None
        self.sides_info = ""
        self.depth_info = ""       # HUD line: root source, hand depths, hand scale
        self.hand_depth = {}       # "L"/"R" -> wrist distance from camera, metres
        # "L"/"R" -> apparent hand size in the image (mean wrist-relative
        # landmark distance). A hand that fills more of the frame was
        # resolved by more pixels, so this is how much to trust this
        # camera's finger geometry when several cameras are fused.
        self.hand_quality = {}
        # idx -> (u, v) in *physical* image-normalised coordinates: the
        # mirror a selfie feed is displayed through is undone, so u grows
        # the way the sensor sees it and the camera frame stays right
        # handed. Sides are already corrected, so index i is the same
        # anatomical point on every camera. This is the raw measurement;
        # everything above is derived from it, which is why the multi-camera
        # solver wants it rather than pose_world.
        self.pose_image = {}
        # (f_px, W, H) for this frame's camera, principal point centred.
        self.cam_model = None
        # The user's own metric dimensions as this camera has measured them:
        # {"shoulder": m, "upper": m, "fore": m}. A fit that has to agree
        # with the body's real proportions is far better conditioned than
        # one that lets every joint float.
        self.body_lengths = {}
        # idx -> how many cameras actually measured this landmark. 1 means
        # the depth is still a monocular guess, whatever else fused it.
        self.pose_observed = {}
        self.solver_info = ""
        # The same lift, left in this camera's own physical frame (x sensor
        # right, y down, z away from the lens), metres. Two cameras' copies
        # of this are the same rigid body seen twice, which is what the rig
        # calibrator aligns to find where the cameras stand.
        self.pose_lifted_cam = {}
        # Joint angles for this frame (see `pose_metrics`). Degrees are the
        # terms the rig is posed in, so they are the readout that says
        # whether a change of engine improved the body form or only the
        # landmark numbers.
        self.joint_angles = None


class _SideResolver:
    """Decides which pose landmarks / hand tracks belong to the user's left
    and right from image geometry, learning the label conventions along the
    way so they can serve as fallback."""

    def __init__(self, user_left_is_image_left):
        self.ul_left = bool(user_left_is_image_left)
        self.pose_swap_score = 0.0    # EMA in [-1, 1]; > 0 => swap pose L/R labels
        self.label_swap_score = 0.0   # EMA in [-1, 1]; > 0 => swap hand labels

    def pose_swap(self, image_lms):
        """Vote on whether MediaPipe's pose L/R labels are flipped for this
        feed, using the shoulders' image x. Returns the current decision."""
        if image_lms and len(image_lms) > POSE_RSHOULDER:
            xl = image_lms[POSE_LSHOULDER][0]
            xr = image_lms[POSE_RSHOULDER][0]
            if abs(xl - xr) > 0.03:
                label_l_on_image_left = xl < xr
                swap = (label_l_on_image_left != self.ul_left)
                self.pose_swap_score = 0.85 * self.pose_swap_score + 0.15 * (1.0 if swap else -1.0)
        return self.pose_swap_score > 0.0

    def hand_sides(self, tracks, pose_wrists):
        """Return a list of "L"/"R" for each hand track.

        `pose_wrists` is {"L": (u, v), "R": (u, v)} in image-normalised coords
        with sides already corrected, or None when the pose is absent."""
        sides = []
        details = []
        for tr in tracks:
            raw = "L" if tr.label == "Left" else "R"
            label_side = raw if self.label_swap_score <= 0.0 else ("R" if raw == "L" else "L")
            side = label_side
            src = "label"
            if pose_wrists and tr.landmarks:
                u, v = tr.landmarks[HAND_WRIST][0], tr.landmarks[HAND_WRIST][1]
                dL = math.hypot(u - pose_wrists["L"][0], v - pose_wrists["L"][1])
                dR = math.hypot(u - pose_wrists["R"][0], v - pose_wrists["R"][1])
                if min(dL, dR) < 0.2 and abs(dL - dR) > 0.04:
                    side = "L" if dL < dR else "R"
                    src = "pose"
                    swap = (side != raw)
                    self.label_swap_score = 0.9 * self.label_swap_score + 0.1 * (1.0 if swap else -1.0)
            sides.append((side, tr, src))
            details.append(src)

        # Two hands on the same side: give the more distant one (or, without
        # a pose, the one on the wrong image side) the other side.
        if len(sides) == 2 and sides[0][0] == sides[1][0]:
            a, b = sides[0][1], sides[1][1]
            ua, ub = a.landmarks[HAND_WRIST][0], b.landmarks[HAND_WRIST][0]
            left_first = ua < ub
            if self.ul_left:
                sides[0] = ("L" if left_first else "R", a, "order")
                sides[1] = ("R" if left_first else "L", b, "order")
            else:
                sides[0] = ("R" if left_first else "L", a, "order")
                sides[1] = ("L" if left_first else "R", b, "order")
        return [s[0] for s in sides], ",".join(details)


class _HandDepth:
    """Absolute wrist position in the camera frame from one hand's image
    landmarks and MediaPipe's metric hand model (perspective-n-point).

    The metric model is an average hand, so its depth is off by the ratio
    of the user's hand size to the model's. That ratio is one number: the
    user can set it (`hand_size`) and it is also learned. Every frame the
    pose lifter offers four candidate wrist depths (elbow near/far x wrist
    near/far); the true one is always among them, at a fixed ratio to the
    unscaled PnP depth, while the wrong ones scatter with the pose. The
    ratio that keeps recurring is the hand scale, so a mode over the
    candidate ratios recovers it from ordinary motion, no special pose
    needed, and puts both depth sources on the same (shoulder-width) scale."""

    _MAX_REPROJ_FRAC = 0.04     # mean reprojection error, fraction of width
    _MIN_AUTO_SAMPLES = 60      # candidate ratios (~15 frames) before learning kicks in
    _LOG_BIN = 0.02             # histogram bin in log(ratio), ~2 %
    _RATIO_LO, _RATIO_HI = 0.5, 2.0

    def __init__(self, hand_size=1.0, auto_scale=True):
        self.hand_size = max(0.3, min(3.0, float(hand_size)))
        self.auto_scale = bool(auto_scale)
        self._z_filters = {s: OneEuroScalar(min_cutoff=0.8, beta=0.4) for s in ("L", "R")}
        self._ratios = deque(maxlen=1200)   # ~5 s of both hands at 15 fps
        self._z_unit = {}           # side -> unscaled PnP wrist depth this frame
        self._auto = 1.0
        self._auto_dirty = 0
        self._reflect = False       # model reflection that fitted last time
        self._fail_logged = False

    def scale(self):
        return self.hand_size * (self._auto if self.auto_scale else 1.0)

    def auto_ratio(self):
        return self._auto if len(self._ratios) >= self._MIN_AUTO_SAMPLES else None

    def add_candidates(self, side, wrist_depths):
        """`wrist_depths`: the lifter's candidate wrist depths this frame."""
        z_unit = self._z_unit.get(side)
        if not z_unit or z_unit <= 0.05:
            return
        for z in wrist_depths:
            r = z / z_unit
            if self._RATIO_LO <= r <= self._RATIO_HI:
                self._ratios.append(math.log(r))
        self._auto_dirty += 1
        if self._auto_dirty >= 8 and len(self._ratios) >= self._MIN_AUTO_SAMPLES:
            self._auto_dirty = 0
            self._auto = self._mode()

    def _mode(self):
        lo = math.log(self._RATIO_LO)
        nbins = int((math.log(self._RATIO_HI) - lo) / self._LOG_BIN) + 1
        hist = [0.0] * nbins
        for v in self._ratios:
            hist[max(0, min(nbins - 1, int((v - lo) / self._LOG_BIN)))] += 1.0
        # triangular smoothing over +-2 bins (~ +-4 %)
        best_i, best_v = 0, -1.0
        for i in range(nbins):
            acc = 0.0
            for k in range(-2, 3):
                j = i + k
                if 0 <= j < nbins:
                    acc += hist[j] * (3 - abs(k))
            if acc > best_v:
                best_i, best_v = i, acc
        return math.exp(lo + (best_i + 0.5) * self._LOG_BIN)

    def begin_frame(self):
        self._z_unit = {}

    def wrist(self, side, image_lms, world_lms, W, H, f_px, mirror, t):
        """(X, Y, Z) camera frame (X right, Y down, Z depth) of the wrist,
        or None when the solve is unavailable or does not fit the image."""
        if not world_lms or len(world_lms) != len(image_lms) or len(image_lms) < 6:
            return None
        try:
            cv2, np = _lazy_cv()
        except Exception:
            return None
        obj0 = np.array(world_lms, dtype=np.float64)
        img = np.array([(lm[0] * W, lm[1] * H) for lm in image_lms], dtype=np.float64)
        K = np.array([[f_px, 0.0, W * 0.5],
                      [0.0, f_px, H * 0.5],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        try:
            flag = getattr(cv2, "SOLVEPNP_SQPNP", cv2.SOLVEPNP_ITERATIVE)
            # MediaPipe reconstructs the hand it sees, so on a mirrored feed
            # the world landmarks are already the reflected hand and fit the
            # mirrored image as-is. Some builds may not; try the reflected
            # model when the plain one does not fit, and remember which did.
            wr = None
            for reflect in (self._reflect, not self._reflect):
                obj = obj0.copy()
                if reflect:
                    obj[:, 0] *= -1.0
                ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=flag)
                if not ok:
                    continue
                proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
                err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)))
                if err > self._MAX_REPROJ_FRAC * W:
                    continue
                R, _ = cv2.Rodrigues(rvec)
                wr = R @ obj[HAND_WRIST] + tvec.reshape(3)
                self._reflect = reflect
                break
            if wr is None:
                return None
        except Exception as e:
            if not self._fail_logged:
                self._fail_logged = True
                print("[FPV Pose Tracker] hand PnP failed, hand depth disabled:", e)
            return None
        z_unit = float(wr[2])
        if not math.isfinite(z_unit) or z_unit < 0.05:
            return None
        self._z_unit[side] = z_unit
        z = self._z_filters[side].filter(z_unit * self.scale(), t)
        # Keep the wrist on its image ray; PnP contributes only the depth.
        u, v = image_lms[HAND_WRIST][0], image_lms[HAND_WRIST][1]
        x = (u - 0.5) * W / f_px * z
        y = (v - 0.5) * H / f_px * z
        return (x, y, z)


class _DepthLifter:
    """Lift 2D pose landmarks to 3D camera coordinates using a pinhole model
    and known limb lengths.

    The lengths are normally learned from the engine's own metric world
    pose. A 2D engine has none, so without measured values every user
    would be lifted as the same average person — which is why `overrides`
    exists: a tape measure beats both a learned estimate and a population
    average, and it is the only source of real proportions the 2D engines
    have. Any override left at 0 keeps the learned-or-default behaviour.
    """

    def __init__(self, hfov_deg, overrides=None):
        self.hfov = math.radians(max(20.0, min(170.0, hfov_deg)))
        ov = overrides or {}
        self._fixed = {k: float(ov.get(k) or 0.0)
                       for k in ("shoulder", "upper", "fore")}
        self._shoulder = deque(maxlen=150)
        self._upper = deque(maxlen=150)
        self._fore = deque(maxlen=150)
        self._z_filter = OneEuroScalar(min_cutoff=0.6, beta=0.05)
        self._sign_ema = {}
        # side -> the four candidate wrist depths (elbow near/far x wrist
        # near/far) from the last frame, for hand-scale learning.
        self.wrist_depth_candidates = {}
        # side -> "hand" / "image": which cue chose the roots last frame
        self.root_source = {}

    def f_px(self, W):
        return (W * 0.5) / math.tan(self.hfov * 0.5)

    # --- limb-length estimates from the metric world landmarks -----------
    def update_lengths(self, world_lms, vis, thr):
        if not world_lms or len(world_lms) <= POSE_RWRIST:
            return
        def ok(i):
            return i < len(vis) and vis[i] >= thr
        def dist(a, b):
            pa, pb = world_lms[a], world_lms[b]
            return math.dist(pa[:3], pb[:3])
        if ok(POSE_LSHOULDER) and ok(POSE_RSHOULDER):
            self._shoulder.append(dist(POSE_LSHOULDER, POSE_RSHOULDER))
        for sh, el, wr in ((POSE_LSHOULDER, POSE_LELBOW, POSE_LWRIST),
                           (POSE_RSHOULDER, POSE_RELBOW, POSE_RWRIST)):
            if ok(sh) and ok(el) and ok(wr):
                self._upper.append(dist(sh, el))
                self._fore.append(dist(el, wr))

    def shoulder_width(self):
        if self._fixed["shoulder"] > 0.0:
            return self._fixed["shoulder"]
        m = _percentile(self._shoulder, 0.5)
        return m if (m and m > 0.15) else _DEFAULT_SHOULDER_WIDTH

    def upper_len(self):
        if self._fixed["upper"] > 0.0:
            return self._fixed["upper"]
        m = _percentile(self._upper, 0.8)
        return m if (m and m > 0.12) else _DEFAULT_UPPER_ARM

    def fore_len(self):
        if self._fixed["fore"] > 0.0:
            return self._fixed["fore"]
        m = _percentile(self._fore, 0.8)
        return m if (m and m > 0.12) else _DEFAULT_FOREARM

    def lengths_measured(self):
        """True when these lengths came from this user rather than a table."""
        if all(v > 0.0 for v in self._fixed.values()):
            return True
        return bool(self._shoulder and self._upper and self._fore)

    # --- geometry -----------------------------------------------------------
    def _ray(self, u, v, W, H, f_px):
        x = (u - 0.5) * W
        y = (v - 0.5) * H
        n = math.sqrt(x * x + y * y + f_px * f_px)
        return (x / n, y / n, f_px / n)

    def _closer_sign(self, key, dz):
        s = -1.0 if dz < 0.0 else 1.0
        prev = self._sign_ema.get(key, s)
        s = 0.7 * prev + 0.3 * s
        self._sign_ema[key] = s
        return -1.0 if s < 0.0 else 1.0

    def _teach_sign(self, key, near):
        """Pull the image-z fallback toward a decision made from a better
        cue, so a momentary hand dropout does not flip the arm."""
        self._sign_ema[key] = -1.0 if near else 1.0

    @staticmethod
    def _ray_roots(ray, parent, length):
        """(t_near, t_far) with |t*ray - parent| = length. Both equal the
        closest approach when the image separation exceeds the limb length
        (noise / bad length)."""
        rp = ray[0] * parent[0] + ray[1] * parent[1] + ray[2] * parent[2]
        pp = parent[0] ** 2 + parent[1] ** 2 + parent[2] ** 2
        disc = rp * rp - pp + length * length
        if disc <= 0.0:
            return rp, rp
        root = math.sqrt(disc)
        return rp - root, rp + root

    @classmethod
    def _solve_along_ray(cls, ray, parent, length, near):
        """Point t*ray with |t*ray - parent| = length. `near` picks the root
        closer to the camera."""
        t_near, t_far = cls._ray_roots(ray, parent, length)
        t = max(t_near if near else t_far, 0.05)
        return (ray[0] * t, ray[1] * t, ray[2] * t)

    # Preference (metres of wrist-depth error) for the pair of roots that
    # agrees with MediaPipe's image-z ordering, used only to break ties.
    _IMAGE_Z_TIEBREAK = 0.02
    # A hand depth further than this outside the reachable range is treated
    # as wrong (bad hand scale) rather than trusted.
    _HAND_DEPTH_SLACK = 0.15

    def _root_candidates(self, r_el, r_wr, p_sh, L1, L2):
        """The four (elbow, wrist, near_el, near_wr) placements consistent
        with the image and the limb lengths."""
        out = []
        for near_el in (True, False):
            p_el = self._solve_along_ray(r_el, p_sh, L1, near_el)
            for near_wr in (True, False):
                p_wr = self._solve_along_ray(r_wr, p_el, L2, near_wr)
                out.append((p_el, p_wr, near_el, near_wr))
        return out

    def _pick_roots_from_hand(self, side, candidates, z_hand, near_el_img, near_wr_img):
        """Choose elbow and wrist roots jointly so the wrist depth matches
        the hand's measured depth. Returns (elbow, wrist) points."""
        best = None
        for p_el, p_wr, near_el, near_wr in candidates:
            cost = abs(p_wr[2] - z_hand)
            if near_el != near_el_img:
                cost += self._IMAGE_Z_TIEBREAK
            if near_wr != near_wr_img:
                cost += self._IMAGE_Z_TIEBREAK
            if best is None or cost < best[0]:
                best = (cost, p_el, p_wr, near_el, near_wr)
        _, p_el, p_wr, near_el, near_wr = best
        self._teach_sign(("el", side), near_el)
        self._teach_sign(("wr", side), near_wr)
        return p_el, p_wr

    def lift(self, image_lms, vis, thr, W, H, t, hand_wrists=None, hand_depth=None):
        """Return {idx: (X, Y, Z)} camera frame (X right, Y down, Z depth) for
        shoulders/elbows/wrists, or None if the shoulders aren't usable.
        `hand_wrists` may give more precise image wrists as {"L": (u,v), "R": (u,v)}
        keyed by *pose* side (already corrected). `hand_depth` gives the
        measured wrist distance from the camera, {"L": z, "R": z}, same keys."""
        self.wrist_depth_candidates = {}
        self.root_source = {}
        if not image_lms or len(image_lms) <= POSE_RWRIST:
            return None
        def ok(i):
            return i < len(vis) and vis[i] >= thr
        if not (ok(POSE_LSHOULDER) and ok(POSE_RSHOULDER)):
            return None
        f_px = self.f_px(W)
        sl, sr = image_lms[POSE_LSHOULDER], image_lms[POSE_RSHOULDER]
        d_px = math.hypot((sl[0] - sr[0]) * W, (sl[1] - sr[1]) * H)
        if d_px < 4.0:
            return None
        z_sh = f_px * self.shoulder_width() / d_px
        z_sh = self._z_filter.filter(z_sh, t)
        z_sh = max(0.2, min(6.0, z_sh))

        out = {}
        for sh_i in (POSE_LSHOULDER, POSE_RSHOULDER):
            r = self._ray(image_lms[sh_i][0], image_lms[sh_i][1], W, H, f_px)
            k = z_sh / r[2]
            out[sh_i] = (r[0] * k, r[1] * k, r[2] * k)

        L1, L2 = self.upper_len(), self.fore_len()
        for side, (sh_i, el_i, wr_i) in (("L", (POSE_LSHOULDER, POSE_LELBOW, POSE_LWRIST)),
                                         ("R", (POSE_RSHOULDER, POSE_RELBOW, POSE_RWRIST))):
            if not ok(el_i):
                continue
            el = image_lms[el_i]
            near_el = self._closer_sign(("el", side), el[2] - image_lms[sh_i][2]) < 0.0
            r_el = self._ray(el[0], el[1], W, H, f_px)
            if not ok(wr_i):
                out[el_i] = self._solve_along_ray(r_el, out[sh_i], L1, near_el)
                continue
            wr = image_lms[wr_i]
            u, v = wr[0], wr[1]
            hand_seen = bool(hand_wrists and side in hand_wrists)
            if hand_seen:
                hu, hv = hand_wrists[side]
                # only trust the hand wrist if it agrees roughly with pose
                if math.hypot(hu - u, hv - v) < 0.12:
                    u, v = hu, hv
                else:
                    hand_seen = False
            near_wr = self._closer_sign(("wr", side), wr[2] - el[2]) < 0.0
            r_wr = self._ray(u, v, W, H, f_px)

            cands = self._root_candidates(r_el, r_wr, out[sh_i], L1, L2)
            if hand_seen:
                self.wrist_depth_candidates[side] = [c[1][2] for c in cands]
            z_hand = hand_depth.get(side) if (hand_depth and hand_seen) else None
            if z_hand is not None and abs(z_hand - z_sh) > L1 + L2 + self._HAND_DEPTH_SLACK:
                z_hand = None        # unreachable: hand scale is off, don't trust it
            if z_hand is not None:
                out[el_i], out[wr_i] = self._pick_roots_from_hand(
                    side, cands, z_hand, near_el, near_wr)
                self.root_source[side] = "hand"
            else:
                out[el_i] = self._solve_along_ray(r_el, out[sh_i], L1, near_el)
                out[wr_i] = self._solve_along_ray(r_wr, out[el_i], L2, near_wr)
                self.root_source[side] = "image"
        return out


class WebcamTracker:
    def __init__(self, camera_index=0, mirror=True, model_complexity=1,
                 show_window=False, preview_width=320, tracking_mode="HANDS",
                 capture_width=0, capture_height=0, pose_y_up=True,
                 view_mode="BODY_CAM", camera_fov=60.0, lift_depth=True,
                 pose_visibility=0.5, hand_depth=True, hand_size=1.0,
                 hand_size_auto=True, record_path=None,
                 target_fps=0, use_mjpg=True, exposure_mode="AUTO",
                 exposure=-6.0, gain=-1.0, infer_width=0, pose_every=1,
                 parallel_infer=True, preview_fps=12.0, delegate="CPU",
                 pose_engine="MEDIAPIPE", engine_quality="BALANCED",
                 engine_layout="BODY", engine_hands="AUTO",
                 engine_device="CPU", engine_det_every=6,
                 engine_kpt_thr=0.3, engine_hand_detector=True,
                 body_shoulder=0.0, body_upper=0.0, body_fore=0.0):
        self.camera_index = camera_index
        self.mirror = mirror
        self.model_complexity = model_complexity
        # --- pose engine selection ---
        self.pose_engine = (pose_engine or "MEDIAPIPE").upper()
        self.engine_quality = (engine_quality or "BALANCED").upper()
        self.engine_layout = (engine_layout or "BODY").upper()
        self.engine_hands = (engine_hands or "AUTO").upper()
        self.engine_device = (engine_device or "CPU").upper()
        self.engine_det_every = max(1, int(engine_det_every or 1))
        self.engine_kpt_thr = float(engine_kpt_thr)
        self.engine_hand_detector = bool(engine_hand_detector)
        self._caps = pose_engines.capabilities(self.pose_engine)
        self._engine_info = ""
        self.show_window = show_window
        self.preview_width = preview_width
        self.tracking_mode = tracking_mode
        self.capture_width = capture_width
        self.capture_height = capture_height
        # --- throughput controls ---
        self.target_fps = int(target_fps or 0)
        self.use_mjpg = bool(use_mjpg)
        self.exposure_mode = (exposure_mode or "AUTO").upper()
        self.exposure = float(exposure)
        self.gain = float(gain)
        # Inference input width. MediaPipe rescales internally anyway, so
        # feeding it a smaller image costs nothing in accuracy that the
        # model was not already going to lose, and saves the colour convert.
        self.infer_width = int(infer_width or 0)
        self.pose_every = max(1, int(pose_every or 1))
        self.parallel_infer = bool(parallel_infer)
        self.preview_fps = float(preview_fps)
        self.delegate = (delegate or "CPU").upper()
        self.pose_y_up = pose_y_up
        self.view_mode = (view_mode or "BODY_CAM").upper()
        self.lift_depth = bool(lift_depth)
        self.pose_visibility = float(pose_visibility)
        self.use_hand_depth = bool(hand_depth)
        self.record_path = record_path
        self._record_file = None
        self.frame_size = (0, 0)

        # Image-frame vector -> user body frame (x right, y forward, z up).
        # MediaPipe image coords: x image-right, y image-down, z away from
        # the camera (smaller = closer).
        #   selfie, mirrored:   user's right = +x, forward (toward cam) = -z
        #   selfie, raw:        user's right = -x, forward = -z
        #   body cam, raw:      user's right = +x, forward (away)  = +z
        #   body cam, mirrored: user's right = -x, forward = +z
        selfie = self.view_mode == "SELFIE"
        self._sx = (1.0 if mirror else -1.0) * (1.0 if selfie else -1.0)
        self._sdepth = -1.0 if selfie else 1.0
        self._sides = _SideResolver(_user_left_is_image_left(self.view_mode, mirror))
        self._lifter = _DepthLifter(camera_fov, {
            "shoulder": body_shoulder, "upper": body_upper,
            "fore": body_fore,
        })
        self._hand_depth = _HandDepth(hand_size=hand_size, auto_scale=hand_size_auto)
        # Depth for engines whose hand landmarks are 2D only; see handplane.
        self._palm_planes = handplane.PalmPlanes()
        # Depth direction of each forearm from the previous frame, which
        # is what tells a flat palm which way it is tilted. Last frame's
        # is used because this frame's lift needs the hands first.
        self._arm_hint = {}

        self._thread = None
        self._grab_thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = None
        self._error = None
        self._state = "idle"
        self._fps_samples = deque(maxlen=30)

        # Newest raw frame from the grabber, with a sequence number so the
        # inference loop can tell "nothing new yet" from "same content".
        self._grab_lock = threading.Lock()
        self._grab_new = threading.Event()
        self._grab = None            # (bgr_frame, capture_time)
        self._grab_seq = 0
        self._grab_dropped = 0       # frames the inference loop never saw
        self._grab_reopens = 0       # times the watchdog revived a stalled device
        self._last_good_grab = 0.0   # time of the last frame that was not a timeout
        self._grab_stalled = False   # currently inside a stall
        self._revive_failures = 0    # consecutive failed revivals, for backoff
        self._backend = None         # (flag, label) of the backend that worked
        self._cap = None             # owned by _run, swapped by the watchdog
        self._cap_info = ""
        self._cap_warning = ""
        self._open_attempts = []
        self._pool = None
        self._timings = {}           # stage -> EMA milliseconds
        self._pose_tick = 0
        self._last_pose_res = None
        self._last_preview_t = 0.0
        self._brightness = None      # mean pixel value of the last checked frame
        self._bright_check_t = 0.0

        mode = (tracking_mode or "HANDS").upper()
        self._use_hands = mode in ("HANDS", "HOLISTIC")
        self._use_pose = mode in ("POSE", "HOLISTIC")

    def start(self):
        if self._thread and self._thread.is_alive():
            return True
        self._stop.clear()
        self._error = None
        self._state = "starting"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._stop.set()
        # Wake the inference loop if it is parked waiting for a frame.
        self._grab_new.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._thread = None
        if self._grab_thread is not None:
            self._grab_thread.join(timeout=2.0)
            self._grab_thread = None
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

    def engine_info(self):
        """Which models this run actually loaded, for the panel."""
        return self._engine_info

    def hand_scale(self):
        """(effective hand-size scale, learned ratio or None)."""
        return self._hand_depth.scale(), self._hand_depth.auto_ratio()

    # --- diagnostics recording ------------------------------------------

    def _record_open(self):
        if not self.record_path:
            return
        try:
            import json
            self._record_file = open(self.record_path, "w", encoding="utf-8")
            header = {"kind": "header", "mirror": bool(self.mirror),
                      "view_mode": self.view_mode, "camera_fov": math.degrees(self._lifter.hfov),
                      "tracking_mode": self.tracking_mode, "model_complexity": self.model_complexity,
                      "pose_engine": self.pose_engine,
                      "engine_quality": self.engine_quality,
                      "engine_layout": self.engine_layout,
                      "engine_hands": self.engine_hands,
                      "pose_visibility": self.pose_visibility, "lift_depth": self.lift_depth,
                      "hand_depth": self.use_hand_depth, "hand_size": self._hand_depth.hand_size,
                      "hand_size_auto": self._hand_depth.auto_scale}
            self._record_file.write(json.dumps(header) + "\n")
            print("[FPV Pose Tracker] recording frames to", self.record_path)
        except Exception as e:
            print("[FPV Pose Tracker] cannot open recording file:", e)
            self._record_file = None

    def _record_frame(self, res, pose_res, W, H, t, pf):
        if self._record_file is None:
            return
        try:
            import json
            rec = {"kind": "frame", "t": t, "W": W, "H": H,
                   "timings": {k: round(v, 2) for k, v in self._timings.items()},
                   "hands": [{"label": h.label, "landmarks": h.landmarks,
                              "world_landmarks": h.world_landmarks} for h in res.hands],
                   "pose": None,
                   "derived": {"sides_info": pf.sides_info, "depth_info": pf.depth_info,
                               "hand_depth": dict(pf.hand_depth),
                               "lifted": {int(k): list(v) for k, v in pf.pose_lifted.items()}}}
            if pose_res is not None and pose_res.present:
                rec["pose"] = {"landmarks": pose_res.landmarks,
                               "image_landmarks": pose_res.image_landmarks,
                               "visibility": pose_res.visibility}
            self._record_file.write(json.dumps(rec) + "\n")
        except Exception as e:
            print("[FPV Pose Tracker] recording stopped:", e)
            self._record_close()

    def _record_close(self):
        f = self._record_file
        self._record_file = None
        if f is not None:
            try:
                f.close()
            except Exception:
                pass

    # --- coordinate conversion -------------------------------------------

    def raw_to_body(self, x, y, z):
        """Un-mirrored camera-frame vector (x sensor-right, y down, z away
        from the lens) -> user body frame (x right, y forward, z up).

        This is `_image_vec_to_body` with the mirror term removed, because
        `pose_image` has already had the mirror taken out of it. A camera
        that is not square-on to the user leaves a constant rotation in the
        result, which the retargeter's torso calibration measures and
        absorbs, so it does not have to be corrected here."""
        selfie = self.view_mode == "SELFIE"
        sx = -1.0 if selfie else 1.0
        sdepth = -1.0 if selfie else 1.0
        return (sx * x, sdepth * z, -y)

    def body_lengths(self):
        return {"shoulder": self._lifter.shoulder_width(),
                "upper": self._lifter.upper_len(),
                "fore": self._lifter.fore_len()}

    def _image_vec_to_body(self, dx, dy, dz):
        """Image-frame vector (x right, y down, z depth) -> body frame."""
        return (self._sx * dx, self._sdepth * dz, -dy)

    def _hand_to_blender(self, lm, ref_scale, wrist, aspect):
        """Hand landmark (image-normalised, wrist-relative) -> user body frame.

        `aspect` = frame_height / frame_width. MediaPipe normalises x by
        width and y by height, so without this a 16:9 feed squashes every
        vertical direction by ~0.56 and skews the palm normal.
        """
        dx = (lm[0] - wrist[0]) * ref_scale
        dy = (lm[1] - wrist[1]) * ref_scale * aspect
        dz = (lm[2] - wrist[2]) * ref_scale
        return self._image_vec_to_body(dx, dy, dz)

    # Auto-exposure has two incompatible conventions: DirectShow uses
    # 0.75 = auto / 0.25 = manual, V4L2 uses 3 = auto / 1 = manual. Drivers
    # accept a value silently either way, so both are tried.
    _AUTO_EXPOSURE_ON = (0.75, 3.0)
    _AUTO_EXPOSURE_OFF = (0.25, 1.0)

    # Mean pixel value (0-255) below which a frame is treated as black.
    # A dim room still reads well above this; a too-short manual exposure
    # reads around 2-3.
    _BLACK_LEVEL = 10.0

    def _set_auto_exposure(self, cv2, cap, enable, notes):
        values = self._AUTO_EXPOSURE_ON if enable else self._AUTO_EXPOSURE_OFF
        for value in values:
            try:
                if cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, value):
                    return True
            except Exception:
                pass
        notes.append("auto-exposure %s rejected" % ("on" if enable else "off"))
        return False

    def _frame_brightness(self, cap, samples=4):
        """Mean pixel value over a few frames, or None if none could be read."""
        vals = []
        for _ in range(samples):
            try:
                ok, f = cap.read()
            except Exception:
                break
            if ok and f is not None:
                try:
                    vals.append(float(f.mean()))
                except Exception:
                    pass
        if not vals:
            return None
        return sum(vals) / len(vals)

    def _recover_black_frame(self, cv2, cap, notes):
        """If the configured exposure produces a black image, put the camera
        back on auto rather than tracking a black frame for the whole
        session. Returns the brightness finally achieved."""
        bright = self._frame_brightness(cap)
        if bright is None or bright >= self._BLACK_LEVEL:
            return bright
        self._cap_warning = (
            "Camera returned a black image (brightness %.0f/255) — "
            "restoring auto exposure." % bright)
        print("[FPV Pose Tracker]", self._cap_warning)
        notes.append("black frame -> auto exposure")
        self._set_auto_exposure(cv2, cap, True, notes)
        # Some drivers only leave manual mode when a longer exposure is
        # written explicitly, so walk it back up as well.
        for value in (-4.0, -2.0, 0.0):
            try:
                cap.set(cv2.CAP_PROP_EXPOSURE, value)
            except Exception:
                break
            time.sleep(0.15)
            bright = self._frame_brightness(cap, samples=2)
            if bright is not None and bright >= self._BLACK_LEVEL:
                break
        if bright is not None and bright < self._BLACK_LEVEL:
            self._cap_warning = (
                "Camera image is black (brightness %.0f/255) even on auto "
                "exposure. Check the privacy shutter, that no other app has "
                "the camera, and the room lighting." % bright)
            print("[FPV Pose Tracker]", self._cap_warning)
        else:
            self._cap_warning += " Recovered."
        return bright

    def cap_warning(self):
        return self._cap_warning

    def brightness(self):
        """Mean pixel value (0-255) of the feed, or None before the first
        check. Low means the camera is seeing nothing — a closed privacy
        shutter looks identical to broken tracking otherwise."""
        return self._brightness

    def is_black(self):
        b = self._brightness
        return b is not None and b < self._BLACK_LEVEL

    def _check_brightness(self, frame, now):
        """Sample the feed's brightness once in a while.

        Continuous rather than startup-only so that covering the lens
        mid-session is caught, and so uncovering it clears the warning
        without needing a restart. Strided sampling keeps it off the
        critical path.
        """
        if (now - self._bright_check_t) < _BRIGHT_CHECK_INTERVAL:
            return
        self._bright_check_t = now
        try:
            self._brightness = float(frame[::8, ::8].mean())
        except Exception:
            pass

    @staticmethod
    def _fourcc_str(cv2, cap):
        try:
            cc = int(cap.get(cv2.CAP_PROP_FOURCC))
        except Exception:
            return ""
        if not cc:
            return ""
        return "".join(chr((cc >> (8 * i)) & 0xFF) for i in range(4)).strip()

    def _apply_format(self, cv2, cap, notes):
        """Negotiate codec and frame size, then check what actually stuck.

        Drivers disagree about the order. Some build their mode list from the
        codec, so FOURCC has to come first; others reset the codec to their
        raw default whenever the frame size changes, silently undoing an MJPG
        request made before it (measured on a Logitech C-series: FOURCC then
        640x480 comes back YUY2, the same calls in the other order come back
        MJPG). Neither order is right for every camera, so we set it, set the
        size, and re-assert whichever one the other knocked out.
        """
        fourcc = None
        if self.use_mjpg:
            try:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            except Exception as e:
                notes.append("mjpg failed: %s" % e)
                fourcc = None

        def set_size():
            if self.capture_width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.capture_width))
            if self.capture_height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.capture_height))

        def size_ok():
            if self.capture_width and int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) != self.capture_width:
                return False
            if self.capture_height and int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) != self.capture_height:
                return False
            return True

        set_size()
        if fourcc is not None and self._fourcc_str(cv2, cap).upper() not in ("MJPG", "MJPEG"):
            # The size reset the codec. Re-assert it, then put the size back
            # if that in turn moved it.
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            if not size_ok():
                set_size()
            if self._fourcc_str(cv2, cap).upper() in ("MJPG", "MJPEG"):
                notes.append("mjpg needed re-asserting after size")

    def _apply_frame_rate(self, cv2, cap, notes):
        """Request a capture rate, but only from a driver that reports one.

        A driver that returns -1 for its own frame rate does not really
        implement the property. Such a driver has been measured to accept the
        request, run fine for that session, and then refuse the *next* open of
        the device — which shows up as a camera that works exactly once per
        Blender run and then "cannot be opened". Not asking costs nothing: the
        camera still runs at its own best rate.
        """
        if not self.target_fps:
            return
        try:
            reported = float(cap.get(cv2.CAP_PROP_FPS))
        except Exception:
            reported = -1.0
        if reported > 0.0:
            cap.set(cv2.CAP_PROP_FPS, float(self.target_fps))
        else:
            notes.append("fps request skipped (driver reports no rate)")

    def _configure_capture(self, cv2, cap, label):
        """Ask the camera for the mode we actually want and report what we
        got. Exposure comes last because setting a mode can reset it.
        """
        notes = []
        self._apply_format(cv2, cap, notes)
        self._apply_frame_rate(cv2, cap, notes)
        # One frame of buffer: we always want the newest frame, and a deep
        # buffer turns a slow inference pass into ever-growing latency.
        # DirectShow rejects this outright, which is harmless.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # Exposure mode is always written, including "Auto".
        #
        # Webcam drivers keep a manual exposure after the process that set it
        # exits, so leaving Auto as "change nothing" makes manual mode a
        # one-way door: once a too-short exposure has been set the camera
        # stays dark for every later run, which looks exactly like broken
        # tracking (no landmarks, calibration never completes, rig frozen).
        if self.exposure_mode == "MANUAL":
            self._set_auto_exposure(cv2, cap, False, notes)
            try:
                cap.set(cv2.CAP_PROP_EXPOSURE, float(self.exposure))
            except Exception as e:
                notes.append("exposure failed: %s" % e)
            if self.gain >= 0.0:
                try:
                    cap.set(cv2.CAP_PROP_GAIN, float(self.gain))
                except Exception:
                    pass
        else:
            self._set_auto_exposure(cv2, cap, True, notes)

        def _get(prop, default=0.0):
            try:
                return cap.get(prop)
            except Exception:
                return default

        bright = self._recover_black_frame(cv2, cap, notes)

        w, h = int(_get(cv2.CAP_PROP_FRAME_WIDTH)), int(_get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = _get(cv2.CAP_PROP_FPS)
        cc_str = self._fourcc_str(cv2, cap) or "?"
        info = "%s %dx%d @%.0f %s" % (label, w, h, fps, cc_str)
        # The driver silently keeps its own codec when it will not honour the
        # request, so say so rather than letting the user assume MJPG applied.
        if self.use_mjpg and cc_str and cc_str.upper() not in ("MJPG", "MJPEG"):
            notes.append("MJPG refused, still %s" % cc_str)
        if self.exposure_mode == "MANUAL":
            info += " exp=%.1f" % _get(cv2.CAP_PROP_EXPOSURE)
        if bright is not None:
            info += " lum=%.0f" % bright
        if notes:
            info += " [" + "; ".join(notes) + "]"
        self._cap_info = info
        print("[FPV Pose Tracker] camera:", info)
        return info

    def _open_capture(self, cv2):
        """Open the camera, exhausting the backend that works before trying
        one that might not.

        The order is not a preference list to walk quickly: on Windows,
        DirectShow is the backend that actually drives most webcams, while
        Media Foundation has been measured taking over 90 seconds to open a
        device it then cannot read, and the generic fallback has been seen to
        return a black 1 fps feed that looks like success and stops the search.
        So every DirectShow retry is spent before either is considered.
        """
        attempts = []
        for backend, label in (
            (getattr(cv2, "CAP_DSHOW", 0), "DSHOW"),
            (getattr(cv2, "CAP_MSMF", 0), "MSMF"),
            (0, "default"),
        ):
            if self._stop.is_set():
                return None
            self._state = "opening camera %d (%s)" % (self.camera_index, label)
            cap, tries = open_camera(
                cv2, self.camera_index, backend=backend,
                configure=lambda c, lb=label: self._configure_capture(cv2, c, lb))
            attempts.extend("%s%s" % (label, t) for t in tries)
            if cap is not None:
                if tries:
                    print("[FPV Pose Tracker] camera %d opened on %s after %d "
                          "failed attempt(s)" % (self.camera_index, label, len(tries)))
                self._state = "camera opened via %s" % label
                self._open_attempts = attempts
                self._backend = (backend, label)
                return cap
        self._open_attempts = attempts
        return None

    # --- capture thread ---------------------------------------------------

    def _grab_loop(self, cv2, cap):
        """Read frames as fast as the camera delivers them, keeping only the
        newest, and revive the device if it stops delivering.

        Without this the inference loop calls `cap.read()` itself, and any
        pass that takes longer than a frame interval leaves the driver's
        queue to decide what happens next — which is what pinned the old
        loop to an exact subharmonic of the camera rate (~15 fps off a
        30 fps camera) instead of simply running as fast as it could.

        The watchdog is here because a stall is permanent. Measured on an
        older Logitech sharing a machine with a second webcam: once it enters
        the state where every `read()` returns after DirectShow's internal
        ~1000 ms timeout, it delivered 0.0 fps for 15 s straight and never
        came back on its own, while a release and reopen restored it
        immediately. That is a camera whose preview shows one frame and then
        freezes for the rest of the session, which is exactly what it looks
        like from the outside. Reading harder cannot fix it; only reopening
        can, so the grab thread owns that recovery.
        """
        while not self._stop.is_set():
            t0 = time.time()
            ok, frame = cap.read()
            if not ok or frame is None:
                if self._stall_expired(t0):
                    cap = self._revive(cv2, cap)
                time.sleep(0.005)
                continue
            now = time.time()
            if (now - t0) * 1000.0 >= _STALL_READ_MS:
                # A read that only returned on the driver's own timeout is not
                # a frame, it is the stall announcing itself.
                if self._stall_expired(t0):
                    cap = self._revive(cv2, cap)
                    continue
            else:
                self._last_good_grab = now
                self._grab_stalled = False
            # Copy: OpenCV may hand back a view onto a buffer it reuses on the
            # next read(), and the inference thread holds this frame while the
            # grabber has already moved on.
            frame = frame.copy()
            self._note_timing("grab", (now - t0) * 1000.0)
            with self._grab_lock:
                if self._grab is not None:
                    self._grab_dropped += 1
                self._grab = (frame, now)
                self._grab_seq += 1
            self._grab_new.set()

    def _stall_expired(self, now):
        """True when nothing usable has arrived for long enough to act."""
        last = self._last_good_grab
        if last <= 0.0:
            # Nothing has ever arrived; measure from the start of the loop.
            self._last_good_grab = now
            return False
        return (now - last) >= _STALL_TIMEOUT

    def _revive(self, cv2, cap):
        """Release a stalled device and open it again, in place.

        Returns the capture to keep using: the new one, or the old one if the
        reopen failed, because a stalled capture that might come back is still
        better than no capture at all.
        """
        self._grab_stalled = True
        print("[FPV Pose Tracker] camera %d stopped delivering frames — "
              "reopening" % self.camera_index)
        try:
            cap.release()
        except Exception:
            pass
        # Give the device time to actually let go. Measured: this camera
        # needs a couple of seconds of rest before it will reopen, which is
        # longer than the open retry budget alone covers — and a device
        # wedged mid-stream sometimes needs several. Back off rather than
        # hammering it, since a failed reopen costs a full retry sweep.
        rest = min(_STALL_REOPEN_REST * (1 + self._revive_failures),
                   _STALL_REOPEN_REST_MAX)
        time.sleep(rest)
        fresh = None
        if self._backend is not None:
            # Retry the backend that was working, spaced far enough apart to
            # respect the rest this device needs. Falling through to another
            # backend "succeeds" but lands in a worse mode: measured, the
            # generic fallback brought this camera back at 5.2 fps where
            # DirectShow brings it back at its full 13.5.
            flag, label = self._backend
            self._state = "reopening camera %d (%s)" % (self.camera_index, label)
            fresh, _ = open_camera(
                cv2, self.camera_index, retries=_STALL_REOPEN_RETRIES,
                delay=_STALL_REOPEN_REST, backend=flag,
                configure=lambda c, lb=label: self._configure_capture(cv2, c, lb))
        if fresh is None:
            fresh = self._open_capture(cv2)
        self._last_good_grab = time.time()
        if fresh is None:
            self._revive_failures += 1
            print("[FPV Pose Tracker] camera %d did not come back (attempt %d)"
                  % (self.camera_index, self._revive_failures))
            self._error = "camera %d stopped responding" % self.camera_index
            return cap
        # It came back, so the camera is not in a failed state any more: a
        # stall the watchdog recovered from must not leave the panel showing
        # FAILED for the rest of the session.
        self._revive_failures = 0
        self._error = None
        self._grab_reopens += 1
        self._cap = fresh
        self._grab_stalled = False
        return fresh

    def grab_reopens(self):
        """How many times the watchdog has had to revive this camera."""
        return self._grab_reopens

    def _take_frame(self, timeout=1.0):
        """Newest grabbed frame, or None if none arrived in time."""
        if not self._grab_new.wait(timeout):
            return None, None
        with self._grab_lock:
            item = self._grab
            self._grab = None
            self._grab_new.clear()
        if item is None:
            return None, None
        return item

    def _note_timing(self, stage, ms):
        # Exponential moving average: one number per stage, cheap to keep
        # and stable enough to read off a panel.
        prev = self._timings.get(stage)
        self._timings[stage] = ms if prev is None else (prev * 0.9 + ms * 0.1)

    def timings(self):
        """Per-stage milliseconds (EMA), plus dropped-frame count."""
        out = dict(self._timings)
        out["dropped"] = float(self._grab_dropped)
        return out

    def cap_info(self):
        return self._cap_info

    # --- inference --------------------------------------------------------

    def _infer(self, hand_backend, pose_backend, want_pose, rgb, ts_ms):
        """Run the landmarkers on one frame, overlapping them when both are
        enabled.

        MediaPipe releases the GIL inside its native inference, so two
        Python threads really do overlap here rather than taking turns. The
        two landmarkers hold separate tracking state and only read the
        frame, so sharing it between them is safe.
        """
        # A ROI hand model needs to know where the body thinks the hands
        # are. With the two running in parallel these boxes are one frame
        # old, which at webcam rates moves a wrist by a few percent of its
        # own crop — far less than the crop's padding, and it buys back the
        # entire cost of a second detector.
        if hand_backend is not None and hasattr(hand_backend, "set_body"):
            hand_backend.set_body(pose_backend, self._last_pose_res)

        def run_hands():
            if hand_backend is None:
                return mp_backend.HandsResult()
            t0 = time.time()
            try:
                return hand_backend.process(rgb, ts_ms)
            finally:
                self._note_timing("hands", (time.time() - t0) * 1000.0)

        def run_pose():
            # On a skipped tick the previous pose is reused: arms move
            # slowly enough in image terms that holding one frame costs
            # far less than the inference it saves.
            if not want_pose:
                return self._last_pose_res
            t0 = time.time()
            try:
                res = pose_backend.process(rgb, ts_ms)
            except Exception as e:
                print("[FPV Pose Tracker] pose process error:", e)
                return self._last_pose_res
            finally:
                self._note_timing("pose", (time.time() - t0) * 1000.0)
            self._last_pose_res = res
            return res

        if self._pool is not None and want_pose:
            try:
                fh = self._pool.submit(run_hands)
                fp = self._pool.submit(run_pose)
                return fh.result(), fp.result()
            except Exception as e:
                print("[FPV Pose Tracker] parallel inference failed, "
                      "falling back to sequential:", e)
                try:
                    self._pool.shutdown(wait=False)
                except Exception:
                    pass
                self._pool = None
        return run_hands(), run_pose()

    def _annotate(self, cv2, frame, res, pose_res, pf, W, H):
        overlay = frame.copy()
        self._draw_hands(overlay, res)
        if pose_res is not None and pose_res.present:
            self._draw_pose(overlay, pose_res, W, H)
        hud = "hands:%d pose:%s L:%s R:%s %s" % (
            len(res.hands),
            "Y" if pf.pose_present else "-",
            "Y" if pf.left_present else "-",
            "Y" if pf.right_present else "-",
            pf.sides_info,
        )
        cv2.putText(overlay, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 2, cv2.LINE_AA)
        if pf.depth_info:
            cv2.putText(overlay, pf.depth_info, (8, 44), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 2, cv2.LINE_AA)
        tm = self._timings
        revived = "" if not self._grab_reopens else " | revived %d" % self._grab_reopens
        perf = "%.1f fps | tot %.0f grab %.0f hand %.0f pose %.0f ovl %.0f | drop %d" % (
            self.fps(), tm.get("total", 0.0), tm.get("grab", 0.0),
            tm.get("hands", 0.0), tm.get("pose", 0.0), tm.get("overlay", 0.0),
            self._grab_dropped) + revived
        cv2.putText(overlay, perf, (8, 66), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 220, 255), 2, cv2.LINE_AA)
        return overlay

    # --- offline reuse ------------------------------------------------

    # A recorded clip has to be turned into PoseFrames by exactly the same
    # maths a live camera is, or a take baked from footage and one baked
    # from the webcam would disagree about depth, sides and hand scale.
    # These three are that seam: `offline` builds a tracker it never
    # starts, purely to call them.

    def build_pose_frame(self, res, pose_res, W, H, t):
        """Assemble a PoseFrame from raw backend results off any source."""
        return self._build_frame(res, pose_res, W, H, t)

    def draw_pose_overlay(self, img, pose_res, w, h):
        """Draw body landmarks onto a BGR frame, in place."""
        self._draw_pose(img, pose_res, w, h)

    def draw_hand_overlay(self, img, res):
        """Draw hand landmarks onto a BGR frame, in place."""
        self._draw_hands(img, res)

    # --- per-frame assembly ------------------------------------------------

    def _build_frame(self, res, pose_res, W, H, t):
        pf = PoseFrame()
        pf.timestamp = t
        aspect = H / float(max(1, W))
        thr = self.pose_visibility

        # ---- pose: decide sides, convert ---------------------------------
        pose_wrists_img = None
        if pose_res is not None and pose_res.present:
            swap = self._sides.pose_swap(pose_res.image_landmarks)
            lms = pose_res.landmarks
            ilms = pose_res.image_landmarks
            vis = pose_res.visibility
            if swap:
                lms = _swap_pose_sides(lms)
                ilms = _swap_pose_sides(ilms)
                vis = _swap_pose_sides(vis)
            conv = (lambda lm: (lm[0], -lm[2], lm[1])) if self.pose_y_up \
                   else (lambda lm: (lm[0], lm[2], -lm[1]))
            for i, lm in enumerate(lms):
                pf.pose_world[i] = conv(lm)
            for i, v in enumerate(vis):
                pf.pose_visibility[i] = float(v)
            pf.pose_present = True
            # Undo the display mirror so the coordinates describe the real
            # camera. Triangulation across two cameras is a statement about
            # physical geometry and a reflected frame is a different (left
            # handed) camera, which would place every point behind the lens.
            flip = bool(self.mirror)
            for i, lm in enumerate(ilms or ()):
                u = (1.0 - lm[0]) if flip else lm[0]
                pf.pose_image[i] = (float(u), float(lm[1]))
            if ilms and len(ilms) > POSE_RWRIST:
                pose_wrists_img = {"L": (ilms[POSE_LWRIST][0], ilms[POSE_LWRIST][1]),
                                   "R": (ilms[POSE_RWRIST][0], ilms[POSE_RWRIST][1])}
            self._lifter.update_lengths(lms, vis, thr)
        else:
            ilms = None
            vis = None

        # ---- hands: assign sides, convert -----------------------------------
        hand_wrists_img = {}
        hand_depth_img = {}
        self._hand_depth.begin_frame()
        f_px = self._lifter.f_px(W)
        pf.cam_model = (float(f_px), int(W), int(H))
        pf.body_lengths = {"shoulder": self._lifter.shoulder_width(),
                           "upper": self._lifter.upper_len(),
                           "fore": self._lifter.fore_len()}
        sides, side_src = self._sides.hand_sides(res.hands, pose_wrists_img)
        for side, track in zip(sides, res.hands):
            if not track.landmarks:
                continue
            # RTMPose and ViTPose hands are pixels with no depth at all,
            # which gives the retargeter a palm whose normal is stuck on
            # the camera axis. Fitting the palm plane puts a usable depth
            # back; a MediaPipe hand already has one and is left alone.
            landmarks = self._palm_planes.lift(side, track.landmarks, aspect,
                                               self._arm_hint.get(side, 0.0))
            wrist = landmarks[HAND_WRIST]
            hand_wrists_img[side] = (wrist[0], wrist[1])
            if self.use_hand_depth:
                wr_cam = self._hand_depth.wrist(side, track.landmarks,
                                                track.world_landmarks,
                                                W, H, f_px, self.mirror, t)
                if wr_cam is not None:
                    hand_depth_img[side] = wr_cam[2]
                    pf.hand_depth[side] = wr_cam[2]
            ref_scale = 0.18   # approximate scale so a hand ~= 0.18 m across
            hand_dict = {}
            for i, lm in enumerate(landmarks):
                hand_dict[i] = self._hand_to_blender(lm, ref_scale, wrist, aspect)
            if hand_dict:
                pf.hand_quality[side] = sum(
                    math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
                    for v in hand_dict.values()) / len(hand_dict)
            if side == "L":
                pf.left_hand = hand_dict
                pf.left_present = True
            else:
                pf.right_hand = hand_dict
                pf.right_present = True

        # ---- depth lifting ----------------------------------------------------
        depth_info = ""
        if pf.pose_present and self.lift_depth and ilms:
            lifted = self._lifter.lift(ilms, vis, thr, W, H, t, hand_wrists_img,
                                       hand_depth_img)
            if lifted:
                flip = -1.0 if self.mirror else 1.0
                for i, p in lifted.items():
                    pf.pose_lifted[i] = self._image_vec_to_body(p[0], p[1], p[2])
                    # The lift ran on the mirrored image, so its x is
                    # reflected; undo that to get the real camera's frame.
                    pf.pose_lifted_cam[i] = (flip * p[0], p[1], p[2])
                for side, (el_i, wr_i) in (("L", (POSE_LELBOW, POSE_LWRIST)),
                                           ("R", (POSE_RELBOW, POSE_RWRIST))):
                    e, w = lifted.get(el_i), lifted.get(wr_i)
                    if e is None or w is None:
                        continue
                    dx, dy, dz = w[0] - e[0], w[1] - e[1], w[2] - e[2]
                    n = math.sqrt(dx * dx + dy * dy + dz * dz)
                    # Unit depth component, so a forearm across the view
                    # contributes nothing rather than a confident zero.
                    self._arm_hint[side] = (dz / n) if n > 1e-6 else 0.0
                for side, cands in self._lifter.wrist_depth_candidates.items():
                    if side in hand_depth_img:
                        self._hand_depth.add_candidates(side, cands)
                src = self._lifter.root_source
                depth_info = "roots[L:%s R:%s]" % (
                    src.get("L", "-")[:3], src.get("R", "-")[:3])
        if hand_depth_img:
            depth_info += " handz[%s] x%.2f" % (
                " ".join("%s=%.2f" % (k, v) for k, v in sorted(hand_depth_img.items())),
                self._hand_depth.scale())

        pf.sides_info = "pose%s hands[%s]" % (
            "*" if self._sides.pose_swap_score > 0 else "", side_src)
        pf.depth_info = depth_info.strip()
        pf.joint_angles = pose_metrics.from_frame(pf, thr)
        return pf

    def _run(self):
        try:
            self._state = "importing cv2/mediapipe"
            import cv2
        except Exception as e:
            self._error = "OpenCV import failed: %s" % e
            self._state = "import failed"
            return

        # Keep trying rather than dying on the first refusal. A flaky camera
        # that refuses roughly half its opens would otherwise be absent for
        # the whole session on a coin flip, which in a multi-camera rig
        # silently halves the tracking the user asked for. Stop is checked
        # every pass, so this never outlives the user pressing Stop.
        cap = None
        sweep = 0
        while cap is None and not self._stop.is_set():
            cap = self._open_capture(cv2)
            if cap is not None:
                break
            sweep += 1
            detail = "; ".join(self._open_attempts[-3:])
            self._error = ("cannot open webcam index %d (is it in use by "
                           "another app?)%s") % (
                self.camera_index, (" [" + detail + "]") if detail else "")
            self._state = "camera open failed, retrying (sweep %d)" % sweep
            if sweep >= _OPEN_SWEEPS:
                break
            self._stop.wait(min(_STALL_REOPEN_REST * sweep,
                                _STALL_REOPEN_REST_MAX))
        if cap is None:
            return
        # It opened, so it is not in a failed state.
        self._error = None

        hand_backend = None
        pose_backend = None
        if self._use_hands:
            self._state = ("loading %s hand model (first run downloads it)"
                           % self.pose_engine.lower())
            hand_backend = pose_engines.make_hands_backend(
                self.pose_engine, hand_source=self.engine_hands,
                body_layout=self.engine_layout, max_hands=2,
                running_mode="VIDEO", delegate=self.delegate,
                device=self.engine_device, kpt_thr=self.engine_kpt_thr,
                hand_detector=self.engine_hand_detector,
            )
            try:
                hand_backend.load()
            except Exception as e:
                self._error = str(e)
                self._state = "hand model init failed"
                cap.release()
                return
        if self._use_pose:
            self._state = ("loading %s body model (first run downloads it)"
                           % self.pose_engine.lower())
            pose_backend = pose_engines.make_pose_backend(
                self.pose_engine, model_complexity=self.model_complexity,
                running_mode="VIDEO", delegate=self.delegate,
                quality=self.engine_quality, body_layout=self.engine_layout,
                device=self.engine_device, det_every=self.engine_det_every,
                kpt_thr=self.engine_kpt_thr,
            )
            try:
                pose_backend.load()
            except Exception as e:
                # Pose failure is non-fatal — continue with hands only.
                print("[FPV Pose Tracker] pose backend disabled:", e)
                pose_backend = None

        which_str = ",".join([b.which() for b in (hand_backend, pose_backend) if b])
        if pose_backend is not None and not self._caps.world_pose \
                and not self._lifter.lengths_measured():
            # A 2D engine cannot measure the user, so unless they typed
            # their own lengths the lift is scaling an average body.
            which_str += " [generic limb lengths]"
        self._engine_info = which_str
        self._state = "waiting for first frame (backend=%s)" % which_str
        self._record_open()
        window_name = "FPV Pose Tracker (press Q here to close)"
        window_opened = False
        t_start = time.time()

        # The watchdog may swap the device out from under us, so the
        # capture to release at the end is whichever one it ended up on.
        self._cap = cap
        self._last_good_grab = 0.0
        self._grab_thread = threading.Thread(
            target=self._grab_loop, args=(cv2, cap), daemon=True)
        self._grab_thread.start()

        if self.parallel_infer and hand_backend is not None and pose_backend is not None:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(max_workers=2,
                                            thread_name_prefix="fpv-infer")

        try:
            while not self._stop.is_set():
                frame, now = self._take_frame()
                if frame is None:
                    if not self._grab_thread.is_alive():
                        break
                    continue
                t_frame = time.time()

                if self.mirror:
                    frame = cv2.flip(frame, 1)
                H, W = frame.shape[:2]
                self.frame_size = (W, H)
                self._check_brightness(frame, now)

                t0 = time.time()
                # Landmarks come back normalised, so a downscaled inference
                # image produces identical coordinates and the depth maths
                # (which works off the real frame width) is unaffected.
                # A top-down engine crops and rescales to its own input
                # size, so shrinking the frame first saves it nothing and
                # costs the hand crops the resolution they exist to get.
                if self._caps.wants_full_frame:
                    infer_bgr = frame
                elif self.infer_width and W > self.infer_width:
                    iw = self.infer_width
                    ih = max(1, int(H * iw / W))
                    infer_bgr = cv2.resize(frame, (iw, ih), interpolation=cv2.INTER_AREA)
                else:
                    infer_bgr = frame
                rgb = cv2.cvtColor(infer_bgr, cv2.COLOR_BGR2RGB)
                self._note_timing("prep", (time.time() - t0) * 1000.0)

                ts_ms = int((now - t_start) * 1000.0)
                self._pose_tick += 1
                want_pose = (pose_backend is not None
                             and (self._pose_tick % self.pose_every) == 0)

                res, pose_res = self._infer(hand_backend, pose_backend, want_pose,
                                            rgb, ts_ms)

                t0 = time.time()
                pf = self._build_frame(res, pose_res, W, H, now)
                self._note_timing("build", (time.time() - t0) * 1000.0)
                self._record_frame(res, pose_res, W, H, now, pf)

                # The overlay is pure presentation: drawing, resizing and two
                # colour converts per frame is real money on the critical
                # path, so it runs at its own much lower rate.
                want_preview = self.show_window or self.preview_fps > 0.0
                interval = 1.0 / self.preview_fps if self.preview_fps > 0 else 0.0
                if want_preview and (now - self._last_preview_t) >= interval:
                    self._last_preview_t = now
                    t0 = time.time()
                    overlay = self._annotate(cv2, frame, res, pose_res, pf, W, H)
                    if self.preview_width and W > self.preview_width:
                        new_w = self.preview_width
                        new_h = max(1, int(H * new_w / W))
                        small = cv2.resize(overlay, (new_w, new_h),
                                           interpolation=cv2.INTER_AREA)
                    else:
                        small = overlay
                    pf.annotated_rgba = cv2.cvtColor(small, cv2.COLOR_BGR2RGBA)
                    self._note_timing("overlay", (time.time() - t0) * 1000.0)

                    if self.show_window:
                        try:
                            cv2.imshow(window_name, overlay)
                            window_opened = True
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord("q"):
                                self._stop.set()
                        except Exception as e:
                            self.show_window = False
                            print("[FPV] imshow disabled:", e)

                with self._lock:
                    self._latest = pf
                self._fps_samples.append(pf.timestamp)
                self._note_timing("total", (time.time() - t_frame) * 1000.0)
                self._state = "tracking %dx%d (%s)" % (W, H, which_str)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._error = "tracker error: %s" % e
            self._state = "error"
        finally:
            # Stop grabbing before releasing the device, or the grabber
            # thread reads from a freed capture.
            self._stop.set()
            if self._grab_thread is not None:
                self._grab_thread.join(timeout=2.0)
                self._grab_thread = None
            if self._pool is not None:
                try:
                    self._pool.shutdown(wait=True)
                except Exception:
                    pass
                self._pool = None
            self._record_close()
            if hand_backend is not None:
                hand_backend.close()
            if pose_backend is not None:
                pose_backend.close()
            (self._cap or cap).release()
            self._cap = None
            if window_opened:
                try:
                    cv2.destroyWindow(window_name)
                    cv2.waitKey(1)
                except Exception:
                    pass

    def _draw_pose(self, img, pose_res, w, h):
        try:
            import cv2
        except Exception:
            return
        if not pose_res.image_landmarks:
            return
        conns = [
            (POSE_LSHOULDER, POSE_RSHOULDER),
            (POSE_LSHOULDER, POSE_LELBOW),
            (POSE_LELBOW, POSE_LWRIST),
            (POSE_RSHOULDER, POSE_RELBOW),
            (POSE_RELBOW, POSE_RWRIST),
        ]
        pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in pose_res.image_landmarks]
        vis = pose_res.visibility

        def seen(i):
            # A 2D engine reports an unconfident joint as visibility 0 with
            # no position at all, so drawing it would fire a bone at the
            # top-left corner. MediaPipe always has a position, and passes
            # this test on any threshold it reports.
            if i >= len(pts):
                return False
            return (i >= len(vis)) or vis[i] > 0.0

        for a, b in conns:
            if seen(a) and seen(b):
                cv2.line(img, pts[a], pts[b], (255, 128, 0), 2, cv2.LINE_AA)
        for i in (POSE_LSHOULDER, POSE_RSHOULDER, POSE_LELBOW, POSE_RELBOW,
                  POSE_LWRIST, POSE_RWRIST):
            if seen(i):
                cv2.circle(img, pts[i], 4, (255, 255, 0), -1, cv2.LINE_AA)

    def _draw_hands(self, img, res):
        try:
            import cv2
        except Exception:
            return
        h, w = img.shape[:2]
        conns = [
            (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
            (0, 5), (5, 6), (6, 7), (7, 8),           # index
            (0, 9), (9, 10), (10, 11), (11, 12),      # middle
            (0, 13), (13, 14), (14, 15), (15, 16),    # ring
            (0, 17), (17, 18), (18, 19), (19, 20),    # pinky
            (5, 9), (9, 13), (13, 17),                # palm
        ]
        for track in res.hands:
            pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in track.landmarks]
            for a, b in conns:
                if a < len(pts) and b < len(pts):
                    cv2.line(img, pts[a], pts[b], (0, 255, 255), 2, cv2.LINE_AA)
            for p in pts:
                cv2.circle(img, p, 3, (0, 0, 255), -1, cv2.LINE_AA)


def test_webcam(camera_index=0, tracking_mode="HANDS", mirror=True,
                pose_engine="MEDIAPIPE", engine_quality="BALANCED",
                engine_layout="BODY", engine_hands="AUTO",
                engine_device="CPU", engine_kpt_thr=0.3):
    """Synchronous single-frame test. Returns (ok, human_report_string).

    Runs the engine the user has selected, not a fixed one: "does my
    webcam work" and "can this machine run ViTPose" are the same question
    at the point where either answer stops the session starting.
    """
    lines = [pose_engines.diagnostics(pose_engine), ""]
    try:
        import cv2
    except Exception as e:
        return False, "\n".join(lines + ["OpenCV import failed: %s" % e])

    lines.append("Opening camera %d..." % camera_index)
    cap = None
    for backend_id, label in (
        (getattr(cv2, "CAP_DSHOW", 0), "DSHOW"),
        (getattr(cv2, "CAP_MSMF", 0), "MSMF"),
        (0, "default"),
    ):
        cap_try = cv2.VideoCapture(camera_index, backend_id) if backend_id else \
                  cv2.VideoCapture(camera_index)
        if cap_try.isOpened():
            ok, frame = cap_try.read()
            if ok and frame is not None:
                cap = cap_try
                lines.append("  opened via %s, shape=%s" % (label, tuple(frame.shape)))
                break
            cap_try.release()
        else:
            lines.append("  %s: not opened" % label)
    if cap is None:
        return False, "\n".join(lines + [
            "ERROR: cannot open camera. Close other webcam apps.",
            "Windows: Settings > Privacy > Camera > 'Let desktop apps access your camera'.",
        ])

    ok, frame = cap.read()
    if not ok:
        cap.release()
        return False, "\n".join(lines + ["ERROR: opened but read() failed"])
    if mirror:
        frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mode = (tracking_mode or "HANDS").upper()
    want_pose = mode in ("POSE", "HOLISTIC")
    want_hands = mode in ("HANDS", "HOLISTIC")
    pose_backend = None
    hand_backend = None
    pose_res = None

    if want_pose:
        lines.append("Loading %s body model (first run downloads it)..."
                     % pose_engine.lower())
        pose_backend = pose_engines.make_pose_backend(
            pose_engine, min_detection=0.3, min_tracking=0.3,
            quality=engine_quality, body_layout=engine_layout,
            device=engine_device, kpt_thr=engine_kpt_thr,
        )
        try:
            pose_backend.load()
            lines.append("  backend: %s" % pose_backend.which())
            pose_res = pose_backend.process(rgb)
        except Exception as e:
            cap.release()
            return False, "\n".join(lines + ["ERROR (body): %s" % e])
        if pose_res.present:
            seen = sum(1 for v in pose_res.visibility if v > 0.0)
            lines.append("Pose: %d/%d landmarks seen, world pose: %s"
                         % (seen, len(pose_res.visibility),
                            "yes" if pose_res.landmarks else "no (2D engine)"))
            ang = pose_metrics.compute(pose_res.image_landmarks,
                                       pose_res.visibility, 0.01, space="2d")
            if ang.summary():
                lines.append("  angles: " + ang.summary())
        else:
            lines.append("Pose: nobody found in this frame")

    if want_hands:
        lines.append("Loading hand model...")
        hand_backend = pose_engines.make_hands_backend(
            pose_engine, hand_source=engine_hands, body_layout=engine_layout,
            max_hands=2, min_detection=0.3, min_tracking=0.3,
            device=engine_device, kpt_thr=engine_kpt_thr,
        )
        try:
            hand_backend.load()
        except Exception as e:
            if pose_backend is not None:
                pose_backend.close()
            cap.release()
            return False, "\n".join(lines + ["ERROR (hands): %s" % e])
        lines.append("  backend: %s" % hand_backend.which())
        if hasattr(hand_backend, "set_body"):
            hand_backend.set_body(pose_backend, pose_res)
        try:
            res = hand_backend.process(rgb)
        except Exception as e:
            hand_backend.close()
            if pose_backend is not None:
                pose_backend.close()
            cap.release()
            return False, "\n".join(lines + ["ERROR during process(): %s" % e])
        lines.append("Detected hands: %d" % len(res.hands))
        for t in res.hands:
            lines.append("  %s hand: %d landmarks" % (t.label, len(t.landmarks)))

    if hand_backend is not None:
        hand_backend.close()
    if pose_backend is not None:
        pose_backend.close()
    cap.release()
    return True, "\n".join(lines)


def autotune_exposure(camera_index=0, width=0, height=0, min_brightness=20.0):
    """Find the exposure that gets the most frames without going dark.

    Webcam frame rate is usually capped by the exposure time, not by the
    sensor: auto-exposure in an ordinary room picks something like 1/15 s,
    which pins capture at 15 fps however high a rate is requested. Each stop
    shorter doubles the achievable rate, until the link's bandwidth becomes
    the limit and further darkening buys nothing.

    So the wanted value is the *brightest* exposure that still reaches the
    plateau. This walks darker only while the rate keeps improving, which
    takes a handful of measurements rather than a full sweep.

    Returns (ok, chosen_exposure, message).
    """
    cv2, _ = _lazy_cv()
    cap = cv2.VideoCapture(camera_index, getattr(cv2, "CAP_DSHOW", 0))
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return False, None, "Could not open camera %d" % camera_index

    def measure(exposure, frames=8):
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
        cap.read()
        time.sleep(0.5)
        times, bright = [], []
        for _ in range(frames):
            t0 = time.time()
            ok, f = cap.read()
            times.append(time.time() - t0)
            if ok and f is not None:
                try:
                    bright.append(float(f[::8, ::8].mean()))
                except Exception:
                    pass
        if not times:
            return 0.0, 0.0
        median = sorted(times)[len(times) // 2]
        fps = 1.0 / median if median > 1e-6 else 0.0
        return fps, (sum(bright) / len(bright) if bright else 0.0)

    results = []
    try:
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        cap.set(cv2.CAP_PROP_FPS, 60.0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try:
            cap.set(cv2.CAP_PROP_GAIN, 0)
        except Exception:
            pass

        best_fps = 0.0
        for exposure in range(-3, -10, -1):
            fps, bright = measure(exposure)
            results.append((exposure, fps, bright))
            if fps > best_fps * 1.15:
                best_fps = max(best_fps, fps)
                continue
            # Rate has stopped improving: darker only costs brightness.
            best_fps = max(best_fps, fps)
            break

        usable = [r for r in results
                  if r[1] >= best_fps * 0.9 and r[2] >= min_brightness]
        if usable:
            # Brightest of the ones that reach the plateau.
            chosen = max(usable, key=lambda r: r[2])
            ok = True
        else:
            # Nothing hit the brightness floor; keep the brightest measured
            # and say so rather than silently choosing a black image.
            chosen = max(results, key=lambda r: r[2]) if results else (-5, 0.0, 0.0)
            ok = False
    finally:
        # Never leave the camera in manual: the caller applies the chosen
        # value itself on the next Start, and a failed tune must not leave a
        # dark camera behind.
        try:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        except Exception:
            pass
        cap.release()

    table = ", ".join("%d:%.0ffps/%.0flum" % r for r in results)
    if ok:
        msg = ("Exposure %d -> %.0f fps at brightness %.0f/255 (was %.0f fps on "
               "auto). Measured %s" % (chosen[0], chosen[1], chosen[2],
                                       results[0][1], table))
    else:
        msg = ("No exposure reached brightness %.0f; the scene is too dark to "
               "run fast. Add light, then retry. Measured %s"
               % (min_brightness, table))
    print("[FPV Pose Tracker] autotune:", msg)
    return ok, chosen[0], msg


def reset_camera(camera_index=0):
    """Put a camera back on automatic exposure and report the result.

    Needed because webcam drivers keep a manual exposure after the process
    that set it exits: a too-short exposure survives closing Blender, and
    every later session gets a black frame with no hint as to why. Tracking
    does not have to be running to use this.

    Returns (ok, message).
    """
    cv2, _ = _lazy_cv()
    # DirectShow only: opening MSMF on a camera in this state can block for
    # minutes, which would hang Blender's UI thread. Retry, because the
    # cameras that most need resetting are the flaky ones that refuse to open
    # on any given try.
    cap, attempts = open_camera(cv2, camera_index)
    if cap is None:
        return False, "Could not open camera %d after %d attempts (%s)" % (
            camera_index, len(attempts), "; ".join(attempts[-2:]))
    try:
        cap.read()
        before = _mean_frame(cap)
        for value in (0.75, 3.0):
            try:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, value)
            except Exception:
                pass
        # Some drivers only leave manual mode once a longer exposure is
        # written explicitly.
        for value in (-4.0, -2.0, 0.0):
            try:
                cap.set(cv2.CAP_PROP_EXPOSURE, value)
            except Exception:
                break
            time.sleep(0.2)
            if (_mean_frame(cap) or 0.0) >= 10.0:
                break
        try:
            cap.set(cv2.CAP_PROP_GAIN, 0.0)
        except Exception:
            pass
        time.sleep(0.3)
        after = _mean_frame(cap)
    finally:
        cap.release()
    b = -1.0 if before is None else before
    a = -1.0 if after is None else after
    if a >= 10.0:
        return True, ("Camera reset to auto exposure — image brightness "
                      "%.0f/255 (was %.0f)." % (a, b))
    return False, ("Camera still returns a near-black image (%.0f/255) after "
                   "resetting to auto exposure. Check the privacy shutter, "
                   "that no other app is using it, and the room lighting." % a)


def _mean_frame(cap, samples=4):
    vals = []
    for _ in range(samples):
        try:
            ok, f = cap.read()
        except Exception:
            break
        if ok and f is not None:
            try:
                vals.append(float(f.mean()))
            except Exception:
                pass
    return (sum(vals) / len(vals)) if vals else None


def enumerate_cameras(max_index=6, deep=False):
    """Device indices that open and deliver a frame.

    On Windows this probes DirectShow only. The Media Foundation backend has
    been measured taking over 90 seconds to open a device it then cannot read
    a frame from, which would turn a six-index scan into a multi-minute freeze
    of Blender's UI. `deep=True` opts into trying it for a camera DirectShow
    will not open.
    """
    try:
        import cv2
    except Exception:
        return []
    import sys
    if sys.platform == "win32":
        backends = [(getattr(cv2, "CAP_DSHOW", 0), "DSHOW")]
        if deep:
            backends.append((getattr(cv2, "CAP_MSMF", 0), "MSMF"))
    else:
        backends = [(0, "default")]
    found = []
    for i in range(max_index):
        for backend, _label in backends:
            cap = cv2.VideoCapture(i, backend) if backend else cv2.VideoCapture(i)
            ok = False
            if cap.isOpened():
                ok, _ = cap.read()
            # Always release, opened or not: a half-built graph keeps the
            # device claimed and the very next open of it fails.
            cap.release()
            if ok:
                found.append(i)
                break
    return found


def register():
    pass


def unregister():
    pass
