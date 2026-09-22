"""Convert PoseFrame landmarks into pose-bone rotations on the target rig.

Per-frame pipeline
------------------
1. One-Euro filter the pose landmarks we read (shoulders, elbows, wrists,
   hips) and the four hand landmarks that define the palm plane.

2. Calibration. The user holds the rig's rest pose (for an FPS-arms rig:
   arms forward toward the camera, palms down) for a short window. From
   that window we derive

     * ``M_pose`` - the linear map from MediaPipe pose_world axes into the
       rig's armature space, built from the user's torso frame (shoulder
       line, shoulders->hips, and forward). Because it is measured rather
       than assumed it absorbs whatever axis convention the installed
       MediaPipe build uses AND the mirror reflection of a selfie feed.
     * ``B_cal`` - each hand's palm basis at rest. The hand bone is then
       driven by the *change* in palm orientation since calibration, applied
       to the bone's rest orientation, so the rig's own hand-bone axis
       convention (which local axis is the palm normal, its sign, ...) never
       has to be known.

3. Arms: analytic 2-bone IK anchored at the rig's rest shoulder joint. In
   selfie mode the wrist target is placed relative to the torso centre and
   scaled by rig/user shoulder width and reach, so hands that touch in front
   of the user touch on the rig even when the rig's proportions differ. The
   shoulder->wrist distance is blended toward what the tracked *elbow angle*
   implies, because MediaPipe's depth is compressed when the arm points at
   the camera and the raw wrist depth reads a straight arm as bent.

4. Hand: full 3D orientation via the calibration delta. Fingers: absolute
   per-phalanx directions in the same frame.

All bone maths is done in armature space. Child bones use the parent's
matrix computed *this frame* (``A @ matrix_basis``, which keeps any pose
scale/location the bone carries) so a chain resolves in one pass without a
depsgraph update.
"""

import math
import time
from collections import deque

import bpy
from mathutils import Matrix, Vector, Quaternion

from . import tracker as T
from .one_euro import OneEuroVec3


_ARM_LANDMARKS = (
    T.POSE_LSHOULDER, T.POSE_RSHOULDER,
    T.POSE_LELBOW, T.POSE_RELBOW,
    T.POSE_LWRIST, T.POSE_RWRIST,
    T.POSE_LHIP, T.POSE_RHIP,
    T.POSE_NOSE,
)

_HAND_BASIS_LANDMARKS = (T.HAND_WRIST, T.HAND_INDEX[0], T.HAND_MIDDLE[0], T.HAND_PINKY[0])

# Anthropometric fallbacks (metres) used until running estimates exist.
_DEFAULT_USER_REACH = 0.58      # shoulder joint -> wrist, adult
_DEFAULT_USER_SHOULDER = 0.38   # shoulder joint -> shoulder joint

# Elbow angle (interior, 180 deg = straight) range over which we stop
# trusting the wrist's tracked depth and trust the tracked elbow angle.
_ELBOW_BLEND_START = math.radians(110.0)
_ELBOW_BLEND_FULL = math.radians(160.0)

# How far (metres, rig scale) the tracked elbow must sit off the
# shoulder->wrist axis before we trust it as the bend direction.
_POLE_TRUST_DIST = 0.06
# The same distance for an elbow two or more cameras triangulated. A
# monocular elbow is a guess along the camera's axis, so it has to be well
# clear of the arm axis before it is worth believing; a triangulated one was
# measured, and holding it to the monocular threshold would throw away the
# best information in the frame.
_POLE_TRUST_DIST_MULTI = 0.02

# Below _UPPER_TRACK_FULL the tracked shoulder->elbow direction drives the
# upper arm outright (the shoulder->wrist axis is ill-conditioned when the
# fist is up near the shoulder); above _UPPER_TRACK_NONE pure IK is used.
_UPPER_TRACK_FULL = math.radians(60.0)
_UPPER_TRACK_NONE = math.radians(100.0)

_Y = Vector((0.0, 1.0, 0.0))
_Z = Vector((0.0, 0.0, 1.0))

# Smoothing weights are authored as "fraction retained per frame at this
# rate" and rescaled to the frame's real dt. Without that the filter
# silently changes strength with the capture rate: a 0.9 weight is a 610 ms
# time constant at 15 fps but 150 ms at 60 fps, so every throughput
# improvement would quietly make the rig twitchier.
# Keep the solved arm a little inside full extension. At d == L1+L2 the
# elbow's rotation *about* the shoulder->wrist axis stops being constrained
# (its offset from that axis goes to zero), so tiny pole-hint noise swings it
# hard. Protracting the clavicle to stay inside this margin is most of why
# the shoulder stops looking janky when the hands move away.
_CLAVICLE_REACH_MARGIN = 0.95

_SMOOTH_REF_DT = 1.0 / 30.0
# Clamp so a stall or a burst cannot turn into a wild filter weight.
_DT_MIN = 1.0 / 240.0
_DT_MAX = 0.25


def _hand_chain(side):
    src = "left" if side == "L" else "right"
    return {
        f"thumb_01_{side}": (src, T.HAND_WRIST, T.HAND_THUMB[1]),
        f"thumb_02_{side}": (src, T.HAND_THUMB[1], T.HAND_THUMB[2]),
        f"thumb_03_{side}": (src, T.HAND_THUMB[2], T.HAND_THUMB[3]),
        f"index_01_{side}": (src, T.HAND_INDEX[0], T.HAND_INDEX[1]),
        f"index_02_{side}": (src, T.HAND_INDEX[1], T.HAND_INDEX[2]),
        f"index_03_{side}": (src, T.HAND_INDEX[2], T.HAND_INDEX[3]),
        f"middle_01_{side}": (src, T.HAND_MIDDLE[0], T.HAND_MIDDLE[1]),
        f"middle_02_{side}": (src, T.HAND_MIDDLE[1], T.HAND_MIDDLE[2]),
        f"middle_03_{side}": (src, T.HAND_MIDDLE[2], T.HAND_MIDDLE[3]),
        f"ring_01_{side}": (src, T.HAND_RING[0], T.HAND_RING[1]),
        f"ring_02_{side}": (src, T.HAND_RING[1], T.HAND_RING[2]),
        f"ring_03_{side}": (src, T.HAND_RING[2], T.HAND_RING[3]),
        f"pinky_01_{side}": (src, T.HAND_PINKY[0], T.HAND_PINKY[1]),
        f"pinky_02_{side}": (src, T.HAND_PINKY[1], T.HAND_PINKY[2]),
        f"pinky_03_{side}": (src, T.HAND_PINKY[2], T.HAND_PINKY[3]),
    }


_FINGER_CHAIN = {"L": _hand_chain("L"), "R": _hand_chain("R")}


def _side_indices(side):
    """(shoulder, elbow, wrist, hip, opposite shoulder) pose indices."""
    if side == "L":
        return (T.POSE_LSHOULDER, T.POSE_LELBOW, T.POSE_LWRIST,
                T.POSE_LHIP, T.POSE_RSHOULDER)
    return (T.POSE_RSHOULDER, T.POSE_RELBOW, T.POSE_RWRIST,
            T.POSE_RHIP, T.POSE_LSHOULDER)


# --------------------------------------------------------------------------
# small maths helpers
# --------------------------------------------------------------------------

def _percentile(values, frac):
    s = sorted(values)
    if not s:
        return None
    i = int(round((len(s) - 1) * frac))
    return s[max(0, min(len(s) - 1, i))]


def _smoothstep(e0, e1, x):
    t = max(0.0, min(1.0, (x - e0) / (e1 - e0)))
    return t * t * (3.0 - 2.0 * t)


def _sign(v):
    return -1.0 if v < 0.0 else 1.0


def _cols(x, y, z):
    """3x3 matrix with the given column vectors."""
    return Matrix((x, y, z)).transposed()


def _gram_schmidt(a, b, c):
    """Orthonormalise (a, b, c) in that priority order, returned as the
    columns of a 3x3. Preserves the handedness of the input triple (a
    mirrored webcam feed gives a left-handed body frame, and we want to
    keep that reflection so it can be undone)."""
    x = a.normalized()
    y = b - x * b.dot(x)
    if y.length < 1e-6:
        y = x.orthogonal()
    y.normalize()
    z = c - x * c.dot(x) - y * c.dot(y)
    if z.length < 1e-6:
        z = x.cross(y)
    z.normalize()
    return _cols(x, y, z)


def _orthonormalize_basis(m):
    """Re-orthonormalise a palm basis, keeping Y (fingers) exact, then Z
    (palm normal), then X = Y x Z (right-handed)."""
    y = Vector(m.col[1]).normalized()
    z = Vector(m.col[2])
    z = z - y * z.dot(y)
    if z.length < 1e-6:
        z = y.orthogonal()
    z.normalize()
    x = y.cross(z).normalized()
    return _cols(x, y, z)


def _mean_vec(vecs):
    acc = Vector((0.0, 0.0, 0.0))
    for v in vecs:
        acc += v
    return acc / max(1, len(vecs))


def _quat_from_to(v_from, v_to):
    a = v_from.normalized()
    b = v_to.normalized()
    d = max(-1.0, min(1.0, a.dot(b)))
    if d > 0.999999:
        return Quaternion((1.0, 0.0, 0.0, 0.0))
    if d < -0.999999:
        axis = a.cross(Vector((1.0, 0.0, 0.0)))
        if axis.length < 1e-4:
            axis = a.cross(Vector((0.0, 1.0, 0.0)))
        axis.normalize()
        return Quaternion(axis, math.pi)
    axis = a.cross(b)
    s = math.sqrt((1 + d) * 2)
    inv_s = 1.0 / s
    q = Quaternion((s * 0.5, axis.x * inv_s, axis.y * inv_s, axis.z * inv_s))
    q.normalize()
    return q


def _two_bone_elbow(shoulder, target, L1, L2, pole_hint):
    """Analytic 2-bone IK for the elbow position.

    Given fixed shoulder and target (wrist) positions in armature space and
    limb lengths L1/L2, place the elbow so |shoulder->elbow|=L1 and
    |elbow->target|=L2. `pole_hint` picks which side of the shoulder-target
    axis the elbow bends to.
    """
    to_target = target - shoulder
    d = to_target.length
    if d < 1e-5:
        return None
    d = min(d, L1 + L2 - 1e-4)
    d = max(d, abs(L1 - L2) + 1e-4)
    u = to_target / d
    a = (L1 * L1 - L2 * L2 + d * d) / (2.0 * d)
    h = math.sqrt(max(L1 * L1 - a * a, 0.0))

    v = pole_hint - u * pole_hint.dot(u)
    if v.length < 1e-5:
        alt = Vector((0.0, 0.0, 1.0))
        if abs(u.dot(alt)) > 0.99:
            alt = Vector((1.0, 0.0, 0.0))
        v = alt - u * alt.dot(u)
    v.normalize()
    return shoulder + u * a + v * h


# --------------------------------------------------------------------------
# retargeter
# --------------------------------------------------------------------------

class Retargeter:
    """Drives one armature from PoseFrames.

    For each bone, the "pre matrix" A is bone-local -> armature-local BEFORE
    this bone's own transform is applied. Setting `rotation_quaternion = q`
    makes the bone's Y axis in armature space `A.to_3x3() @ q @ (0,1,0)`
    (times the sign of the bone's own Y scale). We solve for `q` so that
    this matches the target direction.

    We cache each bone's computed armature-space pose matrix
    (`A @ matrix_basis`) so children can use the parent's just-applied
    transform without waiting for a depsgraph update.
    """

    def __init__(self, armature_obj, bone_map,
                 smoothing=0.5, pose_smoothing=0.9,
                 pose_visibility=0.5,
                 view_mode="BODY_CAM",
                 one_euro_min_cutoff=1.0, one_euro_beta=0.6,
                 use_arm_ik=True, freeze_shoulder=True, shoulder_mode=None,
                 shoulder_protract=30.0,
                 track_palm=True,
                 anchor_mode="AUTO", elbow_from_angle=True,
                 auto_calibrate=True, calibration_delay=3.0,
                 calibration_frames=15, hand_depth_mode="AUTO",
                 pose_source="LIFT", verbose_hand_log=False):
        self.armature = armature_obj
        self.bone_map = dict(bone_map)
        self.smoothing = max(0.0, min(0.99, smoothing))
        self.pose_smoothing = max(0.0, min(0.99, pose_smoothing))
        self.pose_visibility = max(0.0, min(1.0, pose_visibility))
        self.view_mode = (view_mode or "BODY_CAM").upper()
        self.use_arm_ik = bool(use_arm_ik)
        # FREEZE: clavicle pinned to rest (steadiest, but the rig can only
        # reach where its own proportions allow).
        # REACH:  clavicle rotates to carry the shoulder joint toward an
        #         out-of-reach target, driven by geometry rather than by
        #         MediaPipe's shoulder landmarks, whose Z is far too noisy.
        # TRACK:  clavicle follows the tracked shoulder line.
        self.shoulder_mode = (shoulder_mode
                              or ("FREEZE" if freeze_shoulder else "TRACK")).upper()
        self.freeze_shoulder = self.shoulder_mode == "FREEZE"
        self._protract_max = math.radians(max(0.0, float(shoulder_protract)))
        self.track_palm = bool(track_palm)
        self.anchor_mode = (anchor_mode or "AUTO").upper()
        self.elbow_from_angle = bool(elbow_from_angle)
        self.calibration_delay = max(0.0, float(calibration_delay))
        self.calibration_frames = max(1, int(calibration_frames))
        self.hand_depth_mode = (hand_depth_mode or "AUTO").upper()
        # "LIFT": arm joints from the tracker's 2D->3D lifted landmarks, which
        # are already in the body frame (right, forward, up).  "WORLD":
        # MediaPipe pose_world, whose axis convention is measured at
        # calibration (selfie) or assumed (body cam).
        self.pose_source = (pose_source or "LIFT").upper()
        self.verbose_hand_log = bool(verbose_hand_log)

        self._log_counter = 0
        self._last_t = None
        self._dt = _SMOOTH_REF_DT
        self._prev_quats = {}
        self._computed = {}
        self._pole_prev = {}
        self._last_detection = {"pose": False, "left": False, "right": False}
        self._last_bones_applied = 0

        self._lm_filters = {
            i: OneEuroVec3(min_cutoff=one_euro_min_cutoff, beta=one_euro_beta)
            for i in _ARM_LANDMARKS
        }
        self._lift_filters = {
            i: OneEuroVec3(min_cutoff=one_euro_min_cutoff, beta=one_euro_beta)
            for i in _ARM_LANDMARKS
        }
        self._hand_filters = {
            (side, idx): OneEuroVec3(min_cutoff=one_euro_min_cutoff, beta=one_euro_beta)
            for side in ("L", "R") for idx in _HAND_BASIS_LANDMARKS
        }

        # Running body-size estimates (metres, from pose_world).
        self._reach_samples = deque(maxlen=150)
        self._shoulder_samples = deque(maxlen=150)
        # Eye position relative to the shoulder line, as a fraction of the
        # user's own shoulder width, so it transfers to a rig of any size.
        self._eye_samples = deque(maxlen=150)

        # Rig geometry (armature space).
        self._F_rig = Matrix.Identity(3)   # columns: right, forward, up
        self._rig_center = None
        self._rig_shoulder_half = None
        self._rig_hands_point_forward = False
        self._compute_rig_geometry()
        self._M_hand = self._F_rig.copy()

        # Calibration state.
        self._M_pose = None if self._torso_calibration_needed() else self._F_rig.copy()
        self._hand_cal = {"L": None, "R": None}
        self._hand_depth_sign = 1.0
        if self.hand_depth_mode == "FLIPPED":
            self._hand_depth_sign = -1.0
        self._calib_request_time = None
        self._calib_torso = []
        self._calib_hand = {"L": [], "R": []}
        self._calib_status_text = "not calibrated"
        if auto_calibrate:
            self.request_calibration(reset_pose=False)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def last_detection(self):
        return dict(self._last_detection)

    def last_bones_applied(self):
        return self._last_bones_applied

    def frame_dt(self):
        """Seconds between the last two tracked frames."""
        return self._dt

    def smoothing_tau(self, pose=True):
        """Time constant in seconds of the current smoothing weight — the
        number that actually says whether a punch survives."""
        s = self.pose_smoothing if pose else self.smoothing
        if s <= 0.0:
            return 0.0
        if s >= 0.999:
            return float("inf")
        return -_SMOOTH_REF_DT / math.log(s)

    def calib_status(self):
        return self._calib_status_text

    def is_calibrating(self):
        return self._calib_request_time is not None

    def reset_pose(self):
        """Put every mapped bone back to its rest transform. Clears any
        stray pose scale (a -1 scale on an upper arm reflects the whole
        chain) and location as well as rotation."""
        for name in self.bone_map.values():
            if not name:
                continue
            pb = self.armature.pose.bones.get(name)
            if pb is None:
                continue
            pb.location = (0.0, 0.0, 0.0)
            pb.scale = (1.0, 1.0, 1.0)
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        self._prev_quats = {}
        self._pole_prev = {}
        self._computed = {}

    def capture_point_names(self):
        """Labels of the armature-space points the recorder tracks.

        Only the arm chain: those are the joints a tracking glitch throws
        somewhere impossible, and keeping the list short keeps a take's
        outlier score from being dominated by finger jitter.
        """
        names = []
        for side in ("L", "R"):
            for label, slot in (("elbow", f"forearm_{side}"),
                                ("wrist", f"hand_{side}"),
                                ("palm", f"hand_{side}")):
                if self._resolve_bone(slot) is not None:
                    names.append(f"{label}_{side}")
        return names

    def _bone_matrix_now(self, pose_bone):
        """This frame's armature-space matrix for a bone.

        `self._computed` holds what we just solved, which `pose_bone.matrix`
        will not report until the depsgraph catches up.
        """
        m = self._computed.get(pose_bone.name)
        return m if m is not None else pose_bone.matrix

    def capture_state(self, bone_names, point_names):
        """Snapshot for the recorder: (flat quaternions, flat points).

        Quaternions are 4 floats per entry in `bone_names`; points are 3
        floats per entry in `point_names`, both in the caller's order.
        """
        quats = []
        pose_bones = self.armature.pose.bones
        for name in bone_names:
            pb = pose_bones.get(name)
            if pb is None:
                quats.extend((1.0, 0.0, 0.0, 0.0))
                continue
            if pb.rotation_mode == "QUATERNION":
                q = pb.rotation_quaternion
            else:
                q = pb.matrix_basis.to_quaternion()
            quats.extend((q.w, q.x, q.y, q.z))

        points = []
        for label in point_names:
            kind, side = label.rsplit("_", 1)
            slot = f"forearm_{side}" if kind == "elbow" else f"hand_{side}"
            pb = self._resolve_bone(slot)
            if pb is None:
                points.extend((0.0, 0.0, 0.0))
                continue
            m = self._bone_matrix_now(pb)
            p = m.translation
            if kind == "palm":
                # A point out past the wrist, so palm roll and wrist aim
                # register as motion the same way a position does.
                p = p + m.to_3x3().col[1].normalized() * pb.bone.length
            points.extend((p.x, p.y, p.z))
        return quats, points

    def request_calibration(self, reset_pose=True):
        """Start a calibration window. The user should match the rig's rest
        pose; sampling starts after `calibration_delay` seconds and lasts
        `calibration_frames` frames per component (torso, each hand)."""
        if reset_pose:
            self.reset_pose()
        self._calib_request_time = time.time()
        self._calib_torso = []
        self._calib_hand = {"L": [], "R": []}
        self._hand_cal = {"L": None, "R": None}
        if self._torso_calibration_needed():
            self._M_pose = None
        if self.hand_depth_mode == "AUTO":
            self._hand_depth_sign = 1.0
        self._calib_status_text = "calibrating: get into the rig's rest pose"

    def _torso_calibration_needed(self):
        return self.view_mode == "SELFIE" and self.pose_source == "WORLD"

    def _arm_source(self, pf):
        """(landmark dict, matrix-to-armature) for the arm joints this
        frame, or (None, None) when nothing trustworthy is available."""
        if self.pose_source == "LIFT":
            pl = pf.pose_lifted
            if pl and T.POSE_LSHOULDER in pl and T.POSE_RSHOULDER in pl:
                return pl, self._F_rig
            return None, None
        if self._M_pose is None or not pf.pose_present:
            return None, None
        return pf.pose_world, self._M_pose

    # ------------------------------------------------------------------
    # rig geometry
    # ------------------------------------------------------------------

    def _resolve_bone(self, slot):
        name = self.bone_map.get(slot)
        if not name:
            return None
        return self.armature.pose.bones.get(name)

    def _compute_rig_geometry(self):
        ua_L, ua_R = self._resolve_bone("upper_arm_L"), self._resolve_bone("upper_arm_R")
        h_L, h_R = self._resolve_bone("hand_L"), self._resolve_bone("hand_R")

        right = None
        if ua_L is not None and ua_R is not None:
            right = ua_R.bone.head_local - ua_L.bone.head_local
            self._rig_center = (ua_L.bone.head_local + ua_R.bone.head_local) * 0.5
            self._rig_shoulder_half = 0.5 * right.length
        elif h_L is not None and h_R is not None:
            right = h_R.bone.head_local - h_L.bone.head_local
        up = Vector((0.0, 0.0, 1.0))
        if right is None or right.length < 1e-4:
            right = Vector((1.0, 0.0, 0.0))
        right = right - up * right.dot(up)
        if right.length < 1e-4:
            right = Vector((1.0, 0.0, 0.0))
        right.normalize()
        forward = up.cross(right)   # z x x = y : right-handed (right, forward, up)
        self._F_rig = _cols(right, forward, up)

        dots = []
        for pb in (h_L, h_R):
            if pb is None:
                continue
            y = Vector(pb.bone.matrix_local.to_3x3().col[1]).normalized()
            dots.append(y.dot(forward))
        self._rig_hands_point_forward = bool(dots) and (sum(dots) / len(dots)) > 0.5

    def _arm_lengths(self, side):
        ua = self._resolve_bone(f"upper_arm_{side}")
        fa = self._resolve_bone(f"forearm_{side}")
        if ua is None or fa is None:
            return None
        return ua.bone.length, fa.bone.length

    # ------------------------------------------------------------------
    # body-size estimates
    # ------------------------------------------------------------------

    def _vis(self, pf, idx):
        return pf.pose_visibility.get(idx, 0.0)

    def _update_body_estimates(self, pf):
        if not pf.pose_present:
            return
        thr = self.pose_visibility
        pw = pf.pose_world
        sL, sR = pw.get(T.POSE_LSHOULDER), pw.get(T.POSE_RSHOULDER)
        if (sL is not None and sR is not None
                and self._vis(pf, T.POSE_LSHOULDER) >= thr
                and self._vis(pf, T.POSE_RSHOULDER) >= thr):
            self._shoulder_samples.append((Vector(sR) - Vector(sL)).length)
        for side in ("L", "R"):
            sh_i, el_i, wr_i, _, _ = _side_indices(side)
            p = [pw.get(i) for i in (sh_i, el_i, wr_i)]
            if any(v is None for v in p):
                continue
            if any(self._vis(pf, i) < thr for i in (sh_i, el_i, wr_i)):
                continue
            sh, el, wr = (Vector(v) for v in p)
            self._reach_samples.append((el - sh).length + (wr - el).length)

    def _sample_eye_offset(self, pf):
        """Measure where the user's eyes sit relative to their shoulder line.

        Answers "where does a first-person camera belong on this rig" from
        the user's own build rather than a guessed constant. Eye landmarks
        are far steadier than the shoulder Z estimates, and normalising by
        shoulder width removes the user's absolute size.
        """
        pwld = pf.pose_world
        if not pwld:
            return
        need = (T.POSE_LSHOULDER, T.POSE_RSHOULDER, T.POSE_LEYE, T.POSE_REYE)
        if any(pwld.get(i) is None for i in need):
            return
        if min(self._vis(pf, i) for i in need) < max(0.5, self.pose_visibility):
            return
        sh_l = Vector(pwld[T.POSE_LSHOULDER])
        sh_r = Vector(pwld[T.POSE_RSHOULDER])
        width = (sh_r - sh_l).length
        if width < 0.15:                    # implausible; bad frame
            return
        centre = (sh_l + sh_r) * 0.5
        eye = (Vector(pwld[T.POSE_LEYE]) + Vector(pwld[T.POSE_REYE])) * 0.5
        M = self._M_pose if self._M_pose is not None else self._F_rig
        off = M @ (eye - centre)
        local = self._F_rig.transposed() @ off      # (right, forward, up)
        self._eye_samples.append((local.z / width, local.y / width))

    def _user_reach(self):
        # Foreshortening only ever shortens MediaPipe's limb estimate, so a
        # high percentile is closer to the true length than the median.
        m = _percentile(self._reach_samples, 0.8)
        return m if (m is not None and m > 0.25) else _DEFAULT_USER_REACH

    def _user_shoulder_half(self):
        m = _percentile(self._shoulder_samples, 0.5)
        return 0.5 * m if (m is not None and m > 0.15) else 0.5 * _DEFAULT_USER_SHOULDER

    # ------------------------------------------------------------------
    # filtering
    # ------------------------------------------------------------------

    def _filter_pose_landmarks(self, pf):
        if not pf.pose_present:
            return
        t = pf.timestamp
        for i in _ARM_LANDMARKS:
            p = pf.pose_world.get(i)
            if p is not None:
                pf.pose_world[i] = self._lm_filters[i].filter(p, t)
            q = pf.pose_lifted.get(i) if pf.pose_lifted else None
            if q is not None:
                pf.pose_lifted[i] = self._lift_filters[i].filter(q, t)

    def _filter_hand_landmarks(self, pf):
        t = pf.timestamp
        for side, hand in (("L", pf.left_hand), ("R", pf.right_hand)):
            if not hand:
                continue
            for idx in _HAND_BASIS_LANDMARKS:
                p = hand.get(idx)
                if p is None:
                    continue
                hand[idx] = self._hand_filters[(side, idx)].filter(p, t)

    def _apply_hand_depth_sign(self, pf):
        if self._hand_depth_sign > 0.0:
            return
        for hand in (pf.left_hand, pf.right_hand):
            if not hand:
                continue
            for i, p in list(hand.items()):
                hand[i] = (p[0], -p[1], p[2])

    # ------------------------------------------------------------------
    # hand basis
    # ------------------------------------------------------------------

    def _hand_basis(self, hand):
        """Palm basis in tracker space: columns X (across palm, index->pinky
        side), Y (wrist -> middle MCP), Z (palm normal). Right-handed by
        construction. Returns None if landmarks are missing/degenerate."""
        if not hand:
            return None
        w = hand.get(T.HAND_WRIST)
        m = hand.get(T.HAND_MIDDLE[0])
        idx = hand.get(T.HAND_INDEX[0])
        pnk = hand.get(T.HAND_PINKY[0])
        if w is None or m is None or idx is None or pnk is None:
            return None
        w = Vector(w); m = Vector(m); idx = Vector(idx); pnk = Vector(pnk)
        y = m - w
        if y.length < 1e-5:
            return None
        y.normalize()
        across = pnk - idx
        if across.length < 1e-5:
            return None
        z = y.cross(across)
        if z.length < 1e-5:
            return None
        z.normalize()
        x = y.cross(z).normalized()
        z = x.cross(y).normalized()
        return _cols(x, y, z)

    # ------------------------------------------------------------------
    # calibration
    # ------------------------------------------------------------------

    def _sample_torso(self, pf):
        if not pf.pose_present:
            return False
        thr = self.pose_visibility
        pw = pf.pose_world
        need = (T.POSE_LSHOULDER, T.POSE_RSHOULDER, T.POSE_LHIP, T.POSE_RHIP)
        if any(pw.get(i) is None for i in need):
            return False
        if self._vis(pf, T.POSE_LSHOULDER) < thr or self._vis(pf, T.POSE_RSHOULDER) < thr:
            return False
        sL, sR = Vector(pw[T.POSE_LSHOULDER]), Vector(pw[T.POSE_RSHOULDER])
        hL, hR = Vector(pw[T.POSE_LHIP]), Vector(pw[T.POSE_RHIP])
        right = sR - sL
        sh_mid = (sL + sR) * 0.5
        down = (hL + hR) * 0.5 - sh_mid
        if right.length < 1e-4 or down.length < 1e-4:
            return False

        fwd_arms = None
        arm_vecs = []
        for side in ("L", "R"):
            sh_i, _, wr_i, _, _ = _side_indices(side)
            if pw.get(wr_i) is None or self._vis(pf, wr_i) < thr:
                continue
            arm_vecs.append(Vector(pw[wr_i]) - Vector(pw[sh_i]))
        if len(arm_vecs) == 2:
            fwd_arms = _mean_vec(arm_vecs)

        fwd_nose = None
        nose = pw.get(T.POSE_NOSE)
        if nose is not None and self._vis(pf, T.POSE_NOSE) >= thr:
            fwd_nose = Vector(nose) - sh_mid

        self._calib_torso.append((right, down, fwd_arms, fwd_nose))
        return True

    def _finalize_torso(self):
        rights = _mean_vec([s[0] for s in self._calib_torso])
        downs = _mean_vec([s[1] for s in self._calib_torso])
        arms = [s[2] for s in self._calib_torso if s[2] is not None]
        noses = [s[3] for s in self._calib_torso if s[3] is not None]

        up = -downs
        forward = None
        source = "cross-product"
        if self._rig_hands_point_forward and len(arms) >= max(3, len(self._calib_torso) // 2):
            forward = _mean_vec(arms)
            source = "arms"
        elif noses:
            forward = _mean_vec(noses)
            source = "nose"
        if forward is not None:
            # strip the vertical + lateral parts; keep only the depth-ish component
            r = rights.normalized()
            u = up.normalized()
            forward = forward - r * forward.dot(r) - u * forward.dot(u)
            if forward.length < 0.02:
                forward = None
        if forward is None:
            forward = up.cross(rights)
            source = "cross-product (assumes right-handed pose frame)"

        F_user = _gram_schmidt(rights, forward, up)
        self._M_pose = self._F_rig @ F_user.inverted()
        det = self._M_pose.determinant()
        print("[FPV Pose Tracker] torso calibrated from %d frames (forward from %s). "
              "pose->rig det=%.2f  right=%s forward=%s up=%s"
              % (len(self._calib_torso), source, det,
                 tuple(round(v, 2) for v in F_user.col[0]),
                 tuple(round(v, 2) for v in F_user.col[1]),
                 tuple(round(v, 2) for v in F_user.col[2])))

    def _accumulate_hand(self, side, basis):
        self._calib_hand[side].append(basis)
        if len(self._calib_hand[side]) >= self.calibration_frames:
            self._finalize_hand(side)

    def _finalize_hand(self, side):
        samples = self._calib_hand[side]
        mean = _cols(_mean_vec([Vector(b.col[0]) for b in samples]),
                     _mean_vec([Vector(b.col[1]) for b in samples]),
                     _mean_vec([Vector(b.col[2]) for b in samples]))
        basis = _orthonormalize_basis(mean)

        # Auto-detect the depth sign of the hand landmarks: if the rig's
        # rest hands point forward, the user's fingers must too. A negative
        # forward component means this MediaPipe build's hand z has the
        # opposite sign convention, so mirror depth from here on.
        if (self.hand_depth_mode == "AUTO" and self._rig_hands_point_forward
                and self._hand_depth_sign > 0.0 and basis.col[1].y < -0.3):
            self._hand_depth_sign = -1.0
            print("[FPV Pose Tracker] hand depth convention looks flipped "
                  "(fingers y=%.2f at calibration); mirroring hand depth."
                  % basis.col[1].y)
            basis = self._reflect_hand_basis(basis)
            other = "R" if side == "L" else "L"
            self._calib_hand[other] = [self._reflect_hand_basis(b) for b in self._calib_hand[other]]
            if self._hand_cal[other] is not None:
                self._hand_cal[other] = self._reflect_hand_basis(self._hand_cal[other])

        self._hand_cal[side] = basis
        self._calib_hand[side] = []
        print("[FPV Pose Tracker] hand %s calibrated from %d frames: fingers=%s normal=%s"
              % (side, len(samples),
                 tuple(round(v, 2) for v in basis.col[1]),
                 tuple(round(v, 2) for v in basis.col[2])))

    @staticmethod
    def _reflect_hand_basis(basis):
        """Basis the same landmarks would give with their Y (depth) negated."""
        P = Matrix.Diagonal((1.0, -1.0, 1.0))
        x = P @ Vector(basis.col[0])
        y = P @ Vector(basis.col[1])
        z = -(P @ Vector(basis.col[2]))
        return _cols(x, y, z)

    def _run_calibration(self, pf):
        """Advance the calibration window. Returns True while bones must be
        left alone (user is getting into / holding the rest pose)."""
        if self._calib_request_time is None:
            return False
        now = pf.timestamp if pf.timestamp else time.time()
        elapsed = now - self._calib_request_time
        if elapsed < self.calibration_delay:
            self._calib_status_text = "calibrating: hold the rig's rest pose (%d)" % int(
                math.ceil(self.calibration_delay - elapsed))
            return True

        need = self.calibration_frames
        torso_needed = self._torso_calibration_needed()
        if torso_needed and self._M_pose is None:
            self._sample_torso(pf)
            if len(self._calib_torso) >= need:
                self._finalize_torso()
        for side, hand in (("L", pf.left_hand), ("R", pf.right_hand)):
            if self._hand_cal[side] is not None:
                continue
            basis = self._hand_basis(hand)
            if basis is not None:
                self._accumulate_hand(side, basis)

        torso_ok = (not torso_needed) or self._M_pose is not None
        hands_ok = all(self._hand_cal[s] is not None for s in ("L", "R"))
        if torso_ok and hands_ok:
            self._calib_request_time = None
            self._calib_status_text = "calibrated"
            return False

        parts = []
        if torso_needed:
            parts.append("torso %s" % ("ok" if torso_ok else "%d/%d" % (len(self._calib_torso), need)))
        for s in ("L", "R"):
            parts.append("hand %s %s" % (s, "ok" if self._hand_cal[s] is not None
                                         else "%d/%d" % (len(self._calib_hand[s]), need)))
        # Give up waiting after a while; drive what we have and keep
        # sampling missing hands lazily when they show up.
        if elapsed < self.calibration_delay + 8.0 or not torso_ok:
            self._calib_status_text = "calibrating: " + ", ".join(parts)
            return True
        self._calib_request_time = None
        self._calib_status_text = "partial (" + ", ".join(parts) + ") - press Calibrate to retry"
        return False

    # ------------------------------------------------------------------
    # bone driving primitives
    # ------------------------------------------------------------------

    def _eff_smoothing(self, s):
        """This frame's retained fraction for a weight authored at 30 fps."""
        if s <= 0.0:
            return 0.0
        if s >= 0.999 or self._dt == _SMOOTH_REF_DT:
            return s
        try:
            return s ** (self._dt / _SMOOTH_REF_DT)
        except (OverflowError, ValueError):
            return s

    def _pre_matrix(self, pose_bone):
        if pose_bone.parent is None:
            return pose_bone.bone.matrix_local.copy()
        parent_pose_arm = self._computed.get(
            pose_bone.parent.name, pose_bone.parent.matrix.copy()
        )
        rest_rel = (pose_bone.parent.bone.matrix_local.inverted()
                    @ pose_bone.bone.matrix_local)
        return parent_pose_arm @ rest_rel

    def _hold(self, pose_bone):
        """Leave the bone's own transform alone but record its up-to-date
        armature-space matrix for children."""
        if pose_bone is None:
            return
        self._computed[pose_bone.name] = self._pre_matrix(pose_bone) @ pose_bone.matrix_basis

    def _apply_bone(self, pose_bone, target_dir_arm, smoothing=None):
        """Point the bone's Y axis along `target_dir_arm` (armature space)."""
        if pose_bone is None or target_dir_arm is None:
            return False
        if target_dir_arm.length < 1e-5:
            return False

        A = self._pre_matrix(pose_bone)
        A_inv_3 = A.to_3x3().inverted()
        target_local = (A_inv_3 @ target_dir_arm).normalized()
        # A negative Y scale on this bone draws it backwards.
        target_local *= _sign(pose_bone.scale.y)

        prev = self._prev_quats.get(pose_bone.name)
        if prev is None:
            q = _quat_from_to(_Y, target_local)
        else:
            # Incremental: rotate from where the bone was pointing to the
            # new direction. Preserves whatever roll was already applied
            # and never lets tiny perpendicular noise reshuffle the axis.
            prev_dir = (prev @ _Y).normalized()
            dq = _quat_from_to(prev_dir, target_local)
            q = dq @ prev

        s = self._eff_smoothing(self.smoothing if smoothing is None else smoothing)
        if prev is not None and s > 0.0:
            q = prev.slerp(q, 1.0 - s)
        q.normalize()
        self._prev_quats[pose_bone.name] = q.copy()

        pose_bone.rotation_mode = "QUATERNION"
        pose_bone.rotation_quaternion = q
        self._computed[pose_bone.name] = A @ pose_bone.matrix_basis
        return True

    def _apply_bone_orient(self, pose_bone, target_y_arm, target_z_arm, smoothing=None):
        """Drive a bone to a full orientation: local +Y -> target_y, local
        +Z -> target_z (both armature space).

        Done as two shortest-arc steps (aim Y, then roll to match Z) rather
        than a matrix->quaternion conversion so a negative-determinant
        pre-matrix (reflected parent chain) is handled correctly.
        """
        if pose_bone is None or target_y_arm is None or target_z_arm is None:
            return False
        if target_y_arm.length < 1e-6 or target_z_arm.length < 1e-6:
            return False

        A = self._pre_matrix(pose_bone)
        A_inv_3 = A.to_3x3().inverted()
        target_y_local = (A_inv_3 @ target_y_arm).normalized() * _sign(pose_bone.scale.y)
        target_z_local = (A_inv_3 @ target_z_arm).normalized() * _sign(pose_bone.scale.z)

        q_y = _quat_from_to(_Y, target_y_local)
        z_after = (q_y @ _Z).normalized()
        axis = target_y_local
        z_after_proj = z_after - axis * z_after.dot(axis)
        target_z_proj = target_z_local - axis * target_z_local.dot(axis)
        if z_after_proj.length > 1e-4 and target_z_proj.length > 1e-4:
            z_after_proj.normalize()
            target_z_proj.normalize()
            q = _quat_from_to(z_after_proj, target_z_proj) @ q_y
        else:
            q = q_y
        q.normalize()

        s = self._eff_smoothing(self.smoothing if smoothing is None else smoothing)
        prev = self._prev_quats.get(pose_bone.name)
        if prev is not None and s > 0.0:
            if prev.dot(q) < 0.0:
                q = -q
            q = prev.slerp(q, 1.0 - s)
        q.normalize()
        self._prev_quats[pose_bone.name] = q.copy()

        pose_bone.rotation_mode = "QUATERNION"
        pose_bone.rotation_quaternion = q
        self._computed[pose_bone.name] = A @ pose_bone.matrix_basis
        return True

    # ------------------------------------------------------------------
    # arm
    # ------------------------------------------------------------------

    def _resolved_anchor_mode(self):
        if self.anchor_mode in ("TORSO", "SHOULDER"):
            return self.anchor_mode
        return "TORSO" if self.view_mode == "SELFIE" else "SHOULDER"

    def _observed(self, pf, idx):
        """How many cameras measured this landmark. 0 means nobody said, so
        assume the monocular case: every single-camera frame reports it."""
        obs = getattr(pf, "pose_observed", None)
        if not obs:
            return 1
        return int(obs.get(idx, 1))

    def _triangulated(self, pf, idx):
        return self._observed(pf, idx) >= 2

    def _arm_visible(self, pf, pw, side):
        sh_i, _, wr_i, _, _ = _side_indices(side)
        thr = self.pose_visibility
        if pw is None or pw.get(sh_i) is None or pw.get(wr_i) is None:
            return False
        return self._vis(pf, sh_i) >= thr and self._vis(pf, wr_i) >= thr

    def _pole_hint(self, side, M, p_sh, p_el, el_visible, axis_dir, s_reach,
                   trust_dist=_POLE_TRUST_DIST):
        down_rig = -Vector(self._F_rig.col[2])
        hint = down_rig
        if p_el is not None and el_visible:
            el_off = (M @ (p_el - p_sh)) * s_reach
            perp = el_off - axis_dir * el_off.dot(axis_dir)
            if perp.length > 1e-5:
                w = min(1.0, perp.length / max(1e-4, trust_dist))
                hint = perp.normalized() * w + down_rig * (1.0 - w)
        prev = self._pole_prev.get(side)
        if prev is not None and self.pose_smoothing > 0.0:
            hint = prev.lerp(hint, 1.0 - self.pose_smoothing)
        if hint.length < 1e-5:
            hint = down_rig
        self._pole_prev[side] = hint.copy()
        return hint

    def eye_offset_ratio(self):
        """(up, forward) offset of the user's eyes from their shoulder line,
        as a fraction of their own shoulder width, or None.

        Expressed as a ratio so it transfers to a rig of any proportions:
        multiply by the rig's shoulder width to get where a first-person
        camera belongs on that rig.
        """
        if len(self._eye_samples) < 20:
            return None
        ups = sorted(s[0] for s in self._eye_samples)
        fwds = sorted(s[1] for s in self._eye_samples)
        mid = len(ups) // 2
        return (ups[mid], fwds[mid])

    def _drive_clavicle(self, sh_pb, ua_pb, target_abs, reach):
        """Rotate the clavicle so the shoulder joint moves toward a wrist
        target the arm cannot otherwise reach.

        This is what a real shoulder does - the scapula slides forward and
        inward as you reach across yourself. Doing it here means the hand
        offset no longer has to be squashed sideways to make a wide rig's
        hands meet in front of it, so arm motion keeps its true shape.

        The rotation comes purely from reach geometry, never from the tracked
        shoulder landmarks, so it inherits none of their noise.
        """
        if sh_pb is None or self._protract_max <= 0.0:
            return False
        C = self._pre_matrix(sh_pb).translation
        arm_v = ua_pb.bone.head_local - C
        R = arm_v.length
        if R < 1e-4:
            return False
        u_rest = arm_v / R
        to_T = target_abs - C
        D = to_T.length
        if D < 1e-5:
            return False
        u_T = to_T / D

        # Angle between clavicle and target direction at which the shoulder
        # joint sits exactly `reach` from the target.
        cos_needed = (D * D + R * R - reach * reach) / (2.0 * D * R)
        if cos_needed >= 1.0:
            # Out of reach at every clavicle angle: swing as far toward the
            # target as the limit allows.
            theta_needed = 0.0
        elif cos_needed <= -1.0:
            return False        # reachable wherever the clavicle sits
        else:
            theta_needed = math.acos(cos_needed)

        theta_rest = math.acos(max(-1.0, min(1.0, u_rest.dot(u_T))))
        delta = theta_rest - theta_needed
        if delta <= 1e-4:
            return False        # the rest shoulder already reaches
        delta = min(delta, self._protract_max)

        axis = u_rest.cross(u_T)
        if axis.length < 1e-6:
            return False
        u_new = Quaternion(axis.normalized(), delta) @ u_rest
        return self._apply_bone(sh_pb, u_new, smoothing=self.pose_smoothing)

    def _solve_arm_ik(self, side, pf, pw, M, ua_pb, fa_pb):
        """2-bone IK for upper_arm + forearm. Returns number of bones driven."""
        sh_i, el_i, wr_i, _, opp_i = _side_indices(side)
        p_sh = Vector(pw[sh_i])
        p_wr = Vector(pw[wr_i])
        p_el = Vector(pw[el_i]) if pw.get(el_i) is not None else None
        el_visible = p_el is not None and self._vis(pf, el_i) >= self.pose_visibility * 0.5

        L1, L2 = ua_pb.bone.length, fa_pb.bone.length
        reach = L1 + L2
        wrist_measured = self._triangulated(pf, wr_i)
        elbow_measured = self._triangulated(pf, el_i)
        s_reach = reach / self._user_reach()
        F = self._F_rig
        sh_pb = self._resolve_bone("shoulder_" + side)

        # --- wrist target ---------------------------------------------
        # The torso-anchored target is absolute, and has to be resolved
        # before the clavicle moves, because the arm root depends on it.
        target_abs = None
        if (self._resolved_anchor_mode() == "TORSO"
                and self._rig_center is not None
                and pw.get(opp_i) is not None
                and self._vis(pf, opp_i) >= self.pose_visibility):
            center_user = (p_sh + Vector(pw[opp_i])) * 0.5
            off = M @ (p_wr - center_user)
            local = F.transposed() @ off            # (right, forward, up)
            if self.shoulder_mode == "REACH":
                # Uniform: the clavicle absorbs the rig/user width
                # difference, so sideways motion keeps its true scale.
                sx = s_reach
            else:
                # Squash sideways so a wide rig's hands still meet in front
                # of it. Cheap, but it distorts every sideways gesture.
                sx = self._rig_shoulder_half / self._user_shoulder_half()
            local = Vector((local.x * sx, local.y * s_reach, local.z * s_reach))
            target_abs = self._rig_center + F @ local

        if self.shoulder_mode == "REACH" and target_abs is not None:
            self._drive_clavicle(sh_pb, ua_pb, target_abs,
                                 reach * _CLAVICLE_REACH_MARGIN)

        # The arm root is wherever the clavicle just put it.
        sh_rig = self._pre_matrix(ua_pb).translation.copy()

        if target_abs is not None:
            to_t = target_abs - sh_rig
        else:
            to_t = (M @ (p_wr - p_sh)) * s_reach
        d_pos = to_t.length
        if d_pos < 1e-4:
            return 0
        axis_dir = to_t / d_pos

        # --- straightness from the tracked elbow angle -----------------
        d = d_pos
        theta = None
        if el_visible:
            a = p_sh - p_el
            b = p_wr - p_el
            if a.length > 1e-4 and b.length > 1e-4:
                cos_t = max(-1.0, min(1.0, a.normalized().dot(b.normalized())))
                theta = math.acos(cos_t)          # interior elbow angle, pi = straight
                if self.elbow_from_angle and not wrist_measured:
                    # This blend exists because a single camera reads a
                    # straight arm pointing at the lens as a bent one: the
                    # wrist's depth is compressed, so the shoulder->wrist
                    # distance comes out short and the elbow angle is the
                    # better witness. When two cameras triangulated the
                    # wrist there is no compression to undo, and applying
                    # the correction anyway would move a measured wrist to
                    # where an inference says it should be - which is most
                    # of why a second camera used to buy so little.
                    d_angle = math.sqrt(max(L1 * L1 + L2 * L2 - 2.0 * L1 * L2 * cos_t, 0.0))
                    w = _smoothstep(_ELBOW_BLEND_START, _ELBOW_BLEND_FULL, theta)
                    d = d_pos + (d_angle - d_pos) * w
        d = max(abs(L1 - L2) + 1e-3, min(reach * 0.999, d))
        target = sh_rig + axis_dir * d

        pole = self._pole_hint(
            side, M, p_sh, p_el, el_visible, axis_dir, s_reach,
            _POLE_TRUST_DIST_MULTI if elbow_measured else _POLE_TRUST_DIST)
        elbow = _two_bone_elbow(sh_rig, target, L1, L2, pole)
        if elbow is None:
            return 0

        # When the arm is strongly bent the tracked shoulder->elbow direction
        # (mostly image-plane, hence reliable) beats the IK's guess, whose
        # axis degenerates as the wrist approaches the shoulder.
        if theta is not None and el_visible:
            w_track = 1.0 - _smoothstep(_UPPER_TRACK_FULL, _UPPER_TRACK_NONE, theta)
            if elbow_measured:
                # A triangulated shoulder->elbow direction beats the IK's
                # guess over a wider range than a monocular one does.
                w_track = min(1.0, w_track * 1.5)
            if w_track > 1e-3:
                tracked = M @ (p_el - p_sh)
                if tracked.length > 1e-4:
                    ik_dir = (elbow - sh_rig).normalized()
                    blended = ik_dir.lerp(tracked.normalized(), w_track)
                    if blended.length > 1e-4:
                        elbow = sh_rig + blended.normalized() * L1

        drove = 0
        if self._apply_bone(ua_pb, elbow - sh_rig, smoothing=self.pose_smoothing):
            drove += 1
        if self._apply_bone(fa_pb, target - elbow, smoothing=self.pose_smoothing):
            drove += 1
        return drove

    # ------------------------------------------------------------------
    # hand
    # ------------------------------------------------------------------

    def _apply_hand_orient(self, pose_bone, basis_now, basis_cal):
        """Drive the hand bone by the change in palm orientation since
        calibration, applied to the bone's rest orientation."""
        R_delta = basis_now @ basis_cal.transposed()        # tracker frame
        M = self._M_hand
        rest = pose_bone.bone.matrix_local.to_3x3()
        target = M @ R_delta @ M.transposed() @ rest
        if self.verbose_hand_log:
            self._log_counter += 1
            if self._log_counter % 30 == 0:
                ang = math.degrees(R_delta.to_quaternion().angle)
                print("[FPV Pose Tracker] hand %s delta=%.0fdeg target Y=%s Z=%s"
                      % (pose_bone.name, ang,
                         tuple(round(v, 2) for v in target.col[1]),
                         tuple(round(v, 2) for v in target.col[2])))
        return self._apply_bone_orient(pose_bone, Vector(target.col[1]),
                                       Vector(target.col[2]),
                                       smoothing=self.pose_smoothing)

    # ------------------------------------------------------------------
    # per-side driver
    # ------------------------------------------------------------------

    def _drive_side(self, side, pf):
        applied = 0
        sh_pb = self._resolve_bone(f"shoulder_{side}")
        ua_pb = self._resolve_bone(f"upper_arm_{side}")
        fa_pb = self._resolve_bone(f"forearm_{side}")
        hand_pb = self._resolve_bone(f"hand_{side}")
        sh_i, el_i, wr_i, _, opp_i = _side_indices(side)
        pw, M = self._arm_source(pf)
        arm_ok = pw is not None and pf.pose_present and self._arm_visible(pf, pw, side)

        # shoulder
        if sh_pb is not None:
            drove = False
            if (self.shoulder_mode == "TRACK" and self.view_mode == "SELFIE" and arm_ok
                    and pw.get(opp_i) is not None
                    and self._vis(pf, opp_i) >= self.pose_visibility):
                d = M @ (Vector(pw[sh_i]) - Vector(pw[opp_i]))
                drove = self._apply_bone(sh_pb, d, smoothing=self.pose_smoothing)
                applied += 1 if drove else 0
            if not drove:
                self._hold(sh_pb)

        # upper arm + forearm
        drove_arm = False
        if arm_ok and ua_pb is not None and fa_pb is not None:
            if self.use_arm_ik:
                n = self._solve_arm_ik(side, pf, pw, M, ua_pb, fa_pb)
                applied += n
                drove_arm = n > 0
            else:
                if pw.get(el_i) is not None and self._vis(pf, el_i) >= self.pose_visibility:
                    d_u = M @ (Vector(pw[el_i]) - Vector(pw[sh_i]))
                    d_f = M @ (Vector(pw[wr_i]) - Vector(pw[el_i]))
                    if self._apply_bone(ua_pb, d_u, smoothing=self.pose_smoothing):
                        applied += 1
                        drove_arm = True
                    if self._apply_bone(fa_pb, d_f, smoothing=self.pose_smoothing):
                        applied += 1
        if not drove_arm:
            self._hold(ua_pb)
            self._hold(fa_pb)

        # hand
        hand = pf.left_hand if side == "L" else pf.right_hand
        basis = self._hand_basis(hand) if self.track_palm else None
        if hand_pb is not None:
            drove_hand = False
            if basis is not None:
                cal = self._hand_cal[side]
                if cal is None:
                    self._accumulate_hand(side, basis)   # lazy calibration
                elif self._apply_hand_orient(hand_pb, basis, cal):
                    drove_hand = True
                    applied += 1
            if not drove_hand:
                self._hold(hand_pb)

        # fingers
        if hand:
            M = self._M_hand
            for slot, (_, a, b) in _FINGER_CHAIN[side].items():
                pb = self._resolve_bone(slot)
                if pb is None:
                    continue
                pa, pb_ = hand.get(a), hand.get(b)
                if pa is None or pb_ is None:
                    continue
                d = M @ (Vector(pb_) - Vector(pa))
                if self._apply_bone(pb, d, smoothing=self.smoothing):
                    applied += 1
        return applied

    # ------------------------------------------------------------------
    # frame entry point
    # ------------------------------------------------------------------

    def apply_frame(self, pf):
        if pf is None or self.armature is None:
            return
        t = getattr(pf, "timestamp", None)
        if t is not None and self._last_t is not None and t > self._last_t:
            self._dt = max(_DT_MIN, min(_DT_MAX, t - self._last_t))
        if t is not None:
            self._last_t = t
        self._computed = {}
        self._last_detection = {
            "pose": pf.pose_present,
            "left": pf.left_present,
            "right": pf.right_present,
        }

        self._filter_pose_landmarks(pf)
        self._filter_hand_landmarks(pf)
        self._apply_hand_depth_sign(pf)
        self._update_body_estimates(pf)
        self._sample_eye_offset(pf)

        if self._run_calibration(pf):
            self._last_bones_applied = 0
            return

        applied = 0
        for side in ("L", "R"):
            applied += self._drive_side(side, pf)
        self._last_bones_applied = applied


def register():
    pass


def unregister():
    pass
