"""RTMPose / ViTPose backends, built on `rtmlib` + ONNX Runtime.

These are the accuracy-first alternatives to MediaPipe. Both are top-down
2D models: a person detector proposes a box, and a pose head reads the
joints out of that crop. That is the structure MediaPipe hides inside
`Holistic`, exposed — which is the point, because it lets the body and the
hands each run a model matched to their own difficulty instead of sharing
one budget.

    camera -> person detector -> body pose -> wrist/elbow ROIs -> hand pose

What they do not give is depth: `PoseResult.landmarks` (metric world
space) stays empty and `image_landmarks` carry z = 0. The addon's depth
lifter already reconstructs 3D from 2D landmarks, limb lengths and the
camera FOV, and it is the better of the two paths anyway, so the engine
selector forces it on for these backends rather than pretending a world
pose exists.

`rtmlib` is a thin ONNX wrapper — numpy, opencv and onnxruntime, no mmcv
or mmpose — so it installs into Blender's bundled Python without the usual
OpenMMLab build dance. Checkpoints download on first use and are cached by
rtmlib itself.
"""

import math
import os
import threading

from . import pose_schema
from .mp_backend import HandTrack, HandsResult, PoseResult


# --- model zoo -----------------------------------------------------------
_OMM = "https://download.openmmlab.com/mmpose/v1/projects/"
_VIT = "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/"

# Person detectors, indexed by the quality preset that selects them.
DETECTORS = {
    "LITE": (_OMM + "rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
             (416, 416)),
    "BALANCED": (_OMM + "rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
                 (640, 640)),
    "ACCURATE": (_OMM + "rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip",
                 (640, 640)),
}

# (url, input_size, layout) per engine / body layout / quality. Input sizes
# are rtmlib's (width, height).
BODY_MODELS = {
    ("RTMPOSE", "BODY"): {
        "LITE": (_OMM + "rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip",
                 (192, 256), "coco17"),
        "BALANCED": (_OMM + "rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
                     (192, 256), "coco17"),
        "ACCURATE": (_OMM + "rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip",
                     (288, 384), "coco17"),
    },
    ("RTMPOSE", "FEET"): {
        # Halpe-26 adds the foot points, which matter for a standing rig.
        "LITE": (_OMM + "rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.zip",
                 (192, 256), "halpe26"),
        "BALANCED": (_OMM + "rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip",
                     (192, 256), "halpe26"),
        "ACCURATE": (_OMM + "rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip",
                     (288, 384), "halpe26"),
    },
    ("RTMPOSE", "WHOLEBODY"): {
        # RTMW: 133 points, so the hands come out of the body pass. Cheaper
        # than a second model, less precise than one.
        "LITE": (_OMM + "rtmw/onnx_sdk/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip",
                 (192, 256), "wholebody133"),
        "BALANCED": (_OMM + "rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip",
                     (192, 256), "wholebody133"),
        "ACCURATE": (_OMM + "rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip",
                     (288, 384), "wholebody133"),
    },
    ("VITPOSE", "BODY"): {
        "LITE": (_VIT + "coco/vitpose-s-coco.onnx", (192, 256), "coco17"),
        "BALANCED": (_VIT + "coco/vitpose-b-coco.onnx", (192, 256), "coco17"),
        "ACCURATE": (_VIT + "coco/vitpose-l-coco.onnx", (192, 256), "coco17"),
    },
    ("VITPOSE", "FEET"): {
        "LITE": (_VIT + "coco_25/vitpose-s-coco_25.onnx", (192, 256), "coco25"),
        "BALANCED": (_VIT + "coco_25/vitpose-b-coco_25.onnx", (192, 256), "coco25"),
        "ACCURATE": (_VIT + "coco_25/vitpose-l-coco_25.onnx", (192, 256), "coco25"),
    },
    ("VITPOSE", "WHOLEBODY"): {
        "LITE": (_VIT + "wholebody/vitpose-s-wholebody.onnx", (192, 256), "wholebody133"),
        "BALANCED": (_VIT + "wholebody/vitpose-b-wholebody.onnx", (192, 256), "wholebody133"),
        "ACCURATE": (_VIT + "wholebody/vitpose-l-wholebody.onnx", (192, 256), "wholebody133"),
    },
}

# Dedicated hand stack: its own detector for when no body pose is available,
# and a 21-point hand head that runs on a high-resolution crop.
HAND_DET = (_OMM + "rtmposev1/onnx_sdk/rtmdet_nano_8xb32-300e_hand-267f9c8f.zip",
            (320, 320))
HAND_POSE = (_OMM + "rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip",
             (256, 256))

# rtmlib's `device` strings for our Compute enum. It maps these onto ONNX
# Runtime execution providers.
_DEVICES = {"CPU": "cpu", "GPU": "cuda", "CUDA": "cuda", "MPS": "mps"}

# The pose-head class name in rtmlib for each engine.
_POSE_CLASS = {"RTMPOSE": "RTMPose", "VITPOSE": "ViTPose"}


def _device_order(device):
    """GPU first when asked for, always with CPU behind it.

    A missing onnxruntime-gpu raises at session construction, and losing
    tracking entirely because a wheel was CPU-only is a far worse outcome
    than losing the speed-up.
    """
    d = _DEVICES.get((device or "CPU").upper(), "cpu")
    return (d, "cpu") if d != "cpu" else ("cpu",)


def available():
    """True when rtmlib and an ONNX Runtime are both importable."""
    try:
        import rtmlib  # noqa: F401
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return True


def diagnostics():
    lines = []
    try:
        import rtmlib
        lines.append("rtmlib: %s" % getattr(rtmlib, "__version__", "?"))
        lines.append("  path: %s" % os.path.dirname(getattr(rtmlib, "__file__", "?")))
    except Exception as e:
        lines.append("rtmlib import failed: %s" % e)
    try:
        import onnxruntime as ort
        lines.append("onnxruntime: %s" % ort.__version__)
        lines.append("  providers: %s" % ", ".join(ort.get_available_providers()))
    except Exception as e:
        lines.append("onnxruntime import failed: %s" % e)
    return "\n".join(lines)


def _box_from_points(pts, scores, thr, W, H, pad=1.25):
    """xyxy box around the confident points, padded and clamped."""
    xs = [p[0] for p, s in zip(pts, scores) if s >= thr]
    ys = [p[1] for p, s in zip(pts, scores) if s >= thr]
    if len(xs) < 2:
        return None
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    hw = max(8.0, (x1 - x0) * 0.5 * pad)
    hh = max(8.0, (y1 - y0) * 0.5 * pad)
    return [max(0.0, cx - hw), max(0.0, cy - hh),
            min(float(W), cx + hw), min(float(H), cy + hh)]


def _pick_person(bboxes, W, H):
    """The subject, out of everyone the detector found.

    Biggest wins, with a mild pull toward the middle of the frame: in a
    body-cam or desk setup the user is both the nearest and the most
    centred person, and area alone will happily lock onto a large figure
    at the edge of the shot.
    """
    best, best_score = None, -1.0
    cx0, cy0 = W * 0.5, H * 0.5
    diag = math.hypot(W, H) or 1.0
    for b in bboxes:
        x0, y0, x1, y1 = [float(v) for v in b[:4]]
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if area <= 0.0:
            continue
        off = math.hypot((x0 + x1) * 0.5 - cx0, (y0 + y1) * 0.5 - cy0) / diag
        score = (area / (W * H)) * (1.0 - 0.5 * min(1.0, off))
        if score > best_score:
            best, best_score = [x0, y0, x1, y1], score
    return best


class _Loader:
    """Builds an rtmlib model, walking the device list on failure."""

    def __init__(self, device):
        self.order = _device_order(device)
        self.used = "cpu"
        self.errors = []

    def build(self, cls_name, url, input_size, **kwargs):
        import rtmlib
        cls = getattr(rtmlib, cls_name)
        for dev in self.order:
            try:
                model = cls(url, model_input_size=input_size,
                            backend="onnxruntime", device=dev, **kwargs)
            except Exception as e:
                self.errors.append("%s on %s: %s" % (cls_name, dev, e))
                print("[FPV Pose Tracker] %s unavailable on %s: %s"
                      % (cls_name, dev, e))
                continue
            self.used = dev
            return model
        raise RuntimeError("; ".join(self.errors))


class OnnxPoseBackend:
    """Body pose from RTMPose or ViTPose. Duck-typed like `PoseBackend`.

    The detector is the expensive half of a top-down stack and the subject
    of a webcam feed does not teleport, so it runs every `det_every`
    frames and the box is re-derived from the previous frame's keypoints in
    between. That is rtmlib's own `PoseTracker` strategy, and it is what
    makes a two-model stack affordable at interactive rates.
    """

    def __init__(self, engine="RTMPOSE", quality="BALANCED", layout="BODY",
                 device="CPU", det_every=6, kpt_thr=0.3, min_detection=0.3):
        self.engine = (engine or "RTMPOSE").upper()
        self.quality = (quality or "BALANCED").upper()
        self.body_layout = (layout or "BODY").upper()
        self.device = (device or "CPU").upper()
        self.det_every = max(1, int(det_every or 1))
        self.kpt_thr = float(kpt_thr)
        self.min_detection = float(min_detection)

        self._det = None
        self._pose = None
        self._layout = pose_schema.LAYOUTS["coco17"]
        self._which = "none"
        self._device_used = "cpu"
        self._tick = 0
        self._bbox = None
        self._lock = threading.Lock()
        # Hand spans from the last wholebody result, for the hand backend.
        self._wholebody = None

    # --- lifecycle -------------------------------------------------------
    def load(self):
        try:
            import rtmlib  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "rtmlib is not installed (needed for %s). Use 'Install "
                "Engine Dependencies' in the addon preferences.\n  %s"
                % (self.engine, e))
        key = (self.engine, self.body_layout)
        if key not in BODY_MODELS:
            key = (self.engine, "BODY")
        spec = BODY_MODELS[key].get(self.quality) or BODY_MODELS[key]["BALANCED"]
        url, input_size, layout_name = spec
        self._layout = pose_schema.LAYOUTS[layout_name]

        loader = _Loader(self.device)
        det_url, det_size = DETECTORS.get(self.quality, DETECTORS["BALANCED"])
        self._det = loader.build("YOLOX", det_url, det_size, det_mode="human",
                                 score_thr=self.min_detection)
        self._pose = loader.build(_POSE_CLASS[self.engine], url, input_size,
                                  to_openpose=False)
        self._device_used = loader.used
        self._which = "%s %s/%s (%s, %s, det/%d)" % (
            self.engine.lower(), self.quality.lower(), self._layout.name,
            os.path.basename(url).split("_")[0], self._device_used,
            self.det_every)

    def which(self):
        return self._which

    def close(self):
        self._det = None
        self._pose = None

    # --- results shared with the hand backend ----------------------------
    def wholebody_hands(self):
        """The last frame's 21-point hand spans, or None.

        Only a wholebody layout has them; the caller falls back to its own
        hand model otherwise.
        """
        with self._lock:
            return self._wholebody

    def person_bbox(self):
        with self._lock:
            return list(self._bbox) if self._bbox else None

    # --- inference -------------------------------------------------------
    def _detect(self, img, W, H):
        """Current person box: detector on its tick, tracked box otherwise."""
        due = (self._tick % self.det_every) == 0
        if not due and self._bbox is not None:
            return self._bbox
        try:
            bboxes = self._det(img)
        except Exception as e:
            print("[FPV Pose Tracker] person detector failed:", e)
            return self._bbox
        box = _pick_person(bboxes if bboxes is not None else [], W, H)
        if box is None and self._bbox is not None and not due:
            return self._bbox
        return box

    def process(self, rgb_uint8, timestamp_ms=None):
        """Body pose for one RGB frame.

        RGB, not BGR, deliberately: rtmlib's examples feed OpenCV frames
        straight in, but it normalises with MMPose's RGB-ordered
        mean/std and omits MMPose's own BGR->RGB step, so feeding it RGB
        is what actually reproduces the pipeline these checkpoints were
        trained and exported with.
        """
        out = PoseResult()
        if self._pose is None:
            return out
        H, W = rgb_uint8.shape[:2]
        self._tick += 1
        box = self._detect(rgb_uint8, W, H)
        # No box means nobody was found; running the head on the whole frame
        # still recovers a body-cam subject who fills the shot, and costs
        # one crop either way.
        bboxes = [box] if box is not None else []
        try:
            keypoints, scores = self._pose(rgb_uint8, bboxes=bboxes)
        except Exception as e:
            print("[FPV Pose Tracker] %s inference failed: %s" % (self.engine, e))
            return out
        if keypoints is None or len(keypoints) == 0:
            with self._lock:
                self._bbox = None
                self._wholebody = None
            return out

        kpts, kscores = keypoints[0], scores[0]
        out.image_landmarks, out.visibility = pose_schema.to_canonical(
            kpts, kscores, W, H, self._layout, self.kpt_thr)
        out.present = any(v > 0.0 for v in out.visibility)

        with self._lock:
            # Track the box forward from the joints we just found, so the
            # skipped detector ticks still follow the subject.
            self._bbox = _box_from_points(kpts, kscores, self.kpt_thr, W, H) \
                or box
            if self._layout.hands and out.present:
                self._wholebody = (kpts, kscores, W, H)
            else:
                self._wholebody = None
        return out


class RoiHandsBackend:
    """21-point hands from RTMPose, cropped where the body says they are.

    The body pose supplies each hand's region of interest, which is the
    whole reason to run a separate hand model: the box is a few hundred
    pixels of hand instead of a few dozen inside a whole-body crop, so the
    hand head sees articulation the body model never resolves. When no
    body pose is available — hands-only mode, or the subject not found —
    it falls back to its own hand detector over the full frame.

    Setting `wholebody` makes it free: a 133-point body model already
    carries the hand points, and this just reshapes them.
    """

    # How far past the wrist, as a fraction of the forearm's image length,
    # to centre the crop; a hand extends beyond its wrist, not around it.
    _REACH = 0.55
    # Crop half-size as a fraction of forearm length. A hand is roughly as
    # long as the forearm is, and the pose head wants padding around it.
    _SPAN = 0.75

    def __init__(self, device="CPU", kpt_thr=0.3, max_hands=2,
                 use_detector=True, source="RTM"):
        self.device = (device or "CPU").upper()
        self.kpt_thr = float(kpt_thr)
        self.max_hands = int(max_hands)
        self.use_detector = bool(use_detector)
        self.source = (source or "RTM").upper()   # RTM | WHOLEBODY
        self._det = None
        self._pose = None
        self._which = "none"
        self._device_used = "cpu"
        self._lock = threading.Lock()
        self._rois = {}
        self._body = None

    def load(self):
        if self.source == "WHOLEBODY":
            # Nothing to load: the body pass already produced the points.
            self._which = "wholebody hands (from body model)"
            return
        try:
            import rtmlib  # noqa: F401
        except Exception as e:
            raise RuntimeError("rtmlib is not installed (needed for RTMPose "
                               "hands).\n  %s" % e)
        loader = _Loader(self.device)
        self._pose = loader.build("RTMPose", HAND_POSE[0], HAND_POSE[1],
                                  to_openpose=False)
        if self.use_detector:
            try:
                self._det = loader.build("RTMDet", HAND_DET[0], HAND_DET[1],
                                         det_mode="human")
            except Exception as e:
                # A missing hand detector only costs the no-body-pose
                # fallback, so it is not worth failing the whole start.
                print("[FPV Pose Tracker] hand detector unavailable:", e)
                self._det = None
        self._device_used = loader.used
        self._which = "rtmpose-hand (%s%s)" % (
            self._device_used, "" if self._det is not None else ", roi-only")

    def which(self):
        return self._which

    def close(self):
        self._det = None
        self._pose = None

    # --- coupling to the body pass ---------------------------------------
    def set_body(self, pose_backend, pose_result):
        """Hand the hand model the body's view of where the hands are.

        Called by the tracker once per frame before inference. With the two
        models running in parallel these ROIs are one frame old, which at
        webcam rates moves a wrist by a few percent of its own crop — far
        less than the padding — and buys back the whole cost of a second
        detector.
        """
        with self._lock:
            self._body = pose_backend.wholebody_hands() \
                if (pose_backend is not None and self.source == "WHOLEBODY") else None
            self._rois = self._build_rois(pose_result)

    def _build_rois(self, pose_result):
        if pose_result is None or not pose_result.image_landmarks:
            return {}
        ilms = pose_result.image_landmarks
        vis = pose_result.visibility
        thr = max(0.1, self.kpt_thr * 0.6)
        rois = {}
        for label, el_i, wr_i in (("Left", pose_schema.MP_LELBOW, pose_schema.MP_LWRIST),
                                  ("Right", pose_schema.MP_RELBOW, pose_schema.MP_RWRIST)):
            if wr_i >= len(vis) or vis[wr_i] < thr:
                continue
            wx, wy = ilms[wr_i][0], ilms[wr_i][1]
            if el_i < len(vis) and vis[el_i] >= thr:
                ex, ey = ilms[el_i][0], ilms[el_i][1]
                fx, fy = wx - ex, wy - ey
                fore = math.hypot(fx, fy)
            else:
                fx = fy = 0.0
                fore = 0.0
            if fore < 1e-4:
                # No forearm to measure: fall back to a fixed fraction of
                # the frame, which is about a hand's width at desk range.
                fore = 0.18
                fx = fy = 0.0
            else:
                fx, fy = fx / fore, fy / fore
            cx = wx + fx * fore * self._REACH
            cy = wy + fy * fore * self._REACH
            half = fore * self._SPAN
            rois[label] = (cx - half, cy - half, cx + half, cy + half)
        return rois

    # --- inference -------------------------------------------------------
    def process(self, rgb_uint8, timestamp_ms=None):
        out = HandsResult()
        H, W = rgb_uint8.shape[:2]
        with self._lock:
            rois = dict(self._rois)
            body = self._body

        if self.source == "WHOLEBODY":
            if body is None:
                return out
            kpts, scores, bw, bh = body
            for label, span in (("Left", pose_schema.WHOLEBODY_LEFT_HAND),
                                ("Right", pose_schema.WHOLEBODY_RIGHT_HAND)):
                pts, mean = pose_schema.hand_from_layout(
                    kpts, scores, bw, bh, span, self.kpt_thr)
                if pts:
                    out.hands.append(HandTrack(label=label, landmarks=pts))
            return out

        if self._pose is None:
            return out

        labels, boxes = [], []
        for label, (x0, y0, x1, y1) in rois.items():
            box = [max(0.0, x0 * W), max(0.0, y0 * H),
                   min(float(W), x1 * W), min(float(H), y1 * H)]
            if box[2] - box[0] < 12.0 or box[3] - box[1] < 12.0:
                continue
            labels.append(label)
            boxes.append(box)

        if not boxes and self._det is not None:
            try:
                found = self._det(rgb_uint8)
            except Exception as e:
                print("[FPV Pose Tracker] hand detector failed:", e)
                found = []
            for b in (found if found is not None else [])[:self.max_hands]:
                boxes.append([float(v) for v in b[:4]])
                # Without a body pose there is nothing to name the hand
                # from; the tracker's side resolver settles it from image
                # geometry, and "Right" is only its starting guess.
                labels.append("Right")

        if not boxes:
            return out
        try:
            keypoints, scores = self._pose(rgb_uint8, bboxes=boxes)
        except Exception as e:
            print("[FPV Pose Tracker] hand inference failed:", e)
            return out
        for i, label in enumerate(labels[:len(keypoints)]):
            pts, mean = pose_schema.normalise_hand(keypoints[i], scores[i], W, H)
            if mean < self.kpt_thr:
                continue
            out.hands.append(HandTrack(label=label, landmarks=pts))
        return out


def register():
    pass


def unregister():
    pass
