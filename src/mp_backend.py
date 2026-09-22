"""Compatibility layer over MediaPipe's two APIs.

MediaPipe 0.10.x exposes the legacy `mp.solutions.hands.Hands` etc. On some
Python builds (notably the Python 3.13 shipped with Blender 5.x) the wheels
do not include the `solutions` subpackage, and the modern Tasks API
(`mp.tasks.vision.HandLandmarker`) must be used instead. This module tries
both and gives a uniform result shape.

Uniform result: `HandsResult(hands=[HandTrack(label, landmarks)])` where
`landmarks` is a list of 21 (x, y, z) tuples in image-normalised space and
`world_landmarks` the same 21 points in metres, centred on the hand (empty
when the backend does not provide them).
`label` is either "Left" or "Right" (subject's own hand, after mirroring).
"""

import os
import sys
import urllib.request
from dataclasses import dataclass, field
from typing import List, Tuple


HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
                  "hand_landmarker/hand_landmarker/float16/latest/"
                  "hand_landmarker.task")

# Indexed by model_complexity 0/1/2 (lite / full / heavy).
POSE_MODEL_URLS = {
    0: ("pose_landmarker_lite.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"),
    1: ("pose_landmarker_full.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task"),
    2: ("pose_landmarker_heavy.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"),
}


@dataclass
class HandTrack:
    label: str
    landmarks: List[Tuple[float, float, float]]
    world_landmarks: List[Tuple[float, float, float]] = field(default_factory=list)


@dataclass
class HandsResult:
    hands: List[HandTrack] = field(default_factory=list)


@dataclass
class PoseResult:
    """World-space (metric, hip-centered) landmarks in `landmarks`; image-
    normalised (x,y in 0..1, z relative) in `image_landmarks`; per-landmark
    `visibility` in 0..1 (low = predicted, off-camera, unreliable).
    All lists follow the standard 33-point MediaPipe pose order."""
    landmarks: List[Tuple[float, float, float]] = field(default_factory=list)
    image_landmarks: List[Tuple[float, float, float]] = field(default_factory=list)
    visibility: List[float] = field(default_factory=list)
    present: bool = False


def _addon_data_dir():
    d = os.path.join(os.path.dirname(__file__), "_models")
    os.makedirs(d, exist_ok=True)
    return d


def _ensure_hand_model():
    path = os.path.join(_addon_data_dir(), "hand_landmarker.task")
    if not os.path.exists(path):
        urllib.request.urlretrieve(HAND_MODEL_URL, path)
    return path


def _ensure_pose_model(complexity=1):
    name, url = POSE_MODEL_URLS.get(int(complexity), POSE_MODEL_URLS[1])
    path = os.path.join(_addon_data_dir(), name)
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


def _running_mode(tvis, mode):
    return tvis.RunningMode.VIDEO if (mode or "IMAGE").upper() == "VIDEO" \
        else tvis.RunningMode.IMAGE


def _base_options(tpy, model_path, delegate):
    """BaseOptions with an optional GPU delegate.

    The GPU delegate is a large win where it works, but the Windows
    MediaPipe wheels frequently ship without it; callers fall back to CPU
    when construction or the first inference raises.
    """
    kwargs = {"model_asset_path": model_path}
    if (delegate or "CPU").upper() == "GPU":
        kwargs["delegate"] = tpy.BaseOptions.Delegate.GPU
    return tpy.BaseOptions(**kwargs)


def _delegate_order(delegate):
    return ("GPU", "CPU") if (delegate or "CPU").upper() == "GPU" else ("CPU",)


class _VideoClock:
    """MediaPipe VIDEO mode needs strictly increasing millisecond stamps."""

    def __init__(self):
        self._last = -1

    def next(self, ts_ms):
        if ts_ms is None or ts_ms <= self._last:
            ts_ms = self._last + 1
        self._last = ts_ms
        return ts_ms


def diagnostics():
    """Return a multi-line human-readable string describing the environment."""
    lines = []
    lines.append("Python: %s" % sys.version.split()[0])
    lines.append("Executable: %s" % sys.executable)
    try:
        import mediapipe as mp
        lines.append("MediaPipe: %s" % getattr(mp, "__version__", "?"))
        lines.append("  path: %s" % getattr(mp, "__file__", "?"))
        lines.append("  has 'solutions': %s" % hasattr(mp, "solutions"))
        lines.append("  has 'tasks': %s" % hasattr(mp, "tasks"))
        lines.append("  has 'Image': %s" % hasattr(mp, "Image"))
        if hasattr(mp, "solutions"):
            try:
                from mediapipe.python.solutions import hands as _h
                lines.append("  solutions.hands.Hands: %s" % hasattr(_h, "Hands"))
            except Exception as e:
                lines.append("  solutions.hands import error: %s" % e)
    except Exception as e:
        lines.append("MediaPipe import failed: %s" % e)
    try:
        import cv2
        lines.append("OpenCV: %s" % cv2.__version__)
    except Exception as e:
        lines.append("OpenCV import failed: %s" % e)
    return "\n".join(lines)


class HandsBackend:
    """Duck-typed wrapper — call .process(rgb_uint8) -> HandsResult."""

    def __init__(self, min_detection=0.5, min_tracking=0.5, max_hands=2,
                 running_mode="IMAGE", delegate="CPU"):
        self._impl = None
        self._which = "none"
        self._min_det = min_detection
        self._min_track = min_tracking
        self._max_hands = max_hands
        self._init_errors = []
        self._mode = (running_mode or "IMAGE").upper()
        self._delegate = (delegate or "CPU").upper()
        self._delegate_used = "cpu"
        self._clock = _VideoClock()

    def load(self):
        legacy_err = self._try_legacy()
        if self._impl is not None:
            return
        tasks_err = self._try_tasks()
        if self._impl is not None:
            return
        raise RuntimeError(
            "MediaPipe hands API unavailable.\n"
            "  legacy: %s\n"
            "  tasks: %s" % (legacy_err, tasks_err)
        )

    def which(self):
        return self._which

    def _try_legacy(self):
        try:
            from mediapipe.python.solutions.hands import Hands
        except Exception as e:
            try:
                import mediapipe as mp
                Hands = mp.solutions.hands.Hands
            except Exception as e2:
                return "%s | %s" % (e, e2)
        try:
            inst = Hands(
                model_complexity=1,
                max_num_hands=self._max_hands,
                min_detection_confidence=self._min_det,
                min_tracking_confidence=self._min_track,
            )
            self._impl = ("legacy", inst)
            self._which = "legacy solutions.hands"
            return None
        except Exception as e:
            return "instantiation failed: %s" % e

    def _try_tasks(self):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as tpy
            from mediapipe.tasks.python import vision as tvis
        except Exception as e:
            return "import failed: %s" % e
        try:
            model_path = _ensure_hand_model()
        except Exception as e:
            return "model download failed: %s" % e
        err = None
        for delegate in _delegate_order(self._delegate):
            try:
                opts = tvis.HandLandmarkerOptions(
                    base_options=_base_options(tpy, model_path, delegate),
                    num_hands=self._max_hands,
                    min_hand_detection_confidence=self._min_det,
                    min_hand_presence_confidence=self._min_det,
                    min_tracking_confidence=self._min_track,
                    running_mode=_running_mode(tvis, self._mode),
                )
                detector = tvis.HandLandmarker.create_from_options(opts)
            except Exception as e:
                err = "instantiation failed (%s): %s" % (delegate, e)
                print("[FPV Pose Tracker] hand landmarker %s delegate unavailable: %s"
                      % (delegate, e))
                continue
            self._impl = ("tasks", detector, mp)
            self._delegate_used = delegate.lower()
            self._which = "tasks HandLandmarker (%s, %s)" % (
                self._mode.lower(), self._delegate_used)
            return None
        return err

    def process(self, rgb_uint8, timestamp_ms=None):
        if self._impl is None:
            return HandsResult()
        kind = self._impl[0]
        if kind == "legacy":
            inst = self._impl[1]
            rgb_uint8.flags.writeable = False
            res = inst.process(rgb_uint8)
            out = HandsResult()
            multi_lms = getattr(res, "multi_hand_landmarks", None) or []
            multi_hd = getattr(res, "multi_handedness", None) or []
            multi_wl = getattr(res, "multi_hand_world_landmarks", None) or []
            for k, (lms, hd) in enumerate(zip(multi_lms, multi_hd)):
                label = hd.classification[0].label  # "Left"/"Right"
                pts = [(lm.x, lm.y, lm.z) for lm in lms.landmark]
                wpts = []
                if k < len(multi_wl):
                    wpts = [(lm.x, lm.y, lm.z) for lm in multi_wl[k].landmark]
                out.hands.append(HandTrack(label=label, landmarks=pts,
                                           world_landmarks=wpts))
            return out
        else:
            detector = self._impl[1]
            mp = self._impl[2]
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_uint8)
            if self._mode == "VIDEO":
                res = detector.detect_for_video(mp_image, self._clock.next(timestamp_ms))
            else:
                res = detector.detect(mp_image)
            out = HandsResult()
            world = getattr(res, "hand_world_landmarks", None) or []
            for k, (lms, hd) in enumerate(zip(res.hand_landmarks, res.handedness)):
                label = hd[0].category_name if hd else "Right"
                pts = [(lm.x, lm.y, lm.z) for lm in lms]
                wpts = [(lm.x, lm.y, lm.z) for lm in world[k]] if k < len(world) else []
                out.hands.append(HandTrack(label=label, landmarks=pts,
                                           world_landmarks=wpts))
            return out

    def close(self):
        try:
            if self._impl and self._impl[0] == "legacy":
                self._impl[1].close()
            elif self._impl and self._impl[0] == "tasks":
                self._impl[1].close()
        except Exception:
            pass
        self._impl = None


class PoseBackend:
    """Provides subject-pose landmarks (arms, torso, legs).

    MediaPipe pose predicts landmark positions even when they're outside
    the camera frame — critical for FPV body-cam use where shoulders and
    elbows are usually off-screen.
    """

    def __init__(self, min_detection=0.2, min_tracking=0.2, model_complexity=1,
                 running_mode="IMAGE", delegate="CPU"):
        self._impl = None
        self._which = "none"
        self._min_det = min_detection
        self._min_track = min_tracking
        self._model_complexity = model_complexity
        self._mode = (running_mode or "IMAGE").upper()
        self._delegate = (delegate or "CPU").upper()
        self._delegate_used = "cpu"
        self._clock = _VideoClock()

    def load(self):
        legacy_err = self._try_legacy()
        if self._impl is not None:
            return
        tasks_err = self._try_tasks()
        if self._impl is not None:
            return
        raise RuntimeError(
            "MediaPipe pose API unavailable.\n"
            "  legacy: %s\n"
            "  tasks: %s" % (legacy_err, tasks_err)
        )

    def which(self):
        return self._which

    def _try_legacy(self):
        try:
            from mediapipe.python.solutions.pose import Pose
        except Exception as e:
            try:
                import mediapipe as mp
                Pose = mp.solutions.pose.Pose
            except Exception as e2:
                return "%s | %s" % (e, e2)
        try:
            inst = Pose(
                model_complexity=self._model_complexity,
                min_detection_confidence=self._min_det,
                min_tracking_confidence=self._min_track,
                enable_segmentation=False,
                smooth_landmarks=True,
            )
            self._impl = ("legacy", inst)
            self._which = "legacy solutions.pose"
            return None
        except Exception as e:
            return "instantiation failed: %s" % e

    def _try_tasks(self):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as tpy
            from mediapipe.tasks.python import vision as tvis
        except Exception as e:
            return "import failed: %s" % e
        try:
            model_path = _ensure_pose_model(self._model_complexity)
        except Exception as e:
            return "model download failed: %s" % e
        err = None
        for delegate in _delegate_order(self._delegate):
            try:
                opts = tvis.PoseLandmarkerOptions(
                    base_options=_base_options(tpy, model_path, delegate),
                    num_poses=1,
                    min_pose_detection_confidence=self._min_det,
                    min_pose_presence_confidence=self._min_det,
                    min_tracking_confidence=self._min_track,
                    running_mode=_running_mode(tvis, self._mode),
                )
                detector = tvis.PoseLandmarker.create_from_options(opts)
            except Exception as e:
                err = "instantiation failed (%s): %s" % (delegate, e)
                print("[FPV Pose Tracker] pose landmarker %s delegate unavailable: %s"
                      % (delegate, e))
                continue
            self._impl = ("tasks", detector, mp)
            self._delegate_used = delegate.lower()
            self._which = "tasks PoseLandmarker (%s, %s, %s)" % (
                self._mode.lower(), os.path.basename(model_path).replace(".task", ""),
                self._delegate_used)
            return None
        return err

    def process(self, rgb_uint8, timestamp_ms=None):
        if self._impl is None:
            return PoseResult()
        kind = self._impl[0]
        if kind == "legacy":
            inst = self._impl[1]
            rgb_uint8.flags.writeable = False
            res = inst.process(rgb_uint8)
            out = PoseResult()
            wlms = getattr(res, "pose_world_landmarks", None)
            ilms = getattr(res, "pose_landmarks", None)
            if wlms is not None and getattr(wlms, "landmark", None):
                out.landmarks = [(lm.x, lm.y, lm.z) for lm in wlms.landmark]
                out.present = True
            if ilms is not None and getattr(ilms, "landmark", None):
                out.image_landmarks = [(lm.x, lm.y, lm.z) for lm in ilms.landmark]
                # visibility lives on the image-space landmarks in the legacy API
                out.visibility = [getattr(lm, "visibility", 1.0) for lm in ilms.landmark]
                out.present = True
            return out
        else:
            detector = self._impl[1]
            mp = self._impl[2]
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_uint8)
            if self._mode == "VIDEO":
                res = detector.detect_for_video(mp_image, self._clock.next(timestamp_ms))
            else:
                res = detector.detect(mp_image)
            out = PoseResult()
            if getattr(res, "pose_world_landmarks", None):
                lms = res.pose_world_landmarks[0]
                out.landmarks = [(lm.x, lm.y, lm.z) for lm in lms]
                out.present = True
            if getattr(res, "pose_landmarks", None):
                lms = res.pose_landmarks[0]
                out.image_landmarks = [(lm.x, lm.y, lm.z) for lm in lms]
                # Tasks API sets .visibility or .presence per NormalizedLandmark.
                out.visibility = [
                    getattr(lm, "visibility", None)
                    if getattr(lm, "visibility", None) is not None
                    else getattr(lm, "presence", 1.0)
                    for lm in lms
                ]
                out.present = True
            return out

    def close(self):
        try:
            if self._impl and self._impl[0] == "legacy":
                self._impl[1].close()
            elif self._impl and self._impl[0] == "tasks":
                self._impl[1].close()
        except Exception:
            pass
        self._impl = None


def register():
    pass


def unregister():
    pass
