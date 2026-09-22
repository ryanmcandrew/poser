"""Offline pose estimation over a recorded clip.

Three jobs, in the order a clip goes through them:

  `Analyzer`   video file -> per-frame landmarks in an analysis sidecar.
  `Preview`    sidecar + video -> an annotated frame to look at.
  `calibrate`  a clip of the rest pose -> the retargeter's reference.
  `replay`     sidecar -> PoseFrames -> a retargeter -> a recorded take.

Why offline beats live
----------------------
A live tracker has one budget: whatever is left of a frame interval after
Blender has drawn. That budget picks the model, and it is what stops
ViTPose-large being usable at all. Reading from a file removes the budget
entirely, which buys three separate things:

  * the accurate model, at whatever frame rate it manages;
  * every frame, in order, with none dropped, so fast motion keeps the
    samples that describe it;
  * the body pose of *this* frame placing the hand crops, instead of the
    previous frame's, because there is no reason to overlap the two models
    when nothing is waiting on the result.

Splitting estimation from retargeting
-------------------------------------
The sidecar stores raw landmarks, not bone rotations. Estimation is the
expensive, fixed part; retargeting is cheap and depends on things the user
changes constantly (armature, limb lengths, smoothing, camera FOV, IK). So
a clip is analysed once per engine and replayed as often as the rig
changes, and comparing two engines means comparing two sidecars over the
same footage rather than two shoots.
"""

import os
import threading
import time

from . import cliplib
from . import mp_backend
from . import pose_engines
from . import recording
from . import tracker as tracker_mod


# Settings that describe how the frames were interpreted rather than what
# the rig does with them. These go in the sidecar header and are restored
# on replay, so a take always uses the geometry the landmarks were solved
# under.
_GEOMETRY_KEYS = ("mirror", "view_mode", "camera_fov", "pose_visibility",
                  "lift_depth", "hand_depth", "hand_size", "hand_size_auto",
                  "pose_y_up", "tracking_mode", "body_shoulder",
                  "body_upper", "body_fore")


def _math_engine(header):
    """A `WebcamTracker` used only for its per-frame maths.

    The tracker owns side resolution, the depth lifter, hand-size depth and
    the overlay drawing, and none of that has anything to do with where the
    frames came from. Constructing one without starting it opens no device
    and spawns no thread: it is the frame solver, reused rather than
    reimplemented so a clip and a webcam cannot drift apart.
    """
    return tracker_mod.WebcamTracker(
        mirror=bool(header.get("mirror", True)),
        view_mode=header.get("view_mode", "BODY_CAM"),
        camera_fov=float(header.get("camera_fov", 60.0)),
        pose_visibility=float(header.get("pose_visibility", 0.5)),
        lift_depth=bool(header.get("lift_depth", True)),
        hand_depth=bool(header.get("hand_depth", True)),
        hand_size=float(header.get("hand_size", 1.0)),
        hand_size_auto=bool(header.get("hand_size_auto", True)),
        pose_y_up=bool(header.get("pose_y_up", True)),
        tracking_mode=header.get("tracking_mode", "HOLISTIC"),
        pose_engine=header.get("engine", "MEDIAPIPE"),
        engine_quality=header.get("quality", "BALANCED"),
        engine_layout=header.get("layout", "BODY"),
        engine_hands=header.get("hands", "AUTO"),
        body_shoulder=float(header.get("body_shoulder", 0.0)),
        body_upper=float(header.get("body_upper", 0.0)),
        body_fore=float(header.get("body_fore", 0.0)),
        preview_fps=0.0,
    )


def _pack_frame(index, t, pose_res, hands_res):
    """One sidecar record: exactly what the backends returned."""
    rec = {"i": int(index), "t": round(float(t), 4)}
    if pose_res is not None and pose_res.present:
        rec["pose"] = {
            "lm": cliplib.pack_landmarks(pose_res.landmarks),
            "ilm": cliplib.pack_landmarks(pose_res.image_landmarks),
            "vis": [round(float(v), 4) for v in (pose_res.visibility or [])],
        }
    hands = []
    for h in (hands_res.hands if hands_res is not None else []):
        hands.append({"label": h.label,
                      "lm": cliplib.pack_landmarks(h.landmarks),
                      "wlm": cliplib.pack_landmarks(h.world_landmarks)})
    if hands:
        rec["hands"] = hands
    return rec


def _unpack_frame(rec):
    """Sidecar record -> (PoseResult, HandsResult), as inference produced."""
    pose = rec.get("pose")
    pose_res = None
    if pose:
        pose_res = mp_backend.PoseResult(
            landmarks=[tuple(p) for p in pose.get("lm", [])],
            image_landmarks=[tuple(p) for p in pose.get("ilm", [])],
            visibility=list(pose.get("vis", [])),
            present=True,
        )
    hands = [mp_backend.HandTrack(label=h.get("label", ""),
                                  landmarks=[tuple(p) for p in h.get("lm", [])],
                                  world_landmarks=[tuple(p) for p in h.get("wlm", [])])
             for h in rec.get("hands", [])]
    return pose_res, mp_backend.HandsResult(hands=hands)


# ----------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------

class Analyzer:
    """Run one engine over one clip, on a worker thread.

    The UI polls `progress()`, `state()`, `error()` and `finished()`. The
    thread never touches bpy data: it reads a file, runs models, and writes
    another file, so it cannot collide with Blender's own state.
    """

    def __init__(self, video_path, settings, engine="MEDIAPIPE",
                 overwrite=True, stride=1):
        self.video_path = video_path
        self.engine = (engine or "MEDIAPIPE").upper()
        self.settings = dict(settings or {})
        self.overwrite = bool(overwrite)
        # Every Nth frame. 1 is the point of an offline pass; higher trades
        # temporal detail for time on a slow engine and a long clip.
        self.stride = max(1, int(stride or 1))
        self.out_path = cliplib.sidecar_path(video_path, self.engine)

        self._thread = None
        self._stop = threading.Event()
        self._state = "idle"
        self._error = None
        self._progress = 0.0
        self._done = False
        self._total = 0
        self._processed = 0
        self._detected = 0
        self._ms = 0.0
        self._elapsed = 0.0

    # --- public -------------------------------------------------------

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return False
        self._stop.clear()
        self._done = False
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def cancel(self):
        self._stop.set()

    def cancelled(self):
        return self._stop.is_set()

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def finished(self):
        return self._done

    def state(self):
        return self._state

    def error(self):
        return self._error

    def progress(self):
        return self._progress

    def counts(self):
        """(processed, total, detected)."""
        return self._processed, self._total, self._detected

    def summary(self):
        if self._processed <= 0:
            return ""
        return "%d frames, %.0f%% detected, %.0f ms/frame, %.0fs total" % (
            self._processed, 100.0 * self._detected / self._processed,
            self._ms, self._elapsed)

    # --- thread -------------------------------------------------------

    def _build_header(self, info, fps):
        header = {"engine": self.engine,
                  "clip": os.path.basename(self.video_path),
                  "created": time.time(),
                  "width": info.get("width", 0),
                  "height": info.get("height", 0),
                  "video_fps": fps,
                  "stride": self.stride,
                  "quality": self.settings.get("engine_quality", "BALANCED"),
                  "layout": self.settings.get("engine_layout", "BODY"),
                  "hands": self.settings.get("engine_hands", "AUTO"),
                  "device": self.settings.get("engine_device", "CPU"),
                  "kpt_thr": self.settings.get("engine_kpt_thr", 0.3),
                  "model_complexity": self.settings.get("model_complexity", 1)}
        for key in _GEOMETRY_KEYS:
            if key in self.settings:
                header[key] = self.settings[key]
        return header

    def _run(self):
        try:
            import cv2
        except Exception as e:
            self._fail("OpenCV import failed: %s" % e)
            return
        if not self.overwrite and os.path.exists(self.out_path):
            self._fail("an analysis for %s already exists" % self.engine)
            return

        info = cliplib.video_info(self.video_path)
        if not info or not info.get("frames"):
            # A container without a frame count still plays; we just cannot
            # show a percentage for it.
            info = info or {}
        fps = float(info.get("fps") or 0.0)
        total = int(info.get("frames") or 0)
        self._total = total // self.stride if total else 0
        times = cliplib.frame_times(self.video_path, total, fps) if total else []

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self._fail("cannot open video: %s" % self.video_path)
            return

        caps = pose_engines.capabilities(self.engine)
        mode = (self.settings.get("tracking_mode") or "HOLISTIC").upper()
        use_hands = mode in ("HANDS", "HOLISTIC")
        use_pose = mode in ("POSE", "HOLISTIC")
        mirror = bool(self.settings.get("mirror", True))

        pose_backend = hands_backend = None
        records = []
        t_begin = time.time()
        try:
            if use_pose:
                self._state = "loading %s body model" % self.engine.lower()
                pose_backend = pose_engines.make_pose_backend(
                    self.engine,
                    model_complexity=int(self.settings.get("model_complexity", 1)),
                    running_mode="VIDEO",
                    delegate=self.settings.get("mp_delegate", "CPU"),
                    quality=self.settings.get("engine_quality", "BALANCED"),
                    body_layout=self.settings.get("engine_layout", "BODY"),
                    device=self.settings.get("engine_device", "CPU"),
                    # Offline, a detector pass is cheap relative to getting
                    # the crop wrong, and a clip has no real-time budget to
                    # protect: re-detect every frame.
                    det_every=1,
                    kpt_thr=float(self.settings.get("engine_kpt_thr", 0.3)),
                )
                pose_backend.load()
            if use_hands:
                self._state = "loading %s hand model" % self.engine.lower()
                hands_backend = pose_engines.make_hands_backend(
                    self.engine,
                    hand_source=self.settings.get("engine_hands", "AUTO"),
                    body_layout=self.settings.get("engine_layout", "BODY"),
                    max_hands=2, running_mode="VIDEO",
                    delegate=self.settings.get("mp_delegate", "CPU"),
                    device=self.settings.get("engine_device", "CPU"),
                    kpt_thr=float(self.settings.get("engine_kpt_thr", 0.3)),
                    hand_detector=bool(self.settings.get("engine_hand_detector", True)),
                )
                hands_backend.load()

            index = -1
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                index += 1
                if index % self.stride:
                    continue
                if mirror:
                    frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                t = times[index] if index < len(times) else (
                    index / fps if fps > 1e-6 else index / 30.0)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ts_ms = int(t * 1000.0)
                t0 = time.time()
                pose_res = None
                if pose_backend is not None:
                    pose_res = pose_backend.process(rgb, ts_ms)
                hands_res = mp_backend.HandsResult()
                if hands_backend is not None:
                    # This frame's body pose, not the previous one's: with
                    # nothing waiting on the result there is no reason to
                    # overlap the two models, and a current ROI is a
                    # measurably better crop for fast hands.
                    if hasattr(hands_backend, "set_body"):
                        hands_backend.set_body(pose_backend, pose_res)
                    hands_res = hands_backend.process(rgb, ts_ms)
                dt_ms = (time.time() - t0) * 1000.0

                records.append(_pack_frame(len(records), t, pose_res, hands_res))
                self._processed += 1
                if pose_res is not None and pose_res.present:
                    self._detected += 1
                # Exponential mean: a per-frame cost the user can watch
                # settle, rather than one distorted by the first frame's
                # lazy model warm-up.
                self._ms = dt_ms if self._ms <= 0.0 else 0.9 * self._ms + 0.1 * dt_ms
                self._elapsed = time.time() - t_begin
                if self._total:
                    self._progress = min(1.0, self._processed / self._total)
                self._state = "analysing frame %d%s" % (
                    self._processed,
                    " of %d" % self._total if self._total else "")

            if self._stop.is_set():
                self._state = "cancelled"
                self._done = True
                return
            if not records:
                self._fail("no frames could be read from the clip")
                return

            self._elapsed = time.time() - t_begin
            header = self._build_header(info, fps)
            header.update({
                "frames": len(records),
                "duration": records[-1]["t"] - records[0]["t"],
                "detect_rate": self._detected / float(len(records)),
                "ms_per_frame": round(self._ms, 2),
                "analysis_seconds": round(self._elapsed, 1),
                "models": ",".join(b.which() for b in
                                   (pose_backend, hands_backend) if b is not None),
                "generic_limbs": bool(not caps.world_pose
                                      and not float(self.settings.get("body_shoulder", 0.0))),
            })
            self._state = "writing " + os.path.basename(self.out_path)
            cliplib.write_analysis(self.out_path, header, records)
            self._state = "done: " + self.summary()
            self._progress = 1.0
            self._done = True
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._fail("analysis failed: %s" % e)
        finally:
            cap.release()
            for backend in (pose_backend, hands_backend):
                if backend is not None:
                    try:
                        backend.close()
                    except Exception:
                        pass

    def _fail(self, message):
        self._error = message
        self._state = "error"
        self._done = True


# ----------------------------------------------------------------------
# preview
# ----------------------------------------------------------------------

class Preview:
    """An analysed clip, seekable frame by frame with its pose drawn on.

    Holds one open capture, because scrubbing reopens the file otherwise
    and a reopen costs far more than the seek. Landmarks are drawn from the
    sidecar, so what is shown is exactly what would be replayed onto the
    rig -- the point of the preview is to check the estimation, and
    re-running the model to draw it would be checking a different run.
    """

    def __init__(self, video_path, sidecar_path):
        self.video_path = video_path
        self.sidecar_path = sidecar_path
        data = cliplib.read_analysis(sidecar_path)
        self.header = data.get("header", {})
        self.records = data.get("frames", [])
        self._math = _math_engine(self.header)
        self._cap = None
        self._cap_index = -1
        self.error = ""

    # --- lifecycle ----------------------------------------------------

    def close(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def __len__(self):
        return len(self.records)

    def time_at(self, index):
        if not self.records:
            return 0.0
        index = max(0, min(len(self.records) - 1, index))
        return float(self.records[index].get("t", 0.0))

    def detected(self, index):
        if not (0 <= index < len(self.records)):
            return False
        return bool(self.records[index].get("pose"))

    def info_line(self, index):
        h = self.header
        parts = ["%s %s" % (h.get("engine", "?"), h.get("quality", ""))]
        if self.records:
            parts.append("frame %d/%d  t=%.2fs" % (index + 1, len(self.records),
                                                   self.time_at(index)))
        rate = h.get("detect_rate")
        if rate is not None:
            parts.append("%.0f%% detected" % (100.0 * rate))
        if h.get("ms_per_frame"):
            parts.append("%.0f ms/frame" % h["ms_per_frame"])
        return "  |  ".join(p for p in parts if p)

    # --- rendering ----------------------------------------------------

    def _ensure_cap(self):
        import cv2
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.video_path)
            self._cap_index = -1
            if not self._cap.isOpened():
                self._cap = None
                self.error = "cannot open " + os.path.basename(self.video_path)
        return self._cap

    def _read_video_frame(self, index):
        """Decode one frame, reading forward where that is cheaper.

        Scrubbing one frame at a time is the common case and a seek costs a
        keyframe re-decode, so a step forward just reads the next frame.
        """
        import cv2
        cap = self._ensure_cap()
        if cap is None:
            return None
        want = index * max(1, int(self.header.get("stride", 1) or 1))
        if want != self._cap_index + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, want)
            self._cap_index = want - 1
        ok, frame = cap.read()
        if not ok or frame is None:
            # A container whose frame count overstates what it can decode
            # is common; report it once rather than blanking the preview.
            self.error = "cannot decode frame %d" % want
            return None
        self._cap_index = want
        self.error = ""
        return frame

    def rgba(self, index, width=480, draw_pose=True, draw_hands=True):
        """The annotated frame at `index` as an RGBA array, or None."""
        import cv2
        if not self.records:
            return None
        index = max(0, min(len(self.records) - 1, index))
        frame = self._read_video_frame(index)
        if frame is None:
            return None
        if self.header.get("mirror", True):
            # Landmarks were produced on the mirrored image, so the preview
            # has to be mirrored too or the overlay lands on the wrong arm.
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        pose_res, hands_res = _unpack_frame(self.records[index])
        if draw_hands:
            self._math.draw_hand_overlay(frame, hands_res)
        if draw_pose and pose_res is not None:
            self._math.draw_pose_overlay(frame, pose_res, w, h)
        if width and w > width:
            frame = cv2.resize(frame, (width, max(1, int(h * width / w))),
                               interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)


# ----------------------------------------------------------------------
# replay onto the rig
# ----------------------------------------------------------------------

def _open_analysis(sidecar_path, start, end):
    """(header, records, start, end, solver, width, height) for one file."""
    data = cliplib.read_analysis(sidecar_path)
    header = data.get("header", {})
    records = data.get("frames", [])
    if not records:
        return header, [], 0, -1, None, 0, 0
    end = len(records) - 1 if end is None or end < 0 else min(end, len(records) - 1)
    start = max(0, min(start, end))
    return (header, records, start, end, _math_engine(header),
            int(header.get("width") or 0) or 640,
            int(header.get("height") or 0) or 480)


def calibrate(sidecar_path, retargeter, skip_seconds=0.0, hold_seconds=0.0,
              start=0, end=-1):
    """Take the calibration reference from a clip of the rest pose.

    Live tracking can calibrate because the performer is there to be told
    to hold still. A clip of a performance is not that: its opening frames
    are whatever the take starts with, and treating them as the rest pose
    is why an offline take can come out with the wrists rotated away from
    everything they should be relative to. So the reference comes from a
    separate short clip of the performer matching the rig's rest pose --
    recorded once and saved as a profile -- and this is what reads it.

    Returns `(status, frames_used, complete)`. Bones are driven while this
    runs, which the caller is expected to undo or overwrite.
    """
    header, records, start, end, solver, width, height = _open_analysis(
        sidecar_path, start, end)
    if not records:
        return "the calibration clip has no analysed frames", 0, False

    retargeter.request_calibration(reset_pose=False)
    # The window is timed in clip seconds from here on. Placing the origin
    # `calibration_delay` *before* the skip point means sampling starts
    # exactly at `skip_seconds`: on a clip the user has already decided
    # which part is the hold, so the live "get into position" delay would
    # only throw away frames they meant us to use.
    t0 = float(records[start].get("t", 0.0))
    retargeter.set_calibration_origin(
        t0 + float(skip_seconds) - retargeter.calibration_delay)

    used = 0
    for i in range(start, end + 1):
        rec = records[i]
        t = float(rec.get("t", 0.0))
        if hold_seconds and (t - t0 - skip_seconds) > hold_seconds:
            break
        pose_res, hands_res = _unpack_frame(rec)
        pf = solver.build_pose_frame(hands_res, pose_res, width, height, t)
        retargeter.apply_frame(pf)
        used += 1
        if not retargeter.is_calibrating() and retargeter.calibration_complete():
            break
    # A short hold that fell just under the live frame quota is still a
    # measurement; take it rather than report nothing.
    retargeter.finalize_partial_calibration()
    complete = retargeter.calibration_complete()
    # Whatever is missing after the whole clip is not going to arrive, and
    # leaving the window open would let the next thing replayed sample the
    # performance itself -- the failure this function exists to prevent.
    retargeter.cancel_calibration()
    if complete:
        status = "calibrated from %d frames" % used
    else:
        status = ("partial calibration from %d frames (%s) - is the whole "
                  "body and both hands visible?" % (used, retargeter.calib_status()))
    return status, used, complete


def replay(sidecar_path, retargeter, armature, on_progress=None,
           start=0, end=-1, calibrate_from_clip=True):
    """Drive `retargeter` with a whole analysis and record it as a take.

    Returns (samples, take). The take goes through exactly the same review
    and bake path as a live recording, so trimming, outlier removal and
    NLA push are the tools the user already knows.

    `calibrate_from_clip` says where the rest-pose reference comes from.
    True samples it from the opening frames of this clip, which is only
    right when the clip opens on the rest pose; False keeps whatever the
    retargeter was already given (a profile, a calibration clip, a live
    session) and drives bones from the first frame.
    """
    header, records, start, end, math_engine, width, height = _open_analysis(
        sidecar_path, start, end)
    if not records:
        return 0, None

    if calibrate_from_clip:
        # Calibration is timed off the frame stamps the retargeter is
        # handed, and those start at zero rather than at the wall clock,
        # so the window has to be re-anchored or it never opens.
        retargeter.set_calibration_origin(float(records[start].get("t", 0.0)))
    else:
        retargeter.cancel_calibration()

    tk = recording.start(retargeter, armature)
    t0 = records[start].get("t", 0.0)
    count = 0
    try:
        for i in range(start, end + 1):
            rec = records[i]
            pose_res, hands_res = _unpack_frame(rec)
            t = float(rec.get("t", 0.0)) - t0
            pf = math_engine.build_pose_frame(hands_res, pose_res,
                                              width, height, t)
            retargeter.apply_frame(pf)
            # The take is anchored on its own clock, so the baked action
            # runs at the speed the clip was shot at whatever rate the
            # analysis managed.
            if not recording.capture(retargeter, tk.started + t):
                break
            count += 1
            if on_progress is not None and (count % 25) == 0:
                on_progress(count, end - start + 1)
    except Exception:
        recording.finish()
        raise
    return count, tk


def register():
    pass


def unregister():
    pass
