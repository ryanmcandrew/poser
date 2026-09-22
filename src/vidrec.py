"""Record raw webcam video into the clip library.

This deliberately runs no pose model. The whole point of an offline clip
is that estimation happens later, with an accurate engine and as much time
per frame as it needs; doing any inference during capture would drag the
frame rate down and put motion blur and dropped frames into the very
footage the accurate pass is supposed to read. So this thread does three
things only: read, write, and note when each frame arrived.

Frame times are kept because a webcam does not deliver at the rate it
claims, and a container only stores one nominal rate (see `cliplib`).
"""

import os
import threading
import time

from . import cliplib
from . import tracker as tracker_mod


# Written into an .mp4. mp4v is the codec that is present in every
# opencv-python wheel; the better ones (H.264) depend on a system FFmpeg
# build that a Blender user has no reason to have.
_FOURCC_PRIMARY = ("mp4v", ".mp4")
# Fallback when the container/codec pair is refused: MJPG in an AVI is the
# one combination OpenCV can always write. It is much larger per second,
# which is an acceptable price for capturing at all.
_FOURCC_FALLBACK = ("MJPG", ".avi")


class ClipRecorder:
    """Webcam -> video file, on its own thread.

    The caller polls `frames()`, `seconds()`, `state()` and `error()`; the
    UI timer does exactly that. `latest_preview()` hands back the most
    recent frame as RGBA for the panel, at a rate the writer does not pay
    for.
    """

    def __init__(self, camera_index=0, path=None, width=0, height=0,
                 fps=30.0, use_mjpg=True, mirror=False, countdown=0.0,
                 max_seconds=0.0, preview_fps=10.0):
        self.camera_index = int(camera_index)
        self.path = path
        self.width = int(width or 0)
        self.height = int(height or 0)
        self.fps = float(fps or 30.0)
        self.use_mjpg = bool(use_mjpg)
        # Mirroring is a display choice, not a recording one: the analyser
        # applies the mirror it is configured with, so footage stays as the
        # sensor saw it and a clip is never locked to one view mode.
        self.mirror_preview = bool(mirror)
        self.countdown = max(0.0, float(countdown or 0.0))
        self.max_seconds = max(0.0, float(max_seconds or 0.0))
        self.preview_fps = float(preview_fps or 0.0)

        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._preview = None
        self._state = "idle"
        self._error = None
        self._frames = 0
        self._stamps = []
        self._started = 0.0
        self._written_path = None
        self._last_preview_t = 0.0

    # --- public -------------------------------------------------------

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return False
        self._stop.clear()
        self._error = None
        self._frames = 0
        self._stamps = []
        self._state = "starting"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop(self, timeout=6.0):
        """Ask the thread to finish and wait for the file to be closed.

        Waiting matters: the caller wants to scan the library right after,
        and a video whose writer has not been released yet is unreadable.
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        return self._written_path

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def state(self):
        return self._state

    def error(self):
        return self._error

    def frames(self):
        return self._frames

    def seconds(self):
        return self._stamps[-1] if self._stamps else 0.0

    def measured_fps(self):
        if len(self._stamps) < 2:
            return 0.0
        span = self._stamps[-1] - self._stamps[0]
        return (len(self._stamps) - 1) / span if span > 1e-6 else 0.0

    def written_path(self):
        return self._written_path

    def latest_preview(self):
        with self._lock:
            return self._preview

    # --- thread -------------------------------------------------------

    def _configure(self, cv2, cap):
        if self.use_mjpg:
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps:
            cap.set(cv2.CAP_PROP_FPS, self.fps)

    def _open_writer(self, cv2, size, nominal_fps):
        """(writer, path). Falls back to MJPG/AVI if the mp4 is refused."""
        base = cliplib.stem(self.path)
        for fourcc, ext in (_FOURCC_PRIMARY, _FOURCC_FALLBACK):
            path = base + ext
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc),
                                     nominal_fps, size)
            if writer.isOpened():
                return writer, path
            writer.release()
            try:
                # A refused writer can still leave a zero-byte file behind,
                # which would then show up in the library as a dead clip.
                if os.path.exists(path) and os.path.getsize(path) == 0:
                    os.remove(path)
            except OSError:
                pass
        return None, None

    def _run(self):
        try:
            import cv2
        except Exception as e:
            self._error = "OpenCV import failed: %s" % e
            self._state = "import failed"
            return

        self._state = "opening camera %d" % self.camera_index
        cap, attempts = tracker_mod.open_camera(
            cv2, self.camera_index,
            configure=lambda c: self._configure(cv2, c))
        if cap is None:
            detail = "; ".join(attempts[-3:])
            self._error = ("cannot open webcam index %d%s"
                           % (self.camera_index,
                              (" [" + detail + "]") if detail else ""))
            self._state = "camera open failed"
            return

        writer = None
        try:
            ok, frame = cap.read()
            if not ok or frame is None:
                self._error = "camera opened but delivered no frame"
                self._state = "no frame"
                return
            h, w = frame.shape[:2]

            # The rate the device actually reports, since that is what the
            # container will claim. Real per-frame times go in the timing
            # file either way, so this only has to be plausible.
            dev_fps = 0.0
            try:
                dev_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            except Exception:
                pass
            nominal = dev_fps if 1.0 < dev_fps < 240.0 else self.fps
            writer, path = self._open_writer(cv2, (w, h), nominal)
            if writer is None:
                self._error = "cannot open a video writer for " + str(self.path)
                self._state = "writer failed"
                return
            self.path = path

            if self.countdown > 0.0:
                deadline = time.time() + self.countdown
                while not self._stop.is_set():
                    left = deadline - time.time()
                    if left <= 0.0:
                        break
                    self._state = "starting in %d" % int(left + 0.999)
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        self._publish_preview(cv2, frame, force=True)
                    else:
                        time.sleep(0.02)
                if self._stop.is_set():
                    self._state = "cancelled"
                    return

            self._started = time.time()
            self._state = "recording"
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    # One bad read is normal on a USB camera; a run of them
                    # is a dead device, and the stall shows in the frame
                    # count the panel is already displaying.
                    time.sleep(0.005)
                    continue
                now = time.time() - self._started
                writer.write(frame)
                self._frames += 1
                self._stamps.append(now)
                self._publish_preview(cv2, frame)
                if self.max_seconds and now >= self.max_seconds:
                    self._state = "stopped: reached the time limit"
                    break
            else:
                self._state = "stopped"
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._error = "recorder error: %s" % e
            self._state = "error"
        finally:
            if writer is not None:
                writer.release()
            cap.release()
            self._finalize()

    def _finalize(self):
        """Close out the clip: keep it and its timing, or bin an empty one."""
        if self._frames <= 0:
            try:
                if self.path and os.path.exists(self.path):
                    os.remove(self.path)
            except OSError:
                pass
            self._written_path = None
            if not self._error:
                self._state = "nothing recorded"
            return
        cliplib.write_timing(self.path, self._stamps, {
            "frames": self._frames,
            "fps": self.measured_fps(),
            "camera_index": self.camera_index,
            "recorded": time.time(),
        })
        self._written_path = self.path
        if self._state in ("recording", "idle"):
            self._state = "stopped"

    def _publish_preview(self, cv2, frame, force=False):
        if self.preview_fps <= 0.0:
            return
        now = time.time()
        if not force and (now - self._last_preview_t) < 1.0 / self.preview_fps:
            return
        self._last_preview_t = now
        try:
            h, w = frame.shape[:2]
            if w > 480:
                frame = cv2.resize(frame, (480, max(1, int(h * 480 / w))),
                                   interpolation=cv2.INTER_AREA)
            if self.mirror_preview:
                frame = cv2.flip(frame, 1)
            rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        except Exception:
            return
        with self._lock:
            self._preview = rgba


def register():
    pass


def unregister():
    pass
