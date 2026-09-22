"""Operators: start/stop, install deps, auto-map, list cameras."""

import bpy

from . import dependencies
from . import pose_engines
from . import mapping
from . import multicam as multicam_mod
from . import properties as props_mod
from . import recording
from . import retarget as retarget_mod
from . import tracker as tracker_mod


_STATE = {
    "tracker": None,
    "retargeter": None,
    "timer": None,
    "last_frame_t": None,
    # Kept after the tracker is gone: a calibration measured live is the
    # readiest reference for priming a clip replay, which has no moment
    # at which the performer can be asked to hold the rest pose.
    "last_calibration": None,
}

PREVIEW_IMAGE_NAME = "FPV_Preview"


def multi_enabled(s):
    return s.camera_source == props_mod.MULTI_CAM


def enabled_cameras(s):
    """Device rows that will actually be opened, de-duplicated on index.

    Two rows on the same index would have the second camera fail to open on
    Windows and, worse, double-weight the first one's estimate if it did.
    """
    out = []
    seen = set()
    for dev in s.cameras:
        if not dev.enabled or dev.index in seen:
            continue
        seen.add(dev.index)
        out.append(dev)
    return out


def active_camera_index(s):
    """The device the single-camera tools (test, reset, auto-tune) act on."""
    if multi_enabled(s):
        devs = enabled_cameras(s)
        if devs:
            return int(devs[0].index)
    return int(s.camera_index)


def _load_extrinsics(s):
    """The stored camera placement, or None. A corrupt or stale string is
    not worth an error: the rig falls back to the Yaw slider and says so."""
    raw = getattr(s, "multi_extrinsics", "")
    if not raw:
        return None
    try:
        import json
        vals = json.loads(raw)
    except Exception:
        return None
    if not isinstance(vals, dict) or len(vals.get("cams", ())) < 2:
        return None
    return vals


def _save_extrinsics(s, tr):
    """Persist a placement the tracker has just measured, once."""
    if not hasattr(tr, "extrinsics") or tr.is_calibrating():
        return False
    vals = tr.extrinsics()
    if not vals:
        return False
    try:
        import json
        raw = json.dumps({
            "indices": [int(sp.index) for sp in tr.specs],
            "cams": [[round(float(x), 6) for x in cam] for cam in vals],
        })
    except Exception:
        return False
    if raw == getattr(s, "multi_extrinsics", ""):
        return False
    s.multi_extrinsics = raw
    return True


def _camera_specs(s):
    out = []
    for i, d in enumerate(enabled_cameras(s)):
        mode = getattr(d, "exposure_mode", "SHARED")
        out.append(multicam_mod.CameraSpec(
            index=int(d.index), fov=float(d.fov), yaw=float(d.yaw),
            trust=float(d.trust), label="Webcam %d" % (i + 1),
            exposure_mode=None if mode == "SHARED" else mode,
            exposure=float(getattr(d, "exposure", -6.0)),
        ))
    return out


class FPV_OT_install_dependencies(bpy.types.Operator):
    bl_idname = "fpv_pose.install_dependencies"
    bl_label = "Install Dependencies"
    bl_description = ("Install the wheels the selected pose engine needs into "
                      "Blender's bundled Python")

    engine: bpy.props.StringProperty(default="")

    def execute(self, context):
        engine = self.engine or None
        if engine is None:
            scene = getattr(context, "scene", None)
            engine = getattr(scene and scene.fpv_pose, "pose_engine", None)
        ok, log = dependencies.install_dependencies(engine)
        print("[FPV Pose Tracker] install log:\n" + log)
        scene = getattr(context, "scene", None)
        if scene is not None:
            # The tail of the log, in the panel: pip's own words are the
            # only useful thing to read when an install goes wrong, and the
            # system console is not where anyone looks first.
            scene.fpv_pose.test_report = log[-2000:]
        if ok:
            # No restart needed: install_dependencies puts the target
            # directory on sys.path and re-verifies the imports before
            # claiming success.
            self.report({"INFO"},
                        "Installed into %s and verified importable."
                        % dependencies.lib_dir())
        else:
            self.report({"ERROR"}, "Install failed. See the log in the panel.")
        return {"FINISHED"} if ok else {"CANCELLED"}


class FPV_OT_install_gpu_runtime(bpy.types.Operator):
    bl_idname = "fpv_pose.install_gpu_runtime"
    bl_label = "Install GPU Runtime"
    bl_description = ("Replace onnxruntime with onnxruntime-gpu so the "
                      "RTMPose/ViTPose engines can use CUDA. Needs an NVIDIA "
                      "GPU with a matching CUDA/cuDNN install; the CPU "
                      "provider keeps working either way")

    def execute(self, context):
        s = context.scene.fpv_pose
        pkg = dependencies.GPU_PIP.get(s.pose_engine)
        if not pkg:
            self.report({"WARNING"},
                        "The %s engine has no separate GPU wheel" % s.pose_engine)
            return {"CANCELLED"}
        ok, log = dependencies.install_dependencies(packages=[pkg])
        print("[FPV Pose Tracker] gpu runtime install log:\n" + log)
        s.test_report = log[-2000:]
        if ok:
            self.report({"INFO"}, "%s installed — restart Blender, then set "
                                  "Compute to GPU" % pkg)
        else:
            self.report({"ERROR"}, "Install failed. See panel / system console.")
        return {"FINISHED"} if ok else {"CANCELLED"}


class FPV_OT_engine_diagnostics(bpy.types.Operator):
    bl_idname = "fpv_pose.engine_diagnostics"
    bl_label = "Engine Info"
    bl_description = ("Report what every pose engine's dependencies look like "
                      "in this Blender's Python, without opening the camera")

    def execute(self, context):
        s = context.scene.fpv_pose
        lines = []
        for eng in pose_engines.ENGINES:
            ok, msg = dependencies.engine_status(eng)
            lines.append("%-10s %-22s %s" % (
                eng, pose_engines.capabilities(eng).label,
                "OK" if ok else msg))
        lines.append("")
        # Where each module resolved from, and where they are searched for.
        # A wheel can be on disk and still invisible to Blender, which is
        # the one dependency failure that looks like nothing at all.
        lines.append("--- import paths ---")
        lines.append(dependencies.path_report())
        seen = []
        for eng in pose_engines.ENGINES:
            for import_name, _ in pose_engines.required_modules(eng):
                if import_name in seen:
                    continue
                seen.append(import_name)
                origin = dependencies.module_origin(import_name)
                lines.append("  %-14s %s" % (import_name,
                                             origin or "not importable"))
        lines.append("")
        lines.append(pose_engines.diagnostics())
        s.test_report = "\n".join(lines)
        print("[FPV Pose Tracker] engine diagnostics:\n" + s.test_report)
        self.report({"INFO"}, "Engine report in the panel")
        return {"FINISHED"}


class FPV_OT_auto_map(bpy.types.Operator):
    bl_idname = "fpv_pose.auto_map_bones"
    bl_label = "Auto-Map Bones"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        s = context.scene.fpv_pose
        if not s.armature:
            self.report({"ERROR"}, "Pick an armature first")
            return {"CANCELLED"}
        guesses = mapping.auto_map(s.armature)
        b = s.bones
        for slot, name in guesses.items():
            if hasattr(b, slot):
                setattr(b, slot, name)
        self.report({"INFO"}, "Bone map populated (verify in Bone Map panel).")
        return {"FINISHED"}


class FPV_OT_clear_map(bpy.types.Operator):
    bl_idname = "fpv_pose.clear_map"
    bl_label = "Clear Bone Map"

    def execute(self, context):
        b = context.scene.fpv_pose.bones
        for slot in mapping.BONE_SLOTS:
            if hasattr(b, slot):
                setattr(b, slot, "")
        return {"FINISHED"}


class FPV_OT_reinstall_mediapipe(bpy.types.Operator):
    bl_idname = "fpv_pose.reinstall_mediapipe"
    bl_label = "Reinstall MediaPipe"
    bl_description = ("Force-reinstall the mediapipe wheel into Blender's Python. "
                      "Use this if the current install is missing solutions or is broken.")

    def execute(self, context):
        import subprocess
        import sys
        py = sys.executable
        log = []
        for cmd in (
            [py, "-m", "pip", "install", "--upgrade", "pip"],
            [py, "-m", "pip", "uninstall", "-y", "mediapipe"],
            [py, "-m", "pip", "install", "--upgrade", "--no-cache-dir", "mediapipe"],
        ):
            log.append("$ " + " ".join(cmd))
            try:
                p = subprocess.run(cmd, capture_output=True, text=True)
                if p.stdout:
                    log.append(p.stdout.strip()[-500:])
                if p.stderr:
                    log.append(p.stderr.strip()[-500:])
            except Exception as e:
                log.append("error: %s" % e)
        # Refresh sys.path so freshly installed packages are importable.
        try:
            import importlib
            import site
            importlib.reload(site)
        except Exception:
            pass
        context.scene.fpv_pose.test_report = "\n".join(log[-40:])
        print("[FPV Pose Tracker] reinstall log:\n" + "\n".join(log))
        self.report({"INFO"}, "Reinstall done — restart Blender to pick up new wheel.")
        return {"FINISHED"}


class FPV_OT_test_webcam(bpy.types.Operator):
    bl_idname = "fpv_pose.test_webcam"
    bl_label = "Test Webcam + Engine"
    bl_description = ("Synchronously open the webcam, grab one frame, run the "
                      "selected pose engine on it, and print what was "
                      "detected")

    def execute(self, context):
        s = context.scene.fpv_pose
        ok_deps, dep_msg = dependencies.engine_status(s.pose_engine)
        if not ok_deps:
            s.test_report = "%s engine not installed (%s)" % (s.pose_engine,
                                                              dep_msg)
            self.report({"ERROR"}, s.test_report)
            return {"CANCELLED"}
        ok, report = tracker_mod.test_webcam(
            camera_index=active_camera_index(s),
            tracking_mode=s.tracking_mode,
            mirror=bool(s.mirror),
            pose_engine=s.pose_engine,
            engine_quality=s.engine_quality,
            engine_layout=s.engine_layout,
            engine_hands=s.engine_hands,
            engine_device=s.engine_device,
            engine_kpt_thr=float(s.engine_kpt_thr),
        )
        s.test_report = report
        print("[FPV Pose Tracker] test_webcam report:\n" + report)
        if ok:
            self.report({"INFO"}, "Test finished — see panel")
        else:
            self.report({"ERROR"}, "Test failed — see panel")
        return {"FINISHED"}


class FPV_OT_list_bones(bpy.types.Operator):
    bl_idname = "fpv_pose.list_bones"
    bl_label = "List Bones + Auto-Map Preview"
    bl_description = ("Dump the selected armature's bone names and show what "
                      "each would map to. Paste the output back if auto-map "
                      "misses your rig's naming.")

    def execute(self, context):
        s = context.scene.fpv_pose
        if not s.armature:
            self.report({"ERROR"}, "Pick an armature first")
            return {"CANCELLED"}
        report = mapping.diagnose(s.armature)
        s.test_report = report
        print("[FPV Pose Tracker] bone diagnose:\n" + report)
        self.report({"INFO"}, "Bone report written to panel + console")
        return {"FINISHED"}


class FPV_OT_reset_camera(bpy.types.Operator):
    bl_idname = "fpv_pose.reset_camera"
    bl_label = "Reset Camera Exposure"
    bl_description = ("Put the webcam back on automatic exposure. Webcam drivers "
                      "keep a manual exposure even after Blender closes, so a "
                      "too-short one leaves every later session with a black "
                      "image and no tracking. Use this if the preview is dark.")

    def execute(self, context):
        s = context.scene.fpv_pose
        if _STATE.get("tracker") is not None:
            self.report({"ERROR"}, "Stop tracking first — the camera is in use")
            return {"CANCELLED"}
        # Camera-only work: OpenCV is the whole requirement, so it must not
        # be blocked by a pose engine the user is not running.
        if dependencies.missing_base_modules():
            self.report({"ERROR"}, "OpenCV is not installed")
            return {"CANCELLED"}
        try:
            ok, msg = tracker_mod.reset_camera(active_camera_index(s))
        except Exception as e:
            self.report({"ERROR"}, "Camera reset failed: %s" % e)
            return {"CANCELLED"}
        s.cap_info = ""
        s.test_report = msg
        print("[FPV Pose Tracker]", msg)
        self.report({"INFO"} if ok else {"WARNING"}, msg)
        return {"FINISHED"}


class FPV_OT_autotune_exposure(bpy.types.Operator):
    bl_idname = "fpv_pose.autotune_exposure"
    bl_label = "Auto-Tune Exposure"
    bl_description = ("Measure the frame rate at a range of exposures and pick "
                      "the brightest one that still reaches the camera's fastest "
                      "rate. Auto-exposure usually picks a long shutter that pins "
                      "capture at 15 fps. Takes a few seconds; keep the camera "
                      "uncovered and pointed at your normal scene.")

    @classmethod
    def poll(cls, context):
        return _STATE.get("tracker") is None

    def execute(self, context):
        s = context.scene.fpv_pose
        # Camera-only work: OpenCV is the whole requirement, so it must not
        # be blocked by a pose engine the user is not running.
        if dependencies.missing_base_modules():
            self.report({"ERROR"}, "OpenCV is not installed")
            return {"CANCELLED"}
        try:
            ok, exposure, msg = tracker_mod.autotune_exposure(
                active_camera_index(s), int(s.capture_width),
                int(s.capture_height))
        except Exception as e:
            self.report({"ERROR"}, "Auto-tune failed: %s" % e)
            return {"CANCELLED"}
        s.test_report = msg
        if ok and exposure is not None:
            s.exposure_mode = "MANUAL"
            s.exposure = float(exposure)
            self.report({"INFO"}, msg)
        else:
            self.report({"WARNING"}, msg)
        return {"FINISHED"}


class FPV_OT_list_cameras(bpy.types.Operator):
    bl_idname = "fpv_pose.list_cameras"
    bl_label = "List Cameras"
    bl_description = "Scan indices 0..5 and report which webcams are available"

    def execute(self, context):
        import time
        t0 = time.time()
        found = tracker_mod.enumerate_cameras(6)
        took = time.time() - t0
        s = context.scene.fpv_pose
        print("[FPV Pose Tracker] scan took %.1fs, found %r" % (took, found))
        if not found:
            s.cameras_found = "none"
            self.report({"WARNING"}, "No webcams detected on indices 0-5")
        else:
            s.cameras_found = ", ".join(str(i) for i in found)
            self.report({"INFO"}, "Cameras: " + s.cameras_found)
            if multi_enabled(s) and len(s.cameras) < 2:
                props_mod.ensure_camera_slots(s, min(len(found), 2) or 2)
        return {"FINISHED"}


def _current_bone_map(scene):
    b = scene.fpv_pose.bones
    return {slot: getattr(b, slot, "") for slot in mapping.BONE_SLOTS}


def current_bone_map(scene):
    """Public alias: the clip operators build the same map for replay."""
    return _current_bone_map(scene)


def build_retargeter(scene, armature=None, vis_thr=None):
    """A Retargeter configured from the scene settings.

    Live tracking and clip replay have to solve the rig identically -- the
    whole promise of an offline take is that it is the same result with
    better landmarks -- so both come through here rather than each
    spelling out thirty keyword arguments.
    """
    s = scene.fpv_pose
    armature = armature if armature is not None else s.armature
    return retarget_mod.Retargeter(
        armature, _current_bone_map(scene),
        smoothing=float(s.smoothing),
        pose_smoothing=float(s.pose_smoothing),
        pose_visibility=float(s.pose_visibility if vis_thr is None else vis_thr),
        view_mode=s.view_mode,
        one_euro_min_cutoff=float(s.one_euro_min_cutoff),
        one_euro_beta=float(s.one_euro_beta),
        use_arm_ik=bool(s.use_arm_ik),
        freeze_shoulder=bool(s.freeze_shoulder),
        shoulder_mode=s.shoulder_mode,
        shoulder_protract=float(s.shoulder_protract),
        track_palm=bool(s.track_palm),
        anchor_mode=s.anchor_mode,
        elbow_from_angle=bool(s.elbow_from_angle),
        auto_calibrate=bool(s.auto_calibrate),
        calibration_delay=float(s.calibration_delay),
        calibration_frames=int(s.calibration_frames),
        hand_depth_mode=s.hand_depth_mode,
        pose_source=s.pose_source,
        verbose_hand_log=bool(s.verbose_hand_log),
        body_shoulder=float(s.body_shoulder_width),
        body_upper=float(s.body_upper_arm),
        body_fore=float(s.body_forearm),
    )


def _update_preview_image(rgba_np, name=PREVIEW_IMAGE_NAME):
    """Push the annotated frame into a Blender Image datablock and refresh
    its preview so panel drawing shows the latest frame.

    `name` keeps the live tracker and the clip previewer in separate
    datablocks: they can be on screen at once, and sharing one would make
    each overwrite the other every frame.
    """
    try:
        h, w, _ = rgba_np.shape
    except Exception:
        return
    img = bpy.data.images.get(name)
    if img is None or img.size[0] != w or img.size[1] != h:
        if img is not None:
            try:
                bpy.data.images.remove(img)
            except Exception:
                pass
        img = bpy.data.images.new(name, w, h, alpha=True)
    # Blender images store pixels bottom-up; MediaPipe/OpenCV frames are
    # top-down. Flip vertically here.
    try:
        import numpy as np
        flat = (rgba_np[::-1].astype(np.float32) / 255.0).ravel()
        img.pixels.foreach_set(flat)
    except Exception as e:
        print("[FPV] preview update failed:", e)
        return
    # Push into the preview icon so template_icon() shows fresh pixels.
    try:
        if not img.preview:
            img.preview_ensure()
        prev = img.preview
        prev.image_size = (w, h)
        prev.image_pixels_float.foreach_set(flat)
    except Exception:
        pass
    img.update()


def _record_path(s):
    import os
    import tempfile
    p = bpy.path.abspath(s.record_path) if s.record_path else ""
    if not p:
        base = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else tempfile.gettempdir()
        p = os.path.join(base, "fpv_pose_diag.jsonl")
    return p


def _tick_interval(tr):
    """Poll a bit faster than frames arrive.

    A fixed 30 Hz tick throws away everything above 30 fps and adds up to a
    frame of latency below it; polling at ~2.5x the measured rate picks each
    frame up promptly without spinning.
    """
    fps = tr.fps() if tr is not None else 0.0
    if fps <= 1.0:
        fps = 30.0
    return max(1.0 / 240.0, min(1.0 / 20.0, 1.0 / (fps * 2.5)))


def _timer_tick():
    tr = _STATE.get("tracker")
    rt = _STATE.get("retargeter")
    if tr is None:
        return None
    scene = bpy.context.scene
    s = scene.fpv_pose

    err = tr.error()
    if err:
        s.status = "error: " + err
        _stop_internal()
        return None

    frame = tr.latest_frame()
    # Re-retargeting a frame we already applied would burn time and, worse,
    # feed the recorder duplicate samples that read as zero motion.
    if frame is not None and frame.timestamp == _STATE.get("last_frame_t"):
        return _tick_interval(tr)
    if frame is None:
        s.status = tr.state()
    else:
        _STATE["last_frame_t"] = frame.timestamp
        try:
            rt.apply_frame(frame)
        except Exception as e:
            s.status = "retarget error: %s" % e
            print("[FPV Pose Tracker] retarget error:", e)
            return 0.2
        if recording.is_recording():
            try:
                full = not recording.capture(rt, frame.timestamp)
            except Exception as e:
                print("[FPV Pose Tracker] capture error:", e)
                _finish_take(scene)
                s.status = "recording stopped: capture error"
            else:
                tk = recording.capturing()
                if tk is not None:
                    s.rec_count = len(tk.samples)
                    s.rec_seconds = tk.duration()
                if full:
                    # Ending the take beats silently dropping frames; the
                    # panel shows it as ready to review.
                    _finish_take(scene)
                    s.status = "recording stopped: take is full"
        det = rt.last_detection()
        s.detect_pose = det["pose"]
        s.detect_left = det["left"]
        s.detect_right = det["right"]
        s.bones_applied = rt.last_bones_applied()
        s.fps = tr.fps()
        s.status = "%s (%.1f fps)" % (tr.state(), s.fps)
        s.calib_status = rt.calib_status()
        eye = rt.eye_offset_ratio()
        s.eye_offset_info = ("" if eye is None else
                             "eye %.2f up, %.2f fwd (x shoulder width)" % eye)
        tm = tr.timings()
        s.perf_info = ("tot %.0f | grab %.0f  hand %.0f  pose %.0f  ovl %.0f | "
                       "dropped %d" % (
                           tm.get("total", 0.0), tm.get("grab", 0.0),
                           tm.get("hands", 0.0), tm.get("pose", 0.0),
                           tm.get("overlay", 0.0), int(tm.get("dropped", 0))))
        if not s.engine_info and hasattr(tr, "engine_info"):
            s.engine_info = tr.engine_info()
        ang = getattr(frame, "joint_angles", None)
        if ang is not None:
            s.angle_info = ang.summary()
        if not s.cap_info:
            s.cap_info = tr.cap_info()
            s.cap_warning = tr.cap_warning()
        if multi_enabled(s) and hasattr(tr, "unit_fps"):
            # Mirror each camera's real rate onto its row so the device list
            # shows which one is holding the pair back.
            rates = tr.unit_fps()
            for dev, rate in zip(enabled_cameras(s), rates):
                dev.measured_fps = rate
        if hasattr(tr, "calib_status"):
            s.multi_calib_info = tr.calib_status()
            if tr.is_calibrating():
                s.multi_calib_info = "%s  [%.0f%%]" % (
                    s.multi_calib_info, 100.0 * tr.calib_progress())
            elif getattr(tr, "_cams", None) and "calibrated" in tr.calib_status():
                # Only a measured placement is worth keeping; the Yaw guess
                # is recreated from the sliders every run.
                _save_extrinsics(s, tr)
        if hasattr(tr, "per_camera_info"):
            # Refreshed every tick, not cached like cap_info: a camera that
            # degrades mid-session (or dies) is exactly what this reports.
            s.multi_info = "\n".join(tr.per_camera_info())
        b = tr.brightness()
        s.cap_brightness = -1.0 if b is None else b
        if s.show_inline_preview and frame.annotated_rgba is not None:
            _update_preview_image(frame.annotated_rgba)

    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            area.tag_redraw()
    return _tick_interval(tr)


def _finish_take(scene):
    """End the take being captured and set the review state up for it."""
    s = scene.fpv_pose
    tk = recording.finish()
    s.is_recording = False
    if tk is None:
        # Nothing was captured. Only clear the review state if there is no
        # earlier take sitting there waiting to be saved.
        if not recording.has_take():
            s.take_present = False
            scene.fpv_take_frames.clear()
        return None
    s.take_present = True
    s.rec_count = len(tk.samples)
    s.rec_seconds = tk.duration()
    recording.sync_rows(scene)
    s.take_start = 0
    s.take_end = max(0, len(tk.samples) - 1)
    s.take_active_index = 0
    if s.take_auto_detect:
        _detect(scene)
    else:
        recording.compute_scores(tk)
        s.take_threshold_info = ""
    return tk


def finish_take(scene):
    """Public alias: a clip replay ends its take the same way live does."""
    return _finish_take(scene)


def _detect(scene):
    s = scene.fpv_pose
    tk = recording.take()
    if tk is None:
        return 0
    thr, flagged = recording.detect_outliers(
        tk,
        sensitivity=float(s.take_outlier_sensitivity),
        min_jump=float(s.take_outlier_min),
        flag_lost=bool(s.take_flag_lost),
        lo=int(s.take_start), hi=int(s.take_end),
    )
    dropped = recording.apply_flags(scene, bool(s.take_auto_exclude))
    s.take_threshold_info = "threshold %.1f%% of shoulder width" % thr
    return flagged, dropped


def live_calibration():
    """The calibration from the live session, current or last, or None.

    Calibrating live is the one case that works without any setup: the
    performer is at the camera and can be asked to hold the rest pose. So
    it is worth keeping the result after the tracker is gone -- it is the
    cheapest way to prime a clip replay, which has no such moment.
    """
    rt = _STATE.get("retargeter")
    if rt is not None:
        data = rt.export_calibration()
        if data:
            return data
    return _STATE.get("last_calibration")


def _stop_internal():
    tr = _STATE.get("tracker")
    if tr is not None:
        tr.stop()
    rt = _STATE.get("retargeter")
    if rt is not None:
        try:
            _STATE["last_calibration"] = rt.export_calibration()
        except Exception:
            pass
    _STATE["tracker"] = None
    _STATE["retargeter"] = None
    _STATE["last_frame_t"] = None
    recording.set_live(False)
    if _STATE.get("timer") is not None:
        try:
            bpy.app.timers.unregister(_STATE["timer"])
        except Exception:
            pass
        _STATE["timer"] = None


class FPV_OT_camera_add(bpy.types.Operator):
    bl_idname = "fpv_pose.camera_add"
    bl_label = "Add Camera"
    bl_description = "Add another webcam to the multi-camera rig"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        s = context.scene.fpv_pose
        props_mod.ensure_camera_slots(s, len(s.cameras) + 1)
        s.cameras_active_index = len(s.cameras) - 1
        return {"FINISHED"}


class FPV_OT_camera_remove(bpy.types.Operator):
    bl_idname = "fpv_pose.camera_remove"
    bl_label = "Remove Camera"
    bl_description = "Remove this webcam from the multi-camera rig"
    bl_options = {"REGISTER", "UNDO"}

    slot: bpy.props.IntProperty(default=-1)

    def execute(self, context):
        s = context.scene.fpv_pose
        i = self.slot if self.slot >= 0 else len(s.cameras) - 1
        if not (0 <= i < len(s.cameras)):
            return {"CANCELLED"}
        s.cameras.remove(i)
        s.cameras_active_index = max(0, min(s.cameras_active_index,
                                            len(s.cameras) - 1))
        return {"FINISHED"}


class FPV_OT_start(bpy.types.Operator):
    bl_idname = "fpv_pose.start"
    bl_label = "Start Tracking"

    def execute(self, context):
        s = context.scene.fpv_pose
        if _STATE.get("tracker") is not None:
            self.report({"WARNING"}, "Already tracking")
            return {"CANCELLED"}
        ok_deps, dep_msg = dependencies.engine_status(s.pose_engine)
        if not ok_deps:
            self.report({"ERROR"},
                        "%s engine cannot run (%s). Use 'Install Dependencies' "
                        "first." % (s.pose_engine, dep_msg))
            return {"CANCELLED"}
        # A ROI hand model needs a body pose to place its crops, and the
        # MediaPipe hand pairing needs MediaPipe itself.
        hand_src = pose_engines.resolve_hand_source(
            s.pose_engine, s.engine_hands, s.engine_layout)
        if hand_src == "MEDIAPIPE" and s.pose_engine != "MEDIAPIPE" \
                and dependencies.missing_modules("MEDIAPIPE"):
            self.report({"ERROR"},
                        "Hand Source is MediaPipe Hands but MediaPipe is not "
                        "installed. Install it, or pick another hand source.")
            return {"CANCELLED"}
        if hand_src == "RTM" and s.tracking_mode == "HANDS" \
                and not s.engine_hand_detector:
            # The ROI hand model gets its crops from the body pose, and
            # Hands Only never runs one, so without the fallback detector
            # there is nothing to tell it where to look.
            self.report({"ERROR"},
                        "Hands Only with the ROI hand model needs 'Hand "
                        "Fallback Detector' on — there is no body pose to "
                        "place the hand crops.")
            return {"CANCELLED"}
        if hand_src == "WHOLEBODY" and s.tracking_mode == "HANDS":
            self.report({"ERROR"},
                        "Hand Source 'From Body Model' needs the body model "
                        "to run — use Body + Hands, or pick another hand "
                        "source.")
            return {"CANCELLED"}
        if not s.armature:
            self.report({"ERROR"}, "Pick an armature")
            return {"CANCELLED"}

        bone_map = _current_bone_map(context.scene)
        if not any(bone_map.values()):
            self.report({"ERROR"}, "Bone map is empty. Run Auto-Map first.")
            return {"CANCELLED"}

        try:
            context.view_layer.objects.active = s.armature
            if s.armature.mode != "POSE":
                bpy.ops.object.mode_set(mode="POSE")
        except Exception:
            pass

        common = dict(
            mirror=bool(s.mirror),
            model_complexity=int(s.model_complexity),
            tracking_mode=s.tracking_mode,
            capture_width=int(s.capture_width),
            capture_height=int(s.capture_height),
            pose_y_up=bool(s.pose_y_up),
            view_mode=s.view_mode,
            lift_depth=(s.pose_source == "LIFT"),
            pose_visibility=float(s.pose_visibility),
            hand_depth=bool(s.hand_depth_from_size),
            hand_size=float(s.hand_size),
            hand_size_auto=bool(s.hand_size_auto),
            target_fps=int(s.target_fps),
            use_mjpg=bool(s.use_mjpg),
            exposure_mode=s.exposure_mode,
            exposure=float(s.exposure),
            gain=float(s.gain),
            infer_width=int(s.infer_width),
            pose_every=int(s.pose_every),
            parallel_infer=bool(s.parallel_infer),
            delegate=s.mp_delegate,
            pose_engine=s.pose_engine,
            engine_quality=s.engine_quality,
            engine_layout=s.engine_layout,
            engine_hands=s.engine_hands,
            engine_device=s.engine_device,
            engine_det_every=int(s.engine_det_every),
            engine_kpt_thr=float(s.engine_kpt_thr),
            engine_hand_detector=bool(s.engine_hand_detector),
            body_shoulder=float(s.body_shoulder_width),
            body_upper=float(s.body_upper_arm),
            body_fore=float(s.body_forearm),
        )
        caps = pose_engines.capabilities(s.pose_engine)
        if not caps.world_pose and s.pose_source == "WORLD":
            # The engine has no world pose to read, so the depth lift is the
            # only source of 3D there is. Switching it here rather than
            # refusing to start keeps a saved MediaPipe-era scene working.
            s.pose_source = "LIFT"
            self.report({"WARNING"},
                        "%s is a 2D engine — Arm Depth switched to "
                        "'2D + Limb Lengths'" % s.pose_engine)
        common["lift_depth"] = (s.pose_source == "LIFT")
        preview_fps = float(s.preview_fps) if (s.show_preview_window
                                               or s.show_inline_preview) else 0.0
        record_path = _record_path(s) if s.record_diagnostics else None

        if multi_enabled(s):
            specs = _camera_specs(s)
            if not specs:
                self.report({"ERROR"},
                            "Multi-camera is selected but no camera is ticked. "
                            "Add devices under Webcam, or pick a single camera.")
                return {"CANCELLED"}
            if len(specs) < 2:
                # One camera through the fusion path works, but it is the
                # single-camera result with extra machinery, so say so.
                self.report({"WARNING"},
                            "Only one camera enabled — fusion needs two to "
                            "improve on a single webcam")
            tr = multicam_mod.MultiCamTracker(
                specs,
                depth_trust=float(s.multi_depth_trust),
                sync_window=float(s.multi_sync_ms) / 1000.0,
                solver=str(s.multi_solver),
                bone_stiffness=float(s.multi_bone_stiffness),
                extrinsics=_load_extrinsics(s),
                show_window=bool(s.show_preview_window),
                preview_fps=preview_fps,
                record_path=record_path,
                **common
            )
            s.multi_info = "%d cameras: %s" % (
                len(specs), ", ".join("idx %d @ %+.0f deg" % (sp.index, sp.yaw)
                                      for sp in specs))
        else:
            tr = tracker_mod.WebcamTracker(
                camera_index=int(s.camera_index),
                camera_fov=float(s.camera_fov),
                show_window=bool(s.show_preview_window),
                preview_fps=preview_fps,
                record_path=record_path,
                **common
            )
            s.multi_info = ""
        # A narrow FOV means more of the body sits off-frame, so bump the
        # visibility gate to require more confidence before driving arms.
        vis_thr = float(s.pose_visibility)
        narrowest = min([float(d.fov) for d in enabled_cameras(s)] or [180.0])             if multi_enabled(s) else float(s.camera_fov)
        if narrowest < 50.0:
            vis_thr = min(0.9, vis_thr + 0.15)
        rt = build_retargeter(context.scene, s.armature, vis_thr)
        # A preview action would be evaluated on top of every tick, so put
        # the rig back under the tracker's control before it starts.
        recording.clear_preview(s.armature)
        recording.set_live(True)
        if s.reset_pose_on_start:
            rt.reset_pose()
        tr.start()
        _STATE["tracker"] = tr
        _STATE["retargeter"] = rt
        _STATE["timer"] = _timer_tick
        _STATE["last_frame_t"] = None
        s.cap_info = ""
        s.perf_info = ""
        s.engine_info = ""
        s.angle_info = ""
        s.cap_warning = ""
        s.cap_brightness = -1.0
        bpy.app.timers.register(_timer_tick, first_interval=0.1)
        s.status = "starting..."
        s.calib_status = rt.calib_status()
        if s.record_on_start:
            recording.start(rt, s.armature)
            s.is_recording = True
            s.rec_count = 0
            s.rec_seconds = 0.0
        self.report({"INFO"}, "Tracking started")
        return {"FINISHED"}


class FPV_OT_calibrate(bpy.types.Operator):
    bl_idname = "fpv_pose.calibrate"
    bl_label = "Calibrate"
    bl_description = ("Reset the rig to rest and start the calibration countdown. "
                      "Match the rig's rest pose (FPS arms: arms forward toward the "
                      "camera, palms down) and hold still until the status reads 'calibrated'")

    def execute(self, context):
        rt = _STATE.get("retargeter")
        if rt is None:
            self.report({"ERROR"}, "Start tracking first")
            return {"CANCELLED"}
        rt.request_calibration(reset_pose=True)
        context.scene.fpv_pose.calib_status = rt.calib_status()
        self.report({"INFO"}, "Calibrating: hold the rig's rest pose")
        return {"FINISHED"}


class FPV_OT_calibrate_rig(bpy.types.Operator):
    bl_idname = "fpv_pose.calibrate_rig"
    bl_label = "Calibrate Placement"
    bl_description = ("Measure where your cameras actually stand, from your own "
                      "body. Move your arms around your whole working range for "
                      "ten seconds or so. This is what lets two cameras measure "
                      "depth instead of guessing it, and it is far more accurate "
                      "than the Yaw sliders. The result is saved with the scene; "
                      "re-run it if you move a camera")

    def execute(self, context):
        tr = _STATE.get("tracker")
        if tr is None or not hasattr(tr, "request_calibration"):
            self.report({"ERROR"}, "Start tracking with two or more cameras first")
            return {"CANCELLED"}
        if len(getattr(tr, "specs", ())) < 2:
            self.report({"ERROR"},
                        "Camera placement only means something with two cameras")
            return {"CANCELLED"}
        s = context.scene.fpv_pose
        # A stored placement would otherwise sit in front of the new one.
        s.multi_extrinsics = ""
        tr.request_calibration()
        s.multi_calib_info = tr.calib_status()
        self.report({"INFO"},
                    "Measuring camera placement: move your arms through their "
                    "whole range")
        return {"FINISHED"}


class FPV_OT_clear_rig_calibration(bpy.types.Operator):
    bl_idname = "fpv_pose.clear_rig_calibration"
    bl_label = "Clear Placement"
    bl_description = ("Forget the measured camera placement and go back to the "
                      "Yaw sliders. Use this after moving a camera if you cannot "
                      "re-calibrate straight away")

    def execute(self, context):
        s = context.scene.fpv_pose
        s.multi_extrinsics = ""
        s.multi_calib_info = "cleared - restart tracking to use Yaw"
        tr = _STATE.get("tracker")
        if tr is not None and hasattr(tr, "cancel_calibration"):
            tr.cancel_calibration()
        self.report({"INFO"}, "Camera placement cleared")
        return {"FINISHED"}


class FPV_OT_fit_camera(bpy.types.Operator):
    bl_idname = "fpv_pose.fit_camera"
    bl_label = "Fit Camera to FPV View"
    bl_options = {"REGISTER", "UNDO"}
    bl_description = ("Move the scene camera to the rig's eye position, looking "
                      "the way the arms point, so the view reads as first-person. "
                      "Adjust the offsets in the Adjust Last Operation panel.")

    eye_height: bpy.props.FloatProperty(
        name="Eye Above Shoulders", default=0.12, min=-0.5, max=1.0, unit="LENGTH",
        description="How far above the shoulder line the eye sits. On a human "
                    "this is roughly 0.10-0.15 m.",
    )
    eye_forward: bpy.props.FloatProperty(
        name="Eye Forward", default=0.05, min=-0.5, max=1.0, unit="LENGTH",
        description="How far forward of the shoulder line the eye sits. Small "
                    "and positive: the shoulders should end up just behind the "
                    "camera, which is why you do not see them in first person.",
    )
    from_tracking: bpy.props.BoolProperty(
        name="Use Measured Eye Offset", default=True,
        description="If a tracking session has measured where your eyes sit "
                    "relative to your shoulders, scale that by the rig's "
                    "shoulder width and use it instead of the offsets below. "
                    "Needs tracking to have been running with your face visible.",
    )
    set_fov: bpy.props.BoolProperty(
        name="Set Field of View", default=False,
        description="Also set the camera's horizontal FOV. First-person views "
                    "usually run wider than a default 50 mm lens.",
    )
    hfov: bpy.props.FloatProperty(
        name="Horizontal FOV", default=90.0, min=30.0, max=140.0,
        description="Horizontal field of view in degrees",
    )

    def execute(self, context):
        import math
        from mathutils import Matrix, Vector
        scene = context.scene
        s = scene.fpv_pose
        cam = scene.camera
        if cam is None:
            self.report({"ERROR"}, "Scene has no active camera")
            return {"CANCELLED"}
        arm = s.armature
        if arm is None:
            self.report({"ERROR"}, "Pick an armature first")
            return {"CANCELLED"}

        bones = _current_bone_map(scene)
        ua_l = arm.pose.bones.get(bones.get("upper_arm_L", ""))
        ua_r = arm.pose.bones.get(bones.get("upper_arm_R", ""))
        if ua_l is None or ua_r is None:
            self.report({"ERROR"}, "Map both upper arm bones first")
            return {"CANCELLED"}

        # Rig frame from the rig itself rather than assumed axes.
        head_l = ua_l.bone.head_local
        head_r = ua_r.bone.head_local
        centre = (head_l + head_r) * 0.5
        right = (head_r - head_l)
        if right.length < 1e-5:
            self.report({"ERROR"}, "Upper arm bones are in the same place")
            return {"CANCELLED"}
        right.normalize()
        up = Vector((0.0, 0.0, 1.0))
        up = (up - right * up.dot(right))
        if up.length < 1e-5:
            up = Vector((0.0, 1.0, 0.0))
        up.normalize()
        forward = up.cross(right).normalized()

        # Which way the arms actually point decides forward's sign, so a rig
        # built facing either way lands the same.
        h_l = arm.pose.bones.get(bones.get("hand_L", ""))
        h_r = arm.pose.bones.get(bones.get("hand_R", ""))
        if h_l is not None and h_r is not None:
            hands = (h_l.bone.head_local + h_r.bone.head_local) * 0.5
            if (hands - centre).dot(forward) < 0.0:
                forward = -forward
                right = -right

        # Prefer the user's own measured build over the guessed offsets.
        h, f = self.eye_height, self.eye_forward
        measured = None
        rt = _STATE.get("retargeter")
        if self.from_tracking and rt is not None:
            measured = rt.eye_offset_ratio()
        if measured is not None:
            width = (head_r - head_l).length
            h, f = measured[0] * width, measured[1] * width

        eye_local = centre + up * h + forward * f
        M = arm.matrix_world
        eye_world = M @ eye_local
        R = M.to_3x3()
        f_w = (R @ forward).normalized()
        u_w = (R @ up).normalized()
        # Blender cameras look down local -Z with +Y up.
        cam_z = -f_w
        cam_y = (u_w - cam_z * u_w.dot(cam_z)).normalized()
        cam_x = cam_y.cross(cam_z).normalized()
        basis = Matrix((cam_x, cam_y, cam_z)).transposed()

        cam.matrix_world = Matrix.Translation(eye_world) @ basis.to_4x4()
        if self.set_fov and cam.data.type == "PERSP":
            cam.data.sensor_fit = "HORIZONTAL"
            cam.data.lens = cam.data.sensor_width / (
                2.0 * math.tan(math.radians(self.hfov) * 0.5))

        # Report where the shoulders and hands ended up, since "behind the
        # camera" is the whole point and is easy to get wrong.
        inv = cam.matrix_world.inverted()
        def depth(pb):
            return -(inv @ (M @ pb.bone.head_local)).z
        src = "measured" if measured is not None else "manual offsets"
        msg = ("Camera at eye (%s, up %.2f fwd %.2f): shoulders %+.2f m, "
               "hands %+.2f m in front"
               % (src, h, f, 0.5 * (depth(ua_l) + depth(ua_r)),
                  0.5 * (depth(h_l) + depth(h_r)) if (h_l and h_r) else float("nan")))
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class FPV_OT_reset_pose(bpy.types.Operator):
    bl_idname = "fpv_pose.reset_pose"
    bl_label = "Reset Mapped Bones"
    bl_description = "Return every mapped bone to its rest transform (rotation, location and scale)"

    def execute(self, context):
        s = context.scene.fpv_pose
        if not s.armature:
            self.report({"ERROR"}, "Pick an armature first")
            return {"CANCELLED"}
        rt = _STATE.get("retargeter")
        if rt is not None:
            rt.reset_pose()
        else:
            tmp = retarget_mod.Retargeter(s.armature, _current_bone_map(context.scene),
                                          auto_calibrate=False)
            tmp.reset_pose()
        return {"FINISHED"}


class FPV_OT_stop(bpy.types.Operator):
    bl_idname = "fpv_pose.stop"
    bl_label = "Stop Tracking"

    def execute(self, context):
        was_recording = recording.is_recording()
        _stop_internal()
        s = context.scene.fpv_pose
        s.status = "stopped"
        s.fps = 0.0
        s.detect_pose = s.detect_left = s.detect_right = False
        s.bones_applied = 0
        s.calib_status = "not calibrated"
        s.cap_brightness = -1.0
        if was_recording:
            # Stopping the camera ends the take too, rather than leaving a
            # recording that can never receive another frame.
            tk = _finish_take(context.scene)
            if tk is not None:
                recording.refresh_preview(context, force=True)
                _open_review(context, self)
                self.report({"INFO"}, "Tracking stopped — %d frames recorded" % len(tk.samples))
                return {"FINISHED"}
        recording.refresh_preview(context, force=True)
        self.report({"INFO"}, "Tracking stopped")
        return {"FINISHED"}


# ----------------------------------------------------------------------
# animation recording
# ----------------------------------------------------------------------

class FPV_OT_record_start(bpy.types.Operator):
    bl_idname = "fpv_pose.record_start"
    bl_label = "Record Animation"
    bl_description = ("Start capturing every tracked frame as an editable take. "
                      "Nothing is written to an Action until you review and save it.")

    @classmethod
    def poll(cls, context):
        return _STATE.get("retargeter") is not None and not recording.is_recording()

    def execute(self, context):
        s = context.scene.fpv_pose
        rt = _STATE.get("retargeter")
        if rt is None:
            self.report({"ERROR"}, "Start tracking first")
            return {"CANCELLED"}
        if recording.has_take():
            self.report({"WARNING"},
                        "Save or discard the current take before recording another")
            return {"CANCELLED"}
        recording.clear_preview(s.armature)
        recording.start(rt, s.armature)
        s.is_recording = True
        s.rec_count = 0
        s.rec_seconds = 0.0
        self.report({"INFO"}, "Recording")
        return {"FINISHED"}


class FPV_OT_record_stop(bpy.types.Operator):
    bl_idname = "fpv_pose.record_stop"
    bl_label = "Stop Recording"
    bl_description = "End the take and open it for review"

    @classmethod
    def poll(cls, context):
        return recording.is_recording()

    def execute(self, context):
        tk = _finish_take(context.scene)
        if tk is None:
            self.report({"WARNING"}, "Nothing was captured")
            return {"CANCELLED"}
        recording.refresh_preview(context, force=True)
        _open_review(context, self)
        self.report({"INFO"}, "Recorded %d frames (%.1fs)" % (len(tk.samples), tk.duration()))
        return {"FINISHED"}


def _open_review(context, op=None):
    """Show the take review UI, in its own window when asked for."""
    s = context.scene.fpv_pose
    if not s.take_open_window:
        return
    try:
        bpy.ops.fpv_pose.open_review_window()
    except Exception as e:
        print("[FPV Pose Tracker] review window failed:", e)
        if op is not None:
            op.report({"INFO"}, "Take ready — see the Take Review panel in the sidebar")


class FPV_OT_open_review_window(bpy.types.Operator):
    bl_idname = "fpv_pose.open_review_window"
    bl_label = "Open Review Window"
    bl_description = ("Open a separate window for editing the take: the rig on "
                      "top, the take's keyframes below, review controls in the "
                      "sidebar under the FPV Pose tab")

    def _find_v3d(self, context):
        if context.area is not None and context.area.type == "VIEW_3D":
            return context.window, context.area
        for win in context.window_manager.windows:
            for area in win.screen.areas:
                if area.type == "VIEW_3D":
                    return win, area
        return None, None

    def execute(self, context):
        wm = context.window_manager
        win, area = self._find_v3d(context)
        if area is None:
            self.report({"ERROR"}, "Need a 3D Viewport open to clone a window from")
            return {"CANCELLED"}

        before = len(wm.windows)
        region = next((r for r in area.regions if r.type == "WINDOW"), None)
        with context.temp_override(window=win, area=area, region=region):
            bpy.ops.wm.window_new()
        if len(wm.windows) <= before:
            self.report({"ERROR"}, "Could not open a new window")
            return {"CANCELLED"}

        new_win = wm.windows[-1]
        new_win.screen.areas[0].type = "VIEW_3D"
        _layout_review_window(new_win.screen.name)
        self.report({"INFO"},
                    "Review window open — controls under the sidebar's FPV Pose tab")
        return {"FINISHED"}


def _layout_review_window(screen_name):
    """Give the review window a viewport over an Action Editor.

    Done from a timer rather than inline: right after `area_split` the areas
    have not been laid out yet and still report their pre-split geometry, so
    deciding which one is the bottom half immediately gets it wrong.
    """
    state = {"split_done": False, "typed": False, "tries": 0}

    def step():
        state["tries"] += 1
        if state["tries"] > 40:
            return None
        screen = bpy.data.screens.get(screen_name)
        if screen is None:
            return None
        win = next((w for w in bpy.context.window_manager.windows
                    if w.screen == screen), None)
        if win is None:
            return None

        if not state["split_done"]:
            if len(screen.areas) < 2:
                area = screen.areas[0]
                region = next((r for r in area.regions if r.type == "WINDOW"), None)
                try:
                    with bpy.context.temp_override(window=win, screen=screen,
                                                   area=area, region=region):
                        bpy.ops.screen.area_split(direction="HORIZONTAL", factor=0.3)
                except Exception as e:
                    print("[FPV Pose Tracker] area split failed:", e)
                    return None
                return 0.1
            state["split_done"] = True
            return 0.1

        # Wait for the layout: until it runs, both areas report the same
        # position and there is no bottom half to find.
        if len({a.y for a in screen.areas}) < 2:
            return 0.1

        if not state["typed"]:
            bottom = min(screen.areas, key=lambda a: a.y)
            if bottom.type != "DOPESHEET_EDITOR":
                bottom.type = "DOPESHEET_EDITOR"
            state["typed"] = True
            return 0.1

        for a in screen.areas:
            try:
                if a.type == "DOPESHEET_EDITOR":
                    a.spaces.active.mode = "ACTION"
                    a.spaces.active.show_region_ui = True
                    region = next((r for r in a.regions if r.type == "WINDOW"), None)
                    with bpy.context.temp_override(window=win, screen=screen,
                                                   area=a, region=region):
                        bpy.ops.action.view_all()
                elif a.type == "VIEW_3D":
                    a.spaces.active.show_region_ui = True
            except Exception as e:
                print("[FPV Pose Tracker] review window setup:", e)
            a.tag_redraw()
        return None

    bpy.app.timers.register(step, first_interval=0.15)


class FPV_OT_take_scene_range(bpy.types.Operator):
    bl_idname = "fpv_pose.take_scene_range"
    bl_label = "Scene Range to Take"
    bl_description = "Set the scene's frame range to the frames the take covers"

    def execute(self, context):
        scene = context.scene
        if _require_take(self) is None:
            return {"CANCELLED"}
        _, _, _, _, lo, hi = recording.stats(scene)
        if hi <= lo:
            self.report({"WARNING"}, "Take covers no frames")
            return {"CANCELLED"}
        scene.frame_start = lo
        scene.frame_end = hi
        scene.frame_current = min(max(scene.frame_current, lo), hi)
        self.report({"INFO"}, "Scene range set to %d-%d" % (lo, hi))
        return {"FINISHED"}


# ----------------------------------------------------------------------
# take editing
# ----------------------------------------------------------------------

def _require_take(op):
    if not recording.has_take():
        op.report({"ERROR"}, "No take recorded")
        return None
    return recording.take()


def _row_for_frame(scene, frame):
    """The row whose timeline frame is nearest `frame`."""
    rows = scene.fpv_take_frames
    best, best_d = None, None
    for r in rows:
        d = abs(r.frame - frame)
        if best_d is None or d < best_d:
            best, best_d = r, d
    return best


class FPV_OT_take_detect(bpy.types.Operator):
    bl_idname = "fpv_pose.take_detect"
    bl_label = "Find Outliers"
    bl_description = ("Flag samples that sit off the path their neighbours "
                      "describe — where tracking lost the body and snapped the "
                      "rig somewhere it never went")

    def execute(self, context):
        if _require_take(self) is None:
            return {"CANCELLED"}
        flagged, dropped = _detect(context.scene)
        recording.refresh_preview(context, force=True)
        s = context.scene.fpv_pose
        self.report({"INFO"}, "%d flagged, %d dropped — %s"
                    % (flagged, dropped, s.take_threshold_info))
        return {"FINISHED"}


class FPV_OT_take_flagged_action(bpy.types.Operator):
    bl_idname = "fpv_pose.take_flagged_action"
    bl_label = "Flagged Samples"
    bl_options = {"INTERNAL"}

    mode: bpy.props.EnumProperty(items=[
        ("DROP", "Drop Flagged", "Remove every flagged sample from the take"),
        ("KEEP", "Keep Flagged", "Put every flagged sample back in the take"),
        ("RESET", "Keep All", "Put every sample back and clear the trim"),
    ])

    def execute(self, context):
        scene = context.scene
        tk = _require_take(self)
        if tk is None:
            return {"CANCELLED"}
        s = scene.fpv_pose
        rows = scene.fpv_take_frames
        n = 0
        if self.mode == "RESET":
            for r in rows:
                r.set_enabled_quiet(True)
            s.take_start = 0
            s.take_end = max(0, len(tk.samples) - 1)
            n = len(rows)
        else:
            want = self.mode == "KEEP"
            for r in rows:
                if r.flags and r.enabled != want:
                    r.set_enabled_quiet(want)
                    n += 1
        recording.mark_dirty()
        recording.refresh_preview(context, force=True)
        self.report({"INFO"}, "%d samples updated" % n)
        return {"FINISHED"}


class FPV_OT_take_set_trim(bpy.types.Operator):
    bl_idname = "fpv_pose.take_set_trim"
    bl_label = "Set Trim"
    bl_options = {"INTERNAL"}

    edge: bpy.props.EnumProperty(items=[
        ("START", "Start Here", "Cut everything before the playhead"),
        ("END", "End Here", "Cut everything after the playhead"),
    ])

    def execute(self, context):
        scene = context.scene
        if _require_take(self) is None:
            return {"CANCELLED"}
        row = _row_for_frame(scene, scene.frame_current)
        if row is None:
            self.report({"ERROR"}, "Take has no samples")
            return {"CANCELLED"}
        s = scene.fpv_pose
        if self.edge == "START":
            s.take_start = row.index
        else:
            s.take_end = row.index
        recording.refresh_preview(context, force=True)
        return {"FINISHED"}


class FPV_OT_take_toggle_current(bpy.types.Operator):
    bl_idname = "fpv_pose.take_toggle_current"
    bl_label = "Drop Sample at Playhead"
    bl_description = ("Remove (or restore) the sample nearest the playhead. "
                      "Scrub to a bad pose and press this to cut it out; the "
                      "curve interpolates across the hole.")

    def execute(self, context):
        scene = context.scene
        if _require_take(self) is None:
            return {"CANCELLED"}
        row = _row_for_frame(scene, scene.frame_current)
        if row is None:
            self.report({"ERROR"}, "Take has no samples")
            return {"CANCELLED"}
        row.set_enabled_quiet(not row.enabled)
        scene.fpv_pose.take_active_index = row.index
        recording.mark_dirty()
        recording.refresh_preview(context, force=True)
        self.report({"INFO"}, "Sample %d (frame %d) %s"
                    % (row.index, row.frame, "kept" if row.enabled else "dropped"))
        return {"FINISHED"}


class FPV_OT_take_jump(bpy.types.Operator):
    bl_idname = "fpv_pose.take_jump"
    bl_label = "Jump to Sample"
    bl_options = {"INTERNAL"}

    target: bpy.props.EnumProperty(items=[
        ("ACTIVE", "Active", "Playhead to the sample selected in the list"),
        ("NEXT", "Next Flagged", "Playhead to the next flagged sample"),
        ("PREV", "Previous Flagged", "Playhead to the previous flagged sample"),
    ])

    def execute(self, context):
        scene = context.scene
        if _require_take(self) is None:
            return {"CANCELLED"}
        rows = scene.fpv_take_frames
        s = scene.fpv_pose
        if self.target == "ACTIVE":
            idx = max(0, min(int(s.take_active_index), len(rows) - 1))
            if not rows:
                return {"CANCELLED"}
            scene.frame_set(rows[idx].frame)
            return {"FINISHED"}

        here = scene.frame_current
        flagged = [r for r in rows if r.flags]
        if not flagged:
            self.report({"INFO"}, "No flagged samples")
            return {"CANCELLED"}
        if self.target == "NEXT":
            nxt = next((r for r in flagged if r.frame > here), flagged[0])
        else:
            nxt = next((r for r in reversed(flagged) if r.frame < here), flagged[-1])
        scene.frame_set(nxt.frame)
        s.take_active_index = nxt.index
        self.report({"INFO"}, "Sample %d — %.1f%% off path (%s)"
                    % (nxt.index, nxt.score, recording.flag_text(nxt.flags)))
        return {"FINISHED"}


class FPV_OT_take_match_fps(bpy.types.Operator):
    bl_idname = "fpv_pose.take_match_fps"
    bl_label = "Match Scene FPS to Take"
    bl_description = ("Set the scene's frame rate to the rate the take was "
                      "captured at, so no sample has to be thrown away when it "
                      "is baked")

    def execute(self, context):
        scene = context.scene
        tk = _require_take(self)
        if tk is None:
            return {"CANCELLED"}
        fps = int(round(tk.capture_fps))
        if fps < 1:
            self.report({"WARNING"}, "Take has no usable capture rate")
            return {"CANCELLED"}
        scene.render.fps = fps
        scene.render.fps_base = 1.0
        recording.resync_frames(scene)
        recording.refresh_preview(context, force=True)
        self.report({"INFO"}, "Scene frame rate set to %d fps" % fps)
        return {"FINISHED"}


# Arm/hand smoothing weights and One-Euro settings that go together.
# Weights are "fraction retained per frame at 30 fps"; the retargeter
# rescales them to the real frame interval, so the time constants below
# hold at any capture rate.
_MOTION_PRESETS = {
    #            pose   hand  min_cutoff  beta
    "SMOOTH":   (0.90,  0.60, 1.0,        0.6),
    "BALANCED": (0.75,  0.45, 1.5,        1.0),
    "FAST":     (0.50,  0.25, 2.5,        2.0),
}


class FPV_OT_motion_preset(bpy.types.Operator):
    bl_idname = "fpv_pose.motion_preset"
    bl_label = "Motion Response"
    bl_options = {"INTERNAL"}

    mode: bpy.props.EnumProperty(items=[
        ("SMOOTH", "Smooth", "Heaviest filtering (~320 ms). Steadiest for slow, "
                             "deliberate movement and noisy off-frame arms, but a "
                             "punch will barely register"),
        ("BALANCED", "Balanced", "~120 ms. Follows normal gesturing closely while "
                                 "still hiding landmark jitter"),
        ("FAST", "Fast Action", "~50 ms. Keeps quick motion like a punch, at the "
                                "cost of showing every tracking wobble. Pair with a "
                                "high capture rate and a short exposure"),
    ])

    def execute(self, context):
        s = context.scene.fpv_pose
        pose, hand, cutoff, beta = _MOTION_PRESETS[self.mode]
        s.pose_smoothing = pose
        s.smoothing = hand
        s.one_euro_min_cutoff = cutoff
        s.one_euro_beta = beta
        rt = _STATE.get("retargeter")
        if rt is not None:
            # Apply live so the change is felt without a Stop -> Start.
            rt.pose_smoothing = pose
            rt.smoothing = hand
        self.report({"INFO"}, "Motion response: %s (arm response %.0f ms)"
                    % (self.mode.title(), 1000.0 * _tau(pose)))
        return {"FINISHED"}


def _tau(weight):
    """Seconds for a smoothing weight to close 63% of a gap."""
    import math
    if weight <= 0.0:
        return 0.0
    if weight >= 0.999:
        return float("inf")
    return -(1.0 / 30.0) / math.log(weight)


class FPV_OT_take_save(bpy.types.Operator):
    bl_idname = "fpv_pose.take_save"
    bl_label = "Save to Action"
    bl_description = ("Bake the kept samples into a new Action on the rig, and "
                      "optionally push it to an NLA track")

    def execute(self, context):
        scene = context.scene
        tk = _require_take(self)
        if tk is None:
            return {"CANCELLED"}
        s = scene.fpv_pose
        armature = s.armature
        if armature is None or armature.name != tk.armature_name:
            armature = bpy.data.objects.get(tk.armature_name)
        if armature is None:
            self.report({"ERROR"}, "Recorded armature '%s' is gone" % tk.armature_name)
            return {"CANCELLED"}

        indices = recording.included_indices(scene)
        if not indices:
            self.report({"ERROR"}, "Every sample is trimmed or dropped")
            return {"CANCELLED"}

        # Drop the preview first so the rig's own action slot is free and
        # the preview datablock is not left behind.
        recording.clear_preview(armature)

        name = s.take_action_name.strip() or "FPV Take"
        action, slot = recording.bake(
            tk, armature, name, indices,
            scene_fps=recording.scene_fps(scene),
            start_frame=int(s.take_start_frame),
            interpolation=s.take_interpolation,
        )
        if action is None:
            self.report({"ERROR"}, "Nothing to bake")
            return {"CANCELLED"}
        action.use_fake_user = True

        msg = "'%s': %d keyframes over frames %d-%d" % (
            action.name, len(indices), int(action.frame_range[0]), int(action.frame_range[1]))
        if s.take_push_nla:
            try:
                recording.push_nla(armature, action, slot,
                                   s.take_nla_track.strip() or "FPV Takes",
                                   int(action.frame_range[0]))
                msg += ", pushed to NLA track '%s'" % s.take_nla_track
            except Exception as e:
                print("[FPV Pose Tracker] NLA push failed:", e)
                recording.assign_action(armature, action, slot)
                msg += " (NLA push failed: %s)" % e
        else:
            recording.assign_action(armature, action, slot)

        recording.discard()
        scene.fpv_take_frames.clear()
        s.take_present = False
        s.take_threshold_info = ""
        self.report({"INFO"}, "Saved " + msg)
        return {"FINISHED"}


class FPV_OT_take_discard(bpy.types.Operator):
    bl_idname = "fpv_pose.take_discard"
    bl_label = "Discard Take"
    bl_description = "Throw the recording away without writing an Action"

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        scene = context.scene
        s = scene.fpv_pose
        armature = s.armature
        tk = recording.take()
        if tk is not None and (armature is None or armature.name != tk.armature_name):
            armature = bpy.data.objects.get(tk.armature_name)
        recording.clear_preview(armature)
        recording.discard()
        scene.fpv_take_frames.clear()
        s.take_present = False
        s.take_threshold_info = ""
        s.rec_count = 0
        s.rec_seconds = 0.0
        self.report({"INFO"}, "Take discarded")
        return {"FINISHED"}


_CLASSES = (
    FPV_OT_install_dependencies,
    FPV_OT_install_gpu_runtime,
    FPV_OT_engine_diagnostics,
    FPV_OT_reinstall_mediapipe,
    FPV_OT_auto_map,
    FPV_OT_clear_map,
    FPV_OT_list_cameras,
    FPV_OT_reset_camera,
    FPV_OT_autotune_exposure,
    FPV_OT_list_bones,
    FPV_OT_test_webcam,
    FPV_OT_camera_add,
    FPV_OT_camera_remove,
    FPV_OT_start,
    FPV_OT_stop,
    FPV_OT_calibrate,
    FPV_OT_calibrate_rig,
    FPV_OT_clear_rig_calibration,
    FPV_OT_reset_pose,
    FPV_OT_fit_camera,
    FPV_OT_record_start,
    FPV_OT_record_stop,
    FPV_OT_open_review_window,
    FPV_OT_take_detect,
    FPV_OT_take_flagged_action,
    FPV_OT_take_set_trim,
    FPV_OT_take_toggle_current,
    FPV_OT_take_jump,
    FPV_OT_take_scene_range,
    FPV_OT_take_match_fps,
    FPV_OT_motion_preset,
    FPV_OT_take_save,
    FPV_OT_take_discard,
)


def register():
    for c in _CLASSES:
        bpy.utils.register_class(c)


def unregister():
    _stop_internal()
    for c in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(c)
        except Exception:
            pass
