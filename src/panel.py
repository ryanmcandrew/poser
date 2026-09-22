"""N-panel UI in the 3D View sidebar."""

import bpy

from . import dependencies
from . import pose_engines
from . import clip_ops
from . import cliplib
from . import operators as ops
from . import recording


_PANEL_BUILD = "1.26.0"


class _Ready:
    """Common gate for every section panel below the root one.

    OpenCV and numpy are what the whole add-on needs before any of these
    controls mean anything; the root panel owns that message, so the
    section panels simply disappear until it is fixed.
    """

    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "FPV Pose"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return not dependencies.missing_base_modules()


class FPV_PT_main(bpy.types.Panel):
    bl_label = "FPV Pose Tracker  (build " + _PANEL_BUILD + ")"
    bl_idname = "FPV_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "FPV Pose"
    bl_order = 0

    def draw(self, context):
        layout = self.layout
        s = context.scene.fpv_pose

        layout.label(text="panel build: " + _PANEL_BUILD, icon="INFO")

        _draw_black_frame_warning(layout, s)

        # Only OpenCV and numpy gate the whole panel; which estimator is
        # installed is the engine block's business, because a user running
        # RTMPose has no reason to install MediaPipe as well.
        base_missing = dependencies.missing_base_modules()
        if base_missing:
            box = layout.box()
            box.label(text="Missing " + ", ".join(n for n, _ in base_missing),
                      icon="ERROR")
            box.operator("fpv_pose.install_dependencies", icon="IMPORT")
            box.operator("fpv_pose.engine_diagnostics", icon="INFO")
            return

        col = layout.column(align=True)
        col.prop(s, "armature")
        col.prop(s, "tracking_mode")


class FPV_PT_cameras(_Ready, bpy.types.Panel):
    """Which devices are used, how they are framed, and how they capture."""

    bl_label = "Cameras"
    bl_idname = "FPV_PT_cameras"
    bl_order = 1

    def draw(self, context):
        layout = self.layout
        s = context.scene.fpv_pose

        col = layout.column(align=True)
        row = col.row(align=True)
        row.prop(s, "camera_source", text="")
        row.operator("fpv_pose.list_cameras", text="Scan", icon="VIEWZOOM")
        row.operator("fpv_pose.reset_camera", text="", icon="LOOP_BACK")
        if s.cameras_found:
            col.label(text="Found: " + s.cameras_found)
        if ops.multi_enabled(s):
            _draw_multi_cameras(col, s)

        col.prop(s, "view_mode")
        col.prop(s, "mirror")
        if ops.multi_enabled(s):
            col.label(text="Camera FOV is set per device above", icon="INFO")
        else:
            col.prop(s, "camera_fov")
            if s.camera_fov > 100.0 or s.camera_fov < 45.0:
                col.label(text="FOV looks unusual; most webcams are 60-78 deg",
                          icon="ERROR")

        _draw_calibration(layout, s)
        _draw_capture(layout, s)


class FPV_PT_engine(_Ready, bpy.types.Panel):
    """The estimator, and the body geometry its output is read through."""

    bl_label = "Pose Engine + Geometry"
    bl_idname = "FPV_PT_engine"
    bl_order = 2

    def draw(self, context):
        layout = self.layout
        s = context.scene.fpv_pose

        col = layout.column(align=True)
        _draw_engine(col, s)

        col.separator()
        col.prop(s, "pose_visibility", slider=True)
        col.prop(s, "pose_source")
        if s.pose_source == "LIFT":
            col.prop(s, "hand_depth_from_size")
            if s.hand_depth_from_size:
                row = col.row(align=True)
                row.prop(s, "hand_size", slider=True)
                row.prop(s, "hand_size_auto", text="Learn", toggle=True)
        col.prop(s, "use_arm_ik")
        col.prop(s, "anchor_mode")
        col.prop(s, "elbow_from_angle")
        col.prop(s, "shoulder_mode")
        if s.shoulder_mode == "REACH":
            col.prop(s, "shoulder_protract", slider=True)
        col.prop(s, "track_palm")
        col.prop(s, "hand_depth_mode")
        if s.view_mode == "BODY_CAM" or s.pose_source == "WORLD":
            col.prop(s, "pose_y_up")
        col.prop(s, "verbose_hand_log")

        _draw_inference_performance(layout, s)


class FPV_PT_motion(_Ready, bpy.types.Panel):
    """How hard the rig is filtered on its way to the screen."""

    bl_label = "Motion Response"
    bl_idname = "FPV_PT_motion"
    bl_order = 3

    def draw(self, context):
        layout = self.layout
        s = context.scene.fpv_pose

        row = layout.row(align=True)
        for mode, label in (("SMOOTH", "Smooth"), ("BALANCED", "Balanced"),
                            ("FAST", "Fast Action")):
            op = row.operator("fpv_pose.motion_preset", text=label)
            op.mode = mode
        layout.label(text="Arm settles in %s, hands %s"
                          % (_ms(s.pose_smoothing), _ms(s.smoothing)),
                     icon="TIME")
        if s.pose_smoothing >= 0.85:
            layout.label(text="Too slow for fast action (a punch lasts ~130 ms)",
                         icon="INFO")
        col = layout.column(align=True)
        col.prop(s, "smoothing", slider=True)
        col.prop(s, "pose_smoothing", slider=True)
        col.separator()
        col.prop(s, "one_euro_min_cutoff", slider=True)
        col.prop(s, "one_euro_beta", slider=True)


class FPV_PT_preview(_Ready, bpy.types.Panel):
    """Seeing what the camera and the estimator see."""

    bl_label = "Preview + Diagnostics"
    bl_idname = "FPV_PT_preview"
    bl_order = 4

    def draw(self, context):
        layout = self.layout
        s = context.scene.fpv_pose

        col = layout.column(align=True)
        col.prop(s, "show_preview_window")
        col.prop(s, "show_inline_preview")
        if s.show_inline_preview:
            col.prop(s, "inline_preview_scale", slider=True)
        col.prop(s, "preview_fps", slider=True)

        row = layout.row(align=True)
        row.operator("fpv_pose.test_webcam", icon="CAMERA_DATA")
        row.operator("fpv_pose.reinstall_mediapipe", text="Reinstall MP",
                     icon="FILE_REFRESH")

        if s.test_report:
            box = layout.box()
            box.label(text="Test Result:")
            # Wrap long lines so full error text is readable.
            for line in s.test_report.splitlines():
                if not line:
                    box.separator()
                    continue
                chunk = 60
                for i in range(0, len(line), chunk):
                    box.label(text=line[i:i + chunk])

        _draw_inline_preview(layout, s)


class FPV_PT_tracking(_Ready, bpy.types.Panel):
    """Run the tracker, record from it, and watch what it is doing."""

    bl_label = "Tracking"
    bl_idname = "FPV_PT_tracking"
    bl_order = 5
    bl_options = set()          # the panel you actually press buttons in

    def draw(self, context):
        layout = self.layout
        s = context.scene.fpv_pose

        row = layout.row(align=True)
        row.operator("fpv_pose.start", icon="PLAY")
        row.operator("fpv_pose.stop", icon="PAUSE")

        _draw_record_controls(layout, context, s)
        _draw_status(layout, s)


def _draw_calibration(layout, s):
    """Rest-pose reference, and framing the camera to match it."""
    box = layout.box()
    box.label(text="Calibration", icon="ORIENTATION_GIMBAL")
    box.label(text="Match the rig's rest pose, hold still.")
    box.label(text="FPS arms: arms forward to camera, palms down.")
    box.prop(s, "auto_calibrate")
    row = box.row(align=True)
    row.prop(s, "calibration_delay")
    row.prop(s, "calibration_frames")
    box.prop(s, "reset_pose_on_start")
    row = box.row(align=True)
    row.operator("fpv_pose.calibrate", icon="ORIENTATION_GIMBAL")
    row.operator("fpv_pose.reset_pose", text="Reset Bones", icon="LOOP_BACK")
    box.operator("fpv_pose.fit_camera", icon="CAMERA_DATA")
    if s.eye_offset_info:
        box.label(text=s.eye_offset_info, icon="HIDE_OFF")


def _draw_capture(layout, s):
    """What the camera is asked to deliver: size, rate, exposure."""
    box = layout.box()
    box.label(text="Capture", icon="CAMERA_DATA")
    box.label(text="(Stop -> Start to apply)")
    row = box.row(align=True)
    row.prop(s, "capture_width")
    row.prop(s, "capture_height")
    if 0 < s.capture_width < 320 or 0 < s.capture_height < 240:
        box.label(text="Tiny capture size; use 0 for the camera default",
                  icon="ERROR")
    row = box.row(align=True)
    row.prop(s, "target_fps")
    row.prop(s, "use_mjpg", toggle=True)
    row = box.row(align=True)
    row.prop(s, "exposure_mode", text="")
    row.operator("fpv_pose.autotune_exposure", text="Auto-Tune", icon="DRIVER")
    if s.exposure_mode == "MANUAL":
        sub = box.column(align=True)
        sub.prop(s, "exposure", slider=True)
        sub.prop(s, "gain")
        sub.label(text="Shorter exposure = sharper fast hands. Add light.",
                  icon="LIGHT_SUN")
        sub.label(text="Too short gives a black image and no tracking.",
                  icon="ERROR")
        sub.operator("fpv_pose.reset_camera", icon="LOOP_BACK")
    if s.cap_info:
        box.label(text="Camera: " + s.cap_info, icon="CHECKMARK")


def _draw_inference_performance(layout, s):
    """What the estimator costs per frame, and what it reported back."""
    box = layout.box()
    box.label(text="Performance", icon="SORTTIME")
    box.label(text="(Stop -> Start to apply)")
    row = box.row(align=True)
    sub = row.row(align=True)
    # A top-down engine crops to its own input size, so a pre-shrunk
    # frame saves it nothing and only blurs the hand crops.
    sub.enabled = not pose_engines.capabilities(s.pose_engine).wants_full_frame
    sub.prop(s, "infer_width")
    row.prop(s, "pose_every")
    row = box.row(align=True)
    row.prop(s, "parallel_infer", toggle=True)
    row.prop(s, "mp_delegate" if s.pose_engine == "MEDIAPIPE"
             else "engine_device", text="")
    if s.engine_info:
        for line in _wrap("Models: " + s.engine_info, 44):
            box.label(text=line)
    if s.perf_info:
        for line in _wrap(s.perf_info, 44):
            box.label(text=line)
    if s.angle_info:
        for line in _wrap("Angles: " + s.angle_info, 44):
            box.label(text=line, icon="DRIVER_ROTATIONAL_DIFFERENCE")
    box.prop(s, "record_diagnostics")
    if s.record_diagnostics:
        box.prop(s, "record_path")


def _draw_status(layout, s):
    """Live state of the running tracker."""
    box = layout.box()
    box.label(text="Status: %s" % s.status)
    crow = box.row()
    calibrating = s.calib_status.startswith("calibrating")
    crow.alert = calibrating or s.calib_status.startswith("partial")
    crow.label(text="Calibration: %s" % s.calib_status,
               icon="TIME" if calibrating else "CHECKMARK")
    row = box.row(align=True)

    def _mark(label, ok):
        r = row.row(align=True)
        r.alert = not ok
        r.label(text="%s: %s" % (label, "OK" if ok else "--"))

    _mark("Pose", s.detect_pose)
    _mark("L Hand", s.detect_left)
    _mark("R Hand", s.detect_right)
    box.label(text="Bones driven: %d   FPS: %.1f" % (s.bones_applied, s.fps))
    if s.multi_info:
        for i, line in enumerate(s.multi_info.splitlines()):
            r = box.row()
            # A camera contributing almost nothing is the failure mode
            # that otherwise hides behind a healthy fused frame rate.
            r.alert = ("FAILED" in line
                       or any(line.endswith("in %d%% of fused frames" % n)
                              for n in range(0, 30)))
            r.label(text=line,
                    icon="OUTLINER_OB_CAMERA" if i == 0 else "BLANK1")


def _draw_inline_preview(layout, s):
    """The last captured frame, drawn in the sidebar."""
    if not s.show_inline_preview:
        return
    img = bpy.data.images.get(ops.PREVIEW_IMAGE_NAME)
    if img is None:
        layout.label(text="(no preview frame yet)", icon="INFO")
        return
    if not img.preview:
        try:
            img.preview_ensure()
        except Exception:
            pass
    if img.preview and img.preview.icon_id:
        row = layout.row()
        row.template_icon(icon_value=img.preview.icon_id,
                          scale=s.inline_preview_scale)
    else:
        layout.label(text="(preview not available)", icon="INFO")

def _draw_engine(layout, s):
    """Pose-engine picker, and whatever that engine has settings for.

    Only the selected engine's controls are drawn. The alternative --
    every knob on screen with most of them disabled -- makes a panel that
    is mostly dead space and invites the user to tune a model that is not
    running.
    """
    caps = pose_engines.capabilities(s.pose_engine)
    box = layout.box()
    box.label(text="Pose Engine", icon="ARMATURE_DATA")
    box.prop(s, "pose_engine", text="")

    ok, dep_msg = dependencies.engine_status(s.pose_engine)
    if not ok:
        box.label(text=dep_msg, icon="ERROR")
        op = box.operator("fpv_pose.install_dependencies", icon="IMPORT",
                          text="Install %s Dependencies" % caps.label)
        op.engine = s.pose_engine
        # Blender hides the per-user site-packages directory, so a wheel
        # can be installed and still unimportable; Engine Info prints where
        # each module resolved from and which directories were searched.
        box.label(text="Installs into Blender's scripts folder", icon="INFO")
        box.operator("fpv_pose.engine_diagnostics", icon="INFO")
        return
    if caps.note:
        for line in _wrap(caps.note, 44):
            box.label(text=line, icon="INFO")

    if s.pose_engine == "MEDIAPIPE":
        box.prop(s, "model_complexity")
        return

    col = box.column(align=True)
    col.prop(s, "engine_quality")
    col.prop(s, "engine_layout")
    col.prop(s, "engine_hands")
    resolved = pose_engines.resolve_hand_source(
        s.pose_engine, s.engine_hands, s.engine_layout)
    if resolved != s.engine_hands:
        col.label(text="Using: %s hands" % resolved.lower(), icon="INFO")
    if resolved == "MEDIAPIPE" and dependencies.missing_modules("MEDIAPIPE"):
        col.label(text="MediaPipe hands need MediaPipe installed", icon="ERROR")
    if resolved == "RTM":
        col.prop(s, "engine_hand_detector", toggle=True)
    row = col.row(align=True)
    row.prop(s, "engine_det_every")
    row.prop(s, "engine_kpt_thr", slider=True)

    if s.engine_device == "CPU":
        box.operator("fpv_pose.install_gpu_runtime", icon="IMPORT")
    _draw_body_lengths(box, s, caps)
    # A 2D engine has no depth of its own, so the lift is not optional and
    # the near/far decision leans entirely on the hand-size cue.
    if not caps.world_pose:
        if s.pose_source != "LIFT":
            box.label(text="Arm Depth will switch to 2D + Limb Lengths",
                      icon="ERROR")
        if not s.hand_depth_from_size:
            for line in _wrap("No image depth to order joints by: turn on "
                              "'Hand Size Drives Depth' so a reaching arm "
                              "reads as reaching.", 44):
                box.label(text=line, icon="ERROR")
    box.operator("fpv_pose.engine_diagnostics", icon="INFO")


def _draw_body_lengths(layout, s, caps):
    """Measured limb lengths, which only a 2D engine really depends on.

    MediaPipe learns these from its own 3D pose, so they are an optional
    override there. RTMPose and ViTPose have no 3D pose to learn from, so
    leaving them at 0 means the depth lift scales an average body -- which
    is the one accuracy gap switching engines introduces, and the only one
    the user has to close by hand.
    """
    box = layout.box()
    box.label(text="Your Measurements (m)", icon="FIXED_SIZE")
    unset = not (s.body_shoulder_width and s.body_upper_arm and s.body_forearm)
    if unset and not caps.world_pose:
        for line in _wrap("Unset: arm depth will scale an average body. "
                          "Measure these for accurate reach.", 44):
            box.label(text=line, icon="ERROR")
    elif unset:
        box.label(text="0 = learn from the engine's 3D pose", icon="INFO")
    col = box.column(align=True)
    col.prop(s, "body_shoulder_width")
    col.prop(s, "body_upper_arm")
    col.prop(s, "body_forearm")


def _draw_multi_cameras(layout, s):
    """Device assignment for the multi-camera rig."""
    box = layout.box()
    box.label(text="Camera Devices", icon="OUTLINER_OB_CAMERA")

    if s.view_mode != "SELFIE":
        warn = box.column(align=True)
        warn.alert = True
        warn.label(text="Multi-camera needs Selfie view mode", icon="ERROR")
        warn.label(text="Body Cam frames are tied to the camera's own heading")

    if not len(s.cameras):
        box.label(text="No devices yet — Scan, then Add", icon="INFO")

    active = 0
    for i, dev in enumerate(s.cameras):
        sub = box.box()
        head = sub.row(align=True)
        head.prop(dev, "enabled", text="")
        head.label(text="Webcam %d" % (i + 1))
        head.prop(dev, "index", text="Device")
        rm = head.operator("fpv_pose.camera_remove", text="", icon="X")
        rm.slot = i
        if not dev.enabled:
            continue
        active += 1
        row = sub.row(align=True)
        row.prop(dev, "yaw", slider=True)
        row.prop(dev, "fov")
        sub.prop(dev, "trust", slider=True)
        row = sub.row(align=True)
        row.prop(dev, "exposure_mode", text="")
        if dev.exposure_mode == "MANUAL":
            row.prop(dev, "exposure", slider=True)
        if dev.measured_fps > 0.0:
            r = sub.row()
            r.alert = dev.measured_fps < 15.0
            r.label(text="%.1f fps" % dev.measured_fps,
                    icon="SORTTIME" if dev.measured_fps < 15.0 else "CHECKMARK")

    box.operator("fpv_pose.camera_add", icon="ADD")

    dupes = len(s.cameras) - len({d.index for d in s.cameras})
    if dupes > 0:
        r = box.row()
        r.alert = True
        r.label(text="Two rows share a device index", icon="ERROR")
    if active == 1:
        box.label(text="Enable a second camera to gain any accuracy",
                  icon="INFO")
    if active >= 2 and all(d.yaw == 0.0 for d in ops.enabled_cameras(s)):
        r = box.row()
        if s.multi_solver == "AVERAGE":
            # With one yaw for every camera the information matrices are
            # identical, Depth Trust cancels out of the solve, and the
            # fusion degenerates to a plain average. Nothing below will
            # help until these differ.
            r.alert = True
            r.label(text="All cameras at yaw 0: fusion is just an average",
                    icon="ERROR")
        else:
            # Yaw is only a starting guess for the body fit, so leaving it at
            # 0 is not the error here that it is for averaging.
            r.label(text="Yaw only seeds the fit; Calibrate measures it",
                    icon="INFO")

    box.prop(s, "multi_solver", text="")
    if s.multi_solver == "REPROJECT":
        sub = box.box()
        row = sub.row(align=True)
        row.operator("fpv_pose.calibrate_rig", icon="CON_CAMERASOLVER")
        if s.multi_extrinsics:
            row.operator("fpv_pose.clear_rig_calibration", text="", icon="X")
        if s.multi_calib_info:
            measured = "calibrated" in s.multi_calib_info
            for n, line in enumerate(_wrap(s.multi_calib_info, 34)):
                r = sub.row()
                r.alert = ("fall" in line or "not calibrated" in line
                           or "other cameras" in line)
                r.label(text=line,
                        icon=("CHECKMARK" if measured else "INFO") if n == 0
                        else "BLANK1")
        elif not s.multi_extrinsics:
            sub.label(text="Not measured: using the Yaw sliders", icon="INFO")
        sub.prop(s, "multi_bone_stiffness", slider=True)
    else:
        box.prop(s, "multi_depth_trust", slider=True)
    box.prop(s, "multi_sync_ms")
    box.label(text="A camera below ~15 fps holds back the pair", icon="INFO")
    if active >= 2:
        box.label(text="%d x pose+hand inference per frame" % active,
                  icon="SORTTIME")


# Mean pixel value below which the feed counts as black. Mirrors
# tracker._BLACK_LEVEL; a dim room still reads well above this.
_BLACK_LEVEL = 10.0


def _draw_black_frame_warning(layout, s):
    """Banner at the top of the panel when the camera is seeing nothing.

    A black feed produces no landmarks, so calibration never completes and
    the rig never moves — symptoms identical to a broken tracker. Saying so
    up front is the difference between closing a privacy shutter and
    debugging the add-on.
    """
    if s.cap_brightness < 0.0 or s.cap_brightness >= _BLACK_LEVEL:
        return
    box = layout.box()
    col = box.column(align=True)
    col.alert = True
    col.label(text="CAMERA IMAGE IS BLACK (%.0f/255)" % s.cap_brightness,
              icon="ERROR")
    col.label(text="Tracking is running but can see nothing.")
    sub = box.column(align=True)
    sub.label(text="- Check the privacy shutter / lens cover")
    sub.label(text="- Check no other app has the camera")
    sub.label(text="- Check the room lighting")
    if s.exposure_mode == "MANUAL":
        sub.label(text="- Exposure is Manual; try Auto", icon="ERROR")
    box.operator("fpv_pose.reset_camera", icon="LOOP_BACK")


def _ms(weight):
    """Smoothing weight as the time it takes to close 63% of a gap.

    The weight alone is unreadable — 0.9 sounds mild but is a third of a
    second — so the panel shows the number that decides whether fast motion
    survives.
    """
    import math
    if weight <= 0.0:
        return "0 ms"
    if weight >= 0.999:
        return "never"
    return "%.0f ms" % (1000.0 * -(1.0 / 30.0) / math.log(weight))


def _wrap(text, width):
    out = []
    for line in text.splitlines():
        while len(line) > width:
            cut = line.rfind(" ", 0, width)
            cut = cut if cut > 0 else width
            out.append(line[:cut])
            line = line[cut:].lstrip()
        out.append(line)
    return out


def _draw_record_controls(layout, context, s):
    """Record / stop, plus the settings that decide what happens on stop."""
    box = layout.box()
    box.label(text="Animation Recording", icon="REC")

    row = box.row(align=True)
    row.scale_y = 1.3
    if recording.is_recording():
        row.alert = True
        row.operator("fpv_pose.record_stop", icon="SNAP_FACE", text="Stop Recording")
    else:
        row.operator("fpv_pose.record_start", icon="REC", text="Record Animation")

    if recording.is_recording():
        box.label(text="Recording: %d frames  %.1fs" % (s.rec_count, s.rec_seconds),
                  icon="RADIOBUT_ON")
    elif ops._STATE.get("retargeter") is None and not recording.has_take():
        box.label(text="Start tracking to record", icon="INFO")

    col = box.column(align=True)
    col.prop(s, "record_on_start")
    col.prop(s, "take_open_window")
    col.prop(s, "take_auto_detect")

    if recording.has_take() and not recording.is_recording():
        box.separator()
        row = box.row(align=True)
        row.scale_y = 1.2
        row.operator("fpv_pose.open_review_window", icon="WINDOW", text="Review Take…")


def _draw_take_review(layout, context):
    """The take editor. Shared by the sidebar panel and the review window so
    both always show the same state."""
    scene = context.scene
    s = scene.fpv_pose
    tk = recording.take()
    if tk is None:
        layout.label(text="No take recorded.", icon="INFO")
        layout.label(text="Start tracking, then press Record Animation.")
        return

    kept, total, dropped, flagged, frame_lo, frame_hi = recording.stats(scene)

    box = layout.box()
    box.label(text="%d of %d samples kept  (%d removed)" % (kept, total, dropped),
              icon="KEYFRAME_HLT")
    scene_fps = recording.scene_fps(scene)
    box.label(text="Frames %d-%d at %.4g fps  (captured %.1fs at %.1f fps)"
                   % (frame_lo, frame_hi, scene_fps, tk.duration(), tk.capture_fps))
    if tk.capture_fps > scene_fps * 1.15:
        sub = box.column(align=True)
        sub.label(text="Scene runs at %.4g fps but the take was captured at "
                       "%.0f — detail is being dropped"
                       % (scene_fps, tk.capture_fps), icon="ERROR")
        sub.operator("fpv_pose.take_match_fps", icon="PREVIEW_RANGE")
    if recording.is_live():
        r = box.row()
        r.alert = True
        r.label(text="Stop tracking to preview the take", icon="ERROR")

    # --- trim ---
    col = layout.column(align=True)
    col.label(text="Trim", icon="TRIA_RIGHT_BAR")
    col.prop(s, "take_start", slider=True)
    col.prop(s, "take_end", slider=True)
    row = col.row(align=True)
    op = row.operator("fpv_pose.take_set_trim", text="Start at Playhead")
    op.edge = "START"
    op = row.operator("fpv_pose.take_set_trim", text="End at Playhead")
    op.edge = "END"

    # --- outliers ---
    col = layout.column(align=True)
    col.separator()
    col.label(text="Outliers", icon="TRACKING_REFINE_BACKWARDS")
    col.prop(s, "take_outlier_sensitivity", slider=True)
    col.prop(s, "take_outlier_min")
    col.prop(s, "take_flag_lost")
    col.prop(s, "take_auto_exclude")
    col.operator("fpv_pose.take_detect", icon="ZOOM_SELECTED")
    if s.take_threshold_info:
        col.label(text="%d flagged, over %s" % (flagged, s.take_threshold_info))

    row = col.row(align=True)
    op = row.operator("fpv_pose.take_jump", text="", icon="TRIA_LEFT")
    op.target = "PREV"
    op = row.operator("fpv_pose.take_jump", text="Flagged", icon="TRIA_RIGHT")
    op.target = "NEXT"
    op = row.operator("fpv_pose.take_flagged_action", text="Drop All")
    op.mode = "DROP"
    op = row.operator("fpv_pose.take_flagged_action", text="Keep All")
    op.mode = "KEEP"

    row = col.row(align=True)
    row.operator("fpv_pose.take_toggle_current", text="Drop/Keep Here", icon="KEYFRAME")
    op = row.operator("fpv_pose.take_flagged_action", text="Reset Edits", icon="LOOP_BACK")
    op.mode = "RESET"

    # --- per-sample list ---
    col = layout.column(align=True)
    col.separator()
    row = col.row(align=True)
    row.label(text="Samples", icon="PRESET")
    row.prop(s, "take_show_flagged_only", toggle=True)
    col.template_list("FPV_UL_take_frames", "", scene, "fpv_take_frames",
                      s, "take_active_index", rows=6)
    row = col.row(align=True)
    op = row.operator("fpv_pose.take_jump", text="Playhead to Selected", icon="TIME")
    op.target = "ACTIVE"
    row.operator("fpv_pose.take_scene_range", text="Scene Range", icon="ARROW_LEFTRIGHT")

    # --- save ---
    box = layout.box()
    box.label(text="Save", icon="ACTION")
    box.prop(s, "take_action_name")
    row = box.row(align=True)
    row.prop(s, "take_start_frame")
    row.prop(s, "take_interpolation", text="")
    box.prop(s, "take_preview")
    box.prop(s, "take_push_nla")
    if s.take_push_nla:
        box.prop(s, "take_nla_track")
    row = box.row(align=True)
    row.scale_y = 1.4
    row.enabled = kept > 0
    row.operator("fpv_pose.take_save", icon="CHECKMARK")
    row = box.row(align=True)
    row.operator("fpv_pose.take_discard", icon="TRASH")


class FPV_UL_take_frames(bpy.types.UIList):
    """One row per captured sample: keep toggle, frame, how far off the
    neighbours' path it sits, and why it was flagged."""

    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "enabled", text="")
        sub = row.row(align=True)
        sub.active = item.enabled
        sub.alert = bool(item.flags) and item.enabled
        sub.label(text="f%d" % item.frame)
        sub.label(text="%.1f%%" % item.score)
        reason = recording.flag_text(item.flags)
        sub.label(text=reason if reason else "—")

    def filter_items(self, context, data, propname):
        rows = getattr(data, propname)
        flt = [self.bitflag_filter_item] * len(rows)
        if context.scene.fpv_pose.take_show_flagged_only:
            for i, r in enumerate(rows):
                if not r.flags and r.enabled:
                    flt[i] &= ~self.bitflag_filter_item
        return flt, []


class _TakeReviewDraw:
    """Body shared by the review panel wherever it is registered."""

    bl_label = "Take Review"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return recording.has_take()

    def draw(self, context):
        _draw_take_review(self.layout, context)


class FPV_PT_take_review(_TakeReviewDraw, bpy.types.Panel):
    bl_idname = "FPV_PT_take_review"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "FPV Pose"
    bl_order = 6
    bl_options = set()          # open by default: it only shows with a take


class FPV_PT_take_review_dopesheet(_TakeReviewDraw, bpy.types.Panel):
    """Same editor in the Action Editor sidebar, so the review window's
    lower half can drive it without reaching back up to the viewport."""

    bl_idname = "FPV_PT_take_review_dopesheet"
    bl_space_type = "DOPESHEET_EDITOR"
    bl_region_type = "UI"
    bl_category = "FPV Take"
    bl_options = set()


def _draw_clip_library(layout, context, s):
    """Where clips live, and which category/subgroup the next one goes in."""
    box = layout.box()
    row = box.row(align=True)
    row.label(text="Library", icon="FILE_FOLDER")
    row.operator("fpv_pose.clip_scan", text="", icon="FILE_REFRESH")
    row.operator("fpv_pose.clip_open_folder", text="", icon="FILEBROWSER")
    box.prop(s, "clip_root", text="")
    if not s.clip_root.strip():
        box.label(text="Default: " + cliplib.default_root(), icon="INFO")

    col = box.column(align=True)
    row = col.row(align=True)
    row.prop_search(s, "clip_category", context.scene, "fpv_clip_categories",
                    text="Category", icon="OUTLINER_COLLECTION")
    row = col.row(align=True)
    row.prop_search(s, "clip_subgroup", context.scene, "fpv_clip_groups",
                    text="Subgroup", icon="OUTLINER_COLLECTION")
    col.prop(s, "clip_show_all", toggle=True)
    if not s.clip_category:
        col.label(text="No category: clips land in the library root",
                  icon="INFO")


def _draw_clip_record(layout, context, s):
    """Capture raw footage. No estimator runs here, on purpose."""
    box = layout.box()
    box.label(text="Record Clip", icon="REC")
    recording_now = bool(s.clip_rec_active)

    col = box.column(align=True)
    col.enabled = not recording_now
    col.prop(s, "clip_name")
    row = col.row(align=True)
    row.prop(s, "clip_countdown")
    row.prop(s, "clip_max_seconds")

    row = box.row(align=True)
    row.scale_y = 1.3
    if recording_now:
        row.alert = True
        row.operator("fpv_pose.clip_record_stop", icon="SNAP_FACE")
    else:
        row.operator("fpv_pose.clip_record_start", icon="REC")
    if ops._STATE.get("tracker") is not None:
        box.label(text="Live tracking holds the camera — stop it first",
                  icon="ERROR")
    if recording_now:
        box.label(text="%d frames  %.1fs" % (s.clip_rec_frames, s.clip_rec_seconds),
                  icon="RADIOBUT_ON")
    if s.clip_rec_status:
        for line in _wrap(s.clip_rec_status, 44):
            box.label(text=line)
    box.label(text="Resolution, frame rate and exposure come from "
                   "Capture + Performance", icon="INFO")


def _draw_clip_list(layout, context, s):
    scene = context.scene
    layout.template_list("FPV_UL_clips", "", scene, "fpv_clips",
                         s, "clip_active_index", rows=4)
    clip = clip_ops.active_clip(scene)
    if clip is None:
        layout.label(text="No clips here yet — record one, or Refresh",
                     icon="INFO")
        return None
    col = layout.column(align=True)
    col.label(text="%d frames  %.4g fps  %.1fs  %.0f MB"
                   % (clip.frames, clip.fps, clip.duration, clip.size_mb))
    if clip.category:
        col.label(text="In %s%s" % (clip.category,
                                    "/" + clip.subgroup if clip.subgroup else ""))
    row = col.row(align=True)
    row.operator("fpv_pose.clip_open_video", icon="PLAY")
    row.operator("fpv_pose.clip_delete", icon="TRASH")
    return clip


def _draw_clip_analysis(layout, context, s):
    """Run an estimator over the clip, and say what it will cost."""
    box = layout.box()
    box.label(text="Pose Estimation", icon="ARMATURE_DATA")
    col = box.column(align=True)
    col.prop(s, "clip_engine", text="")
    ok, msg = dependencies.engine_status(s.clip_engine)
    if not ok:
        sub = col.column(align=True)
        sub.alert = True
        sub.label(text=msg, icon="ERROR")
        op = sub.operator("fpv_pose.install_dependencies", icon="IMPORT")
        op.engine = s.clip_engine
    caps = pose_engines.capabilities(s.clip_engine)
    if caps.note:
        for line in _wrap(caps.note, 44):
            col.label(text=line, icon="INFO")
    if s.clip_engine == "MEDIAPIPE":
        col.prop(s, "model_complexity")
    else:
        row = col.row(align=True)
        row.prop(s, "clip_quality", text="")
        row.prop(s, "clip_device", text="")
        col.prop(s, "engine_layout", text="")
        col.prop(s, "engine_hands", text="")
        # Worth saying where the palm angle comes from: these engines
        # return hand keypoints in pixels with no depth, so the palm
        # plane is reconstructed from the shape of the knuckles rather
        # than measured, and it is less certain than MediaPipe's.
        for line in _wrap("Hands from this engine are 2D; palm tilt is "
                          "fitted from the knuckle shape", 44):
            col.label(text=line, icon="INFO")
    col.prop(s, "clip_stride")
    col.prop(s, "clip_overwrite")

    row = box.row(align=True)
    row.scale_y = 1.2
    if s.clip_analysis_active:
        row.operator("fpv_pose.clip_analyze_cancel", icon="X")
        box.label(text="%.0f%%  %s" % (100.0 * s.clip_analysis_progress,
                                       s.clip_analysis_status),
                  icon="TIME")
    else:
        row.operator("fpv_pose.clip_analyze", icon="PLAY")
        if s.clip_analysis_status:
            for line in _wrap(s.clip_analysis_status, 44):
                box.label(text=line)


def _draw_clip_preview(layout, context, s):
    """The estimated pose drawn over the footage, frame by frame."""
    box = layout.box()
    box.label(text="Result Preview", icon="SEQ_PREVIEW")
    box.prop(s, "clip_analysis", text="")
    engine, sidecar = clip_ops.selected_analysis(s)
    if sidecar is None:
        box.label(text="Analyse the clip to get a result to preview",
                  icon="INFO")
        return

    row = box.row(align=True)
    row.operator("fpv_pose.clip_preview_open", icon="HIDE_OFF")
    row.operator("fpv_pose.clip_preview_close", text="", icon="X")
    row.operator("fpv_pose.clip_delete_analysis", text="", icon="TRASH")

    length = clip_ops.preview_length(s)
    if not length:
        return
    col = box.column(align=True)
    sub = col.row(align=True)
    for label, icon, target in (("", "REW", "FIRST"),
                                ("", "TRIA_LEFT", "PREV"),
                                ("", "TRIA_RIGHT", "NEXT"),
                                ("", "FF", "LAST")):
        op = sub.operator("fpv_pose.clip_preview_step", text=label, icon=icon)
        op.target = target
    op = sub.operator("fpv_pose.clip_preview_step", text="Next Gap",
                      icon="TRACKING_CLEAR_FORWARDS")
    op.target = "NEXT_LOST"
    # No slider: the clip length is not known to the property, and a
    # slider without a real maximum is worse than a drag field.
    col.prop(s, "clip_preview_frame")
    row = col.row(align=True)
    row.prop(s, "clip_preview_pose", toggle=True)
    row.prop(s, "clip_preview_hands", toggle=True)
    col.prop(s, "clip_preview_scale", slider=True)
    if s.clip_preview_info:
        for line in _wrap(s.clip_preview_info, 44):
            box.label(text=line)

    img = bpy.data.images.get(clip_ops.CLIP_PREVIEW_IMAGE)
    if img is None:
        return
    if not img.preview:
        try:
            img.preview_ensure()
        except Exception:
            pass
    if img.preview and img.preview.icon_id:
        row = box.row()
        row.template_icon(icon_value=img.preview.icon_id,
                          scale=s.clip_preview_scale)


def _draw_clip_calibration(layout, context, s):
    """Where the rest-pose reference comes from.

    This gets its own box because it is the difference between a clip
    replay that matches the performance and one whose wrists are rotated
    off somewhere: the hand bones are driven by the change in palm
    orientation *since* a reference, and a clip cannot be asked to hold
    still after the fact.
    """
    box = layout.box()
    box.label(text="Calibration", icon="OUTLINER_OB_EMPTY")
    box.prop(s, "clip_calib_source", text="")

    if s.clip_calib_source == "PROFILE":
        row = box.row(align=True)
        row.prop(s, "clip_calib_profile", text="")
        row.operator("fpv_pose.clip_calib_delete", text="", icon="TRASH")
        if s.clip_calib_profile in ("", "NONE"):
            box.label(text="Record a few seconds of the rig's rest pose, "
                           "analyse it, then Calibrate from Clip", icon="INFO")
    elif s.clip_calib_source == "LIVE":
        if ops.live_calibration() is None:
            box.label(text="No live calibration yet — track once and "
                           "calibrate", icon="ERROR")
        else:
            box.label(text="Using the live session's calibration",
                      icon="CHECKMARK")
    else:
        box.label(text="Only correct if this clip opens on the rest pose",
                  icon="ERROR")

    col = box.column(align=True)
    col.operator("fpv_pose.clip_record_calibration", icon="REC")
    col.label(text="Then analyse it and make a profile from it:")
    row = col.row(align=True)
    row.prop(s, "clip_calib_skip")
    row.prop(s, "clip_calib_hold")
    col.prop(s, "clip_calib_name")
    row = col.row(align=True)
    row.operator("fpv_pose.clip_calibrate", icon="OUTLINER_OB_ARMATURE")
    row.operator("fpv_pose.clip_calib_from_live", icon="REC")
    if s.clip_calib_status:
        box.label(text=s.clip_calib_status, icon="INFO")


def _draw_clip_apply(layout, context, s):
    """Put an analysed clip on the rig, and play it back."""
    box = layout.box()
    box.label(text="Drive the Rig", icon="ARMATURE_DATA")
    if s.armature is None:
        box.label(text="Pick an armature above", icon="ERROR")
    if recording.is_live():
        r = box.row()
        r.alert = True
        r.label(text="Stop live tracking before replaying a clip", icon="ERROR")
    if recording.has_take():
        box.label(text="Save or discard the current take first", icon="INFO")
    if s.clip_calib_source == "PROFILE" and s.clip_calib_profile in ("", "NONE"):
        r = box.row()
        r.alert = True
        r.label(text="No calibration profile — see Calibration above",
                icon="ERROR")
    elif s.clip_calib_source == "LIVE" and ops.live_calibration() is None:
        r = box.row()
        r.alert = True
        r.label(text="No live calibration to reuse", icon="ERROR")
    box.prop(s, "clip_apply_action")
    row = box.row(align=True)
    row.scale_y = 1.3
    row.operator("fpv_pose.clip_apply", icon="PLAY")
    row = box.row(align=True)
    row.operator("fpv_pose.clip_play", icon="PLAY_SOUND")
    box.label(text="Replays through the same retargeter as live tracking, "
                   "with no camera running", icon="INFO")


class FPV_PT_clips(bpy.types.Panel):
    """Record footage, estimate pose over it offline, and drive the rig
    from the result -- the whole path that does not need tracking mode."""

    bl_label = "Clip Library + Offline Pose"
    bl_idname = "FPV_PT_clips"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "FPV Pose"
    bl_order = 7
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        s = context.scene.fpv_pose

        base_missing = dependencies.missing_base_modules()
        if base_missing:
            box = layout.box()
            box.label(text="Missing " + ", ".join(n for n, _ in base_missing),
                      icon="ERROR")
            box.operator("fpv_pose.install_dependencies", icon="IMPORT")
            return

        _draw_clip_library(layout, context, s)
        _draw_clip_record(layout, context, s)
        clip = _draw_clip_list(layout, context, s)
        if clip is None:
            return
        _draw_clip_analysis(layout, context, s)
        _draw_clip_preview(layout, context, s)
        _draw_clip_calibration(layout, context, s)
        _draw_clip_apply(layout, context, s)


class FPV_PT_bones(bpy.types.Panel):
    bl_label = "Bone Map"
    bl_idname = "FPV_PT_bones"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "FPV Pose"
    bl_order = 8
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        s = context.scene.fpv_pose
        row = layout.row(align=True)
        row.operator("fpv_pose.auto_map_bones", icon="AUTO")
        row.operator("fpv_pose.clear_map", icon="X")
        layout.operator("fpv_pose.list_bones", icon="OUTLINER_OB_ARMATURE")

        armature = s.armature
        if not armature:
            layout.label(text="Pick an armature above", icon="INFO")
            return

        b = s.bones

        def bone_row(parent, slot, label):
            r = parent.row(align=True)
            r.label(text=label)
            r.prop_search(b, slot, armature.data, "bones", text="")

        col = layout.column(align=True)
        col.label(text="Left Arm")
        bone_row(col, "shoulder_L", "Shoulder")
        bone_row(col, "upper_arm_L", "Upper Arm")
        bone_row(col, "forearm_L", "Forearm")
        bone_row(col, "hand_L", "Hand")

        col.separator()
        col.label(text="Right Arm")
        bone_row(col, "shoulder_R", "Shoulder")
        bone_row(col, "upper_arm_R", "Upper Arm")
        bone_row(col, "forearm_R", "Forearm")
        bone_row(col, "hand_R", "Hand")

        for side, side_label in (("L", "Left Hand"), ("R", "Right Hand")):
            col.separator()
            col.label(text=side_label + " Fingers")
            for finger in ("thumb", "index", "middle", "ring", "pinky"):
                sub = col.column(align=True)
                sub.label(text=finger.capitalize())
                for i in (1, 2, 3):
                    slot = f"{finger}_0{i}_{side}"
                    bone_row(sub, slot, f"  Joint {i}")


_CLASSES = (FPV_UL_take_frames, FPV_PT_main, FPV_PT_cameras,
            FPV_PT_engine, FPV_PT_motion, FPV_PT_preview,
            FPV_PT_tracking, FPV_PT_take_review,
            FPV_PT_take_review_dopesheet, FPV_PT_clips, FPV_PT_bones)


def register():
    for c in _CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(c)
        except Exception:
            pass
