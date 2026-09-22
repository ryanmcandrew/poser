"""Operators for the clip library: record, analyse, preview, apply.

The three long-running jobs here (capture, analysis, replay) all run off
the main thread or in one blocking pass, and each reports through the
scene so the panel can draw progress:

  record   `vidrec.ClipRecorder` thread + a UI timer mirroring its counters
  analyse  `offline.Analyzer` thread + a UI timer mirroring its progress
  apply    `offline.replay`, run inline: it is only matrix maths over
           landmarks that already exist, so it finishes in about a second
           for a minute of footage and a modal would cost more than it saves

Applying produces a take, not an Action, so recorded footage lands in the
same review editor as a live recording and gets the same trimming and
outlier removal before it is baked.

Applying also has to answer a question live tracking answers by asking the
user to stand still: what the rest pose looked like. `prime_calibration`
settles it from a saved profile, the live session, or (only when the clip
opens on the rest pose) the clip itself.
"""

import os
import subprocess
import sys

import bpy

from . import cliplib
from . import dependencies
from . import offline
from . import operators as ops
from . import pose_engines
from . import recording
from . import vidrec


CLIP_PREVIEW_IMAGE = "FPV_Clip_Preview"

_CLIP_STATE = {
    "recorder": None,
    "rec_timer": None,
    "analyzer": None,
    "analysis_timer": None,
    "preview": None,        # offline.Preview for the selected analysis
    "preview_key": None,    # (video path, sidecar path) it was opened for
}


# ----------------------------------------------------------------------
# library state, shared with the panel and the property callbacks
# ----------------------------------------------------------------------

def library_root(s):
    return cliplib.resolve_root(s.clip_root)


def active_clip(scene):
    """The selected clip row, or None."""
    clips = scene.fpv_clips
    i = scene.fpv_pose.clip_active_index
    return clips[i] if 0 <= i < len(clips) else None


def selected_analysis(s):
    """(engine, sidecar path) chosen for preview/apply, or (None, None)."""
    path = (s.clip_selected_path or "").strip()
    engine = s.clip_analysis
    if not path or not engine or engine == "NONE":
        return None, None
    sidecar = cliplib.sidecar_path(path, engine)
    return (engine, sidecar) if os.path.exists(sidecar) else (None, None)


def refresh_clip_list(context):
    """Rebuild the scene mirror of the library folders and their clips."""
    scene = context.scene
    s = scene.fpv_pose
    root = library_root(s)

    scene.fpv_clip_categories.clear()
    for name in cliplib.list_categories(root):
        scene.fpv_clip_categories.add().name = name
    scene.fpv_clip_groups.clear()
    for name in cliplib.list_subgroups(root, s.clip_category):
        scene.fpv_clip_groups.add().name = name

    category = "" if s.clip_show_all else s.clip_category
    subgroup = "" if s.clip_show_all else s.clip_subgroup
    paths = cliplib.list_clips(root, category, subgroup)

    scene.fpv_clips.clear()
    for path in paths:
        row = scene.fpv_clips.add()
        row.path = path
        row.name = os.path.splitext(os.path.basename(path))[0]
        row.label = cliplib.relative_label(root, path)
        row.category, row.subgroup = cliplib.split_location(root, path)
        # Reading the container per clip is a decode of one header, not of
        # the video, so the list can afford to say how long each take is.
        info = cliplib.video_info(path)
        row.frames = int(info.get("frames", 0))
        row.fps = float(info.get("fps", 0.0))
        row.duration = float(info.get("duration", 0.0))
        try:
            row.size_mb = os.path.getsize(path) / (1024.0 * 1024.0)
        except OSError:
            row.size_mb = 0.0
        row.analyses = ", ".join(e for e, _ in cliplib.list_analyses(path))

    if scene.fpv_clips:
        s.clip_active_index = min(s.clip_active_index, len(scene.fpv_clips) - 1)
    else:
        s.clip_active_index = 0
    clip_selection_changed(context)
    return len(scene.fpv_clips)


def clip_selection_changed(context):
    """Point the preview at whatever clip is now selected."""
    scene = context.scene
    s = scene.fpv_pose
    clip = active_clip(scene)
    path = clip.path if clip is not None else ""
    if path == s.clip_selected_path:
        return
    s.clip_selected_path = path
    close_preview()
    s.clip_preview_info = ""
    # The analysis enum is rebuilt from the new clip; the previous choice
    # is meaningless, and Blender clamps an out-of-range enum for us.
    try:
        s.clip_preview_frame = 0
    except Exception:
        pass


def close_preview():
    prev = _CLIP_STATE.get("preview")
    if prev is not None:
        prev.close()
    _CLIP_STATE["preview"] = None
    _CLIP_STATE["preview_key"] = None


def _preview_for(s):
    """The open `offline.Preview` for the current selection, opening it if
    needed. Returns None when there is nothing analysed to show."""
    engine, sidecar = selected_analysis(s)
    if sidecar is None:
        close_preview()
        return None
    key = (s.clip_selected_path, sidecar)
    if _CLIP_STATE.get("preview_key") == key:
        return _CLIP_STATE["preview"]
    close_preview()
    try:
        prev = offline.Preview(s.clip_selected_path, sidecar)
    except Exception as e:
        print("[FPV Pose Tracker] cannot open analysis:", e)
        return None
    _CLIP_STATE["preview"] = prev
    _CLIP_STATE["preview_key"] = key
    return prev


def refresh_clip_preview(context):
    """Render the selected analysis at the scrub position."""
    s = context.scene.fpv_pose
    prev = _preview_for(s)
    if prev is None:
        s.clip_preview_info = ""
        return False
    index = max(0, min(len(prev) - 1, int(s.clip_preview_frame)))
    rgba = prev.rgba(index,
                     draw_pose=bool(s.clip_preview_pose),
                     draw_hands=bool(s.clip_preview_hands))
    if rgba is None:
        s.clip_preview_info = prev.error or "no frame"
        return False
    ops._update_preview_image(rgba, CLIP_PREVIEW_IMAGE)
    s.clip_preview_info = prev.info_line(index)
    _tag_redraw()
    return True


def preview_length(s):
    prev = _CLIP_STATE.get("preview")
    return len(prev) if prev is not None else 0


def _tag_redraw():
    try:
        for area in bpy.context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
    except Exception:
        pass


# ----------------------------------------------------------------------
# recording clips
# ----------------------------------------------------------------------

def _rec_tick():
    rec = _CLIP_STATE.get("recorder")
    if rec is None:
        _CLIP_STATE["rec_timer"] = None
        return None
    s = bpy.context.scene.fpv_pose
    s.clip_rec_frames = rec.frames()
    s.clip_rec_seconds = rec.seconds()
    err = rec.error()
    s.clip_rec_status = err if err else rec.state()
    frame = rec.latest_preview()
    if frame is not None:
        ops._update_preview_image(frame, CLIP_PREVIEW_IMAGE)
    _tag_redraw()
    if not rec.is_running():
        # The thread finished on its own (time limit, or a dead camera):
        # collect it exactly as the Stop button would.
        _finish_recording(bpy.context)
        return None
    return 0.1


def _finish_recording(context):
    rec = _CLIP_STATE.get("recorder")
    _CLIP_STATE["recorder"] = None
    if _CLIP_STATE.get("rec_timer") is not None:
        try:
            bpy.app.timers.unregister(_CLIP_STATE["rec_timer"])
        except Exception:
            pass
        _CLIP_STATE["rec_timer"] = None
    s = context.scene.fpv_pose
    s.clip_rec_active = False
    if rec is None:
        return None
    path = rec.stop()
    s.clip_rec_frames = rec.frames()
    s.clip_rec_seconds = rec.seconds()
    err = rec.error()
    if err:
        s.clip_rec_status = err
    elif path:
        s.clip_rec_status = "saved %s  (%d frames, %.1fs, %.1f fps)" % (
            os.path.basename(path), rec.frames(), rec.seconds(),
            rec.measured_fps())
    else:
        s.clip_rec_status = rec.state()
    refresh_clip_list(context)
    if path:
        # Select what was just shot: it is what the user wants to analyse.
        for i, row in enumerate(context.scene.fpv_clips):
            if os.path.normcase(row.path) == os.path.normcase(path):
                s.clip_active_index = i
                break
    return path


class FPV_OT_clip_scan(bpy.types.Operator):
    bl_idname = "fpv_pose.clip_scan"
    bl_label = "Refresh Library"
    bl_description = ("Re-read the clip library from disk. Categories and "
                      "subgroups are plain folders, so anything you move in "
                      "Explorer shows up here")

    def execute(self, context):
        try:
            count = refresh_clip_list(context)
        except Exception as e:
            self.report({"ERROR"}, "Cannot read the library: %s" % e)
            return {"CANCELLED"}
        self.report({"INFO"}, "%d clips in %s"
                    % (count, library_root(context.scene.fpv_pose)))
        return {"FINISHED"}


class FPV_OT_clip_open_folder(bpy.types.Operator):
    bl_idname = "fpv_pose.clip_open_folder"
    bl_label = "Open Folder"
    bl_description = "Open the clip library folder in the system file browser"

    def execute(self, context):
        path = library_root(context.scene.fpv_pose)
        cliplib.ensure_dir(path)
        try:
            _desktop_open(path)
        except Exception as e:
            self.report({"ERROR"}, "Cannot open %s: %s" % (path, e))
            return {"CANCELLED"}
        return {"FINISHED"}


def _desktop_open(path):
    """Hand a path to whatever the desktop opens it with. Raises on failure."""
    if sys.platform.startswith("win"):
        os.startfile(path)      # noqa: S606 - the shell's own file handler
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def prime_calibration(context, rt):
    """Give `rt` its rest-pose reference. Returns (ok, message).

    Palm direction and wrist orientation are driven by the change since a
    reference pose, so this is not a nicety: with the wrong reference the
    whole take comes out with the wrists rotated, which is exactly what
    calibrating off the first frames of a performance produces.
    """
    s = context.scene.fpv_pose
    source = s.clip_calib_source
    if source == "CLIP":
        return True, "calibrating from this clip's opening frames"
    if source == "LIVE":
        data = ops.live_calibration()
        if not data:
            return False, ("No live calibration in this session. Track once "
                           "and calibrate, or use a saved profile.")
        return True, rt.load_calibration(data)
    name = s.clip_calib_profile
    if not name or name == "NONE":
        return False, ("No calibration profile. Record a clip of the rig's "
                       "rest pose, analyse it, then press Calibrate from Clip.")
    data = cliplib.read_profile(library_root(s), name)
    if not data:
        return False, "Cannot read calibration profile '%s'" % name
    return True, rt.load_calibration(data.get("calibration") or data)


class FPV_OT_clip_open_video(bpy.types.Operator):
    bl_idname = "fpv_pose.clip_open_video"
    bl_label = "Play in Player"
    bl_description = ("Open the selected clip in the system's default video "
                      "player, to watch the footage itself at full size")

    @classmethod
    def poll(cls, context):
        return bool((context.scene.fpv_pose.clip_selected_path or "").strip())

    def execute(self, context):
        path = (context.scene.fpv_pose.clip_selected_path or "").strip()
        if not os.path.exists(path):
            self.report({"ERROR"}, "That clip is no longer on disk")
            return {"CANCELLED"}
        # The preview holds the file open, and a player that wants an
        # exclusive handle (several do, on Windows) would just fail.
        close_preview()
        try:
            _desktop_open(path)
        except Exception as e:
            self.report({"ERROR"}, "Cannot open %s: %s" % (path, e))
            return {"CANCELLED"}
        self.report({"INFO"}, "Opened %s" % os.path.basename(path))
        return {"FINISHED"}


class FPV_OT_clip_record_start(bpy.types.Operator):
    bl_idname = "fpv_pose.clip_record_start"
    bl_label = "Record Clip"
    bl_description = ("Record webcam video into the library. No pose model "
                      "runs while recording, so the footage keeps the frame "
                      "rate and sharpness the accurate offline pass needs")

    @classmethod
    def poll(cls, context):
        return _CLIP_STATE.get("recorder") is None

    def execute(self, context):
        s = context.scene.fpv_pose
        if ops._STATE.get("tracker") is not None:
            self.report({"ERROR"},
                        "Stop live tracking first — it is holding the camera")
            return {"CANCELLED"}
        missing = dependencies.missing_base_modules()
        if missing:
            self.report({"ERROR"}, "Needs " + ", ".join(n for n, _ in missing))
            return {"CANCELLED"}

        root = library_root(s)
        directory = cliplib.clip_dir(root, s.clip_category, s.clip_subgroup)
        try:
            cliplib.ensure_dir(directory)
        except OSError as e:
            self.report({"ERROR"}, "Cannot create %s: %s" % (directory, e))
            return {"CANCELLED"}
        path = cliplib.unique_path(directory, s.clip_name)

        rec = vidrec.ClipRecorder(
            camera_index=ops.active_camera_index(s),
            path=path,
            width=int(s.capture_width),
            height=int(s.capture_height),
            fps=float(s.target_fps or 30),
            use_mjpg=bool(s.use_mjpg),
            mirror=bool(s.mirror),
            countdown=float(s.clip_countdown),
            max_seconds=float(s.clip_max_seconds),
            preview_fps=10.0,
        )
        if not rec.start():
            self.report({"ERROR"}, "Recorder is already running")
            return {"CANCELLED"}
        _CLIP_STATE["recorder"] = rec
        _CLIP_STATE["rec_timer"] = _rec_tick
        s.clip_rec_active = True
        s.clip_rec_frames = 0
        s.clip_rec_seconds = 0.0
        s.clip_rec_status = "starting"
        bpy.app.timers.register(_rec_tick, first_interval=0.1)
        self.report({"INFO"}, "Recording to " + os.path.basename(path))
        return {"FINISHED"}


class FPV_OT_clip_record_stop(bpy.types.Operator):
    bl_idname = "fpv_pose.clip_record_stop"
    bl_label = "Stop Clip"
    bl_description = "Stop recording and add the clip to the library"

    @classmethod
    def poll(cls, context):
        return _CLIP_STATE.get("recorder") is not None

    def execute(self, context):
        path = _finish_recording(context)
        if path is None:
            self.report({"WARNING"}, context.scene.fpv_pose.clip_rec_status
                        or "Nothing was recorded")
            return {"CANCELLED"}
        self.report({"INFO"}, "Saved " + os.path.basename(path))
        return {"FINISHED"}


class FPV_OT_clip_delete(bpy.types.Operator):
    bl_idname = "fpv_pose.clip_delete"
    bl_label = "Delete Clip"
    bl_description = ("Delete the selected clip and every analysis of it "
                      "from disk. This cannot be undone")

    @classmethod
    def poll(cls, context):
        return active_clip(context.scene) is not None

    def invoke(self, context, event):
        # Footage cannot be re-shot, so this one always asks.
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        clip = active_clip(context.scene)
        if clip is None:
            return {"CANCELLED"}
        name = os.path.basename(clip.path)
        close_preview()
        removed = cliplib.delete_clip(clip.path)
        refresh_clip_list(context)
        self.report({"INFO"}, "Deleted %s (%d files)" % (name, removed))
        return {"FINISHED"}


# ----------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------

def _analysis_settings(s):
    """Everything the analyser needs that is not about the clip itself.

    The geometry half (mirror, view mode, FOV, limb lengths) is the same
    set the live tracker is built with, and is stored in the sidecar so a
    replay months later solves depth the way the analysis did.
    """
    return {
        "tracking_mode": s.tracking_mode,
        "mirror": bool(s.mirror),
        "view_mode": s.view_mode,
        "camera_fov": float(s.camera_fov),
        "pose_visibility": float(s.pose_visibility),
        "lift_depth": (s.pose_source == "LIFT"),
        "hand_depth": bool(s.hand_depth_from_size),
        "hand_size": float(s.hand_size),
        "hand_size_auto": bool(s.hand_size_auto),
        "pose_y_up": bool(s.pose_y_up),
        "body_shoulder": float(s.body_shoulder_width),
        "body_upper": float(s.body_upper_arm),
        "body_fore": float(s.body_forearm),
        "model_complexity": int(s.model_complexity),
        "mp_delegate": s.mp_delegate,
        "engine_quality": s.clip_quality,
        "engine_layout": s.engine_layout,
        "engine_hands": s.engine_hands,
        "engine_device": s.clip_device,
        "engine_kpt_thr": float(s.engine_kpt_thr),
        "engine_hand_detector": bool(s.engine_hand_detector),
    }


def _analysis_tick():
    an = _CLIP_STATE.get("analyzer")
    if an is None:
        _CLIP_STATE["analysis_timer"] = None
        return None
    scene = bpy.context.scene
    s = scene.fpv_pose
    s.clip_analysis_progress = an.progress()
    err = an.error()
    s.clip_analysis_status = err if err else an.state()
    _tag_redraw()
    if an.is_running() or not an.finished():
        return 0.25

    _CLIP_STATE["analyzer"] = None
    _CLIP_STATE["analysis_timer"] = None
    s.clip_analysis_active = False
    if not err:
        # The new sidecar has to reach the list and the preview menu, and
        # the enum has to land on the engine that just finished.
        close_preview()
        refresh_clip_list(bpy.context)
        try:
            s.clip_analysis = an.engine
        except TypeError:
            pass
        s.clip_preview_frame = 0
        refresh_clip_preview(bpy.context)
    return None


class FPV_OT_clip_analyze(bpy.types.Operator):
    bl_idname = "fpv_pose.clip_analyze"
    bl_label = "Analyze Clip"
    bl_description = ("Run the chosen pose engine over the whole clip and "
                      "store the result beside it. Runs in the background; "
                      "an accurate engine can take longer than the clip")

    @classmethod
    def poll(cls, context):
        return (active_clip(context.scene) is not None
                and _CLIP_STATE.get("analyzer") is None)

    def execute(self, context):
        s = context.scene.fpv_pose
        clip = active_clip(context.scene)
        if clip is None:
            self.report({"ERROR"}, "Select a clip first")
            return {"CANCELLED"}
        ok, msg = dependencies.engine_status(s.clip_engine)
        if not ok:
            self.report({"ERROR"},
                        "%s cannot run (%s). Install it in Preferences first."
                        % (s.clip_engine, msg))
            return {"CANCELLED"}
        hand_src = pose_engines.resolve_hand_source(
            s.clip_engine, s.engine_hands, s.engine_layout)
        if hand_src == "MEDIAPIPE" and s.clip_engine != "MEDIAPIPE" \
                and dependencies.missing_modules("MEDIAPIPE"):
            self.report({"ERROR"},
                        "Hand Source is MediaPipe Hands but MediaPipe is not "
                        "installed. Install it, or pick another hand source.")
            return {"CANCELLED"}
        out = cliplib.sidecar_path(clip.path, s.clip_engine)
        if os.path.exists(out) and not s.clip_overwrite:
            self.report({"ERROR"},
                        "%s has already been analysed with %s — tick Replace "
                        "Existing to redo it" % (clip.name, s.clip_engine))
            return {"CANCELLED"}

        close_preview()
        an = offline.Analyzer(clip.path, _analysis_settings(s),
                              engine=s.clip_engine,
                              overwrite=bool(s.clip_overwrite),
                              stride=int(s.clip_stride))
        an.start()
        _CLIP_STATE["analyzer"] = an
        _CLIP_STATE["analysis_timer"] = _analysis_tick
        s.clip_analysis_active = True
        s.clip_analysis_progress = 0.0
        s.clip_analysis_status = "starting"
        bpy.app.timers.register(_analysis_tick, first_interval=0.2)
        self.report({"INFO"}, "Analysing %s with %s" % (clip.name, s.clip_engine))
        return {"FINISHED"}


class FPV_OT_clip_analyze_cancel(bpy.types.Operator):
    bl_idname = "fpv_pose.clip_analyze_cancel"
    bl_label = "Cancel Analysis"
    bl_description = "Stop the running analysis. Nothing is written"

    @classmethod
    def poll(cls, context):
        return _CLIP_STATE.get("analyzer") is not None

    def execute(self, context):
        an = _CLIP_STATE.get("analyzer")
        if an is not None:
            an.cancel()
        context.scene.fpv_pose.clip_analysis_status = "cancelling"
        return {"FINISHED"}


class FPV_OT_clip_delete_analysis(bpy.types.Operator):
    bl_idname = "fpv_pose.clip_delete_analysis"
    bl_label = "Delete Result"
    bl_description = "Delete the selected analysis, keeping the clip itself"

    @classmethod
    def poll(cls, context):
        return selected_analysis(context.scene.fpv_pose)[1] is not None

    def execute(self, context):
        s = context.scene.fpv_pose
        engine, sidecar = selected_analysis(s)
        if sidecar is None:
            return {"CANCELLED"}
        close_preview()
        try:
            os.remove(sidecar)
        except OSError as e:
            self.report({"ERROR"}, "Cannot delete: %s" % e)
            return {"CANCELLED"}
        refresh_clip_list(context)
        self.report({"INFO"}, "Deleted the %s result" % engine)
        return {"FINISHED"}


# ----------------------------------------------------------------------
# preview
# ----------------------------------------------------------------------

class FPV_OT_clip_preview_open(bpy.types.Operator):
    bl_idname = "fpv_pose.clip_preview_open"
    bl_label = "Show Pose Preview"
    bl_description = ("Draw the selected analysis over the footage so the "
                      "estimation can be checked before it drives the rig")

    @classmethod
    def poll(cls, context):
        return selected_analysis(context.scene.fpv_pose)[1] is not None

    def execute(self, context):
        if not refresh_clip_preview(context):
            s = context.scene.fpv_pose
            self.report({"ERROR"}, s.clip_preview_info or "Cannot read the clip")
            return {"CANCELLED"}
        return {"FINISHED"}


class FPV_OT_clip_preview_close(bpy.types.Operator):
    bl_idname = "fpv_pose.clip_preview_close"
    bl_label = "Close Preview"
    bl_description = "Release the clip being previewed"

    def execute(self, context):
        close_preview()
        context.scene.fpv_pose.clip_preview_info = ""
        img = bpy.data.images.get(CLIP_PREVIEW_IMAGE)
        if img is not None:
            try:
                bpy.data.images.remove(img)
            except Exception:
                pass
        _tag_redraw()
        return {"FINISHED"}


class FPV_OT_clip_preview_step(bpy.types.Operator):
    bl_idname = "fpv_pose.clip_preview_step"
    bl_label = "Step"
    bl_description = "Move the preview through the analysed clip"

    target: bpy.props.EnumProperty(
        items=[("FIRST", "First", ""), ("PREV", "Previous", ""),
               ("NEXT", "Next", ""), ("LAST", "Last", ""),
               ("NEXT_LOST", "Next Gap", "Jump to the next frame where the "
                                         "body was not detected")],
        default="NEXT",
        options={"HIDDEN"},
    )

    def execute(self, context):
        s = context.scene.fpv_pose
        prev = _preview_for(s)
        if prev is None or not len(prev):
            self.report({"ERROR"}, "Nothing analysed to preview")
            return {"CANCELLED"}
        last = len(prev) - 1
        i = max(0, min(last, int(s.clip_preview_frame)))
        if self.target == "FIRST":
            i = 0
        elif self.target == "LAST":
            i = last
        elif self.target == "PREV":
            i = max(0, i - 1)
        elif self.target == "NEXT":
            i = min(last, i + 1)
        else:
            # A lost stretch is what makes a take unusable, and finding it
            # by dragging a slider over 2000 frames is hopeless.
            found = next((j for j in range(i + 1, last + 1)
                          if not prev.detected(j)), None)
            if found is None:
                self.report({"INFO"}, "No further frames without a detection")
                return {"CANCELLED"}
            i = found
        s.clip_preview_frame = i
        refresh_clip_preview(context)
        return {"FINISHED"}


# ----------------------------------------------------------------------
# applying an analysis to the rig
# ----------------------------------------------------------------------

def _save_profile(context, data, source, used, complete):
    """Write a calibration to the library and select it. Returns the path."""
    s = context.scene.fpv_pose
    name = cliplib.safe_name(s.clip_calib_name, "default")
    path = cliplib.write_profile(library_root(s), name, {
        "version": 1,
        "name": name,
        "source": source,
        "frames": int(used),
        "complete": bool(complete),
        "calibration": data,
    })
    try:
        s.clip_calib_name = name
        s.clip_calib_profile = name
        s.clip_calib_source = "PROFILE"
    except (TypeError, ValueError):
        # The enum rebuilds from disk, so the new name is there; if
        # Blender still refuses it the file is written either way.
        pass
    return path


class FPV_OT_clip_record_calibration(bpy.types.Operator):
    bl_idname = "fpv_pose.clip_record_calibration"
    bl_label = "Record Calibration Clip"
    bl_description = ("Record a short clip of the rig's rest pose, into a "
                      "'calibration' category. Stand facing the camera with "
                      "your arms and hands where the armature's are, and "
                      "hold still until it stops")

    @classmethod
    def poll(cls, context):
        return FPV_OT_clip_record_start.poll(context)

    def execute(self, context):
        s = context.scene.fpv_pose
        # A calibration clip is a different thing from a take, so it gets
        # its own category and its own timings: long enough to settle and
        # hold, short enough that nobody is tempted to move.
        s.clip_category = "calibration"
        s.clip_subgroup = ""
        s.clip_name = "rest_pose"
        s.clip_countdown = max(int(s.clip_countdown), 5)
        s.clip_max_seconds = 6
        result = bpy.ops.fpv_pose.clip_record_start()
        if "FINISHED" not in result:
            return result
        self.report({"INFO"}, "Hold the rig's rest pose until it stops")
        return {"FINISHED"}


class FPV_OT_clip_calibrate(bpy.types.Operator):
    bl_idname = "fpv_pose.clip_calibrate"
    bl_label = "Calibrate from Clip"
    bl_description = ("Take the rest-pose reference from the selected clip "
                      "and save it as a calibration profile. Shoot a few "
                      "seconds of yourself matching the rig's rest pose, "
                      "analyse it, then press this once")

    @classmethod
    def poll(cls, context):
        s = context.scene.fpv_pose
        return (selected_analysis(s)[1] is not None
                and s.armature is not None
                and not recording.is_live())

    def execute(self, context):
        scene = context.scene
        s = scene.fpv_pose
        engine, sidecar = selected_analysis(s)
        if sidecar is None:
            self.report({"ERROR"}, "Analyse the calibration clip first")
            return {"CANCELLED"}
        if s.armature is None:
            self.report({"ERROR"}, "Pick an armature")
            return {"CANCELLED"}
        if recording.is_live():
            self.report({"ERROR"}, "Stop live tracking first")
            return {"CANCELLED"}
        if not any(ops.current_bone_map(scene).values()):
            self.report({"ERROR"}, "Bone map is empty. Run Auto-Map first.")
            return {"CANCELLED"}

        rt = ops.build_retargeter(scene, s.armature)
        try:
            status, used, complete = offline.calibrate(
                sidecar, rt,
                skip_seconds=float(s.clip_calib_skip),
                hold_seconds=float(s.clip_calib_hold))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.report({"ERROR"}, "Calibration failed: %s" % e)
            return {"CANCELLED"}
        # Reading a calibration clip drives the rig as a side effect; the
        # pose it left behind is one frame of a hold, not anything the
        # user asked to keep.
        rt.reset_pose()

        data = rt.export_calibration()
        if not data:
            s.clip_calib_status = status
            self.report({"ERROR"},
                        "Nothing to calibrate from: %s" % status)
            return {"CANCELLED"}
        clip = active_clip(scene)
        path = _save_profile(context, data,
                             "%s (%s)" % (clip.name if clip else "clip", engine),
                             used, complete)
        s.clip_calib_status = status
        level = {"INFO"} if complete else {"WARNING"}
        self.report(level, "%s - saved as %s"
                    % (status, os.path.basename(path)))
        return {"FINISHED"}


class FPV_OT_clip_calib_from_live(bpy.types.Operator):
    bl_idname = "fpv_pose.clip_calib_from_live"
    bl_label = "Save Live Calibration"
    bl_description = ("Save the calibration from the current or last live "
                      "tracking session as a profile, so clips can be "
                      "replayed against it")

    @classmethod
    def poll(cls, context):
        return ops.live_calibration() is not None

    def execute(self, context):
        data = ops.live_calibration()
        if not data:
            self.report({"ERROR"}, "No live calibration in this session")
            return {"CANCELLED"}
        path = _save_profile(context, data, "live session", 0, True)
        context.scene.fpv_pose.clip_calib_status = (
            "saved the live calibration as %s" % os.path.basename(path))
        self.report({"INFO"}, "Saved %s" % os.path.basename(path))
        return {"FINISHED"}


class FPV_OT_clip_calib_delete(bpy.types.Operator):
    bl_idname = "fpv_pose.clip_calib_delete"
    bl_label = "Delete Profile"
    bl_description = "Delete the selected calibration profile from the library"

    @classmethod
    def poll(cls, context):
        name = context.scene.fpv_pose.clip_calib_profile
        return bool(name) and name != "NONE"

    def execute(self, context):
        s = context.scene.fpv_pose
        name = s.clip_calib_profile
        if not cliplib.delete_profile(library_root(s), name):
            self.report({"ERROR"}, "Cannot delete profile '%s'" % name)
            return {"CANCELLED"}
        s.clip_calib_status = "deleted profile '%s'" % name
        self.report({"INFO"}, "Deleted '%s'" % name)
        return {"FINISHED"}


class FPV_OT_clip_apply(bpy.types.Operator):
    bl_idname = "fpv_pose.clip_apply"
    bl_label = "Apply to Rig"
    bl_description = ("Replay the selected analysis through the retargeter "
                      "and record it as a take, ready to review and bake "
                      "into an Action. No camera and no tracking involved")

    @classmethod
    def poll(cls, context):
        s = context.scene.fpv_pose
        return (selected_analysis(s)[1] is not None
                and s.armature is not None
                and not recording.is_live())

    def execute(self, context):
        scene = context.scene
        s = scene.fpv_pose
        engine, sidecar = selected_analysis(s)
        if sidecar is None:
            self.report({"ERROR"}, "Analyse the clip first")
            return {"CANCELLED"}
        armature = s.armature
        if armature is None:
            self.report({"ERROR"}, "Pick an armature")
            return {"CANCELLED"}
        if recording.is_live():
            self.report({"ERROR"}, "Stop live tracking first")
            return {"CANCELLED"}
        if recording.has_take():
            self.report({"WARNING"},
                        "Save or discard the current take before applying a clip")
            return {"CANCELLED"}
        bone_map = ops.current_bone_map(scene)
        if not any(bone_map.values()):
            self.report({"ERROR"}, "Bone map is empty. Run Auto-Map first.")
            return {"CANCELLED"}

        try:
            context.view_layer.objects.active = armature
            if armature.mode != "POSE":
                bpy.ops.object.mode_set(mode="POSE")
        except Exception:
            pass

        # The preview action would be evaluated on top of every replayed
        # frame, so the rig has to be its own again before the replay.
        recording.clear_preview(armature)
        rt = ops.build_retargeter(scene, armature)
        ok, calib_msg = prime_calibration(context, rt)
        if not ok:
            self.report({"ERROR"}, calib_msg)
            return {"CANCELLED"}
        s.clip_calib_status = calib_msg
        wm = context.window_manager
        wm.progress_begin(0, 100)

        def report_progress(done, total):
            wm.progress_update(int(100.0 * done / max(1, total)))

        try:
            count, tk = offline.replay(
                sidecar, rt, armature, on_progress=report_progress,
                calibrate_from_clip=(s.clip_calib_source == "CLIP"))
        except Exception as e:
            import traceback
            traceback.print_exc()
            recording.discard()
            wm.progress_end()
            self.report({"ERROR"}, "Replay failed: %s" % e)
            return {"CANCELLED"}
        wm.progress_end()
        if not count or tk is None:
            recording.discard()
            self.report({"ERROR"}, "The analysis produced no usable frames")
            return {"CANCELLED"}

        take = ops.finish_take(scene)
        if take is None:
            self.report({"ERROR"}, "Nothing was captured from the clip")
            return {"CANCELLED"}
        clip = active_clip(scene)
        if clip is not None and not s.take_action_name.strip().startswith(clip.name):
            # Name the action after the footage it came from; a library of
            # takes called "FPV Take" is a library of nothing.
            s.take_action_name = "%s_%s" % (clip.name, engine.lower())

        if s.clip_apply_action:
            recording.refresh_preview(context, force=True)
            bpy.ops.fpv_pose.take_save()
            self.report({"INFO"}, "Baked %d frames from %s (%s)"
                        % (count, clip.name if clip else "clip", engine))
            return {"FINISHED"}

        recording.refresh_preview(context, force=True)
        ops._open_review(context, self)
        self.report({"INFO"}, "Replayed %d frames (%s) — review and save"
                    % (count, engine))
        return {"FINISHED"}


class FPV_OT_clip_play(bpy.types.Operator):
    bl_idname = "fpv_pose.clip_play"
    bl_label = "Play on Rig"
    bl_description = ("Set the scene range to the rig's current action and "
                      "play it, so a recorded performance can be watched "
                      "without any tracking running")

    @classmethod
    def poll(cls, context):
        s = context.scene.fpv_pose
        armature = s.armature
        return (armature is not None and armature.animation_data is not None
                and armature.animation_data.action is not None)

    def execute(self, context):
        scene = context.scene
        action = scene.fpv_pose.armature.animation_data.action
        start, end = (int(round(v)) for v in action.frame_range)
        if end <= start:
            self.report({"ERROR"}, "Action '%s' has no keyframes" % action.name)
            return {"CANCELLED"}
        scene.frame_start = start
        scene.frame_end = end
        scene.frame_set(start)
        try:
            bpy.ops.screen.animation_cancel(restore_frame=False)
            bpy.ops.screen.animation_play()
        except Exception as e:
            self.report({"WARNING"},
                        "Range set to %d-%d; press Play (%s)" % (start, end, e))
            return {"FINISHED"}
        self.report({"INFO"}, "Playing '%s' (frames %d-%d)"
                    % (action.name, start, end))
        return {"FINISHED"}


class FPV_UL_clips(bpy.types.UIList):
    """One row per clip: where it sits, how long it is, what has been run
    over it."""

    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_propname, index):
        row = layout.row(align=True)
        row.label(text=item.label or item.name, icon="FILE_MOVIE")
        sub = row.row(align=True)
        sub.alignment = "RIGHT"
        if item.duration:
            sub.label(text="%.1fs" % item.duration)
        if item.analyses:
            sub.label(text=item.analyses, icon="CHECKMARK")
        else:
            sub.label(text="not analysed", icon="DOT")


_CLASSES = (
    FPV_UL_clips,
    FPV_OT_clip_scan,
    FPV_OT_clip_open_folder,
    FPV_OT_clip_open_video,
    FPV_OT_clip_record_start,
    FPV_OT_clip_record_stop,
    FPV_OT_clip_delete,
    FPV_OT_clip_analyze,
    FPV_OT_clip_analyze_cancel,
    FPV_OT_clip_delete_analysis,
    FPV_OT_clip_preview_open,
    FPV_OT_clip_preview_close,
    FPV_OT_clip_preview_step,
    FPV_OT_clip_record_calibration,
    FPV_OT_clip_calibrate,
    FPV_OT_clip_calib_from_live,
    FPV_OT_clip_calib_delete,
    FPV_OT_clip_apply,
    FPV_OT_clip_play,
)


def _shutdown():
    """Stop anything this module owns. Called on unregister, and on reload
    of the addon, where a live thread would otherwise keep writing to a
    file whose module is gone."""
    rec = _CLIP_STATE.get("recorder")
    if rec is not None:
        rec.stop(timeout=2.0)
    _CLIP_STATE["recorder"] = None
    an = _CLIP_STATE.get("analyzer")
    if an is not None:
        an.cancel()
    _CLIP_STATE["analyzer"] = None
    for key in ("rec_timer", "analysis_timer"):
        fn = _CLIP_STATE.get(key)
        if fn is not None:
            try:
                bpy.app.timers.unregister(fn)
            except Exception:
                pass
        _CLIP_STATE[key] = None
    close_preview()


def register():
    for c in _CLASSES:
        bpy.utils.register_class(c)


def unregister():
    _shutdown()
    for c in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(c)
        except Exception:
            pass
