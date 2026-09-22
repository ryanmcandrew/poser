"""Scene properties for the addon UI."""

import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty,
                       FloatProperty, IntProperty, PointerProperty,
                       StringProperty)
from bpy.types import PropertyGroup

from . import recording


# Set while code (rather than the user) writes a take property, so the
# preview is rebuilt once at the end instead of once per row touched.
_MUTE = {"on": False}


def _take_update(self, context):
    """A take edit changed: rebuild the preview action."""
    if _MUTE["on"]:
        return
    recording.mark_dirty()
    try:
        recording.refresh_preview(context)
    except Exception as e:
        print("[FPV Pose Tracker] preview refresh failed:", e)


def _frame_map_update(self, context):
    """Start frame or scene rate changed: the rows' timeline frames move."""
    if _MUTE["on"]:
        return
    try:
        recording.resync_frames(context.scene)
    except Exception as e:
        print("[FPV Pose Tracker] frame resync failed:", e)
    _take_update(self, context)


def _preview_update(self, context):
    """Turning the preview off has to take the preview Action back off the
    rig, not just stop rebuilding it."""
    if _MUTE["on"]:
        return
    if self.take_preview:
        _take_update(self, context)
        return
    tk = recording.take()
    armature = self.armature
    if tk is not None and (armature is None or armature.name != tk.armature_name):
        armature = bpy.data.objects.get(tk.armature_name)
    recording.clear_preview(armature)


def _last_sample():
    tk = recording.take()
    return max(0, len(tk.samples) - 1) if tk is not None else 0


def _start_update(self, context):
    """Keep the trim handles in range and in order. Dragging the start past
    the end collapses the range rather than inverting it."""
    if _MUTE["on"]:
        return
    n = _last_sample()
    _MUTE["on"] = True
    try:
        self.take_start = max(0, min(self.take_start, n))
        if self.take_end < self.take_start:
            self.take_end = self.take_start
    finally:
        _MUTE["on"] = False
    _take_update(self, context)


def _end_update(self, context):
    if _MUTE["on"]:
        return
    n = _last_sample()
    _MUTE["on"] = True
    try:
        self.take_end = max(0, min(self.take_end, n))
        if self.take_start > self.take_end:
            self.take_start = self.take_end
    finally:
        _MUTE["on"] = False
    _take_update(self, context)


MULTI_CAM = "MULTI"

# Blender keeps only a borrowed reference to the strings an EnumProperty
# items callback returns, so they have to outlive the call.
_CAM_ITEMS = []


def scanned_camera_indices(s):
    """Device indices from the last 'Scan', as a list of ints."""
    out = []
    for token in (s.cameras_found or "").replace(",", " ").split():
        if token.isdigit():
            out.append(int(token))
    return sorted(set(out))


def _camera_items(self, context):
    """Devices to offer, plus the multi-camera entry.

    Every item carries an explicit number so a rescan that finds a
    different set of cameras cannot silently reassign the saved choice.
    """
    found = scanned_camera_indices(self)
    if not found:
        found = [0, 1, 2, 3]
    if int(self.camera_index) not in found:
        found.append(int(self.camera_index))
    items = [("CAM%d" % i, "Webcam %d" % i,
              "Capture from device index %d" % i, "CAMERA_DATA", i)
             for i in sorted(set(found))]
    items.append((MULTI_CAM, "Source Multiple Cameras",
                  "Capture from several webcams at once and fuse their pose "
                  "estimates. Each camera measures across its own view far "
                  "better than along it, so two cameras at an angle to each "
                  "other cover each other's depth blind spot",
                  "OUTLINER_OB_CAMERA", 999))
    _CAM_ITEMS[:] = items
    return _CAM_ITEMS


def ensure_camera_slots(s, count=2):
    """Fill the device list with as many rows as we can name devices for."""
    found = scanned_camera_indices(s)
    while len(s.cameras) < max(1, count):
        dev = s.cameras.add()
        used = {c.index for c in s.cameras[:-1]}
        free = [i for i in found if i not in used]
        if free:
            dev.index = free[0]
        else:
            dev.index = max(used) + 1 if used else int(s.camera_index)


def _camera_source_update(self, context):
    value = self.camera_source
    if value != MULTI_CAM and value.startswith("CAM"):
        try:
            self.camera_index = int(value[3:])
        except ValueError:
            pass
        return
    if value == MULTI_CAM and len(self.cameras) < 2:
        ensure_camera_slots(self, 2)


# ----------------------------------------------------------------------
# clip library
# ----------------------------------------------------------------------

def _clip_dirs_update(self, context):
    """Category or subgroup changed: re-list what is in that folder."""
    if _MUTE["on"]:
        return
    try:
        from . import clip_ops
        clip_ops.refresh_clip_list(context)
    except Exception as e:
        print("[FPV Pose Tracker] clip rescan failed:", e)


def _clip_select_update(self, context):
    """A different clip is active: its analyses and preview follow it."""
    if _MUTE["on"]:
        return
    try:
        from . import clip_ops
        clip_ops.clip_selection_changed(context)
    except Exception as e:
        print("[FPV Pose Tracker] clip selection failed:", e)


def _clip_preview_update(self, context):
    if _MUTE["on"]:
        return
    try:
        from . import clip_ops
        clip_ops.refresh_clip_preview(context)
    except Exception as e:
        print("[FPV Pose Tracker] clip preview failed:", e)


# Blender keeps no reference to the strings an EnumProperty callback
# returns, so a list built inline is freed while the menu still points at
# it and the UI draws garbage. Holding the last result fixes that.
_ANALYSIS_ITEMS = []


def _analysis_items(self, context):
    """Which analyses exist for the selected clip."""
    from . import cliplib
    _ANALYSIS_ITEMS.clear()
    path = (self.clip_selected_path or "").strip()
    if path:
        for engine, sidecar in cliplib.list_analyses(path):
            header = cliplib.read_header(sidecar)
            detail = "%s, %d frames" % (header.get("quality", "?"),
                                        int(header.get("frames", 0)))
            rate = header.get("detect_rate")
            if rate is not None:
                detail += ", %.0f%% detected" % (100.0 * rate)
            _ANALYSIS_ITEMS.append((engine, "%s (%s)" % (engine, detail), detail))
    if not _ANALYSIS_ITEMS:
        _ANALYSIS_ITEMS.append(("NONE", "No analysis yet",
                                "Run Analyze Clip to estimate pose over this clip"))
    return _ANALYSIS_ITEMS


# Same lifetime trap as `_ANALYSIS_ITEMS`: the list has to outlive the
# call that builds it.
_PROFILE_ITEMS = []


def _profile_items(self, context):
    """Calibration profiles saved in the current library."""
    from . import cliplib
    _PROFILE_ITEMS.clear()
    try:
        root = cliplib.resolve_root(self.clip_root)
        for name in cliplib.list_profiles(root):
            data = cliplib.read_profile(root, name) or {}
            detail = data.get("source") or "saved calibration"
            _PROFILE_ITEMS.append((name, name, detail))
    except Exception as e:
        print("[FPV Pose Tracker] cannot list calibration profiles:", e)
    if not _PROFILE_ITEMS:
        _PROFILE_ITEMS.append(
            ("NONE", "No profile saved",
             "Record a clip of the rig's rest pose, analyse it, then press "
             "Calibrate from Clip"))
    return _PROFILE_ITEMS


class FPVNameItem(PropertyGroup):
    """A folder name, mirrored into the scene so prop_search can offer it."""


class FPVClipItem(PropertyGroup):
    """One recorded clip in the library, as the UI list needs it."""

    path: StringProperty()
    label: StringProperty()
    category: StringProperty()
    subgroup: StringProperty()
    frames: IntProperty()
    fps: FloatProperty()
    duration: FloatProperty()
    size_mb: FloatProperty()
    analyses: StringProperty(
        description="Engines this clip has already been analysed with")


class FPVCameraDevice(PropertyGroup):
    """One webcam in a multi-camera rig."""

    index: IntProperty(
        name="Device", default=0, min=0, max=16,
        description="OpenCV device index for this webcam. Use Scan to see "
                    "which indices exist; two rows must not share one",
    )
    enabled: BoolProperty(
        name="Use", default=True,
        description="Include this camera in the fused pose. Untick to drop a "
                    "camera without losing its settings",
    )
    fov: FloatProperty(
        name="FOV", default=65.0, min=30.0, max=120.0,
        description="Horizontal field of view of THIS camera in degrees. Every "
                    "depth estimate it contributes scales with it, so a second "
                    "camera with a different lens needs its own value",
    )
    yaw: FloatProperty(
        name="Yaw", default=0.0, min=-180.0, max=180.0,
        description="Where this camera stands around you in degrees, measured "
                    "from straight in front, positive to your right. 0 = facing "
                    "you head on, 90 = off your right shoulder. This is what "
                    "lets the fused pose take each axis from the camera that "
                    "measures it across its view instead of along it, so get it "
                    "roughly right (to within ~15 deg)",
    )
    trust: FloatProperty(
        name="Trust", default=1.0, min=0.0, max=4.0,
        description="Weight this camera's estimates by this much relative to "
                    "the others. Lower it for a camera that is further away, "
                    "lower resolution or badly lit",
    )
    exposure_mode: EnumProperty(
        name="Exposure",
        items=[
            ("SHARED", "Shared", "Use the Exposure setting from Capture "
                                 "Throughput for this camera"),
            ("AUTO", "Auto", "Force automatic exposure on this camera"),
            ("MANUAL", "Manual", "Pin a short exposure on this camera alone. "
                                 "An older webcam in a dim room will drop to "
                                 "half its frame rate on auto exposure, and it "
                                 "is usually the older camera in a pair that "
                                 "needs pinning while the newer one does not"),
        ],
        default="SHARED",
        description="Exposure control for this camera. Cameras in a pair rarely "
                    "want the same value: exposure is the single biggest lever "
                    "on an old webcam's frame rate",
    )
    exposure: FloatProperty(
        name="Exp", default=-6.0, min=-13.0, max=0.0,
        description="Manual exposure for this camera, in the camera's own log2 "
                    "seconds: -6 is about 1/64 s. Lower = shorter = sharper and "
                    "faster, but darker",
    )
    measured_fps: FloatProperty(default=0.0)


class FPVTakeFrame(PropertyGroup):
    """One captured sample, mirrored into the scene for the review UI."""

    index: IntProperty()
    frame: IntProperty()
    score: FloatProperty()
    step: FloatProperty()
    flags: IntProperty()
    enabled: BoolProperty(
        name="Keep", default=True, update=_take_update,
        description="Key this sample. Unticked samples are skipped, so the "
                    "curve interpolates straight across them",
    )

    def set_enabled_quiet(self, value):
        _MUTE["on"] = True
        try:
            self.enabled = value
        finally:
            _MUTE["on"] = False


class FPVBoneMap(PropertyGroup):
    shoulder_L: StringProperty(name="Shoulder L")
    upper_arm_L: StringProperty(name="Upper Arm L")
    forearm_L: StringProperty(name="Forearm L")
    hand_L: StringProperty(name="Hand L")
    shoulder_R: StringProperty(name="Shoulder R")
    upper_arm_R: StringProperty(name="Upper Arm R")
    forearm_R: StringProperty(name="Forearm R")
    hand_R: StringProperty(name="Hand R")


def _add_finger_props():
    ann = FPVBoneMap.__annotations__
    for side in ("L", "R"):
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            for i in (1, 2, 3):
                slot = f"{finger}_0{i}_{side}"
                ann[slot] = StringProperty(name=slot)


_add_finger_props()


class FPVPoseSettings(PropertyGroup):
    armature: PointerProperty(
        name="Armature",
        type=bpy.types.Object,
        poll=lambda self, obj: obj and obj.type == "ARMATURE",
    )
    camera_index: IntProperty(name="Camera Index", default=0, min=0, max=16)
    camera_source: EnumProperty(
        name="Webcam",
        items=_camera_items,
        update=_camera_source_update,
        description="Which webcam to track from, or 'Source Multiple Cameras' "
                    "to combine several. Press Scan to populate the list",
    )
    cameras: CollectionProperty(type=FPVCameraDevice)
    cameras_active_index: IntProperty(default=0)
    multi_solver: EnumProperty(
        name="Fusion",
        items=[
            ("REPROJECT", "Fit Body to Both Views",
             "Fit one skeleton to every camera's 2D landmarks at once, using "
             "your measured limb lengths. A joint two cameras can see gets a "
             "measured depth instead of an averaged guess, which is most of "
             "the accuracy a second camera can give"),
            ("AVERAGE", "Average Each Camera's 3D",
             "Combine the 3D each camera infers on its own, weighted by Yaw "
             "and Depth Trust. Cheaper, and it needs no camera placement, but "
             "both cameras guess depth the same way so averaging them cancels "
             "little of the error"),
        ],
        default="REPROJECT",
        description="How several cameras are combined into one pose",
    )
    multi_bone_stiffness: FloatProperty(
        name="Skeleton Stiffness",
        default=0.025, min=0.005, max=0.15, precision=3, unit="LENGTH",
        description="How far the fit may stretch a limb from the length you "
                    "were measured at, in metres. Tighter keeps the skeleton "
                    "rigid and rejects more landmark noise; looser lets it "
                    "follow the cameras when the measured lengths are off. "
                    "Only used by the body fit",
    )
    multi_extrinsics: StringProperty(
        default="",
        description="Where each camera was measured to be, saved with the "
                    "scene. Cleared by Calibrate Placement",
    )
    multi_calib_info: StringProperty(default="")
    multi_depth_trust: FloatProperty(
        name="Depth Trust",
        default=0.35, min=0.05, max=1.0,
        description="How much a camera's forward (depth) axis is worth next to "
                    "its lateral ones when the cameras are fused. A single "
                    "webcam infers depth from foreshortening and gets it far "
                    "less accurately than it measures sideways motion, so the "
                    "default 0.35 lets a camera off to the side supply the "
                    "depth the front camera is guessing at. 1.0 weights every "
                    "axis equally, which is a plain average",
    )
    multi_sync_ms: FloatProperty(
        name="Sync Window (ms)",
        default=120.0, min=20.0, max=500.0,
        description="A camera whose newest frame is older than this is skipped "
                    "for that fused frame rather than smearing a stale limb "
                    "into the current pose. Raise it if a slower camera keeps "
                    "dropping out of the fusion",
    )
    mirror: BoolProperty(
        name="Mirror Webcam", default=True,
        description="Flip the feed horizontally. Keep ON for Selfie mode (the rig's left follows your left) and OFF for Body Cam mode. Pose/hand left-right labels are corrected automatically for either setting.",
    )
    pose_engine: EnumProperty(
        name="Pose Engine",
        items=[
            ("MEDIAPIPE", "MediaPipe Holistic",
             "33-point pose plus 21-point hands, with MediaPipe's own metric "
             "3D pose. By far the fastest and the only engine here that "
             "reports depth itself, but the least accurate on shoulders, "
             "elbows and wrists."),
            ("RTMPOSE", "RTMPose (accurate, real-time)",
             "Top-down RTMPose over ONNX Runtime: a person detector, then a "
             "body model, then a dedicated hand model on crops the body "
             "found. Markedly better arm and torso landmarks at close to "
             "MediaPipe's frame rate. 2D only, so arm depth comes from the "
             "depth lifter. Needs rtmlib + onnxruntime."),
            ("VITPOSE", "ViTPose (most accurate, slowest)",
             "The same top-down stack with a Vision Transformer body "
             "backbone — the most accurate body landmarks available here, "
             "and the heaviest. Sensible with a CUDA GPU; expect well under "
             "real time on CPU. Needs rtmlib + onnxruntime."),
        ],
        default="MEDIAPIPE",
        description="Which pose estimation system drives the rig. Changing "
                    "this requires Stop -> Start; models download on first use.",
    )
    model_complexity: IntProperty(
        name="Model Complexity", default=1, min=0, max=2,
        description="MediaPipe pose model: 0 = lite (fastest), 1 = full, 2 = heavy (most accurate landmarks, slowest). Changing this requires Stop -> Start; the model downloads on first use.",
    )
    engine_quality: EnumProperty(
        name="Quality",
        items=[
            ("LITE", "Lite", "Smallest body model and detector (RTMPose-s / "
                             "ViTPose-s, YOLOX-tiny). Fastest, and still "
                             "ahead of MediaPipe on arm landmarks."),
            ("BALANCED", "Balanced", "RTMPose-m / ViTPose-b with YOLOX-m. The "
                                     "recommended setting."),
            ("ACCURATE", "Accurate", "RTMPose-x at 384x288 / ViTPose-l with "
                                     "YOLOX-x. Best landmarks, several times "
                                     "the cost."),
        ],
        default="BALANCED",
        description="Model size for the RTMPose/ViTPose engines. Changing "
                    "this requires Stop -> Start.",
    )
    engine_layout: EnumProperty(
        name="Keypoints",
        items=[
            ("BODY", "Body (17)", "COCO-17: head, shoulders, elbows, wrists, "
                                  "hips, knees, ankles. Everything the arm "
                                  "retargeter uses, at the lowest cost."),
            ("FEET", "Body + Feet (26)", "Adds heels and toes, which a "
                                         "standing full-body rig wants. "
                                         "Halpe-26 / COCO-25."),
            ("WHOLEBODY", "Whole Body (133)",
             "Body, feet, face and both hands from one model (RTMW / "
             "ViTPose-wholebody). The hands come out of the body pass for "
             "free, but at body resolution — less precise than the dedicated "
             "hand model."),
        ],
        default="BODY",
        description="Which keypoint set the body model predicts. Changing "
                    "this requires Stop -> Start.",
    )
    engine_hands: EnumProperty(
        name="Hand Source",
        items=[
            ("AUTO", "Auto", "Use the body model's own hand points when it "
                             "has them (Whole Body), otherwise the dedicated "
                             "RTMPose hand model."),
            ("RTM", "RTMPose Hand (ROI)",
             "A 21-point hand model run on a high-resolution crop around "
             "each wrist, using the body pose to place the crop. The best "
             "option for fingers, and the architecture MediaPipe's Holistic "
             "uses internally."),
            ("WHOLEBODY", "From Body Model",
             "Take the hand points out of a 133-keypoint body result. Free, "
             "but limited by the body crop's resolution. Falls back to the "
             "ROI hand model if the body model is not Whole Body."),
            ("MEDIAPIPE", "MediaPipe Hands",
             "Pair an RTMPose/ViTPose body with MediaPipe's hand landmarker, "
             "which is fast and supplies metric hand geometry the depth "
             "estimate can use. Needs MediaPipe installed as well."),
        ],
        default="AUTO",
        description="Where finger landmarks come from when the engine is not "
                    "MediaPipe. Changing this requires Stop -> Start.",
    )
    engine_device: EnumProperty(
        name="Compute",
        items=[
            ("CPU", "CPU", "Always available."),
            ("GPU", "GPU (CUDA)", "Use the CUDA execution provider, falling "
                                  "back to CPU if onnxruntime-gpu is not "
                                  "installed. Several times faster, and close "
                                  "to required for ViTPose."),
            ("MPS", "GPU (Apple)", "Use the CoreML execution provider on "
                                   "Apple silicon, falling back to CPU."),
        ],
        default="CPU",
        description="Execution provider for the ONNX engines. Changing this "
                    "requires Stop -> Start.",
    )
    engine_det_every: IntProperty(
        name="Detect Every",
        default=6, min=1, max=60,
        description="Run the person detector every N frames; in between, the "
                    "person box is tracked forward from the previous frame's "
                    "keypoints. The detector is the expensive half of a "
                    "top-down stack and a webcam subject does not teleport, "
                    "so 5-10 costs almost nothing in accuracy. 1 detects "
                    "every frame.",
    )
    engine_kpt_thr: FloatProperty(
        name="Keypoint Threshold",
        default=0.3, min=0.0, max=1.0,
        description="Keypoints scoring below this are reported as unseen "
                    "rather than driving a bone. Raise it if the rig twitches "
                    "toward guessed joints, lower it to keep partly occluded "
                    "arms alive.",
    )
    engine_hand_detector: BoolProperty(
        name="Hand Fallback Detector",
        default=True,
        description="Also load RTMPose's own hand detector, used when no body "
                    "pose is available to place the hand crops (Hands Only "
                    "mode, or the subject not found). Costs one more model in "
                    "memory.",
    )
    body_shoulder_width: FloatProperty(
        name="Shoulder Width",
        default=0.0, min=0.0, max=1.0, subtype="DISTANCE",
        description="Your shoulder-to-shoulder width in metres, used to scale "
                    "the depth lift. 0 = learn it from the engine's 3D pose, "
                    "which only MediaPipe provides — with RTMPose or ViTPose "
                    "an unset value falls back to an average body, so measure "
                    "yourself here for accurate arm depth. Changing this "
                    "requires Stop -> Start.",
    )
    body_upper_arm: FloatProperty(
        name="Upper Arm",
        default=0.0, min=0.0, max=1.0, subtype="DISTANCE",
        description="Shoulder-to-elbow length in metres. 0 = learn or assume. "
                    "Changing this requires Stop -> Start.",
    )
    body_forearm: FloatProperty(
        name="Forearm",
        default=0.0, min=0.0, max=1.0, subtype="DISTANCE",
        description="Elbow-to-wrist length in metres. 0 = learn or assume. "
                    "Changing this requires Stop -> Start.",
    )
    engine_info: StringProperty(default="")
    angle_info: StringProperty(default="")
    tracking_mode: EnumProperty(
        name="Tracking Mode",
        items=[
            ("HANDS", "Hands Only", "MediaPipe Hands — fingers only, arms untouched. Use if you don't want any pose estimation."),
            ("HOLISTIC", "Body + Hands", "MediaPipe Pose + Hands. Drives arms/shoulders from pose (even partly off-camera) plus per-finger tracking."),
            ("POSE", "Pose Only", "Body pose only, no per-finger tracking."),
        ],
        default="HOLISTIC",
    )
    view_mode: EnumProperty(
        name="View Mode",
        items=[
            ("BODY_CAM", "Body Cam",
             "Camera worn on the user's body, looking forward at their own arms/hands (FPV). "
             "Arm predictions can be noisy because shoulders/elbows are usually off-frame."),
            ("SELFIE", "Selfie",
             "Camera in front of the user, seeing their torso, arms and hands. "
             "A short calibration (hold the rig's rest pose) measures the pose-axis convention and each palm's rest orientation; "
             "arms and hands are then driven relative to that reference."),
        ],
        default="BODY_CAM",
        description="How the webcam is positioned relative to the user. Changing this requires Stop -> Start.",
    )
    init_frames: IntProperty(name="Init Frames (unused)", default=6, min=1, max=60)
    test_report: StringProperty(default="")
    smoothing: FloatProperty(
        name="Hand Smoothing",
        default=0.6, min=0.0, max=0.99,
        description="Smoothing applied to per-finger bones, as the fraction "
                    "retained per frame at 30 fps. Rescaled to the real frame "
                    "interval, so the response stays the same in milliseconds "
                    "whatever the capture rate. Higher = smoother, more lag.",
    )
    pose_smoothing: FloatProperty(
        name="Arm Smoothing",
        default=0.9, min=0.0, max=0.99,
        description="Smoothing applied to shoulder/upper-arm/forearm/hand bones, "
                    "as the fraction retained per frame at 30 fps, rescaled to the "
                    "real frame interval. The default 0.9 is a ~320 ms settling "
                    "time, chosen to hide noisy off-frame pose predictions — far "
                    "too slow to capture a punch. Use the Motion Response presets.",
    )
    pose_visibility: FloatProperty(
        name="Arm Visibility Threshold",
        default=0.5, min=0.0, max=1.0,
        description="Skip arm updates when a source pose landmark's visibility is below this. Off-camera joints (shoulders, elbows in body-cam view) come back with low visibility; freezing them prevents random jitter.",
    )
    use_arm_ik: BoolProperty(
        name="Arm IK",
        default=True,
        description="Solve upper_arm + forearm together with 2-bone IK anchored on the rig's rest shoulder joint, driven by the (filtered) pose wrist. Removes shoulder/elbow hallucinations as a noise source.",
    )
    shoulder_mode: EnumProperty(
        name="Shoulders",
        items=[
            ("REACH", "Reach (clavicle)",
             "Rotate the collar bone so the shoulder joint travels toward a wrist "
             "the arm cannot otherwise reach, the way a real scapula slides "
             "forward when you reach across yourself. The rotation comes from "
             "reach geometry, not from MediaPipe's noisy shoulder landmarks, and "
             "it lets sideways hand motion keep its true scale instead of being "
             "squashed to make a wide rig's hands meet."),
            ("FREEZE", "Frozen",
             "Lock the collar bones to rest. Steadiest, but the rig can only "
             "reach where its own proportions allow, so hand positions get "
             "distorted sideways to compensate."),
            ("TRACK", "Track Shoulder Line",
             "Drive the collar bones from the tracked shoulder landmarks. "
             "MediaPipe's shoulder Z jitters by ~20 cm frame to frame, so this "
             "is usually the worst of the three."),
        ],
        default="REACH",
    )
    shoulder_protract: FloatProperty(
        name="Max Protraction",
        default=30.0, min=0.0, max=70.0,
        description="How far the collar bone may rotate from rest, in degrees. "
                    "0 behaves like Frozen. Large values let the rig reach "
                    "further but make the shoulders visibly roll.",
    )
    freeze_shoulder: BoolProperty(
        name="Freeze Shoulders",
        default=True,
        description="Lock the shoulder bones to their rest pose. MediaPipe's 3D shoulder Z estimates jitter by ~20cm frame-to-frame even when the shoulders are visible; freezing looks vastly cleaner than trying to drive them. Turn off if you have a full-body view and want to try tracking the shoulder line.",
    )
    pose_y_up: BoolProperty(
        name="Pose Y-Up (Body Cam only)",
        default=True,
        description="Axis convention assumed for pose_world landmarks in Body Cam mode. Selfie mode measures the convention during calibration and ignores this. Untick if the arms end up upside-down in Body Cam mode.",
    )
    track_palm: BoolProperty(
        name="Track Palm Orientation",
        default=True,
        description="Drive the wrist bone with a full 3D orientation built from the tracked hand landmarks so palm-up vs palm-down actually rotates the wrist. Without this the wrist bone follows only the fingers direction, leaving roll (palm rotation) undefined.",
    )
    # Deprecated: kept so old scenes still load. Calibration replaced them.
    flip_palm: BoolProperty(name="Flip Palm Normal (unused)", default=False)
    flip_fingers: BoolProperty(name="Flip Finger Direction (unused)", default=False)

    reset_pose_on_start: BoolProperty(
        name="Reset Pose on Start",
        default=True,
        description="Return every mapped bone to its rest transform (rotation, location AND scale) when tracking starts. A stray -1 scale or leftover rotation on an arm bone reflects/offsets the whole chain and makes hands point the wrong way.",
    )
    auto_calibrate: BoolProperty(
        name="Calibrate on Start",
        default=True,
        description="Automatically run the calibration countdown when tracking starts. Hold the rig's rest pose (for FPS arms: arms forward toward the camera, palms down, fingers relaxed) until the status says 'calibrated'.",
    )
    calibration_delay: FloatProperty(
        name="Calibration Delay (s)",
        default=3.0, min=0.0, max=15.0,
        description="Seconds between pressing Start/Calibrate and the first sample, so you have time to get into the rest pose.",
    )
    calibration_frames: IntProperty(
        name="Calibration Frames",
        default=15, min=1, max=120,
        description="Frames averaged for each calibration component (torso frame, left palm, right palm). More = steadier reference, longer hold.",
    )
    anchor_mode: EnumProperty(
        name="Wrist Anchor",
        items=[
            ("AUTO", "Auto", "Torso centre in Selfie mode, shoulder in Body Cam mode."),
            ("TORSO", "Torso Centre",
             "Place each wrist relative to the point between the shoulders, scaled by rig/user shoulder width (sideways) and reach (forward/up). Hands that touch in front of you touch on the rig even if the rig's shoulders are much wider than yours."),
            ("SHOULDER", "Shoulder",
             "Use only the shoulder->wrist direction from tracking, anchored at the rig's own shoulder joint. Robust when the far shoulder is off-camera, but hand-to-hand distance follows the rig's proportions, not yours."),
        ],
        default="AUTO",
    )
    elbow_from_angle: BoolProperty(
        name="Elbow Angle Drives Reach",
        default=True,
        description="When the tracked elbow reads as nearly straight, set the shoulder->wrist distance from that angle instead of the tracked wrist depth. MediaPipe compresses depth when an arm points at the camera, which otherwise makes a straight arm look bent.",
    )
    hand_depth_mode: EnumProperty(
        name="Hand Depth Sign",
        items=[
            ("AUTO", "Auto", "Detect at calibration: if the rig's rest hands point forward, the user's fingers must too; flip depth if they read backwards."),
            ("NORMAL", "Normal", "MediaPipe convention: smaller z = closer to camera."),
            ("FLIPPED", "Flipped", "Force the opposite depth sign for hand landmarks."),
        ],
        default="AUTO",
    )
    pose_source: EnumProperty(
        name="Arm Depth",
        items=[
            ("LIFT", "2D + Limb Lengths (perspective)",
             "Lift the 2D pose landmarks to 3D along camera rays using Camera FOV and your limb lengths (estimated from tracking). Foreshortening in the image becomes real depth, so an arm pointed at the camera reads as extended. Set Camera FOV correctly."),
            ("WORLD", "MediaPipe World",
             "Use MediaPipe's own 3D world landmarks. Depth is a learned guess and compresses when a limb points at the camera. Selfie mode measures its axis convention at calibration."),
        ],
        default="LIFT",
        description="Where the arm joints' 3D positions come from. Changing this requires Stop -> Start.",
    )
    hand_depth_from_size: BoolProperty(
        name="Hand Size Drives Depth",
        default=True,
        description="Measure each hand's distance from the camera from its size in the image (MediaPipe's metric hand model + a perspective-n-point solve) and use it to decide whether the elbow and wrist sit toward or away from the camera. Without this the near/far choice relies on MediaPipe's image depth ordering, which cannot tell an arm reaching forward from one reaching back. Changing this requires Stop -> Start.",
    )
    hand_size: FloatProperty(
        name="Hand Size",
        default=1.0, min=0.5, max=2.0,
        description="Your hand size relative to MediaPipe's average hand model. Scales the measured hand depth: 1.1 = hands 10%% larger than average. Applied on top of the automatic estimate. Changing this requires Stop -> Start.",
    )
    hand_size_auto: BoolProperty(
        name="Learn Hand Size",
        default=True,
        description="Refine the hand size automatically from frames where the arm lies in the image plane (hanging down or out to the side), where the pose lifter's own depth needs no near/far cue. The learned factor is shown in the preview HUD as 'x1.03'. Changing this requires Stop -> Start.",
    )
    record_diagnostics: BoolProperty(
        name="Record Frames",
        default=False,
        description="Write every tracked frame's raw MediaPipe output (pose + hands, image and metric landmarks, frame size, FOV) to a .jsonl file so a session can be replayed offline with tools/replay_diag.py. Off by default; a few MB per minute. Applies on Stop -> Start.",
    )
    record_path: StringProperty(
        name="Record File",
        default="",
        subtype="FILE_PATH",
        description="Where to write the recording. Empty = 'fpv_pose_diag.jsonl' next to the .blend file (or in the temp folder if the file is unsaved).",
    )
    calib_status: StringProperty(default="not calibrated")
    verbose_hand_log: BoolProperty(
        name="Verbose Hand Log",
        default=False,
        description="Print rate-limited hand-landmark and computed-basis values to the system console. Use to diagnose orientation bugs — turn on, reproduce the pose, then read the console output.",
    )
    one_euro_min_cutoff: FloatProperty(
        name="Filter Min Cutoff",
        default=1.0, min=0.05, max=10.0,
        description="One-Euro min cutoff (Hz). Lower = smoother at rest, more lag when moving.",
    )
    one_euro_beta: FloatProperty(
        name="Filter Beta",
        default=0.6, min=0.0, max=5.0,
        description="One-Euro speed coefficient. Higher = filter opens up faster during fast motion.",
    )
    sensitivity: FloatProperty(name="Sensitivity", default=1.0, min=0.1, max=3.0)
    fpv_offset_forward: FloatProperty(name="Forward Offset", default=0.0)

    camera_fov: FloatProperty(
        name="Camera FOV (deg)",
        default=65.0, min=30.0, max=120.0,
        description="Horizontal field of view of the webcam in degrees. Every depth estimate (shoulders, elbows, wrists, hand size) scales with it, so it must be close to the real value: too small exaggerates depth, too large flattens it and above ~110 breaks it. Typical laptop/USB webcams: 60-78 deg (check the camera's spec sheet; 'diagonal' FOV values are larger than horizontal).",
    )
    capture_width: IntProperty(
        name="Capture Width",
        default=0, min=0, max=4096,
        description="Requested webcam capture width in pixels. 0 = whatever the camera defaults to. Some cameras change FOV when you change resolution.",
    )
    capture_height: IntProperty(
        name="Capture Height",
        default=0, min=0, max=4096,
        description="Requested webcam capture height in pixels. 0 = camera default.",
    )

    show_preview_window: BoolProperty(
        name="Show Preview Window",
        default=True,
        description="Open a separate OpenCV window with the annotated webcam feed",
    )
    show_inline_preview: BoolProperty(
        name="Preview in Panel",
        default=True,
        description="Show a small live preview of the webcam feed inside this panel",
    )
    inline_preview_scale: FloatProperty(
        name="Panel Preview Scale",
        default=8.0, min=1.0, max=20.0,
        description="Size of the inline preview icon",
    )

    # ---- animation recording ----
    record_on_start: BoolProperty(
        name="Record on Start",
        default=False,
        description="Begin capturing a take as soon as tracking starts. Off by "
                    "default so calibration and framing-up do not end up in the take.",
    )
    take_open_window: BoolProperty(
        name="Open Review Window",
        default=True,
        description="Open a separate window to trim and clean the take when a "
                    "recording ends. The same controls live in the Take Review "
                    "panel of this sidebar either way.",
    )
    take_auto_detect: BoolProperty(
        name="Find Outliers on Stop",
        default=True,
        description="Run outlier detection automatically when a recording ends",
    )
    take_preview: BoolProperty(
        name="Live Preview",
        default=True,
        description="Keep a preview Action on the rig showing exactly what the "
                    "current edits would save, so scrubbing the timeline shows "
                    "the cleaned result. Replaced by the real Action on save.",
        update=_preview_update,
    )

    # ---- take editing ----
    take_start: IntProperty(
        name="Trim Start",
        default=0, min=0, max=recording.MAX_SAMPLES,
        description="First captured sample to keep. Raise it to cut the "
                    "get-into-position frames off the front of the take.",
        update=_start_update,
    )
    take_end: IntProperty(
        name="Trim End",
        default=0, min=0, max=recording.MAX_SAMPLES,
        description="Last captured sample to keep",
        update=_end_update,
    )
    take_outlier_sensitivity: FloatProperty(
        name="Sensitivity",
        default=0.5, min=0.0, max=1.0,
        description="How eagerly to flag samples that sit off the path the "
                    "neighbouring samples describe. Higher flags more.",
    )
    take_outlier_min: FloatProperty(
        name="Min Jump %",
        default=5.0, min=0.1, max=100.0,
        description="Never flag a sample that is off its neighbours' path by "
                    "less than this, as a percentage of the rig's shoulder "
                    "width. Stops a still take from flagging its own noise.",
    )
    take_flag_lost: BoolProperty(
        name="Flag Lost Tracking",
        default=True,
        description="Also flag samples captured while the tracker saw neither "
                    "the body nor a hand",
    )
    take_auto_exclude: BoolProperty(
        name="Drop When Flagged",
        default=True,
        description="Untick 'Keep' on a sample as soon as it is flagged, "
                    "instead of only marking it",
    )
    take_threshold_info: StringProperty(default="")
    take_active_index: IntProperty(name="Active Sample", default=0)
    take_show_flagged_only: BoolProperty(
        name="Flagged Only",
        default=True,
        description="List only flagged or dropped samples instead of every "
                    "captured sample",
    )

    # ---- saving ----
    take_action_name: StringProperty(
        name="Action Name",
        default="FPV Take",
        description="Name for the saved Action. A number is appended if the "
                    "name is taken.",
    )
    take_start_frame: IntProperty(
        name="Start Frame",
        default=1,
        description="Timeline frame the first kept sample lands on. Sample "
                    "timing is real-time, resampled to the scene's frame rate.",
        update=_frame_map_update,
    )
    take_interpolation: EnumProperty(
        name="Interpolation",
        items=[
            ("BEZIER", "Bezier", "Smooth. Blender's default; eases through each key."),
            ("LINEAR", "Linear", "Straight between keys. Truer to the captured timing, "
                                 "and will not overshoot across a removed sample."),
        ],
        default="BEZIER",
        update=_take_update,
    )
    take_push_nla: BoolProperty(
        name="Push to NLA",
        default=True,
        description="Also place the saved Action as a strip on an NLA track",
    )
    take_nla_track: StringProperty(
        name="NLA Track",
        default="FPV Takes",
        description="NLA track to push the strip onto. Created if missing.",
    )

    # ---- capture throughput (Stop -> Start to apply) ----
    target_fps: IntProperty(
        name="Target FPS",
        default=60, min=0, max=240,
        description="Frame rate to request from the camera. 0 = leave the "
                    "camera's default. Most webcams will only reach 50-60 with "
                    "MJPG enabled; the panel shows what the camera actually gave.",
    )
    use_mjpg: BoolProperty(
        name="MJPG",
        default=True,
        description="Request the MJPG codec. Webcams default to raw YUY2, whose "
                    "USB bandwidth caps 640x480 at 30 fps (often 15); the same "
                    "camera on MJPG will usually do 60. Turn off if the feed "
                    "comes back corrupted or the camera refuses to open.",
    )
    exposure_mode: EnumProperty(
        name="Exposure",
        items=[
            ("AUTO", "Auto", "Let the camera choose. In a dim room it will pick a "
                             "long exposure, which both halves the frame rate and "
                             "smears fast hands into an unreadable blur."),
            ("MANUAL", "Manual", "Pin a short exposure. This is the single biggest "
                                 "win for fast motion: it stops the driver dropping "
                                 "to 15 fps AND keeps a moving fist sharp enough for "
                                 "MediaPipe to find. Needs good lighting."),
        ],
        default="AUTO",
        description="Camera exposure control. Changing this requires Stop -> Start.",
    )
    exposure: FloatProperty(
        name="Exposure Value",
        default=-5.0, min=-13.0, max=0.0,
        description="Manual exposure, in the camera's own log2 seconds: -6 is about "
                    "1/64 s, -7 about 1/128 s. Lower = shorter = sharper fast motion "
                    "but darker. Drop it until a moving hand stops smearing, then add "
                    "light (not gain) to compensate.",
    )
    gain: FloatProperty(
        name="Gain",
        default=-1.0, min=-1.0, max=255.0,
        description="Sensor gain to offset a short exposure. -1 leaves it alone. "
                    "Raising it brightens without adding blur, at the cost of noise.",
    )
    infer_width: IntProperty(
        name="Infer Width",
        default=0, min=0, max=1920,
        description="Downscale frames to this width before MediaPipe sees them. "
                    "0 = full capture resolution. Landmarks come back normalised so "
                    "positions are unaffected; 480 or 640 is usually plenty and "
                    "saves the colour convert on every frame.",
    )
    pose_every: IntProperty(
        name="Pose Every",
        default=1, min=1, max=6,
        description="Run the body-pose model only every Nth frame, reusing the last "
                    "result in between, while hands keep running every frame. Arms "
                    "move slowly in image terms compared to fingers, so 2 buys most "
                    "of pose's cost back with little visible change.",
    )
    parallel_infer: BoolProperty(
        name="Parallel",
        default=True,
        description="Run the pose and hand models on two threads. They are "
                    "independent and MediaPipe releases the GIL during inference, so "
                    "this nearly halves per-frame time when both are enabled. Turn "
                    "off if the tracker becomes unstable.",
    )
    mp_delegate: EnumProperty(
        name="Compute",
        items=[
            ("CPU", "CPU", "Always available."),
            ("GPU", "GPU", "Try MediaPipe's GPU delegate, falling back to CPU if the "
                           "installed wheel lacks it. Can be several times faster."),
        ],
        default="CPU",
        description="Changing this requires Stop -> Start.",
    )
    preview_fps: FloatProperty(
        name="Preview FPS",
        default=12.0, min=0.0, max=60.0,
        description="How often the annotated preview image is built. Drawing "
                    "landmarks, resizing and converting colour costs real time on "
                    "the tracking thread, so this runs far slower than tracking. "
                    "0 = never (fastest; the preview goes blank).",
    )
    cap_info: StringProperty(default="")
    perf_info: StringProperty(default="")
    cap_warning: StringProperty(default="")
    eye_offset_info: StringProperty(default="")
    cap_brightness: FloatProperty(default=-1.0)

    # --- clip library ------------------------------------------------
    clip_root: StringProperty(
        name="Library",
        subtype="DIR_PATH",
        default="",
        description="Folder that holds recorded clips, as "
                    "<library>/<category>/<subgroup>/<clip>.mp4. Leave it "
                    "blank to keep them in the clips folder inside the addon",
    )
    clip_category: StringProperty(
        name="Category",
        default="",
        update=_clip_dirs_update,
        description="Top-level grouping, for example punches. A folder of "
                    "this name is created under the library when you record",
    )
    clip_subgroup: StringProperty(
        name="Subgroup",
        default="",
        update=_clip_dirs_update,
        description="Optional second level inside the category, for example "
                    "jab. Leave blank to record straight into the category",
    )
    clip_show_all: BoolProperty(
        name="Whole Library",
        default=False,
        update=_clip_dirs_update,
        description="List every clip in the library instead of only the ones "
                    "in the selected category and subgroup",
    )
    clip_active_index: IntProperty(
        default=0,
        update=_clip_select_update,
        description="Selected clip",
    )
    clip_selected_path: StringProperty(default="")

    # --- clip recording ----------------------------------------------
    clip_name: StringProperty(
        name="Clip Name",
        default="take",
        description="Base name for the next recording. A number is appended "
                    "rather than overwriting footage you cannot re-shoot",
    )
    clip_countdown: IntProperty(
        name="Countdown",
        default=3, min=0, max=15,
        description="Seconds of preview before recording starts, to get into "
                    "position and away from the keyboard",
    )
    clip_max_seconds: IntProperty(
        name="Max Length",
        default=120, min=0, max=3600,
        description="Stop automatically after this many seconds. 0 = no "
                    "limit; raw video runs roughly 5-20 MB per second",
    )
    clip_rec_active: BoolProperty(default=False)
    clip_rec_frames: IntProperty(default=0)
    clip_rec_seconds: FloatProperty(default=0.0)
    clip_rec_status: StringProperty(default="")

    # --- offline analysis --------------------------------------------
    clip_engine: EnumProperty(
        name="Engine",
        items=[
            ("MEDIAPIPE", "MediaPipe Holistic",
             "Fastest, and the only engine with its own metric 3D pose. "
             "Least accurate on shoulders, elbows and wrists."),
            ("RTMPOSE", "RTMPose (ONNX)",
             "Markedly better arm and torso landmarks. 2D only, so arm depth "
             "comes from the depth lifter. Needs rtmlib + onnxruntime."),
            ("VITPOSE", "ViTPose (ONNX)",
             "The most accurate body landmarks available here and the "
             "slowest, which offline is a price worth paying because nothing "
             "is waiting on the frame. Needs rtmlib + onnxruntime."),
        ],
        default="RTMPOSE",
        description="Estimator to run over the recorded clip. Each engine "
                    "writes its own result beside the clip, so several can "
                    "be compared on the same footage",
    )
    clip_quality: EnumProperty(
        name="Quality",
        items=[
            ("LITE", "Lite", "Smallest models. Rarely the right choice "
                             "offline."),
            ("BALANCED", "Balanced", "The live default."),
            ("ACCURATE", "Accurate", "Largest models. Several times the cost "
                                     "per frame, and the reason to analyse "
                                     "offline at all."),
        ],
        default="ACCURATE",
        description="Model size for the RTMPose and ViTPose engines",
    )
    clip_device: EnumProperty(
        name="Compute",
        items=[
            ("CPU", "CPU", "Always available."),
            ("GPU", "GPU (CUDA)", "Use the CUDA execution provider, falling "
                                  "back to CPU if onnxruntime-gpu is not "
                                  "installed."),
            ("MPS", "GPU (Apple)", "Use the CoreML execution provider on "
                                   "Apple silicon, falling back to CPU."),
        ],
        default="CPU",
        description="Execution provider for the ONNX engines",
    )
    clip_stride: IntProperty(
        name="Every Nth Frame",
        default=1, min=1, max=10,
        description="Analyse every Nth frame only. 1 is the point of an "
                    "offline pass; raise it to trade temporal detail for "
                    "time on a long clip and a slow engine",
    )
    clip_overwrite: BoolProperty(
        name="Replace Existing",
        default=True,
        description="Overwrite this engine's previous analysis of the clip. "
                    "Untick to refuse rather than discard a long run",
    )
    clip_analysis_active: BoolProperty(default=False)
    clip_analysis_progress: FloatProperty(default=0.0, min=0.0, max=1.0)
    clip_analysis_status: StringProperty(default="")

    # --- analysis preview ---------------------------------------------
    clip_analysis: EnumProperty(
        name="Result",
        items=_analysis_items,
        description="Which analysis of this clip to preview and apply",
    )
    clip_preview_frame: IntProperty(
        name="Frame",
        default=0, min=0,
        update=_clip_preview_update,
        description="Frame of the analysed clip to show, with its estimated "
                    "pose drawn over the footage",
    )
    clip_preview_scale: FloatProperty(
        name="Preview Size",
        default=8.0, min=1.0, max=30.0,
        description="On-screen size of the preview image",
    )
    clip_preview_pose: BoolProperty(
        name="Body",
        default=True,
        update=_clip_preview_update,
        description="Draw the body landmarks over the frame",
    )
    clip_preview_hands: BoolProperty(
        name="Hands",
        default=True,
        update=_clip_preview_update,
        description="Draw the hand landmarks over the frame",
    )
    clip_preview_info: StringProperty(default="")

    # --- calibration --------------------------------------------------
    clip_calib_source: EnumProperty(
        name="Calibration",
        items=[
            ("PROFILE", "Saved Profile",
             "Use a calibration recorded once from a clip of the rest pose. "
             "The reliable option: the reference is a pose you posed for, "
             "not whatever the take happens to open on"),
            ("LIVE", "Live Session",
             "Reuse the calibration from the current or last live tracking "
             "session in this Blender session"),
            ("CLIP", "This Clip's Opening",
             "Sample the reference from the first seconds of this clip. Only "
             "right when the clip itself opens on the rig's rest pose"),
        ],
        default="PROFILE",
        description="Where the rest-pose reference comes from when this clip "
                    "is replayed onto the rig. Palm direction and wrist "
                    "orientation are driven by the change since that "
                    "reference, so a wrong one rotates every wrist in the take",
    )
    clip_calib_profile: EnumProperty(
        name="Profile",
        items=_profile_items,
        description="Which saved calibration to prime the rig with",
    )
    clip_calib_name: StringProperty(
        name="Save As",
        default="default",
        description="Name to save the calibration under, in the library's "
                    "_calibration folder. One per performer and camera setup",
    )
    clip_calib_skip: FloatProperty(
        name="Skip",
        default=0.5, min=0.0, max=30.0, subtype="TIME_ABSOLUTE", unit="TIME",
        description="Seconds to ignore at the start of a calibration clip, "
                    "before the performer has settled into the rest pose",
    )
    clip_calib_hold: FloatProperty(
        name="Hold",
        default=3.0, min=0.0, max=60.0, subtype="TIME_ABSOLUTE", unit="TIME",
        description="Seconds of the calibration clip to sample from. 0 uses "
                    "the whole clip",
    )
    clip_calib_status: StringProperty(default="")

    # --- applying to the rig ------------------------------------------
    clip_apply_action: BoolProperty(
        name="Bake Straight to an Action",
        default=False,
        description="Bake the replayed clip to an Action immediately instead "
                    "of leaving it in Take Review. Untick to trim it and drop "
                    "bad frames first, which recorded footage usually wants",
    )

    bones: PointerProperty(type=FPVBoneMap)

    # runtime status (updated by the timer tick)
    status: StringProperty(default="idle")
    fps: FloatProperty(default=0.0)
    detect_pose: BoolProperty(default=False)
    detect_left: BoolProperty(default=False)
    detect_right: BoolProperty(default=False)
    bones_applied: IntProperty(default=0)
    cameras_found: StringProperty(default="")
    multi_info: StringProperty(default="")
    is_recording: BoolProperty(default=False)
    rec_count: IntProperty(default=0)
    rec_seconds: FloatProperty(default=0.0)
    take_present: BoolProperty(default=False)


_CLASSES = (FPVTakeFrame, FPVBoneMap, FPVCameraDevice, FPVNameItem,
            FPVClipItem, FPVPoseSettings)


def register():
    for c in _CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.fpv_pose = PointerProperty(type=FPVPoseSettings)
    bpy.types.Scene.fpv_take_frames = CollectionProperty(type=FPVTakeFrame)
    bpy.types.Scene.fpv_clips = CollectionProperty(type=FPVClipItem)
    bpy.types.Scene.fpv_clip_categories = CollectionProperty(type=FPVNameItem)
    bpy.types.Scene.fpv_clip_groups = CollectionProperty(type=FPVNameItem)


def unregister():
    if hasattr(bpy.types.Scene, "fpv_pose"):
        del bpy.types.Scene.fpv_pose
    if hasattr(bpy.types.Scene, "fpv_take_frames"):
        del bpy.types.Scene.fpv_take_frames
    for attr in ("fpv_clips", "fpv_clip_categories", "fpv_clip_groups"):
        if hasattr(bpy.types.Scene, attr):
            delattr(bpy.types.Scene, attr)
    for c in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(c)
        except Exception:
            pass
