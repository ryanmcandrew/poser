"""Pose-engine registry: which estimator drives the rig, and what it can do.

The addon started MediaPipe-only, and the rest of the pipeline grew around
what MediaPipe hands back. This module is the seam that lets a different
estimator in without any of that pipeline learning a second vocabulary: it
answers three questions and nothing else —

  * `capabilities(engine)`  what this engine can and cannot supply
  * `make_pose_backend()`   a body backend duck-typed like `PoseBackend`
  * `make_hands_backend()`  a hand backend duck-typed like `HandsBackend`

Engines differ in more than accuracy, and the differences are load-bearing:

  MEDIAPIPE  33 landmarks, its own metric world pose, fastest by a wide
             margin, one model for body and one for hands.
  RTMPOSE    top-down COCO/Halpe/WholeBody 2D via ONNX Runtime. Markedly
             better shoulders, elbows and wrists; 2D only, so the depth
             lifter does the 3D. Near real time on CPU, fast on CUDA.
  VITPOSE    the same top-down structure with a ViT backbone. The most
             accurate 2D body available here and the slowest; sensible
             with a GPU, heavy without one.

Anything an engine cannot supply is reported here rather than discovered
at frame time, so the UI can gray the option out and the operator can
force a compatible setting instead of starting a tracker that will quietly
produce nothing.
"""

from . import mp_backend
from . import rtm_backend


class Capabilities:
    """What one engine supplies, in the terms the rest of the addon uses."""

    def __init__(self, label, modules, world_pose, own_hands, hand_sources,
                 wants_full_frame, note=""):
        self.label = label
        # (import_name, pip_name) pairs this engine needs on top of the base.
        self.modules = modules
        # True when the engine reports metric 3D landmarks of its own, i.e.
        # `pose_source = WORLD` is a real choice.
        self.world_pose = world_pose
        # True when the engine also provides hand landmarks itself.
        self.own_hands = own_hands
        # Hand backends that can be paired with this engine.
        self.hand_sources = hand_sources
        # True when the engine should see the full-resolution frame. A
        # top-down model crops and rescales to its own input size, so
        # pre-shrinking the frame buys it nothing and costs the hand crops
        # the very resolution they exist to get.
        self.wants_full_frame = wants_full_frame
        self.note = note


BASE_MODULES = (
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
)

MEDIAPIPE_MODULES = (("mediapipe", "mediapipe"),)
RTM_MODULES = (
    ("rtmlib", "rtmlib"),
    ("onnxruntime", "onnxruntime"),
)

CAPABILITIES = {
    "MEDIAPIPE": Capabilities(
        label="MediaPipe Holistic",
        modules=MEDIAPIPE_MODULES,
        world_pose=True,
        own_hands=True,
        hand_sources=("MEDIAPIPE",),
        wants_full_frame=False,
    ),
    "RTMPOSE": Capabilities(
        label="RTMPose (ONNX)",
        modules=RTM_MODULES,
        world_pose=False,
        own_hands=True,
        hand_sources=("AUTO", "RTM", "WHOLEBODY", "MEDIAPIPE"),
        wants_full_frame=True,
        note="2D only — arm depth comes from the lifter.",
    ),
    "VITPOSE": Capabilities(
        label="ViTPose (ONNX)",
        modules=RTM_MODULES,
        world_pose=False,
        own_hands=True,
        hand_sources=("AUTO", "RTM", "WHOLEBODY", "MEDIAPIPE"),
        wants_full_frame=True,
        note="Most accurate body, slowest — use a GPU if you have one.",
    ),
}

ENGINES = tuple(CAPABILITIES.keys())


def capabilities(engine):
    return CAPABILITIES.get((engine or "MEDIAPIPE").upper(),
                            CAPABILITIES["MEDIAPIPE"])


def required_modules(engine):
    """(import_name, pip_name) pairs this engine needs, base included."""
    return tuple(BASE_MODULES) + tuple(capabilities(engine).modules)


def resolve_hand_source(engine, hand_source, body_layout):
    """The hand backend this configuration actually ends up using.

    `AUTO` prefers the hand points the body model already produced, since
    a 133-point body pass has them for free, and otherwise runs the
    dedicated ROI hand model. A choice the engine cannot honour is
    downgraded rather than refused — a pairing that is merely suboptimal
    should still track.
    """
    eng = (engine or "MEDIAPIPE").upper()
    src = (hand_source or "AUTO").upper()
    if eng == "MEDIAPIPE":
        return "MEDIAPIPE"
    wholebody = (body_layout or "BODY").upper() == "WHOLEBODY"
    if src == "AUTO":
        return "WHOLEBODY" if wholebody else "RTM"
    if src == "WHOLEBODY" and not wholebody:
        # The body model was not asked for 133 points, so there are no hand
        # points in its output to reshape.
        return "RTM"
    return src


def make_pose_backend(engine="MEDIAPIPE", *, model_complexity=1,
                      running_mode="VIDEO", delegate="CPU",
                      min_detection=0.2, min_tracking=0.2,
                      quality="BALANCED", body_layout="BODY",
                      device="CPU", det_every=6, kpt_thr=0.3):
    """Body backend for `engine`, duck-typed like `mp_backend.PoseBackend`."""
    eng = (engine or "MEDIAPIPE").upper()
    if eng == "MEDIAPIPE":
        return mp_backend.PoseBackend(
            min_detection=min_detection, min_tracking=min_tracking,
            model_complexity=model_complexity, running_mode=running_mode,
            delegate=delegate,
        )
    return rtm_backend.OnnxPoseBackend(
        engine=eng, quality=quality, layout=body_layout, device=device,
        det_every=det_every, kpt_thr=kpt_thr, min_detection=min_detection,
    )


def make_hands_backend(engine="MEDIAPIPE", *, hand_source="AUTO",
                       body_layout="BODY", max_hands=2, running_mode="VIDEO",
                       delegate="CPU", min_detection=0.5, min_tracking=0.5,
                       device="CPU", kpt_thr=0.3, hand_detector=True):
    """Hand backend for `engine`, duck-typed like `mp_backend.HandsBackend`.

    The returned object may expose `set_body(pose_backend, pose_result)`;
    the tracker calls it when present so a ROI-based hand model knows where
    to crop. MediaPipe's own hand backend does not need it — it runs its
    own palm detector — and simply does not define it.
    """
    src = resolve_hand_source(engine, hand_source, body_layout)
    if src == "MEDIAPIPE":
        return mp_backend.HandsBackend(
            min_detection=min_detection, min_tracking=min_tracking,
            max_hands=max_hands, running_mode=running_mode, delegate=delegate,
        )
    return rtm_backend.RoiHandsBackend(
        device=device, kpt_thr=kpt_thr, max_hands=max_hands,
        use_detector=hand_detector, source=src,
    )


def diagnostics(engine=None):
    """Environment report for one engine, or for all of them."""
    engines = (engine,) if engine else ENGINES
    out = []
    for eng in engines:
        eng = (eng or "MEDIAPIPE").upper()
        out.append("=== %s (%s) ===" % (eng, capabilities(eng).label))
        if eng == "MEDIAPIPE":
            out.append(mp_backend.diagnostics())
        else:
            out.append(rtm_backend.diagnostics())
        out.append("")
    return "\n".join(out).strip()


def register():
    pass


def unregister():
    pass
