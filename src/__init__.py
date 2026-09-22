bl_info = {
    "name": "FPV Pose Tracker",
    "author": "ryanmcandrew",
    "version": (1, 25, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar (N) > FPV Pose",
    "description": "Drive arm and hand armatures from one or several webcams using a selectable pose estimator (MediaPipe, RTMPose or ViTPose), live or from recorded clips analysed offline, oriented for first-person body-cam view.",
    "category": "Animation",
}

import bpy

from . import dependencies
from . import properties
from . import mapping
from . import pose_schema
from . import mp_backend
from . import rtm_backend
from . import pose_engines
from . import pose_metrics
from . import handplane
from . import retarget
from . import recording
from . import cliplib
from . import vidrec
from . import tracker
from . import calib
from . import bodyfit
from . import multicam
from . import operators
from . import offline
from . import clip_ops
from . import panel
from . import preferences


_modules = (
    dependencies,
    properties,
    mapping,
    pose_schema,
    mp_backend,
    rtm_backend,
    pose_engines,
    pose_metrics,
    handplane,
    retarget,
    recording,
    cliplib,
    vidrec,
    tracker,
    calib,
    bodyfit,
    multicam,
    operators,
    offline,
    clip_ops,
    panel,
    preferences,
)


def register():
    print("[FPV Pose Tracker] register build %s from"
          % ".".join(str(n) for n in bl_info["version"]), __file__)
    for m in _modules:
        if hasattr(m, "register"):
            m.register()


def unregister():
    for m in reversed(_modules):
        if hasattr(m, "unregister"):
            try:
                m.unregister()
            except Exception:
                pass
