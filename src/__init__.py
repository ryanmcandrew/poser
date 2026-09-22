bl_info = {
    "name": "FPV Pose Tracker",
    "author": "ryanmcandrew",
    "version": (1, 22, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar (N) > FPV Pose",
    "description": "Drive arm and hand armatures from one or several webcams using MediaPipe pose estimation, oriented for first-person body-cam view.",
    "category": "Animation",
}

import bpy

from . import dependencies
from . import properties
from . import mapping
from . import mp_backend
from . import retarget
from . import recording
from . import tracker
from . import calib
from . import bodyfit
from . import multicam
from . import operators
from . import panel
from . import preferences


_modules = (
    dependencies,
    properties,
    mapping,
    mp_backend,
    retarget,
    recording,
    tracker,
    calib,
    bodyfit,
    multicam,
    operators,
    panel,
    preferences,
)


def register():
    print("[FPV Pose Tracker] register build 1.22.0 from", __file__)
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
