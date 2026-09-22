"""Addon preferences shown in Edit > Preferences > Add-ons > FPV Pose Tracker."""

import bpy

from . import dependencies
from . import pose_engines


class FPVPosePreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    def draw(self, context):
        layout = self.layout

        base_missing = dependencies.missing_base_modules()
        if base_missing:
            layout.label(
                text="Missing base modules: "
                     + ", ".join(name for name, _ in base_missing),
                icon="ERROR",
            )
            layout.operator("fpv_pose.install_dependencies", icon="IMPORT")
        else:
            layout.label(text="OpenCV and numpy installed.", icon="CHECKMARK")

        # One row per engine, each installable on its own: the engines do
        # not share a runtime, and the ONNX stack is a few hundred
        # megabytes that a MediaPipe user should never be asked to fetch.
        box = layout.box()
        box.label(text="Pose Engines", icon="ARMATURE_DATA")
        for engine in pose_engines.ENGINES:
            caps = pose_engines.capabilities(engine)
            ok, msg = dependencies.engine_status(engine)
            row = box.row(align=True)
            row.label(text=caps.label,
                      icon="CHECKMARK" if ok else "ERROR")
            if ok:
                row.label(text="ready")
            else:
                row.label(text=msg)
                op = row.operator("fpv_pose.install_dependencies",
                                  text="Install", icon="IMPORT")
                op.engine = engine
        box.operator("fpv_pose.engine_diagnostics", icon="INFO")

        layout.label(text="Panel: 3D Viewport > Sidebar (N) > FPV Pose")


def register():
    bpy.utils.register_class(FPVPosePreferences)


def unregister():
    try:
        bpy.utils.unregister_class(FPVPosePreferences)
    except Exception:
        pass
