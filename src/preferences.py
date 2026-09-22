"""Addon preferences shown in Edit > Preferences > Add-ons > FPV Pose Tracker."""

import bpy

from . import dependencies


class FPVPosePreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    def draw(self, context):
        layout = self.layout
        if dependencies.dependencies_ok():
            layout.label(text="Dependencies installed.", icon="CHECKMARK")
        else:
            missing = [name for name, _ in dependencies.missing_modules()]
            layout.label(
                text="Missing modules: " + ", ".join(missing),
                icon="ERROR",
            )
            layout.operator("fpv_pose.install_dependencies", icon="IMPORT")
        layout.label(text="Panel: 3D Viewport > Sidebar (N) > FPV Pose")


def register():
    bpy.utils.register_class(FPVPosePreferences)


def unregister():
    try:
        bpy.utils.unregister_class(FPVPosePreferences)
    except Exception:
        pass
