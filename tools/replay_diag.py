"""Replay a recording made with the addon's "Record Frames" option through the
tracker's depth pipeline, outside Blender.

    python tools/replay_diag.py path/to/fpv_pose_diag.jsonl [--fov 65] [--no-hand-depth]
                               [--hand-size 1.0] [--csv out.csv]

Prints, per frame, the forward position of each wrist relative to the
shoulders (metres, + = toward the camera), the hand's measured distance from
the camera, which cue chose the arm roots and the learned hand-size factor.
Options let you re-run the same frames with a different FOV or with the hand
depth disabled, to compare.

Needs opencv-python and numpy (already installed for the addon); bpy is not
needed.
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "fpv_pose_tracker")


def _load_tracker():
    """Import tracker.py without running the package __init__ (needs bpy)."""
    pkg = types.ModuleType("fpv_pose_tracker")
    pkg.__path__ = [PKG]
    sys.modules["fpv_pose_tracker"] = pkg

    def load(name):
        spec = importlib.util.spec_from_file_location(
            "fpv_pose_tracker." + name, os.path.join(PKG, name + ".py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = m
        spec.loader.exec_module(m)
        return m

    load("mp_backend")
    load("one_euro")
    return load("tracker"), sys.modules["fpv_pose_tracker.mp_backend"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--fov", type=float, default=None, help="override the recorded camera FOV")
    ap.add_argument("--no-hand-depth", action="store_true")
    ap.add_argument("--hand-size", type=float, default=None)
    ap.add_argument("--no-learn", action="store_true")
    ap.add_argument("--csv", default=None, help="also write a per-frame CSV")
    ap.add_argument("--every", type=int, default=1, help="print every Nth frame")
    args = ap.parse_args()

    T, mp_backend = _load_tracker()

    with open(args.path, encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    header = next((l for l in lines if l.get("kind") == "header"), {})
    frames = [l for l in lines if l.get("kind") == "frame"]
    if not frames:
        print("no frames in", args.path)
        return

    fov = args.fov if args.fov is not None else header.get("camera_fov", 65.0)
    tr = T.WebcamTracker(
        mirror=header.get("mirror", True),
        tracking_mode=header.get("tracking_mode", "HOLISTIC"),
        view_mode=header.get("view_mode", "SELFIE"),
        camera_fov=fov,
        lift_depth=header.get("lift_depth", True),
        pose_visibility=header.get("pose_visibility", 0.5),
        hand_depth=(not args.no_hand_depth) and header.get("hand_depth", True),
        hand_size=args.hand_size if args.hand_size is not None else header.get("hand_size", 1.0),
        hand_size_auto=(not args.no_learn) and header.get("hand_size_auto", True),
    )
    print("recording: %d frames, %dx%d, fov %.0f (using %.0f), mirror=%s"
          % (len(frames), frames[0]["W"], frames[0]["H"], header.get("camera_fov", -1), fov,
             header.get("mirror")))
    print("%7s %6s %6s | %6s %6s | %6s %6s | %-8s %5s" % (
        "t", "fwdL", "fwdR", "hzL", "hzR", "elbL", "elbR", "roots", "scale"))

    csv = open(args.csv, "w", encoding="utf-8") if args.csv else None
    if csv:
        csv.write("t,fwdL,fwdR,handzL,handzR,elbowL,elbowR,roots,scale,rec_fwdL,rec_fwdR\n")
    t0 = frames[0]["t"]
    for k, fr in enumerate(frames):
        res = mp_backend.HandsResult()
        for h in fr["hands"]:
            res.hands.append(mp_backend.HandTrack(
                label=h["label"], landmarks=[tuple(p) for p in h["landmarks"]],
                world_landmarks=[tuple(p) for p in h.get("world_landmarks", [])]))
        pose_res = None
        if fr.get("pose"):
            pose_res = mp_backend.PoseResult(
                landmarks=[tuple(p) for p in fr["pose"]["landmarks"]],
                image_landmarks=[tuple(p) for p in fr["pose"]["image_landmarks"]],
                visibility=list(fr["pose"]["visibility"]), present=True)
        pf = tr._build_frame(res, pose_res, fr["W"], fr["H"], fr["t"])

        def fwd(side):
            sh_i, el_i, wr_i = (T.POSE_LSHOULDER, T.POSE_LELBOW, T.POSE_LWRIST) if side == "L" \
                else (T.POSE_RSHOULDER, T.POSE_RELBOW, T.POSE_RWRIST)
            pl = pf.pose_lifted
            if wr_i not in pl or sh_i not in pl:
                return None, None
            f = pl[wr_i][1] - pl[sh_i][1]
            ang = None
            if el_i in pl:
                a = [pl[sh_i][i] - pl[el_i][i] for i in range(3)]
                b = [pl[wr_i][i] - pl[el_i][i] for i in range(3)]
                na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
                if na > 1e-6 and nb > 1e-6:
                    c = max(-1.0, min(1.0, sum(a[i] * b[i] for i in range(3)) / (na * nb)))
                    ang = math.degrees(math.acos(c))
            return f, ang

        fL, aL = fwd("L")
        fR, aR = fwd("R")
        hz = pf.hand_depth
        roots = tr._lifter.root_source
        scale = tr._hand_depth.scale()
        rec = fr.get("derived", {}).get("lifted", {})

        def rec_fwd(sh_i, wr_i):
            a, b = rec.get(str(sh_i)), rec.get(str(wr_i))
            return (b[1] - a[1]) if (a and b) else None

        fmt = lambda v: "  --  " if v is None else "%+6.2f" % v
        fmta = lambda v: "  --  " if v is None else "%6.0f" % v
        if k % args.every == 0:
            print("%7.2f %s %s | %s %s | %s %s | L:%s R:%s %5.2f" % (
                fr["t"] - t0, fmt(fL), fmt(fR), fmt(hz.get("L")), fmt(hz.get("R")),
                fmta(aL), fmta(aR), roots.get("L", "-")[:3], roots.get("R", "-")[:3], scale))
        if csv:
            csv.write("%.3f,%s,%s,%s,%s,%s,%s,%s,%.3f,%s,%s\n" % (
                fr["t"] - t0,
                "" if fL is None else "%.3f" % fL, "" if fR is None else "%.3f" % fR,
                "" if "L" not in hz else "%.3f" % hz["L"], "" if "R" not in hz else "%.3f" % hz["R"],
                "" if aL is None else "%.1f" % aL, "" if aR is None else "%.1f" % aR,
                "L%s/R%s" % (roots.get("L", "-")[:3], roots.get("R", "-")[:3]), scale,
                "" if rec_fwd(11, 15) is None else "%.3f" % rec_fwd(11, 15),
                "" if rec_fwd(12, 16) is None else "%.3f" % rec_fwd(12, 16)))
    if csv:
        csv.close()
        print("wrote", args.csv)


if __name__ == "__main__":
    main()
