"""Canonical landmark schema and converters for the pose engines.

Every engine in this addon reports into one schema so the retargeter, depth
lifter and multi-camera solver never need to know which model produced a
frame. That schema is MediaPipe's 33-point pose order, because it was here
first and because every other layout is a subset of it plus feet detail:
converting the newcomers costs a lookup table, while converting the
existing pipeline would cost a rewrite of the parts that already work.

Slots a given layout does not carry are emitted with visibility 0.0, which
is exactly how the rest of the addon already spells "this joint was not
measured" for a landmark MediaPipe could not see.

Hand landmarks need no conversion at all: MMPose/COCO-WholeBody number the
21 hand points in the same order as MediaPipe (wrist, then thumb through
pinky, proximal to distal), so a hand from RTMPose drops straight into the
slots the finger retargeter already reads.
"""

# --- canonical (MediaPipe pose) indices ---------------------------------
NUM_POSE = 33

MP_NOSE = 0
MP_LEYE_INNER, MP_LEYE, MP_LEYE_OUTER = 1, 2, 3
MP_REYE_INNER, MP_REYE, MP_REYE_OUTER = 4, 5, 6
MP_LEAR, MP_REAR = 7, 8
MP_MOUTH_L, MP_MOUTH_R = 9, 10
MP_LSHOULDER, MP_RSHOULDER = 11, 12
MP_LELBOW, MP_RELBOW = 13, 14
MP_LWRIST, MP_RWRIST = 15, 16
MP_LPINKY, MP_LINDEX, MP_LTHUMB = 17, 18, 19
MP_RPINKY, MP_RINDEX, MP_RTHUMB = 20, 21, 22
MP_LHIP, MP_RHIP = 23, 24
MP_LKNEE, MP_RKNEE = 25, 26
MP_LANKLE, MP_RANKLE = 27, 28
MP_LHEEL, MP_RHEEL = 29, 30
MP_LFOOT, MP_RFOOT = 31, 32

NUM_HAND = 21

# --- source layouts ------------------------------------------------------
# COCO-17, the order shared by RTMPose-body, ViTPose-coco and every
# top-down COCO model: nose, eyes, ears, shoulders, elbows, wrists, hips,
# knees, ankles. Left/right are the subject's own, as in MediaPipe.
COCO17_TO_MP33 = {
    0: MP_NOSE,
    1: MP_LEYE, 2: MP_REYE,
    3: MP_LEAR, 4: MP_REAR,
    5: MP_LSHOULDER, 6: MP_RSHOULDER,
    7: MP_LELBOW, 8: MP_RELBOW,
    9: MP_LWRIST, 10: MP_RWRIST,
    11: MP_LHIP, 12: MP_RHIP,
    13: MP_LKNEE, 14: MP_RKNEE,
    15: MP_LANKLE, 16: MP_RANKLE,
}

# Halpe-26 / RTMPose-body7-halpe26: COCO-17, then head, neck, mid-hip and
# six foot points. The pelvis (19) has no MediaPipe slot of its own — the
# retargeter derives the root from the two hips — so it is dropped.
HALPE26_TO_MP33 = dict(COCO17_TO_MP33)
HALPE26_TO_MP33.update({
    20: MP_LFOOT, 21: MP_RFOOT,      # big toes
    24: MP_LHEEL, 25: MP_RHEEL,
})

# COCO-25 ("coco_25", ViTPose fine-tuned on COCO+feet): COCO-17 then
# left/right big toe, small toe, heel.
COCO25_TO_MP33 = dict(COCO17_TO_MP33)
COCO25_TO_MP33.update({
    17: MP_LFOOT, 19: MP_LHEEL,
    20: MP_RFOOT, 22: MP_RHEEL,
})

# COCO-WholeBody-133 (RTMW, DWPose, ViTPose-wholebody):
#   0-16    body, COCO-17 order
#   17-22   feet: l_big_toe, l_small_toe, l_heel, r_big_toe, r_small_toe, r_heel
#   23-90   face, 68-point order
#   91-111  left hand, 21 points
#   112-132 right hand, 21 points
WHOLEBODY133_TO_MP33 = dict(COCO17_TO_MP33)
WHOLEBODY133_TO_MP33.update({
    17: MP_LFOOT, 19: MP_LHEEL,
    20: MP_RFOOT, 22: MP_RHEEL,
    # 68-point face landmarks 48 and 54 are the mouth corners.
    23 + 48: MP_MOUTH_L, 23 + 54: MP_MOUTH_R,
    # Hand roots double as the finger-tip-ish slots MediaPipe's pose
    # carries, so the pose alone still knows roughly where the fingers are
    # when the hand landmarks are switched off.
    91 + 20: MP_LPINKY, 91 + 8: MP_LINDEX, 91 + 4: MP_LTHUMB,
    112 + 20: MP_RPINKY, 112 + 8: MP_RINDEX, 112 + 4: MP_RTHUMB,
})

WHOLEBODY_LEFT_HAND = (91, 112)     # [start, stop) of the 21 left-hand points
WHOLEBODY_RIGHT_HAND = (112, 133)


class Layout:
    """One keypoint ordering an engine can emit."""

    def __init__(self, name, count, mapping, hands=None):
        self.name = name
        self.count = count
        self.mapping = mapping
        # (left_slice, right_slice) when the layout carries hand points.
        self.hands = hands

    def __repr__(self):
        return "Layout(%s, %d)" % (self.name, self.count)


LAYOUTS = {
    "coco17": Layout("coco17", 17, COCO17_TO_MP33),
    "coco25": Layout("coco25", 25, COCO25_TO_MP33),
    "halpe26": Layout("halpe26", 26, HALPE26_TO_MP33),
    "wholebody133": Layout("wholebody133", 133, WHOLEBODY133_TO_MP33,
                           hands=(WHOLEBODY_LEFT_HAND, WHOLEBODY_RIGHT_HAND)),
}


def layout_for(count):
    """Best-effort layout for a keypoint count an engine reported.

    Model zoos grow, and a checkpoint whose head is one point wider than
    expected should still drive the arms rather than refuse to run: every
    layout here starts with the same COCO-17 body, so falling back to that
    prefix loses feet and hands but keeps everything the retargeter
    actually needs.
    """
    for lay in LAYOUTS.values():
        if lay.count == count:
            return lay
    return LAYOUTS["coco17"]


def to_canonical(keypoints, scores, width, height, layout, kpt_thr=0.0):
    """Convert one person's 2D keypoints into the canonical 33-slot lists.

    `keypoints` is (K, 2) in pixels, `scores` is (K,). Returns
    `(image_landmarks, visibility)` where `image_landmarks` holds 33
    `(x, y, z)` tuples with x/y normalised to 0..1 and z always 0.0 — a 2D
    model has no depth to report, and inventing one would be worse than
    saying so, because the depth lifter's near/far test reads z as a real
    measurement.

    Points below `kpt_thr` are reported with visibility 0.0 rather than
    dropped, so index i always means the same joint.
    """
    w = float(max(1, width))
    h = float(max(1, height))
    image_landmarks = [(0.0, 0.0, 0.0)] * NUM_POSE
    visibility = [0.0] * NUM_POSE
    n = min(len(keypoints), len(scores))
    for src, dst in layout.mapping.items():
        if src >= n:
            continue
        s = float(scores[src])
        if s < kpt_thr:
            continue
        x, y = keypoints[src][0], keypoints[src][1]
        image_landmarks[dst] = (float(x) / w, float(y) / h, 0.0)
        visibility[dst] = s
    return image_landmarks, visibility


def hand_from_layout(keypoints, scores, width, height, span, kpt_thr=0.0):
    """Pull one 21-point hand out of a wholebody result.

    Returns `(landmarks, mean_score)` with landmarks normalised to the
    image, or `(None, 0.0)` when the span is missing or too weak to use.
    """
    start, stop = span
    if len(keypoints) < stop:
        return None, 0.0
    w = float(max(1, width))
    h = float(max(1, height))
    pts = []
    total = 0.0
    for i in range(start, stop):
        pts.append((float(keypoints[i][0]) / w, float(keypoints[i][1]) / h, 0.0))
        total += float(scores[i]) if i < len(scores) else 0.0
    mean = total / float(stop - start)
    if mean < kpt_thr:
        return None, mean
    return pts, mean


def normalise_hand(keypoints, scores, width, height):
    """Normalise a standalone 21-point hand result to image space."""
    w = float(max(1, width))
    h = float(max(1, height))
    pts = [(float(p[0]) / w, float(p[1]) / h, 0.0) for p in keypoints[:NUM_HAND]]
    mean = (sum(float(s) for s in scores[:NUM_HAND]) / float(NUM_HAND)) \
        if len(scores) else 0.0
    return pts, mean


def register():
    pass


def unregister():
    pass
