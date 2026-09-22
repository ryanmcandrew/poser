"""Bone name maps for common rig conventions plus auto-detection.

Supports UE Mannequin (`hand_l`, `index_01_l`), Mixamo
(`mixamorig:LeftHandIndex1`), Rigify (`hand.L`, `f_index.01.L`),
Blender defaults (`Hand.L`), and generic underscore/dot naming.
"""

import re

# Logical bone slots the retargeter fills.
BONE_SLOTS = [
    "shoulder_L", "upper_arm_L", "forearm_L", "hand_L",
    "shoulder_R", "upper_arm_R", "forearm_R", "hand_R",
    "thumb_01_L", "thumb_02_L", "thumb_03_L",
    "index_01_L", "index_02_L", "index_03_L",
    "middle_01_L", "middle_02_L", "middle_03_L",
    "ring_01_L", "ring_02_L", "ring_03_L",
    "pinky_01_L", "pinky_02_L", "pinky_03_L",
    "thumb_01_R", "thumb_02_R", "thumb_03_R",
    "index_01_R", "index_02_R", "index_03_R",
    "middle_01_R", "middle_02_R", "middle_03_R",
    "ring_01_R", "ring_02_R", "ring_03_R",
    "pinky_01_R", "pinky_02_R", "pinky_03_R",
]


_FINGER_TOKENS = {
    "thumb":  ("thumb",),
    "index":  ("index", "pointer"),
    "middle": ("middle", "mid"),
    "ring":   ("ring",),
    "pinky":  ("pinky", "little"),
}
_ALL_FINGER_WORDS = tuple({w for tokens in _FINGER_TOKENS.values() for w in tokens})

# Anatomical joint synonyms — driven bones only.
#   1 = proximal phalanx (MCP -> PIP)
#   2 = intermediate/middle phalanx (PIP -> DIP)
#   3 = distal phalanx (DIP -> TIP)
_JOINT_SYNONYMS = {
    1: ("proximal", "proxi", "prox"),
    2: ("intermediate", "inter", "middle", "mid"),
    3: ("distal", "dist"),
}
# Metacarpal bones live between wrist and MCP. They are not driven by
# MediaPipe, so we treat "meta" as a weak match for joint 1 only when the
# rig has no explicit proximal bone.
_META_WORDS = ("meta", "metacarpal", "metacarp", "mc")


def _tokens(name):
    """Split a bone name into lowercase word tokens.

    "mixamorig:LeftHandIndex1" -> ["mixamorig", "left", "hand", "index", "1"]
    "hand.L" -> ["hand", "l"]
    "f_index_01_L" -> ["f", "index", "01", "l"]
    """
    n = name.lower()
    # Split camelCase: "LeftHand" -> "Left Hand"
    n = re.sub(r"([a-z])([A-Z])", r"\1 \2", name).lower()
    # Split letter/digit boundaries: "index1" -> "index 1"
    n = re.sub(r"([a-z])([0-9])", r"\1 \2", n)
    n = re.sub(r"([0-9])([a-z])", r"\1 \2", n)
    # Any non-alphanum becomes a separator.
    parts = re.split(r"[^a-z0-9]+", n)
    return [p for p in parts if p]


_SIDE_L = {"l", "lt", "lft", "left"}
_SIDE_R = {"r", "rt", "rgt", "right"}


def _side_of(tokens):
    for t in tokens:
        if t in _SIDE_L:
            return "L"
        if t in _SIDE_R:
            return "R"
    return None


def _finger_of(tokens):
    """First token that is a known finger word. Returns (finger, token_index)
    or (None, None). Consumes only the first occurrence — later finger-word
    tokens (like the second `middle` in `middle_middle.L`) are available as
    joint qualifiers."""
    for i, t in enumerate(tokens):
        for f, words in _FINGER_TOKENS.items():
            if t in words:
                return f, i
    return None, None


def _score_arm_bone(tokens, kind, side):
    if _side_of(tokens) != side:
        return -1
    # Arm bones must NOT contain finger tokens.
    if any(w in tokens for w in _ALL_FINGER_WORDS):
        return -1

    keywords = {
        "shoulder":  ("shoulder", "clavicle"),
        "upper_arm": ("upperarm", "humerus", "biceps"),
        "forearm":   ("forearm", "lowerarm", "elbow", "radius", "ulna"),
        "hand":      ("hand", "wrist", "palm"),
    }[kind]

    joined = " ".join(tokens)
    # Compound tokens (some rigs keep "upperarm" as one word).
    if any(k in joined for k in keywords):
        # Strong: exact token match.
        if any(k in tokens for k in keywords):
            return 30
        return 22

    # Fallback: split-token forms.
    if kind == "upper_arm":
        if ("upper" in tokens and "arm" in tokens) or "arm" in tokens:
            return 20 if "upper" in tokens else 12
    if kind == "forearm":
        if "lower" in tokens and "arm" in tokens:
            return 20

    return -1


def _score_finger_bone(tokens, target_finger, target_idx, target_side):
    if _side_of(tokens) != target_side:
        return -1
    finger, fidx = _finger_of(tokens)
    if finger != target_finger:
        return -1

    # Everything except the token we consumed as the finger name.
    body = [t for i, t in enumerate(tokens) if i != fidx]

    # Reject end/nub/tip control bones — they are zero-length helpers.
    if "tip" in body or "end" in body or "nub" in body:
        return -1

    # (1) Digit match — strongest signal (e.g. "index_02_l", "index1").
    for t in body:
        if t.isdigit():
            v = int(t)
            if 1 <= v <= 3:
                return 40 if v == target_idx else -1

    # Thumb anatomy differs from fingers:
    #   metacarpal -> thumb_01 (the wrist-side rotatable bone)
    #   proximal   -> thumb_02
    #   distal     -> thumb_03
    # If the rig has no explicit metacarpal, `proximal` falls back to thumb_01.
    if target_finger == "thumb":
        for t in body:
            if t in _META_WORDS:
                return 30 if target_idx == 1 else -1
        for t in body:
            if t in _JOINT_SYNONYMS[1]:  # proximal
                if target_idx == 2:
                    return 25
                if target_idx == 1:
                    return 15  # fallback for thumbs with no meta bone
                return -1
            if t in _JOINT_SYNONYMS[2]:  # middle / intermediate
                return 25 if target_idx == 2 else -1
            if t in _JOINT_SYNONYMS[3]:  # distal
                return 25 if target_idx == 3 else -1
        return 5 if target_idx == 1 else -1

    # Non-thumb fingers: proximal / middle / distal correspond to joints 1/2/3.
    for t in body:
        for j, syns in _JOINT_SYNONYMS.items():
            if t in syns:
                return 25 if j == target_idx else -1

    # Metacarpal — weak fallback for finger joint 1 only.
    for t in body:
        if t in _META_WORDS:
            return 8 if target_idx == 1 else -1

    # No joint qualifier — likely a single-bone finger; accept for joint 1.
    return 5 if target_idx == 1 else -1


def _pick_best(bones, scorer, exclude):
    best_score = 0
    best_name = ""
    for b in bones:
        if b in exclude:
            continue
        toks = _tokens(b)
        s = scorer(toks)
        if s > best_score:
            best_score = s
            best_name = b
    return best_name, best_score


def auto_map(armature_obj):
    """Return dict slot -> bone_name (or "") using best-guess matching.

    Assigns exclusively — one bone can only fill one slot. Uses a
    score-first greedy assignment so a bone whose best fit is joint 2 (e.g.
    a thumb `proximal` when a `meta` bone also exists) doesn't get stolen
    by joint 1 just because slots are iterated in order.
    """
    result = {slot: "" for slot in BONE_SLOTS}
    if not armature_obj or armature_obj.type != "ARMATURE":
        return result

    bones = [b.name for b in armature_obj.data.bones]
    tokens = {b: _tokens(b) for b in bones}

    # Fingers first — arms then compete for what's left.
    finger_candidates = []
    for side in ("L", "R"):
        for finger in _FINGER_TOKENS.keys():
            for i in (1, 2, 3):
                slot = f"{finger}_0{i}_{side}"
                for b in bones:
                    s = _score_finger_bone(tokens[b], finger, i, side)
                    if s > 0:
                        finger_candidates.append((s, slot, b))
    finger_candidates.sort(key=lambda x: (-x[0], x[1], x[2]))

    used = set()
    filled = set()
    for score, slot, bone in finger_candidates:
        if slot in filled or bone in used:
            continue
        result[slot] = bone
        filled.add(slot)
        used.add(bone)

    arm_candidates = []
    for side in ("L", "R"):
        for kind in ("hand", "forearm", "upper_arm", "shoulder"):
            slot = f"{kind}_{side}"
            for b in bones:
                if b in used:
                    continue
                s = _score_arm_bone(tokens[b], kind, side)
                if s > 0:
                    arm_candidates.append((s, slot, b))
    arm_candidates.sort(key=lambda x: (-x[0], x[1], x[2]))

    for score, slot, bone in arm_candidates:
        if slot in filled or bone in used:
            continue
        result[slot] = bone
        filled.add(slot)
        used.add(bone)

    # Thumb promotion: if the rig has no metacarpal bone and `proximal`
    # ended up on thumb_02, promote it to thumb_01 (single-bone thumbs
    # should drive the base, not the tip).
    for side in ("L", "R"):
        j1 = f"thumb_01_{side}"
        j2 = f"thumb_02_{side}"
        b2 = result[j2]
        if result[j1] or not b2:
            continue
        # b2 must actually be a valid fallback candidate for joint 1.
        if _score_finger_bone(tokens[b2], "thumb", 1, side) <= 0:
            continue
        # Only promote if nothing else on this side could fill joint 1.
        others = [b for b in bones
                  if b != b2 and b not in used
                  and _score_finger_bone(tokens[b], "thumb", 1, side) > 0]
        if others:
            continue
        result[j1] = b2
        result[j2] = ""

    return result


def diagnose(armature_obj):
    """Return human-readable text: bone list + which slot each fills."""
    if not armature_obj or armature_obj.type != "ARMATURE":
        return "No armature selected."
    bones = [b.name for b in armature_obj.data.bones]
    guesses = auto_map(armature_obj)
    reverse = {v: k for k, v in guesses.items() if v}
    lines = ["Armature: %s   Bones: %d" % (armature_obj.name, len(bones)), ""]
    lines.append("--- Bone -> slot ---")
    for b in bones:
        slot = reverse.get(b, "-")
        lines.append("  %s  ->  %s" % (b, slot))
    lines.append("")
    lines.append("--- Unfilled slots ---")
    missing = [s for s in BONE_SLOTS if not guesses[s]]
    if not missing:
        lines.append("  (none, all slots mapped)")
    else:
        for s in missing:
            lines.append("  " + s)
    return "\n".join(lines)


def register():
    pass


def unregister():
    pass
