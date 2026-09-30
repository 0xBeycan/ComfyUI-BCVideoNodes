"""`prompt_pose` - prompt mode's track, and on every frame where the pose shows the track lost a
whole forearm-and-hand or lower leg, that limb's drawn keypoints as positive points on the tracked
object on that frame; on every frame of a short run where the mask drops a hand-sized region of her
it holds on both sides, points inside that region. The sequence is Meta's SAM 3 video API chained as
its predictor runs it (easy-sam3 @ 88fe578 vendors it): the text prompt with its full
detector-and-tracker pass, points on the existing object, then the tracker-only re-propagation its
action history asks for.

1. Pass 1 is `segment_by_prompt`, the track, unchanged (refine.first_pass); it only hands over what
   the rest reads (its capture, the conditioning outputs whole). Prompt mode's own repairs
   (refine.segment_by_prompt_repaired) are not run: the triggers below judge the track, so no part is
   refined twice and the clip is tracked again at most once. Where pass 1 tracked the frames before a
   gain again (prompt.gain_frame), the capture's birth is the gain frame, so below "the birth" is that
   frame and the frames before it are pass 1's backward fill.
2. The frames to refine and their points are chosen from pass 1's masks once pass 1 is done, by two
   triggers whose frames are joined: the pose-driven one (`refine_points`), from the keypoints the
   pose images draw, each frame on its own; and the mask-driven one (`dropped_regions`), the Mask
   Guard's mask_loss detection on closed runs (guard.mask.closed_losses), from the region itself.
   A frame both pick gets the pose's points first, then the region's.
3. No frame chosen: pass 1's masks are the result, the track's bit for bit. Meta does not propagate
   again without a new prompt.
4. Otherwise each chosen frame is refined, pass 1's conditioning frames near a refined frame are
   demoted and the tracker alone tracks the clip again from the birth; the frames before the first
   one the refine can reach keep pass 1's mask (refine.refine_and_track).

The result: pass 1's backward fill before the birth, pass 1's mask on the kept conditioning frames
and on every frame before the first one the refine can reach, the refine on the refined frames and
the second pass on every other frame. Nothing is removed on pose grounds: there are no negative
points.
"""
from typing import TypedDict

import numpy as np
import torch

from ...libs import log
from ...libs.keypoints import (L_ANKLE, L_ELBOW, L_FOOT, L_KNEE, L_WRIST, R_ANKLE, R_ELBOW, R_FOOT, R_KNEE, R_WRIST,
                               in_frame)
from ...libs.mask import count_masked_frames, masked_frames
from ...models.sam3_1_multiplex.adapter import SAM3_1_MULTIPLEX_SIZE
from .config import report_counts
from .refine import first_pass, held_words, refine_and_track, region_frames, to_tracker


# The keypoints the rule reads, in pose order: pose_data's 20 body keypoints (libs/keypoints.py),
# then the left hand's 21, then the right hand's 21. A frame's points are sent in this order.
BODY_KEYPOINTS, HAND_KEYPOINTS = 20, 21
HANDS = {"left": BODY_KEYPOINTS, "right": BODY_KEYPOINTS + HAND_KEYPOINTS}   # each hand's first keypoint
KEYPOINT_SETS = (("keypoints_body", 0), ("keypoints_left_hand", HANDS["left"]),
                 ("keypoints_right_hand", HANDS["right"]))
KEYPOINT_COUNT = BODY_KEYPOINTS + 2 * HAND_KEYPOINTS


def _hand(side):
    return tuple(range(HANDS[side], HANDS[side] + HAND_KEYPOINTS))


# The limbs a refine is for, as pose-order keypoints: each forearm with its hand (the hand hangs
# on the wrist) and each lower leg with its toe. The head, neck, shoulders and hips never trigger.
LIMBS = {"right forearm and hand": (R_ELBOW, R_WRIST) + _hand("right"),
         "left forearm and hand": (L_ELBOW, L_WRIST) + _hand("left"),
         "right lower leg": (R_KNEE, R_ANKLE, R_FOOT),
         "left lower leg": (L_KNEE, L_ANKLE, L_FOOT)}


def _distal_parts():
    """[K, K] 0/1: row k marks the distal part of limb keypoint k, which the rule's C5 reads: for a
    hand keypoint or a wrist, the wrist and its hand; an elbow, the elbow, wrist and hand; a knee,
    the knee, ankle and toe; an ankle or a toe, the ankle and toe. Other rows are empty."""
    parts = {}
    for side, elbow, wrist in (("right", R_ELBOW, R_WRIST), ("left", L_ELBOW, L_WRIST)):
        hand = (wrist,) + _hand(side)
        parts.update({k: hand for k in hand})
        parts[elbow] = (elbow,) + hand
    for knee, ankle, toe in ((R_KNEE, R_ANKLE, R_FOOT), (L_KNEE, L_ANKLE, L_FOOT)):
        parts.update({knee: (knee, ankle, toe), ankle: (ankle, toe), toe: (ankle, toe)})
    matrix = np.zeros((KEYPOINT_COUNT, KEYPOINT_COUNT), dtype=np.int64)
    for k, part in parts.items():
        matrix[k, list(part)] = 1
    return matrix


DISTAL = _distal_parts()
# The rule's constants: they define the rule, so they are no config fields. A limb fires with
# LIMB_POINTS qualifying keypoints or more (C4); a keypoint's distal part must lie at least
# OUTSIDE_TENTHS in ten outside the mask (C5).
LIMB_POINTS = 3
OUTSIDE_TENTHS = 9


# --- the pose: which keypoints the pose images draw ----------------------------------------------

def drawn_keypoints(pose_metas, threshold, hidden, H, W):
    """(xy [N, K, 2], drawn [N, K]): every frame's keypoints in pose order, where pose_data holds
    them (x / W, y / H), and which of them the pose images draw (pipelines/pose.draw): the body and
    hand keypoints (face keypoints never are) of confidence `threshold` (pose_data's
    draw_threshold) or more, on the canvas (0 <= x < W, 0 <= y < H; one off it is dropped, not
    clamped), a hand keypoint only at int(x) and int(y) of 1 or more (the draw code's eps rule),
    and none an enabled draw rule leaves out (`hidden`, libs/draw_rules.hidden_by_rules). What
    this cannot see: draw_head and the stick widths are Pose Detection widgets, not in pose_data,
    so a part a stick width of 0 leaves out still counts as drawn (the head is in no limb anyway)."""
    N = len(pose_metas)
    xy = np.zeros((N, KEYPOINT_COUNT, 2))
    drawn = np.zeros((N, KEYPOINT_COUNT), dtype=bool)
    for f, (meta, parts) in enumerate(zip(pose_metas, hidden)):
        rows = np.concatenate([np.asarray(meta[key], dtype=np.float64).reshape(-1, 3) for key, _ in KEYPOINT_SETS])
        pixels = rows[:, :2] * np.array([W, H])
        on = (rows[:, 2] >= threshold) & in_frame(pixels, W, H)
        on[BODY_KEYPOINTS:] &= (np.trunc(pixels[BODY_KEYPOINTS:]) >= 1).all(axis=1)
        on[parts["body"]] = False
        for side in parts["hands"]:
            on[list(_hand(side))] = False
        xy[f], drawn[f] = rows[:, :2], on
    return xy, drawn


# --- the rule: which frames get points, and which points ----------------------------------------

def refine_points(masks, xy, drawn, birth, distance):
    """{frame: [keypoint, ...]}: the frames prompt_pose refines, ascending, and the pose-order
    keypoints it sends as positive points on each, in pose order. `masks` [N, H, W] are pass 1's,
    whose track was born on frame `birth` (-1: none); `xy` and `drawn` are drawn_keypoints'.

    Each frame f from the birth to the last whose mask is not empty is judged on its own: it is
    refined when one of the LIMBS has LIMB_POINTS or more keypoints (C4) that each
    - (C1) lie outside the mask on f, `distance` pixels or more from its nearest pixel;
    - (C5) have their distal part at least OUTSIDE_TENTHS in ten outside the mask on f, counting
      its keypoints drawn on f: a whole limb lost, not a fingertip past the edge.
    Every such keypoint of every limb that fires goes onto the frame, so a loss that lasts several
    frames refines each frame it qualifies on, the birth included. The frames before the birth
    (pass 1's backward fill) and a frame whose mask is empty are never refined: the rule is for a
    limb lost from a tracked body."""
    import cv2
    N, H, W = masks.shape
    fired = {}
    if birth < 0:
        return fired
    pixels = xy * np.array([W, H])
    cells = np.where(drawn[..., None], pixels, 0.0).astype(np.int64)    # the pixel each drawn keypoint is in
    rows, cols = torch.from_numpy(cells[..., 1]), torch.from_numpy(cells[..., 0])
    inside = (masks[torch.arange(N)[:, None], rows, cols] > 0).numpy() & drawn
    present = masked_frames(masks).tolist()
    limb_keypoints = np.zeros(KEYPOINT_COUNT, dtype=bool)
    limb_keypoints[[k for limb in LIMBS.values() for k in limb]] = True
    for f in range(birth, N):
        if not present[f]:
            continue
        outside = drawn[f] & ~inside[f]
        qualified = limb_keypoints & outside
        qualified &= 10 * (DISTAL @ outside.astype(np.int64)) >= OUTSIDE_TENTHS * (DISTAL @ drawn[f].astype(np.int64))  # C5
        if not any(qualified[list(limb)].sum() >= LIMB_POINTS for limb in LIMBS.values()):
            continue
        # C1: each pixel's exact Euclidean distance to the nearest pixel of the mask
        to_mask = cv2.distanceTransform((masks[f] <= 0).numpy().astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        qualified &= to_mask[cells[f, :, 1], cells[f, :, 0]] >= distance
        chosen = sorted(k for limb in LIMBS.values() if qualified[list(limb)].sum() >= LIMB_POINTS
                        for k in limb if qualified[k])                                                 # C4
        if chosen:
            fired[f] = chosen
    return fired


# --- the mask-driven trigger: a region the mask drops for a few frames ---------------------------

def dropped_regions(masks, pose_metas, draw_threshold, birth):
    """{frame: (points, holding)}: the frames the mask-driven trigger refines, ascending, the
    positive points it sends on each as (x, y) pixel centres and the frames that hold its regions
    around it (refine.region_frames). `masks` [N, H, W] are pass 1's, whose track was born on frame
    `birth` (-1: none).

    The regions are the Mask Guard's mask_loss detection on closed runs (guard.mask.closed_losses):
    a region of her the Wan Animate workflow's final mask (grow 10, blockify 32) loses for 1 to 8
    frames and holds on the frames on both sides, a whole block of its grid, hand-sized (LOSS_HAND
    of her mask) or bigger, with the pose's body in it (the guard's evidence: her body keypoints
    drawn at `draw_threshold`, no draw rule left out; a limb that crosses the region on an end frame
    and is not drawn on the run counts). Every frame of such a run is refined, except, as with the
    pose-driven rule, the frames before the birth and a frame whose mask is empty."""
    from ..guard.mask import closed_losses
    return region_frames(masks, birth, lambda booleans: closed_losses(booleans, pose_metas, draw_threshold))


# --- the mode ------------------------------------------------------------------------------------

# The counts segment_by_prompt_pose adds to pass 1's for the log, keyed by the label the log
# shows, in the order they are added (refine_and_track fills its own among them); "frames
# segmented" replaces pass 1's with the result's.
PromptPoseCounts = TypedDict("PromptPoseCounts", {"refined frames": int, "refined for a dropped region": int,
                                                  "points": int, "stability fallbacks": int, "demoted": int,
                                                  "re-tracked": int, "kept from the first pass": int,
                                                  "frames segmented": int},
                             total=False)


def segment_by_prompt_pose(model, clip, images, prompt, config, pose_metas, draw_threshold, hidden, result=None,
                           logits=None):
    """[N, H, W] float masks of the person in `images` [N, H, W, 3]: prompt mode's track, with Meta's
    point refine where the pose shows a limb the track lost, and where the mask drops a region of
    her for a few frames (see the module docstring).
    `pose_metas`, `draw_threshold` and `hidden` are pose_data's keypoints, the threshold its images
    are drawn at and what its draw rules leave out of them (track.prompt_pose_inputs). `result`
    and `logits` are segment_by_prompt's; in the logits record the refined frames and the
    re-decoded frames the result shows are "prompt" frames whose logits are the ones before the
    cleaning ("raw" true), and every other frame keeps pass 1's entry.

    Pass 1 reads every [prompt] and [prompt, max_objects 1] field. A refine decodes from its
    points and pass 1's logits of the frame (raw, or on the birth its conditioning output's). The
    second pass reads input_range, fill_hole_area, obj_ptr_token, memory_selection and
    max_conditioning_frames, and encodes its frames' memory from the decoder's raw logits whatever
    memory_mask says, as Meta's re-propagation does (refine.refine_and_track)."""
    c = config
    N, H, W, _ = images.shape
    xy, drawn = drawn_keypoints(pose_metas, draw_threshold, hidden, H, W)
    counts: PromptPoseCounts = {"refined frames": 0, "refined for a dropped region": 0, "points": 0,
                                "stability fallbacks": 0, "demoted": 0, "re-tracked": 0, "kept from the first pass": 0}

    masks, capture, pass_one = first_pass(model, clip, images, prompt, c, result, logits)
    birth = capture.get("birth", -1)
    chosen = refine_points(masks, xy, drawn, birth, c.pose_point_distance * min(H, W))
    dropped = dropped_regions(masks, pose_metas, draw_threshold, birth)
    if not chosen and not dropped:
        log.info("prompt_pose: no frame needed points; the mask is the track's")
        report_counts(result, counts)
        return masks

    # the refines: one round, on pass 1's frames (refine_and_track). A frame both triggers pick gets
    # the pose's points first; past MAX_REFINE_POINTS the refine keeps the first half and the last half.
    refines = {}
    for g in sorted(set(chosen) | set(dropped)):
        keypoints = chosen.get(g, [])
        pixels, holding = dropped.get(g, ([], []))
        points = [tuple(p) for p in (xy[g, keypoints] * SAM3_1_MULTIPLEX_SIZE).tolist()] + to_tracker(pixels, H, W)
        limbs = [name for name, limb in LIMBS.items() if set(limb) & set(keypoints)]
        where = [f"on the {' and '.join(limbs)}"] if limbs else []
        if holding:
            where.append(held_words(holding))
        refines[g] = (points, " and ".join(where))
    counts["refined for a dropped region"] = len(dropped)
    refine_and_track(model, images, c, masks, capture, refines, counts, logits, "prompt_pose", pass_one)
    counts["frames segmented"] = count_masked_frames(masks)
    report_counts(result, counts)
    return masks
