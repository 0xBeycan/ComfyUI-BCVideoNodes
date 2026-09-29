"""`prompt_pose` - prompt mode's track, and on every frame where the pose shows the track lost a
whole forearm-and-hand or lower leg, that limb's drawn keypoints as positive points on the tracked
object on that frame. The sequence is Meta's SAM 3 video API chained as its predictor runs it
(easy-sam3 @ 88fe578 vendors it): the text prompt with its full detector-and-tracker pass, points on
the existing object, then the tracker-only re-propagation its action history asks for.

1. Pass 1 is `segment_by_prompt`, unchanged; it only hands over what the rest reads (its capture).
   Where it tracked the frames before a gain again (prompt.gain_frame), the capture's birth is the
   gain frame, so below "the birth" is that frame and the frames before it are pass 1's backward fill.
2. The frames to refine and their points are chosen from pass 1's masks and the keypoints the pose
   images draw (`refine_points`), once pass 1 is done, each frame on its own.
3. No frame chosen: pass 1's masks are the result, prompt mode's bit for bit. Meta does not
   propagate again without a new prompt.
4. Otherwise each chosen frame is refined (Meta's point refine, `refine_with_points`, with pass 1's
   mask logits of the frame as the dense prompt: the raw ones where pass 1 propagated the frame, the
   conditioning output's on the birth, as Meta's lookup finds them) and becomes a conditioning
   frame, a refine on the birth or an anchor replacing that conditioning; pass 1's conditioning
   frames (the birth and the fired anchors) within DEMOTION_WINDOW of a refined frame and not
   refined themselves are demoted to ordinary frames; and the tracker alone tracks the clip again
   from the birth: no detection, no re-anchor, no probation, every frame but the conditioning ones
   re-decoded, each reading the conditioning frames closest to it on both sides.
5. The frames of the second pass before the first one the refine can reach (first_influenced) show
   pass 1's mask: the refine cannot change them, so all the second pass would add there is its
   drift from pass 1, tracking without the detector and with raw-logit memory.

The result: pass 1's backward fill before the birth, pass 1's mask on the kept conditioning frames
and on every frame before the first one the refine can reach, the refine on the refined frames and
the second pass on every other frame. Nothing is removed on pose grounds: there are no negative
points.
"""
import dataclasses
import time
from typing import TypedDict

import numpy as np
import torch

from ...libs import log
from ...libs.keypoints import (L_ANKLE, L_ELBOW, L_FOOT, L_KNEE, L_WRIST, R_ANKLE, R_ELBOW, R_FOOT, R_KNEE, R_WRIST,
                               in_frame)
from ...libs.mask import count_masked_frames, to_frame_size
from ...models.sam3_1_multiplex.adapter import (SAM3_1_MULTIPLEX_SIZE, backbone_frame, memory_lookback, multiplex_parts,
                                                propagation_backbone, refine_with_points, track_and_clean)
from ...models.sam3_1_multiplex.postprocess import clean_channel_logits, low_res_logits
from .config import BEST_IOU, RAW, SIGNED_RANGE, report_counts
from .prompt import PromptCapture, encode_frame_memory, keep_memory, memory_score, memory_view, segment_by_prompt


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
# Meta's refinement_detector_cond_frame_removal_window (easy-sam3 sam3_video_inference.py): the
# detector's conditioning frames this close to a refined frame are demoted to ordinary frames.
DEMOTION_WINDOW = 16


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
    present = masks.flatten(1).any(dim=1).numpy()
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


# --- the second pass: demotion and the memory it reads ------------------------------------------

def demote(conditioning, refined):
    """(kept, demoted): pass 1's conditioning frames `conditioning` ({frame: output}, the birth and
    every fired anchor as created) split into those farther than DEMOTION_WINDOW from every refined
    frame in `refined`, which stay conditioning frames, and the frames of the rest, ascending, which
    Meta's refine demotes to ordinary frames (clear_detector_added_cond_frame_in_tracker). A refined
    frame is never demoted: its refine replaces what pass 1 conditioned it with."""
    kept = {t: out for t, out in conditioning.items() if all(abs(t - g) > DEMOTION_WINDOW for g in refined)}
    return kept, sorted(set(conditioning) - set(kept) - set(refined))


def closest_conditioning(conditioning, frame_idx, keep):
    """(selected, unselected): Meta's select_closest_cond_frames with keep_first_cond_frame off.
    Of more than `keep` conditioning frames, the closest before `frame_idx`, the closest at or
    after it, then the temporally closest of the rest (the earlier on a tie) up to `keep`, and the
    others; fewer are all selected."""
    if len(conditioning) <= keep:
        return dict(conditioning), {}
    selected = {}
    before = max((t for t in conditioning if t < frame_idx), default=None)
    after = min((t for t in conditioning if t >= frame_idx), default=None)
    for t in (before, after):
        if t is not None:
            selected[t] = conditioning[t]
    rest = sorted((t for t in conditioning if t not in selected), key=lambda t: (abs(t - frame_idx), t))
    selected.update((t, conditioning[t]) for t in rest[:keep - len(selected)])
    return selected, {t: out for t, out in conditioning.items() if t not in selected}


def pass_two_view(conditioning, stored, frame_idx, keep, count, selection):
    """The output_dict the second pass hands the tracker at `frame_idx`, as Meta's re-propagation
    reads its memory: the `keep` conditioning frames closest to it on both sides
    (closest_conditioning; core takes object pointers only from those at or before it, and gives
    those after it the far temporal slot), and as ordinary memory the second pass's own frames
    `stored`, read as prompt mode reads its propagated frames: with memory `selection` those
    prompt.memory_view picks, without it by distance. Without selection an unselected conditioning
    frame before `frame_idx` is ordinary memory too, as Meta attends it where the lookup reads it
    (sam3_tracker_base.py); with selection the one frame the lookup reads that is not in `stored`
    is the frame before, which is always a selected conditioning frame when it is one."""
    selected, unselected = closest_conditioning(conditioning, frame_idx, keep)
    if selection:
        return memory_view({"cond_frame_outputs": selected, "non_cond_frame_outputs": stored}, frame_idx, count)
    earlier = {t: out for t, out in unselected.items() if t < frame_idx}
    return {"cond_frame_outputs": selected, "non_cond_frame_outputs": {**earlier, **stored}}


def first_influenced(conditioning, baseline, refined, birth, N, keep, count, selection):
    """The first frame from `birth` on that the refine action (the refined frames `refined` and the
    demotions) can influence in the second pass, N if none: the first whose view (pass_two_view,
    both slots, before the pass stores a frame) holds a refined frame, or other conditioning frames
    than it holds with pass 1's `baseline` in place of the second pass's `conditioning`, none
    demoted and none refined. Every frame after it is influenced too, as the pass runs forwards and
    each frame's ordinary memory comes from the frames before it; every frame before it is tracked
    as it would be with no refine action at all."""
    def held(cond, f):
        view = pass_two_view(cond, {}, f, keep, count, selection)
        return {slot: sorted(outputs) for slot, outputs in view.items()}

    for f in range(birth, N):
        view = held(conditioning, f)
        if any(t in refined for frames in view.values() for t in frames) or view != held(baseline, f):
            return f
    return N


# --- the mode ------------------------------------------------------------------------------------

def _clock(device):
    """time.perf_counter() once the work queued on `device` is done, so a step's seconds are its own."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


# The counts segment_by_prompt_pose adds to pass 1's for the log, keyed by the label the log
# shows, in the order they are added; "frames segmented" replaces pass 1's with the result's.
PromptPoseCounts = TypedDict("PromptPoseCounts", {"refined frames": int, "points": int, "stability fallbacks": int,
                                                  "demoted": int, "re-tracked": int, "kept from the first pass": int,
                                                  "frames segmented": int},
                             total=False)


def segment_by_prompt_pose(model, clip, images, prompt, config, pose_metas, draw_threshold, hidden, result=None,
                           logits=None):
    """[N, H, W] float masks of the person in `images` [N, H, W, 3]: prompt mode's, with Meta's
    point refine where the pose shows a limb the track lost (see the module docstring).
    `pose_metas`, `draw_threshold` and `hidden` are pose_data's keypoints, the threshold its images
    are drawn at and what its draw rules leave out of them (track.prompt_pose_inputs). `result`
    and `logits` are segment_by_prompt's; in the logits record the refined frames and the
    re-decoded frames the result shows are "prompt" frames whose logits are the ones before the
    cleaning ("raw" true), and every other frame keeps pass 1's entry.

    Pass 1 reads every [prompt] and [prompt, max_objects 1] field. A refine decodes from its
    points and pass 1's logits of the frame (raw, or on the birth its conditioning output's). The
    second pass reads input_range, fill_hole_area, obj_ptr_token, memory_selection and
    max_conditioning_frames, and encodes its frames' memory from the decoder's raw logits whatever
    memory_mask says, as Meta's re-propagation does."""
    from comfy import model_management as mm
    from comfy.utils import ProgressBar
    c = config
    N, H, W, _ = images.shape
    xy, drawn = drawn_keypoints(pose_metas, draw_threshold, hidden, H, W)
    device = mm.get_torch_device()
    capture: PromptCapture = {"raw": {}}
    counts: PromptPoseCounts = {"refined frames": 0, "points": 0, "stability fallbacks": 0, "demoted": 0,
                                "re-tracked": 0}

    start = _clock(device)
    masks = segment_by_prompt(model, clip, images, prompt, c, result=result, logits=logits, capture=capture)
    pass_one = _clock(device) - start
    birth = capture.get("birth", -1)
    chosen = refine_points(masks, xy, drawn, birth, c.pose_point_distance * min(H, W))
    if not chosen:
        log.info("prompt_pose: no frame needed points; the mask is prompt mode's")
        report_counts(result, counts)
        return masks

    dtype = model.model.get_dtype()
    mm.load_model_gpu(model)
    _, _, tracker, backbone = multiplex_parts(model)
    backbone_fn = propagation_backbone(backbone)
    frames = images[..., :3].movedim(-1, 1)
    size, signed = tracker.image_size, c.input_range == SIGNED_RANGE
    mux = capture["mux"]
    best_iou = c.obj_ptr_token == BEST_IOU

    def put(f, shown, dumped, index=None):
        """Frame f shows the mask logits `shown`; the logits record gets `dumped`, the ones before the
        cleaning."""
        masks[f] = to_frame_size(shown, H, W)
        if logits is not None:
            logits["logits"][f], logits["cut"][f], logits["raw"][f] = low_res_logits(dumped), "prompt", True
            if "mask_index" in logits:
                logits["mask_index"][f] = index

    # the refines: one round, on pass 1's frames, each from its points and pass 1's logits of the
    # frame: the raw ones where pass 1 propagated it; on the birth, which it did not, its conditioning
    # output's, the previous output Meta's refine looks up there. From the points without that mask
    # (Meta's first refine of a frame) the test clips' refined frame kept the forearms and hands but
    # lost the head, torso and dress.
    refined = {}
    start = _clock(device)
    with torch.inference_mode():
        for g, keypoints in chosen.items():
            frame, vision_feats, vision_pos, feat_sizes, _, trunk_out = backbone_frame(
                tracker, backbone_fn, frames, g, device, dtype, size, signed)
            points = [tuple(p) for p in (xy[g, keypoints] * SAM3_1_MULTIPLEX_SIZE).tolist()]
            output, info = refine_with_points(tracker, backbone, frame, trunk_out, vision_feats, vision_pos,
                                              feat_sizes, points, mux,
                                              capture["cond"][g]["pred_masks"] if g == birth else capture["raw"][g])
            refined[g] = output
            put(g, clean_channel_logits(output["pred_masks"], c.fill_hole_area), output["pred_masks"])
            counts["refined frames"] += 1
            counts["points"] += len(info["points"])
            counts["stability fallbacks"] += int(info["fallback"])
            sent = len(info["points"])
            stable = "-" if info["stability"] is None else f"{info['stability']:.3f}"
            limbs = [name for name, limb in LIMBS.items() if set(limb) & set(keypoints)]
            log.info(f"prompt_pose: frame {g} refined from {sent if sent == len(points) else f'{sent} of {len(points)}'} "
                     f"point(s) on the {' and '.join(limbs)}, with the first pass mask; object score "
                     f"{float(output['object_score_logits'].float().flatten()[0]):.2f}, token-0 stability {stable}"
                     f"{', the best-IoU mask taken' if info['fallback'] else ''}")
    refines = _clock(device) - start

    kept, demoted = demote(capture["cond"], refined)
    conditioning = dict(sorted({**kept, **refined}.items()))
    counts["demoted"] = len(demoted)

    # the second pass: the tracker alone, from the birth, every frame but the conditioning ones
    retracked = [f for f in range(birth, N) if f not in conditioning]
    counts["re-tracked"] = len(retracked)
    selection = c.memory_selection
    count = min(N, tracker.max_obj_ptrs_in_encoder) - 1   # the frames memory selection gathers
    # the frames before the first one the refine can reach keep pass 1's mask; the pass still tracks
    # them, from the birth, for its memory
    first = first_influenced(conditioning, capture["cond"], refined, birth, N, c.max_conditioning_frames, count,
                             selection)
    counts["kept from the first pass"] = first
    log.info(f"prompt_pose: frames {log.frame_ranges(range(first))} keep the first pass: the refine cannot reach them"
             if first else "prompt_pose: the refine can reach every frame")
    lookback = memory_lookback(tracker)
    raw_memory = dataclasses.replace(c, memory_mask=RAW)
    stored = {}
    pbar = ProgressBar(len(retracked))
    trunk = {"seconds": 0.0, "start": 0.0}   # the trunk's own seconds, for a trunk cache's saving

    def trunk_start(module, args):
        trunk["start"] = _clock(device)

    def trunk_end(module, args, output):
        trunk["seconds"] += _clock(device) - trunk["start"]

    start = _clock(device)
    hooks = (backbone.trunk.register_forward_pre_hook(trunk_start), backbone.trunk.register_forward_hook(trunk_end))
    try:
        with torch.inference_mode():
            for f in retracked:
                frame, vision_feats, vision_pos, feat_sizes, high_res, _ = backbone_frame(
                    tracker, backbone_fn, frames, f, device, dtype, size, signed)
                view = pass_two_view(conditioning, stored, f, c.max_conditioning_frames, count, selection)
                current, raw = track_and_clean(tracker, f, vision_feats, vision_pos, feat_sizes, view, N, high_res,
                                               mux, c.fill_hole_area, best_iou, selection)
                if selection:
                    current["memory_score"] = memory_score(current)
                encode_frame_memory(tracker, current, raw, vision_feats, feat_sizes, mux, device, raw_memory)
                stored[f] = current
                keep_memory(stored, f, lookback, count, selection)
                if f >= first:
                    put(f, current["pred_masks"], raw, int(current["mask_index"][0]) if best_iou else None)
                pbar.update(1)
    finally:
        for hook in hooks:
            hook.remove()
    pass_two = _clock(device) - start

    log.info(f"prompt_pose: conditioning frames demoted {log.frame_ranges(demoted) or 'none'}, kept "
             f"{log.frame_ranges(sorted(kept)) or 'none'}; re-tracked {len(retracked)} frame(s) from frame {birth}; "
             f"seconds: pass 1 {pass_one:.1f}, refines {refines:.1f}, pass 2 {pass_two:.1f} "
             f"(its trunk {trunk['seconds']:.1f})")
    counts["frames segmented"] = count_masked_frames(masks)
    report_counts(result, counts)
    return masks
