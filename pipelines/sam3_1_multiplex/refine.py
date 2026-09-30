"""What prompt_pose runs after prompt mode's track: Meta's point refine on the tracked object, on the
frames chosen from the track's masks, then the tracker-only re-propagation its action history asks
for. The sequence is Meta's SAM 3 video API chained as its predictor runs it (easy-sam3 @ 88fe578
vendors it):

- `first_pass`: the track (prompt.segment_by_prompt) and the capture the rest reads;
- `region_frames`: the frames of a run where the mask drops a region of her, and points inside it;
- `refine_and_track`: the refines, the demotion and the second pass (step 4 of prompt_pose's
  docstring).
"""
import dataclasses
import time
from typing import TypedDict

import numpy as np
import torch

from ...libs import log
from ...libs.mask import to_frame_size
from ...models.sam3_1_multiplex.adapter import (MAX_REFINE_POINTS, SAM3_1_MULTIPLEX_SIZE, backbone_frame,
                                                memory_lookback, multiplex_parts, propagation_backbone,
                                                refine_with_points, track_and_clean)
from ...models.sam3_1_multiplex.postprocess import clean_channel_logits, low_res_logits
from .config import BEST_IOU, RAW, SIGNED_RANGE
from .prompt import PromptCapture, encode_frame_memory, keep_memory, memory_score, memory_view, segment_by_prompt


# Meta's refinement_detector_cond_frame_removal_window (easy-sam3 sam3_video_inference.py): the
# detector's conditioning frames this close to a refined frame are demoted to ordinary frames.
DEMOTION_WINDOW = 16


def _clock(device):
    """time.perf_counter() once the work queued on `device` is done, so a step's seconds are its own."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def first_pass(model, clip, images, prompt, config, result=None, logits=None):
    """(masks, capture, seconds): prompt mode's track of `images` (segment_by_prompt, with `result` and
    `logits` as it reads them), what it hands over to the refine (PromptCapture), and its seconds."""
    from comfy import model_management as mm
    device = mm.get_torch_device()
    capture: PromptCapture = {"raw": {}}
    start = _clock(device)
    masks = segment_by_prompt(model, clip, images, prompt, config, result=result, logits=logits, capture=capture)
    return masks, capture, _clock(device) - start


# --- a region the mask drops for a few frames ----------------------------------------------------

class _Booleans:
    """[N, H, W] float masks as the mask_loss detections read them: a frame's booleans (mask > 0), or a
    box of them, at a time, so the clip is never copied whole."""

    def __init__(self, masks):
        self.masks, self.shape = masks, tuple(masks.shape)

    def __getitem__(self, key):
        return (self.masks[key] > 0).numpy()


def region_points(piece, corner, count=MAX_REFINE_POINTS):
    """Up to `count` points well inside the region `piece` ([h, w] booleans, its top-left pixel at
    `corner`, (y, x)) and spread over it, as (x, y) pixel centres of the frame. The region's depth
    is its deepest pixel's distance from the outline (the frame edge counts as outline). Among the
    pixels at least half that depth inside: the deepest (the first in raster order on a tie), then
    the deepest farther than half that depth from every point taken, and so on."""
    import cv2
    h, w = piece.shape
    depth = cv2.distanceTransform(np.pad(piece, 1).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    half = float(depth.max()) / 2
    free = depth >= half
    ys, xs = np.mgrid[:h, :w]
    points = []
    while len(points) < count and free.any():
        y, x = np.unravel_index(int(np.argmax(np.where(free, depth, -1.0))), depth.shape)
        points.append((corner[1] + int(x) + 0.5, corner[0] + int(y) + 0.5))
        free &= (ys - y) ** 2 + (xs - x) ** 2 > half ** 2
    return points


def region_frames(masks, birth, find):
    """{frame: (points, holding)}: the frames a mask-driven trigger refines, ascending, the positive
    points it sends on each as (x, y) pixel centres (region_points of every region the frame lacks, in
    the order found) and the frames that hold those regions around it, ascending. `masks` [N, H, W] are
    the track's, born on frame `birth` (-1: none); `find` reads their booleans and yields the regions as
    guard.mask's detections do, (run, anchors, region, corner, share). Every frame of a region's run is
    refined, except, as with prompt_pose's pose-driven rule, the frames before the birth and a frame
    whose mask is empty."""
    found = {}
    if birth < 0:
        return found
    for run, anchors, piece, corner, _ in find(_Booleans(masks)):
        points = region_points(piece, corner)
        for f in run:
            if f >= birth and masks[f].any():
                pixels, holding = found.setdefault(f, ([], set()))
                pixels += points
                holding.update(anchors)
    return {f: (pixels, sorted(holding)) for f, (pixels, holding) in sorted(found.items())}


def held_words(holding):
    """The log's words for points in a region the frames `holding` hold around a frame that drops it."""
    return f"in a region the mask dropped (held on frames {', '.join(map(str, holding[:-1]))} and {holding[-1]})"


def to_tracker(pixels, H, W):
    """(x, y) pixel positions of an H x W frame in the tracker's SAM3_1_MULTIPLEX_SIZE square."""
    return [(x / W * SAM3_1_MULTIPLEX_SIZE, y / H * SAM3_1_MULTIPLEX_SIZE) for x, y in pixels]


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


# --- the refine action ---------------------------------------------------------------------------

# The counts refine_and_track adds to its caller's, keyed by the label the log shows, in the order
# they are added.
RefineCounts = TypedDict("RefineCounts", {"refined frames": int, "points": int, "stability fallbacks": int,
                                          "demoted": int, "re-tracked": int, "kept from the first pass": int},
                         total=False)


def refine_and_track(model, images, config, masks, capture, refines, counts: RefineCounts, logits, name, pass_one):
    """Meta's refine action on the track of `images` [N, H, W, 3]: `masks` [N, H, W] and `capture` are
    first_pass's (the masks changed in place), `refines` the frames to refine, {frame: (points, where)}
    ascending, with the positive points (x, y) in the tracker's 1008 x 1008 space in the order to send
    them and the log's words for where they lie. `counts` gets RefineCounts, `logits` is the logits
    record the track filled (None: none), `name` starts each log line and `pass_one` is the track's
    seconds, for the last one.

    1. Each frame is refined from its points and the track's mask logits of it as the dense prompt:
       the raw ones where the track propagated it, the conditioning output's on the birth, as Meta's
       lookup finds them (refine_with_points; past MAX_REFINE_POINTS it keeps the first half and the
       last half). From the points without that mask (Meta's first refine of a frame) the test clips'
       refined frame kept the forearms and hands but lost the head, torso and dress. The frame becomes
       a conditioning frame, a refine on the birth or an anchor replacing that conditioning.
    2. The track's conditioning frames (the birth and the fired anchors) within DEMOTION_WINDOW of a
       refined frame and not refined themselves are demoted to ordinary frames (demote).
    3. The tracker alone tracks the clip again from the birth: no detection, no re-anchor, no
       probation, every frame but the conditioning ones re-decoded, each reading the conditioning
       frames closest to it on both sides (pass_two_view), its memory encoded from the decoder's raw
       logits whatever memory_mask says, as Meta's re-propagation does.
    4. The frames before the first one the refine can reach (first_influenced) keep the track's mask:
       the refine cannot change them, so all the second pass would add there is its drift from the
       track, tracking without the detector and with raw-logit memory.

    The refined frames and the re-decoded frames the result shows are "prompt" frames of the logits
    record whose logits are the ones before the cleaning ("raw" true); every other frame keeps the
    track's entry. The second pass reads input_range, fill_hole_area, obj_ptr_token, memory_selection
    and max_conditioning_frames."""
    from comfy import model_management as mm
    from comfy.utils import ProgressBar
    c = config
    N, H, W, _ = images.shape
    device = mm.get_torch_device()
    dtype = model.model.get_dtype()
    mm.load_model_gpu(model)
    _, _, tracker, backbone = multiplex_parts(model)
    backbone_fn = propagation_backbone(backbone)
    frames = images[..., :3].movedim(-1, 1)
    size, signed = tracker.image_size, c.input_range == SIGNED_RANGE
    mux, birth = capture["mux"], capture["birth"]
    best_iou = c.obj_ptr_token == BEST_IOU

    def put(f, shown, dumped, index=None):
        """Frame f shows the mask logits `shown`; the logits record gets `dumped`, the ones before the
        cleaning."""
        masks[f] = to_frame_size(shown, H, W)
        if logits is not None:
            logits["logits"][f], logits["cut"][f], logits["raw"][f] = low_res_logits(dumped), "prompt", True
            if "mask_index" in logits:
                logits["mask_index"][f] = index

    refined = {}
    start = _clock(device)
    with torch.inference_mode():
        for g, (points, where) in refines.items():
            frame, vision_feats, vision_pos, feat_sizes, _, trunk_out = backbone_frame(
                tracker, backbone_fn, frames, g, device, dtype, size, signed)
            output, info = refine_with_points(tracker, backbone, frame, trunk_out, vision_feats, vision_pos,
                                              feat_sizes, points, mux,
                                              capture["raw"][g] if g in capture["raw"] else capture["cond"][g]["pred_masks"])
            refined[g] = output
            put(g, clean_channel_logits(output["pred_masks"], c.fill_hole_area), output["pred_masks"])
            counts["refined frames"] += 1
            counts["points"] += len(info["points"])
            counts["stability fallbacks"] += int(info["fallback"])
            sent = len(info["points"])
            stable = "-" if info["stability"] is None else f"{info['stability']:.3f}"
            log.info(f"{name}: frame {g} refined from {sent if sent == len(points) else f'{sent} of {len(points)}'} "
                     f"point(s) {where}, with the first pass mask; object score "
                     f"{float(output['object_score_logits'].float().flatten()[0]):.2f}, token-0 stability {stable}"
                     f"{', the best-IoU mask taken' if info['fallback'] else ''}")
    refine_seconds = _clock(device) - start

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
    log.info(f"{name}: frames {log.frame_ranges(range(first))} keep the first pass: the refine cannot reach them"
             if first else f"{name}: the refine can reach every frame")
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

    log.info(f"{name}: conditioning frames demoted {log.frame_ranges(demoted) or 'none'}, kept "
             f"{log.frame_ranges(sorted(kept)) or 'none'}; re-tracked {len(retracked)} frame(s) from frame {birth}; "
             f"seconds: pass 1 {pass_one:.1f}, refines {refine_seconds:.1f}, pass 2 {pass_two:.1f} "
             f"(its trunk {trunk['seconds']:.1f})")
    return masks
