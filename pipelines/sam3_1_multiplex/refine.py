"""What prompt mode and prompt_pose run after prompt mode's track: Meta's point refine on the tracked
object on frames chosen from the track's masks, as Meta's SAM 3 video API chains it after the text
pass (easy-sam3 @ 88fe578 vendors it), and in prompt_pose the tracker-only re-propagation its action
history asks for:

- `first_pass`: the track (prompt.segment_by_prompt) and the capture the rest reads;
- `region_frames`: the frames of a run where the mask drops a region of her, and points inside it;
- `refine_and_track`: prompt_pose's refines, the demotion and the second pass (step 4 of
  prompt_pose's docstring);
- `segment_by_prompt_repaired`: prompt mode itself, the track and its two repairs from the mask
  alone: the frames after a part lost for good tracked again from the mask before it, then a refine
  of each frame of a run where the mask drops a part of her for a few frames. Nothing repaired, the
  result is the track's tensor itself.
"""
import dataclasses
import time
from typing import TypedDict

import numpy as np
import torch

from ...libs import log
from ...libs.mask import count_masked_frames, to_frame_size
from ...models.sam3_1_multiplex.adapter import (MAX_REFINE_POINTS, SAM3_1_MULTIPLEX_SIZE, backbone_frame,
                                                memory_lookback, multiplex_parts, propagation_backbone,
                                                refine_with_points, track_and_clean)
from ...models.sam3_1_multiplex.postprocess import clean_channel_logits, low_res_logits
from .config import BEST_IOU, RAW, SIGNED_RANGE, report_counts
from .prompt import (PromptCapture, encode_frame_memory, keep_memory, lost_for_good, memory_score, memory_view,
                     propagate_from, segment_by_prompt)


# Meta's refinement_detector_cond_frame_removal_window (easy-sam3 sam3_video_inference.py): the
# detector's conditioning frames this close to a refined frame are demoted to ordinary frames.
DEMOTION_WINDOW = 16


def _clock(device):
    """time.perf_counter() once the work queued on `device` is done, so a step's seconds are its own."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def first_pass(model, clip, images, prompt, config, result=None, logits=None, conditioning_outputs=True):
    """(masks, capture, seconds): prompt mode's track of `images` (segment_by_prompt, with `result` and
    `logits` as it reads them), what it hands over to the rest (PromptCapture), and its seconds. The
    capture holds the conditioning outputs whole with `conditioning_outputs` (prompt_pose's second pass
    conditions on them), else their mask logits alone (prompt mode's repairs)."""
    from comfy import model_management as mm
    device = mm.get_torch_device()
    capture: PromptCapture = {"raw": {}, "cond": {}} if conditioning_outputs else {"raw": {}}
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


# --- what both modes' refines share ------------------------------------------------------------------

def tracker_parts(model, images, config):
    """What tracking or refining `images` [N, H, W, 3] again after the track needs, the model loaded:
    (tracker, backbone, backbone_fn, frames, device, dtype, size, signed), as backbone_frame reads them."""
    from comfy import model_management as mm
    device = mm.get_torch_device()
    dtype = model.model.get_dtype()
    mm.load_model_gpu(model)
    _, _, tracker, backbone = multiplex_parts(model)
    frames = images[..., :3].movedim(-1, 1)
    return (tracker, backbone, propagation_backbone(backbone), frames, device, dtype, tracker.image_size,
            config.input_range == SIGNED_RANGE)


def frame_logits(capture, f):
    """Frame f's mask logits as the track has them, the dense prompt of a refine on it (Meta's lookup
    finds them there): the decoder's raw ones where the track propagated it (an anchor included), the
    conditioning output's on the birth."""
    return capture["raw"][f] if f in capture["raw"] else capture["cond"][f]["pred_masks"]


def show(masks, logits, f, shown, dumped, raw, index=None):
    """Frame f of `masks` [N, H, W] shows the mask logits `shown`; the logits record `logits` (None:
    none) gets `dumped` as a "prompt" frame, `raw` saying whether they are the ones before the cleaning,
    and the mask index `index` where it keeps them."""
    _, H, W = masks.shape
    masks[f] = to_frame_size(shown, H, W)
    if logits is not None:
        logits["logits"][f], logits["cut"][f], logits["raw"][f] = low_res_logits(dumped), "prompt", raw
        if "mask_index" in logits:
            logits["mask_index"][f] = index


def refine_frame(parts, g, points, mux, previous, counts, name, where, dense):
    """Meta's point refine on frame g (refine_with_points) from the positive `points` (x, y) in the
    tracker's 1008 x 1008 space, in the order to send them, and the dense prompt `previous`: the
    output. `parts` are tracker_parts'; `counts` gets the refined frame, its points and a stability
    fallback; one log line, which `name` starts, says where the points lie (`where`) and what the dense
    prompt was (`dense`)."""
    tracker, backbone, backbone_fn, frames, device, dtype, size, signed = parts
    frame, vision_feats, vision_pos, feat_sizes, _, trunk_out = backbone_frame(
        tracker, backbone_fn, frames, g, device, dtype, size, signed)
    output, info = refine_with_points(tracker, backbone, frame, trunk_out, vision_feats, vision_pos, feat_sizes,
                                      points, mux, previous)
    counts["refined frames"] += 1
    counts["points"] += len(info["points"])
    counts["stability fallbacks"] += int(info["fallback"])
    sent = len(info["points"])
    stable = "-" if info["stability"] is None else f"{info['stability']:.3f}"
    log.info(f"{name}: frame {g} refined from {sent if sent == len(points) else f'{sent} of {len(points)}'} "
             f"point(s) {where}, with {dense}; object score "
             f"{float(output['object_score_logits'].float().flatten()[0]):.2f}, token-0 stability {stable}"
             f"{', the best-IoU mask taken' if info['fallback'] else ''}")
    return output


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
    from comfy.utils import ProgressBar
    c = config
    N = images.shape[0]
    parts = tracker_parts(model, images, c)
    tracker, backbone, backbone_fn, frames, device, dtype, size, signed = parts
    mux, birth = capture["mux"], capture["birth"]
    best_iou = c.obj_ptr_token == BEST_IOU

    refined = {}
    start = _clock(device)
    with torch.inference_mode():
        for g, (points, where) in refines.items():
            output = refine_frame(parts, g, points, mux, frame_logits(capture, g), counts, name, where,
                                  "the first pass mask")
            refined[g] = output
            show(masks, logits, g, clean_channel_logits(output["pred_masks"], c.fill_hole_area), output["pred_masks"],
                 True)
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
                    show(masks, logits, f, current["pred_masks"], raw, True,
                         int(current["mask_index"][0]) if best_iou else None)
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


# --- prompt mode ---------------------------------------------------------------------------------

# The counts segment_by_prompt_repaired adds to the track's for the log, keyed by the label the log
# shows, in the order they are added: the frames tracked again after a part lost for good, the frames
# refined for a dropped part, their points and stability fallbacks, then "frames segmented", which
# replaces the track's with the result's.
PromptRepairCounts = TypedDict("PromptRepairCounts", {"tracked again": int, "refined frames": int, "points": int,
                                                      "stability fallbacks": int, "frames segmented": int},
                               total=False)


def segment_by_prompt_repaired(model, clip, images, prompt, config, result=None, logits=None):
    """[N, H, W] float masks of the person in `images` [N, H, W, 3]: prompt mode at max_objects 1. The
    track (segment_by_prompt, its gain re-track included), then two repairs read from its masks alone,
    no pose, each changing only the frames it finds:

    1. A part lost for good (prompt.lost_for_good: a piece of LOST_SHARE of the mask or more gone, the
       area at or under BACK_AREA of the frame before's to the end of the stretch): the frames from the
       one that lost it to the end of their stretch are tracked again, forwards, by the tracker alone,
       on a memory of their own that holds nothing but the mask the track had on the frame before
       (prompt.propagate_from, its decoder logits cleaned as the track cleaned them). The track's memory
       was built with the detector's anchors, whose masks leave out a limb the detector does not see as
       her, and there the tracker let the part go; a memory seeded from her whole mask keeps it.
    2. Then, on those masks, a part dropped for 1 to LOSS_WINDOW frames between two frames that hold it
       (guard.mask.dropped_parts: each of the two frames loses a hand-sized part holding a whole block
       of the Wan Animate workflow's final, grow 10 and blockify 32, the two parts overlap, and neither
       moved away): every frame of that run from the birth on, its mask not empty, is refined
       (refine_frame) from up to MAX_REFINE_POINTS points inside the region both frames hold
       (region_frames), with the frame's raw logits as the dense prompt, and shows the union of its
       mask and the refine's, taken where the decoder's logits are, 288 x 288: the larger of the two
       cleaned logits. No frame is tracked again for it and no conditioning frame changes.

    With neither, the result is the track's tensor itself, its log and counts unchanged. The capture
    this needs (first_pass, the conditioning frames' masks alone) holds every propagated frame's raw
    logits on the CPU, 162 KiB a frame (35 MiB for 233 frames, 97 MiB for 612).

    `result` and `logits` are segment_by_prompt's; in the logits record the frames tracked again are
    "prompt" frames whose logits are the ones before the cleaning ("raw" true) and "mask_index" the
    second track's, and the refined frames "prompt" frames of the union's logits ("raw" false,
    "mask_index" None). The track reads every [prompt] and [prompt, max_objects 1] field; the repairs
    read input_range, fill_hole_area, obj_ptr_token, memory_selection and memory_mask as the track's
    backward pass does."""
    from ..guard.mask import dropped_parts
    c = config
    counts: PromptRepairCounts = {"tracked again": 0, "refined frames": 0, "points": 0, "stability fallbacks": 0}
    masks, capture, _ = first_pass(model, clip, images, prompt, c, result, logits, conditioning_outputs=False)
    birth = capture.get("birth", -1)
    best_iou = c.obj_ptr_token == BEST_IOU
    track = {**capture, "raw": dict(capture.get("raw", {}))}   # the logits the refine reads, a re-track's in its frames
    parts = None

    def tracked(f):
        """Frame f's mask as the tracker has it: its decoder's raw logits cleaned as the track cleaned them,
        or on the birth its conditioning mask."""
        return clean_channel_logits(track["raw"][f], c.fill_hole_area) if f in track["raw"] else \
            track["cond"][f]["pred_masks"]

    with torch.inference_mode():
        for t, end, share in lost_for_good(masks, birth):
            parts = parts or tracker_parts(model, images, c)
            tracker, backbone, backbone_fn, frames, device, dtype, _, _ = parts
            start = _clock(device)

            def emit(f, current, raw):
                track["raw"][f] = raw.to("cpu", copy=True)
                show(masks, logits, f, current["pred_masks"], raw, True,
                     int(current["mask_index"][0]) if best_iou else None)
            propagate_from(tracker, backbone, backbone_fn, frames, tracked(t - 1).to(device), t - 1, end + 1, emit,
                           device, dtype, c)
            counts["tracked again"] += end - t + 1
            log.info(f"prompt: frames {t}-{end} tracked again from frame {t - 1}'s mask: it lost a part for good "
                     f"({share:.2f} of its mask); {_clock(device) - start:.1f} s")

        _, H, W = masks.shape
        for g, (pixels, holding) in region_frames(masks, birth, dropped_parts).items():
            parts = parts or tracker_parts(model, images, c)
            output = refine_frame(parts, g, to_tracker(pixels, H, W), capture["mux"], frame_logits(track, g), counts,
                                  "prompt", held_words(holding), "the tracked mask")
            _, _, _, _, device, *_ = parts
            union = torch.maximum(tracked(g).to(device).float(),
                                  clean_channel_logits(output["pred_masks"], c.fill_hole_area).float())
            show(masks, logits, g, union, union, False)

    if parts is None:
        return masks
    counts["frames segmented"] = count_masked_frames(masks)
    report_counts(result, counts)
    return masks
