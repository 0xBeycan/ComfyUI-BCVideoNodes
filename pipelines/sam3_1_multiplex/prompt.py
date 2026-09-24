"""`prompt` - SAM 3 finds the person itself. Every frame is scored against one text prompt and
nothing else: no detector box, no keypoints, no points of any kind. The first detection that
beats `birth_threshold` starts the track and the tracker's memory carries the mask from
there. This is Meta's own SAM 3 video policy, run over core's weight-carrying primitives:

- a detection scores sigmoid(query) * sigmoid(presence), not the query alone: the presence
  head is the model saying whether the concept is in this frame at all, and on this prompt
  it is the entire signal (see SAM3_1MultiplexConfig)
- detections are thinned by NMS on plain IoU, because several queries fire on one body
- a track is born from the first detection that beats the birth threshold
- for its first `hotstart_frames` frames a track is on probation: if the detector fails to
  find it on `hotstart_unmatched` of them it was never a person - a poster, a reflection, a
  figure in the background that scored once - so it is killed, everything it produced is
  thrown away and the search for a birth starts again. That is why the output is buffered
  until the track is out of its window.
- every `recondition_every` frames a detection that still agrees with the track re-anchors
  it as a conditioning frame, and the non-conditioning memories around it are cleared so the
  fresh anchor is not outvoted by the drift it is there to correct
- the tracker only moves forwards, so once the clip is done the frames before the birth are
  filled by propagating over them in a mirrored index space (see `propagate_backwards`)

With `max_objects` above 1 the same policy runs once per track and Meta's multi-object rules
decide between the tracks (`segment_by_prompt_multi`); at 1 none of that code runs.
"""
from typing import TypedDict

import torch

from ...libs import log
from ...libs.mask import count_masked_frames, to_frame_size
from ...models.sam3_1_multiplex.adapter import (DEFAULT_SAM3_1_MULTIPLEX, backbone_frame, clear_memory,
                                                condition_on_mask, encode_memory, encode_memory_into, encode_prompt,
                                                forget_older, memory_lookback, multiplex_parts, new_memory, new_mux,
                                                new_output_dict, object_logit, object_score, propagation_backbone,
                                                store_output, track_and_clean)
from ...models.sam3_1_multiplex.postprocess import clean_logits, low_res_logits, shown_logits
from .config import META, OURS, logits_record, output_cut, report_counts


# What the person is called to SAM. A bare noun scores higher and more evenly across clips,
# but it also tracks whoever the detector likes best, which on these clips is as often a
# figure in the background; the phrase names the one the generation is about. The
# thresholds in SAM3_1MultiplexConfig were measured with this prompt.
PROMPT = "main person in the foreground"


# Conditioning frames are attended in full and forever, so a clip's worth of them would
# grow the memory attention without bound. Two is what the policy needs: the birth frame,
# which says which person this is, and the newest anchor, which says where they are now.
MAX_CONDITIONING_FRAMES = 2
# Meta's side of the switches: max_cond_frames_in_attn (A3), the tracker object-score logit a
# re-anchored track must have (A5, HIGH_CONF_THRESH on the raw logit), and the memory
# selection threshold on the rescaled object score (A4, mf_threshold).
META_CONDITIONING_FRAMES = 4
META_ANCHOR_OBJECT_LOGIT = 0.8
META_MEMORY_SCORE = 0.01


def iou(masks_a, masks_b):
    """[A, B] intersection over union of two sets of mask logits, thresholded at 0."""
    a = (masks_a > 0).float().flatten(1)
    b = (masks_b > 0).float().flatten(1)
    intersection = a @ b.T
    return intersection / (a.sum(1, keepdim=True) + b.sum(1, keepdim=True).T - intersection).clamp(min=1)


def detect_person(detector, backbone, trunk_out, embedding, text_mask, config):
    """The detections on one frame, sorted by score and thinned by NMS: (masks, scores).

    This is core's `forward_from_trunk` with one line added - it throws away `dec_out`, and
    with it the presence logit that the score is meant to be multiplied by."""
    from comfy.ops import cast_to_input
    features = [conv(trunk_out) for conv in backbone.convs]
    positions = [cast_to_input(backbone.position_encoding(f), f) for f in features]
    _, scores, masks, dec_out = detector._detect(features, positions, embedding, text_mask)
    scores = (scores.sigmoid() * dec_out["presence"].sigmoid())[0].float()
    masks = masks[0]

    order = scores.argsort(descending=True)
    order = order[scores[order] >= config.detection_threshold]
    masks, scores = masks[order], scores[order]
    keep = []
    for i in range(masks.shape[0]):
        if keep and iou(masks[i:i + 1], masks[keep]).max() >= config.nms_iou:
            continue
        keep.append(i)
    return masks[keep], scores[keep]


def _frame_detections(tracker, backbone_fn, frames, frame_idx, device, dtype, size, detector, backbone, embedding,
                      text_mask, config):
    """Frame `frame_idx` as the tracker takes it with its backbone features (backbone_frame), and
    the detections on it (detect_person): (frame, vision_feats, vision_pos, feat_sizes, high_res,
    trunk_out, det_masks, det_scores)."""
    frame, vision_feats, vision_pos, feat_sizes, high_res, trunk_out = backbone_frame(
        tracker, backbone_fn, frames, frame_idx, device, dtype, size)
    det_masks, det_scores = detect_person(detector, backbone, trunk_out, embedding, text_mask, config)
    return frame, vision_feats, vision_pos, feat_sizes, high_res, trunk_out, det_masks, det_scores


def seed_from(det_masks, index, config):
    """The detection that starts or re-anchors a track, as it is handed to the tracker: cleaned
    (ours) or raw (Meta, A9)."""
    if config.seed_cleaning == OURS:
        return clean_logits(det_masks[index:index + 1], config.fill_hole_area)
    return det_masks[index:index + 1].float()


def choose_anchor(det_scores, overlap, config):
    """The detection that re-anchors the single track, or None: of the detections scoring
    recondition_score and overlapping the track by recondition_iou, the first - the best
    scoring, they are sorted - (ours) or the last (Meta's per-detection loop, where a later
    detection overwrites an earlier one for the same track, A6). `overlap` is [D]."""
    fresh = ((det_scores >= config.recondition_score) & (overlap >= config.recondition_iou)).nonzero()
    if not fresh.shape[0]:
        return None
    return int(fresh[0]) if config.anchor_matching == OURS else int(fresh[-1])


def anchor_detections(det_scores, overlap, tracks, config):
    """{track: detection} re-anchoring several tracks; `overlap` is [D, K], `tracks` the columns
    still live. Ours: every track takes the first detection agreeing with it, so two tracks can
    take one detection. Meta (A6): every agreeing detection goes to the one track it overlaps
    most, the last detection winning a track two agree with."""
    anchors = {}
    if config.anchor_matching == OURS:
        for k in tracks:
            fresh = ((det_scores >= config.recondition_score) & (overlap[:, k] >= config.recondition_iou)).nonzero()
            if fresh.shape[0]:
                anchors[k] = int(fresh[0])
        return anchors
    if not tracks:
        return anchors
    columns = torch.tensor(tracks, device=overlap.device)
    for d in range(overlap.shape[0]):
        best, j = overlap[d, columns].max(dim=0)
        if float(det_scores[d]) >= config.recondition_score and float(best) >= config.recondition_iou:
            anchors[tracks[int(j)]] = d
    return anchors


def counts_unmatched(matched, track_masks, config):
    """Whether a probation frame counts against the track: unmatched, and for Meta (A7) only
    when the track's mask is not empty."""
    return not matched and (config.unmatched_counting == OURS or bool((track_masks > 0).any()))


def prune_conditioning(conditioned, how):
    """Drop conditioning frames past the limit (A3). Ours keeps the birth frame, which says
    which person this is, and the newest anchor, which says where they are now; Meta keeps the
    META_CONDITIONING_FRAMES temporally closest, which for a forward pass are the newest."""
    limit, keep_first = (MAX_CONDITIONING_FRAMES, True) if how == OURS else (META_CONDITIONING_FRAMES, False)
    while len(conditioned) > limit:
        del conditioned[sorted(conditioned)[1 if keep_first else 0]]


def object_present(out):
    """Meta's memory score for a stored frame, by the object score alone (A4): the logit,
    rescaled to 0..1 above 0, has to pass META_MEMORY_SCORE. Meta multiplies in the decoder's
    predicted IoU, which core's track_step does not return."""
    logit = object_logit(out)
    return logit > 0 and (torch.sigmoid(torch.tensor(logit)).item() * 2 - 1) > META_MEMORY_SCORE


def memory_view(output_dict, frame_idx, tracker):
    """The output_dict the tracker should read on `frame_idx` under Meta's memory selection
    (A4): the frame before, plus the newest stored frames the person was present on, placed at
    frame_idx - 1, frame_idx - 2, ... in that order. Core's lookup reads memories and object
    pointers by index, so re-indexing them is how the selected frames reach it - the temporal
    positions become ranks, as in Meta's frame_filter."""
    stored = output_dict["non_cond_frame_outputs"]
    selected = [t for t in sorted(stored, reverse=True) if t < frame_idx and object_present(stored[t])]
    selected = selected[:tracker.max_obj_ptrs_in_encoder - 1]
    if frame_idx - 1 in stored and frame_idx - 1 not in selected:
        selected.insert(0, frame_idx - 1)   # the frame before is always read, present or not
    return {"cond_frame_outputs": output_dict["cond_frame_outputs"],
            "non_cond_frame_outputs": {frame_idx - rank: stored[t] for rank, t in enumerate(selected, start=1)}}


def forget_old_memory(stored, frame_idx, lookback, how):
    """Drop the non-conditioning outputs the lookup cannot reach any more. Ours: older than
    `lookback` frames. Meta (A4): all but the newest frame and the newest lookback - 1 frames the
    person was present on, however old."""
    if how == OURS:
        forget_older(stored, frame_idx, lookback)
        return
    present = [t for t in sorted(stored, reverse=True) if t != frame_idx and object_present(stored[t])]
    keep = {frame_idx, *present[:lookback - 1]}
    for old in list(stored):
        if old not in keep:
            del stored[old]


def propagate_backwards(tracker, backbone, backbone_fn, frames, seed, birth, emit, device, dtype, config):
    """Fill in the frames before `birth` by tracking backwards from the detection that
    started the track: `emit(frame, output, raw)` is called with the tracker's output for every
    frame from birth - 1 down to 0, and the logits it had before they were cleaned.

    Core's memory lookups only run forwards - `collect_memory_tokens` asks for frame_idx - 1,
    frame_idx - 2 and so on - so the frames go in mirrored: frame t is tracked under the
    index MIRROR - t, which turns "the frame before" into the real frame after it, the one
    already tracked. Nothing else in the tracker is direction-aware, because every position
    encoding it builds from those indices is relative. This pass propagates and nothing
    else: no detection, no reconditioning, no probation. It is filling a gap at the head of
    the clip from a track that has already proved itself, not deciding anything."""
    N = frames.shape[0]
    mirror = 2 * N
    size = tracker.image_size
    output_dict, mux = new_memory(tracker, device, dtype)

    frame, vision_feats, vision_pos, feat_sizes, high_res, trunk_out = backbone_frame(
        tracker, backbone_fn, frames, birth, device, dtype, size)
    condition_on_mask(tracker, seed, mirror - birth, vision_feats, vision_pos, feat_sizes,
                      high_res, output_dict, N, mux, backbone, frame, trunk_out)

    for f in range(birth - 1, -1, -1):
        frame, vision_feats, vision_pos, feat_sizes, high_res, _ = backbone_frame(
            tracker, backbone_fn, frames, f, device, dtype, size)
        current, raw = track_and_clean(tracker, mirror - f, vision_feats, vision_pos, feat_sizes, output_dict, N,
                                       high_res, mux, config.fill_hole_area)
        encode_memory(tracker, current, vision_feats, feat_sizes, mux, device)
        store_output(output_dict, mirror - f, current, memory_lookback(tracker))
        emit(f, current, raw)


def _prompt_setup(model, clip, images, prompt, config):
    """What both prompt modes set up before their first frame: the model loaded and taken
    apart, the prompt encoded, the frames in the tracker's layout, the memory lookback and the
    output cut, as (c, N, H, W, device, dtype, frames, detector, tracker, backbone, embedding,
    text_mask, size, backbone_fn, lookback, cut). Raises when the checkpoint has no text
    encoder."""
    from comfy import model_management as mm
    if clip is None:
        raise ValueError("the SAM3 checkpoint carries no text encoder, so the prompt cannot be "
                         f"encoded; use a full SAM3 checkpoint such as {DEFAULT_SAM3_1_MULTIPLEX}")
    c = config
    N, H, W, _ = images.shape
    device, dtype = mm.get_torch_device(), model.model.get_dtype()
    frames = images[..., :3].movedim(-1, 1)

    mm.load_model_gpu(model)
    _, detector, tracker, backbone = multiplex_parts(model)
    embedding, text_mask = encode_prompt(clip, detector, prompt, device, dtype)
    size = tracker.image_size
    backbone_fn = propagation_backbone(backbone)
    lookback = memory_lookback(tracker)
    cut = output_cut(c)
    return (c, N, H, W, device, dtype, frames, detector, tracker, backbone, embedding, text_mask,
            size, backbone_fn, lookback, cut)


# The counts segment_by_prompt reports, keyed by the label the log shows, in the order they are
# added; "tracked backwards" only when the frames before the birth were filled.
PromptCounts = TypedDict("PromptCounts", {"false starts": int, "reconditioned": int, "tracked backwards": int,
                                          "frames segmented": int, "tracked from frame": int}, total=False)


def segment_by_prompt(model, clip, images, prompt, config, result=None, logits=None):
    """[N, H, W] float masks of the person in `images` [N, H, W, 3], from the text prompt
    alone. `result`, if given, is filled with what happened for the log; `logits`, if given, a
    dict, receives each output frame's low-res logits (see `logits_record`).

    The [prompt] A/B switches of `config` pick ours or Meta's side of each policy step; at
    their defaults this is the policy described in the module docstring."""
    from comfy.utils import ProgressBar
    (c, N, H, W, device, dtype, frames, detector, tracker, backbone, embedding, text_mask,
     size, backbone_fn, lookback, cut) = _prompt_setup(model, clip, images, prompt, config)
    # M4: birth and anchor frames show the detection cut at `cut`, not the conditioning mask
    detection_shown = c.uniform_mask_threshold and c.m4_anchor_frames == "output"
    record = logits_record(logits, N)
    if record is not None:
        record["raw"] = [False] * N

    masks = torch.zeros(N, H, W)
    output_dict = new_output_dict()
    mux = seed = None
    birth = -1            # the frame the live track was born on, -1 while there is none
    unmatched = 0         # frames of the probation window the track went unmatched
    quiet_until = -1      # no non-conditioning memory is kept up to here, after an anchor
    pending = []          # (frame, mask, logits, raw, how) produced by a track still on probation
    counts: PromptCounts = {"false starts": 0, "reconditioned": 0}
    pbar = ProgressBar(N)

    def put(f, mask, low, raw, how="prompt"):
        masks[f] = mask
        if record is not None:
            record["logits"][f], record["cut"][f], record["raw"][f] = low, how, raw

    with torch.inference_mode():
        for f in range(N):
            (frame, vision_feats, vision_pos, feat_sizes, high_res, trunk_out, det_masks,
             det_scores) = _frame_detections(tracker, backbone_fn, frames, f, device, dtype, size, detector, backbone,
                                             embedding, text_mask, c)
            how = "prompt"

            if birth < 0:
                if det_scores.numel() == 0 or det_scores[0] < c.birth_threshold:
                    pbar.update(1)
                    continue
                mux = new_mux(tracker, device, dtype)
                seed = seed_from(det_masks, 0, c)
                current = condition_on_mask(
                    tracker, seed, f, vision_feats, vision_pos, feat_sizes, high_res,
                    output_dict, N, mux, backbone, frame, trunk_out)
                birth, unmatched, quiet_until, how = f, 0, -1, "birth"
                # Meta tracks the raw detection but outputs it cleaned (A9). `dumped`: the
                # logits before the output's cleaning, None when the frame shows the
                # conditioning mask itself
                if c.seed_cleaning == OURS and not detection_shown:
                    shown, dumped = current["pred_masks"], None
                else:
                    dumped = det_masks[:1].unsqueeze(1)
                    shown = shown_logits(dumped, cut, c.fill_hole_area)
            else:
                lookup = output_dict if c.memory_selection == OURS else memory_view(output_dict, f, tracker)
                current, raw = track_and_clean(tracker, f, vision_feats, vision_pos, feat_sizes, lookup, N, high_res,
                                               mux, c.fill_hole_area)
                shown, dumped = shown_logits(raw, cut, c.fill_hole_area, current["pred_masks"]), raw

                overlap = iou(det_masks, current["pred_masks"][:, 0])[:, 0] if det_masks.shape[0] else None
                matched = overlap is not None and float(overlap.max()) >= c.match_iou

                if f - birth < c.hotstart_frames and counts_unmatched(matched, current["pred_masks"], c):
                    unmatched += 1
                    if unmatched >= c.hotstart_unmatched:
                        # The detector cannot find this track on half of its first frames, so
                        # it was never the person; drop it and everything it drew.
                        log.info(f"the track born on frame {birth} went unmatched on {unmatched} of its "
                                 f"first {f - birth + 1} frames, so it was not the person; dropped")
                        counts["false starts"] += 1
                        output_dict = new_output_dict()
                        mux, birth, pending = None, -1, []
                        pbar.update(1)
                        continue

                anchor = None
                if f % c.recondition_every == 0 and overlap is not None and (
                        c.anchor_score_gate == OURS
                        or object_logit(current) > META_ANCHOR_OBJECT_LOGIT):
                    anchor = choose_anchor(det_scores, overlap, c)
                if anchor is None:
                    if f > quiet_until:
                        encode_memory(tracker, current, vision_feats, feat_sizes, mux, device)
                    output_dict["non_cond_frame_outputs"][f] = current
                    forget_old_memory(output_dict["non_cond_frame_outputs"], f, lookback, c.memory_selection)
                else:
                    # An anchor only anchors if it is alone: drop the spatial memory of the
                    # frames around it, the ones carrying the drift. Everything still in the
                    # dict is within reach of the lookup, and the frames after it are held off
                    # by quiet_until. The object pointers stay - it is the memory that is
                    # cleared, not the frame. (A1: clear_past only clears, Meta does neither.)
                    if c.anchor_memory != META:
                        clear_memory(output_dict["non_cond_frame_outputs"])
                    if c.anchor_memory == OURS:
                        quiet_until = f + c.memory_gap
                    propagated = current
                    current = condition_on_mask(
                        tracker, seed_from(det_masks, anchor, c), f, vision_feats,
                        vision_pos, feat_sizes, high_res, output_dict, N, mux, backbone, frame, trunk_out)
                    shown, dumped, how = current["pred_masks"], None, "anchor"
                    if c.anchor_output == META:
                        # A2: the frame shows the propagated mask, and the conditioning entry
                        # remembers that mask - encoded as a propagated one, the way Meta
                        # re-encodes every frame's memory from the propagated masks
                        encode_memory_into(tracker, propagated, current, vision_feats, feat_sizes, mux, device)
                        shown = shown_logits(raw, cut, c.fill_hole_area, propagated["pred_masks"])
                    elif detection_shown:
                        dumped = det_masks[anchor:anchor + 1].unsqueeze(1)
                        shown = shown_logits(dumped, cut, c.fill_hole_area)
                    counts["reconditioned"] += 1
                    prune_conditioning(output_dict["cond_frame_outputs"], c.conditioning_frames)

            mask = to_frame_size(shown, H, W, cut)
            low = low_res_logits(shown if dumped is None else dumped) if record is not None else None
            if f - birth >= c.hotstart_frames:
                for held in pending:
                    put(*held)
                pending = []
                put(f, mask, low, dumped is not None, how)
            else:
                pending.append((f, mask, low, dumped is not None, how))
            pbar.update(1)

        for held in pending:
            put(*held)
        if birth > 0:
            log.info(f"the track starts on frame {birth}; tracking backwards to fill frames 0-{birth - 1}")
            def emit(f, current, raw):
                put(f, to_frame_size(shown_logits(raw, cut, c.fill_hole_area, current["pred_masks"]), H, W, cut),
                    low_res_logits(raw) if record is not None else None, True)
            propagate_backwards(tracker, backbone, backbone_fn, frames, seed, birth, emit, device, dtype, c)
            counts["tracked backwards"] = birth

    segmented = count_masked_frames(masks)
    if segmented == 0:
        log.warning(f"no frame of this clip scored above {c.birth_threshold} for '{prompt}'; there is no mask")
    else:
        log.info(f"segmented {segmented} of {N} frame(s) from '{prompt}', tracked from frame {birth}"
                 + (f", reconditioned {counts['reconditioned']} time(s)" if counts["reconditioned"] else ""))
    counts["frames segmented"] = segmented
    counts["tracked from frame"] = birth if segmented else 0
    report_counts(result, counts)
    return masks


# --- prompt mode, several objects ----------------------------------------------------------

def suppress_recently_occluded(masks, last_occluded, threshold):
    """[K] bool: which of the K tracks' mask logits [K, h, w] to blank on this frame. Meta's
    `_get_objects_to_suppress_based_on_most_recently_occluded`: of two tracks that overlap by
    `threshold` IoU or more, the one whose mask was empty (or suppressed) more recently gives
    way. `last_occluded` [K] is each track's last such frame, -1 for never."""
    K = masks.shape[0]
    to_suppress = torch.zeros(K, dtype=torch.bool, device=masks.device)
    if K <= 1:
        return to_suppress
    pairs = torch.triu(iou(masks, masks) >= threshold, diagonal=1)
    li, lj = last_occluded[:, None], last_occluded[None, :]
    suppress_i = pairs & (li > lj) & (lj > -1)
    suppress_j = pairs & (lj > li) & (li > -1)
    return suppress_i.any(dim=1) | suppress_j.any(dim=0)


def suppress_shrunk(masks, keep_share):
    """The K tracks' mask logits [K, h, w] as the memory encoder should see them. Meta's
    `_suppress_object_pw_area_shrinkage`, as core writes it for its own multi-object memory
    encode: give every pixel to the track scoring highest there, and a track left with less
    than `keep_share` of its area is noise under another one and is blanked. The masks
    themselves are not made disjoint here - that happens at output."""
    K = masks.shape[0]
    if K <= 1:
        return masks
    winner = masks.argmax(dim=0, keepdim=True)
    own = torch.arange(K, device=masks.device)[:, None, None]
    exclusive = torch.where(winner == own, masks, masks.clamp(max=-10.0))
    before = (masks > 0).sum(dim=(-1, -2)).float().clamp(min=1)
    after = (exclusive > 0).sum(dim=(-1, -2)).float()
    keep = (after / before) >= keep_share
    return torch.where(keep[:, None, None], masks, masks.clamp(max=-10.0))


def non_overlapping(binary, probs):
    """[K, H, W] bool masks made disjoint, Meta's `_apply_object_wise_non_overlapping_constraints`
    at output: a pixel two objects claim goes to the one with the higher object score `probs`
    [K] (the earlier-born one on a tie)."""
    K = binary.shape[0]
    if K <= 1:
        return binary
    scores = torch.where(binary, probs[:, None, None], torch.zeros((), dtype=probs.dtype))
    winner = scores.argmax(dim=0)
    return binary & (winner[None] == torch.arange(K)[:, None, None])


# The counts segment_by_prompt_multi reports, keyed by the label the log shows, in the order they
# are added.
MultiCounts = TypedDict("MultiCounts", {"false starts": int, "duplicates": int, "reconditioned": int,
                                        "suppressed": int, "objects tracked": int, "frames segmented": int},
                        total=False)


def segment_by_prompt_multi(model, clip, images, prompt, config, max_objects, object_index, result=None):
    """[N, H, W] float masks of up to `max_objects` people in `images` [N, H, W, 3], from the
    text prompt alone: the union of all of them when `object_index` is -1, else only object
    `object_index`, objects numbered in birth order (on the same frame, the higher score
    first). Raises when fewer objects than that were tracked.

    Every track runs the single-object policy of `segment_by_prompt` on its own tracker state -
    birth, probation, reconditioning, memory clearing, backwards fill - and Meta's
    multi-object parts decide between them: association (a detection touching any track by
    `assoc_iou` is not a new object; new objects past `max_objects` are dropped, lowest score
    first; the first track is born at `birth_threshold`, further ones at
    `new_object_threshold`), duplicate removal on probation, suppression of the more recently occluded of two
    overlapping tracks, the shrinkage check before memory encoding, and disjoint masks at
    output. Every output is held until the clip is done, so a track dropped on probation
    leaves nothing behind. Meta's keep-alive counters and masklet confirmation are not here:
    both are switched off in the configuration easy-sam3 ships (keep-alive only suppresses
    with suppress_unmatched_only_within_hotstart=False, confirmation with
    masklet_confirmation_enable=True), so they would be dead code. Of the [prompt] A/B switches
    only anchor_matching (A6) and unmatched_counting (A7) are read here, and the
    uniform_mask_threshold cut (M4)."""
    from comfy.utils import ProgressBar
    (c, N, H, W, device, dtype, frames, detector, tracker, backbone, embedding, text_mask,
     size, backbone_fn, lookback, cut) = _prompt_setup(model, clip, images, prompt, config)

    born = []             # every track ever started, in birth order
    live = []             # the tracks still running
    duplicates = {}       # (older id, younger id) -> probation frames they shared a detection
    outputs = [dict() for _ in range(N)]   # frame -> {track id: (low-res logits on the CPU, score)}
    counts: MultiCounts = {"false starts": 0, "duplicates": 0, "reconditioned": 0, "suppressed": 0}
    pbar = ProgressBar(N)

    def drop(track, why):
        log.info(f"the track born on frame {track['birth']} {why}; dropped")
        track["removed"] = True
        for f in range(track["birth"], N):
            outputs[f].pop(track["id"], None)

    with torch.inference_mode():
        for f in range(N):
            (frame, vision_feats, vision_pos, feat_sizes, high_res, trunk_out, det_masks,
             det_scores) = _frame_detections(tracker, backbone_fn, frames, f, device, dtype, size, detector, backbone,
                                             embedding, text_mask, c)
            D = det_masks.shape[0]

            for t in live:
                current, t["raw"] = track_and_clean(tracker, f, vision_feats, vision_pos, feat_sizes, t["output"], N,
                                                    high_res, t["mux"], c.fill_hole_area)
                t["current"] = current
                t["score"] = object_score(current)
            K = len(live)
            overlap = (iou(det_masks, torch.cat([t["current"]["pred_masks"][:, 0] for t in live]))
                       if K and D else torch.zeros(D, K, device=device))

            # association: a detection that touches any track is that track's
            threshold = c.new_object_threshold if K else c.birth_threshold
            new = [d for d in range(D) if float(det_scores[d]) >= threshold
                   and not bool((overlap[d] >= c.assoc_iou).any())]
            new = new[:max(0, max_objects - K)]      # detections are sorted, best first

            # probation: unmatched tracks, and the younger of two tracks sharing a detection
            for k, t in enumerate(live):
                matched = D > 0 and float(overlap[:, k].max()) >= c.match_iou
                if f - t["birth"] < c.hotstart_frames and counts_unmatched(matched, t["current"]["pred_masks"], c):
                    t["unmatched"] += 1
                    if t["unmatched"] >= c.hotstart_unmatched:
                        drop(t, f"went unmatched on {t['unmatched']} of its first {f - t['birth'] + 1} frames")
                        counts["false starts"] += 1
            for d in range(D):
                sharing = [k for k in range(K) if float(overlap[d, k]) >= c.assoc_iou]
                if len(sharing) < 2:
                    continue
                oldest = min(sharing, key=lambda k: live[k]["id"])
                for k in sharing:
                    if k != oldest:
                        key = (live[oldest]["id"], live[k]["id"])
                        duplicates[key] = duplicates.get(key, 0) + 1
            for (older, younger), shared in duplicates.items():
                t = born[younger]
                if (not t["removed"] and f - t["birth"] < c.hotstart_frames
                        and shared >= c.duplicate_frames):
                    drop(t, f"shared a detection with the track born on frame {born[older]['birth']} "
                            f"on {shared} frames")
                    counts["duplicates"] += 1
            kept = [k for k, t in enumerate(live) if not t["removed"]]

            # reconditioning, per track
            anchors = {}
            if f % c.recondition_every == 0 and D:
                anchors = anchor_detections(det_scores, overlap, kept, c)

            # the tracks against each other, then memory
            if kept:
                stacked = torch.cat([live[k]["current"]["pred_masks"][:, 0] for k in kept])
                last = torch.tensor([live[k]["last_occluded"] for k in kept], device=device)
                suppressed = suppress_recently_occluded(stacked, last, c.occlusion_iou)
                empty = ~(stacked > 0).any(dim=(-1, -2))
                for j, k in enumerate(kept):
                    if k in anchors:
                        suppressed[j] = False
                    if bool(suppressed[j]) or bool(empty[j]):
                        live[k]["last_occluded"] = f
                    if bool(suppressed[j]):
                        live[k]["current"]["pred_masks"][:] = -10.0
                        stacked[j] = -10.0
                        counts["suppressed"] += 1
                memory = suppress_shrunk(stacked, c.shrink_keep)
            for j, k in enumerate(kept):
                t = live[k]
                output = t["output"]
                if k not in anchors:
                    current = t["current"]
                    if f > t["quiet_until"]:
                        encoded = {"pred_masks": memory[j:j + 1, None]}
                        encode_memory_into(tracker, encoded, current, vision_feats, feat_sizes, t["mux"], device)
                    store_output(output, f, current, lookback)
                else:
                    # as in segment_by_prompt: an anchor only anchors if it is alone
                    clear_memory(output["non_cond_frame_outputs"])
                    t["quiet_until"] = f + c.memory_gap
                    a = anchors[k]
                    t["current"] = condition_on_mask(
                        tracker, clean_logits(det_masks[a:a + 1], c.fill_hole_area), f, vision_feats, vision_pos,
                        feat_sizes, high_res, output, N, t["mux"], backbone, frame, trunk_out)
                    counts["reconditioned"] += 1
                    conditioned = output["cond_frame_outputs"]
                    prune_conditioning(conditioned, OURS)
                shown = t["current"]["pred_masks"]
                if k not in anchors and not bool(suppressed[j]):
                    shown = shown_logits(t["raw"], cut, c.fill_hole_area, shown)
                outputs[f][t["id"]] = (shown.float().cpu(), t["score"])
            live = [live[k] for k in kept]

            # births, best score first
            for d in new:
                seed = clean_logits(det_masks[d:d + 1], c.fill_hole_area)
                t = {"id": len(born), "birth": f, "seed": seed, "unmatched": 0, "quiet_until": -1,
                     "last_occluded": -1, "removed": False, "score": float(det_scores[d]),
                     "mux": new_mux(tracker, device, dtype),
                     "output": new_output_dict()}
                t["current"] = condition_on_mask(
                    tracker, seed, f, vision_feats, vision_pos, feat_sizes, high_res,
                    t["output"], N, t["mux"], backbone, frame, trunk_out)
                outputs[f][t["id"]] = (t["current"]["pred_masks"].float().cpu(), t["score"])
                born.append(t)
                live.append(t)
            pbar.update(1)

        objects = [t for t in born if not t["removed"]]
        for t in objects:
            if t["birth"] > 0:
                log.info(f"object {objects.index(t)} starts on frame {t['birth']}; tracking backwards "
                         f"to fill frames 0-{t['birth'] - 1}")

                def emit(f, current, raw, track=t):
                    score = object_score(current)
                    shown = shown_logits(raw, cut, c.fill_hole_area, current["pred_masks"])
                    outputs[f][track["id"]] = (shown.float().cpu(), score)
                propagate_backwards(tracker, backbone, backbone_fn, frames, t["seed"], t["birth"], emit,
                                    device, dtype, c)

    if object_index >= len(objects):
        raise ValueError(f"object_index {object_index}: {len(objects)} object(s) were tracked "
                         f"for '{prompt}' (numbered from 0)")
    masks = torch.zeros(N, H, W)
    ids = [t["id"] for t in objects]
    for f in range(N):
        present = [i for i in ids if i in outputs[f]]
        if not present:
            continue
        binary = torch.stack([to_frame_size(outputs[f][i][0], H, W, cut) > 0 for i in present])
        binary = non_overlapping(binary, torch.tensor([outputs[f][i][1] for i in present]))
        if object_index < 0:
            masks[f] = binary.any(dim=0).float()
        elif ids[object_index] in present:
            masks[f] = binary[present.index(ids[object_index])].float()

    segmented = count_masked_frames(masks)
    log.info(f"tracked {len(objects)} object(s) from '{prompt}', born on frame(s) "
             f"{[t['birth'] for t in objects]}; {segmented} of {N} frame(s) segmented")
    counts["objects tracked"] = len(objects)
    counts["frames segmented"] = segmented
    report_counts(result, counts)
    return masks
