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
  it as a conditioning frame. At the defaults (easy-sam3's policy) the detection gives the
  frame its object pointer while the frame's memory is the tracker's own mask, nothing is
  cleared, and only an anchor the tracker itself is sure of fires. The earlier policy - the
  detector mask as the anchor's memory, the memories around it cleared - is still selectable
  in SAM3_1MultiplexConfig
- the tracker only moves forwards, so once the clip is done the frames before the birth are
  filled by propagating over them in a mirrored index space (see `propagate_backwards`)

With `max_objects` above 1 the earlier single-track policy runs once per track, on the shared
defaults (input range, pointer token, memory_gap, thresholds), and Meta's multi-object rules
decide between the tracks (`segment_by_prompt_multi`); at 1 none of that code runs.
"""
from typing import Optional, TypedDict

import numpy as np
import torch

from ...libs import log
from ...libs.mask import count_masked_frames, to_frame_size
from ...models.sam3_1_multiplex.adapter import (DEFAULT_SAM3_1_MULTIPLEX, backbone_frame, clear_memory,
                                                condition_on_mask, encode_memory_into, encode_prompt,
                                                forget_older, memory_lookback, multiplex_parts, new_memory, new_mux,
                                                new_output_dict, object_score, propagation_backbone, store_output,
                                                track_and_clean)
from ...models.sam3_1_multiplex.postprocess import clean_logits, low_res_logits
from .config import BEST_IOU, CLEANED, DETECTION, OURS, PROPAGATED, SIGNED_RANGE, logits_record, report_counts


# What the person is called to SAM. A bare noun scores higher and more evenly across clips,
# but it also tracks whoever the detector likes best, which on these clips is as often a
# figure in the background; the phrase names the one the generation is about. The
# defaults in SAM3_1MultiplexConfig were validated with this prompt.
PROMPT = "main person in the foreground"


# Conditioning frames are attended in full and forever, so a clip's worth of them would
# grow the memory attention without bound. Two is what the policy needs: the birth frame,
# which says which person this is, and the newest anchor, which says where they are now.
# (segment_by_prompt reads its count from max_conditioning_frames, whose default this is.)
MAX_CONDITIONING_FRAMES = 2

# easy-sam3's memory selection (memory_selection): a propagated frame counts as memory only
# when its memory_score is above this (its mf_threshold).
MEMORY_SELECTION_SCORE = 0.01


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
        tracker, backbone_fn, frames, frame_idx, device, dtype, size, config.input_range == SIGNED_RANGE)
    det_masks, det_scores = detect_person(detector, backbone, trunk_out, embedding, text_mask, config)
    return frame, vision_feats, vision_pos, feat_sizes, high_res, trunk_out, det_masks, det_scores


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


def prune_conditioning(conditioned, keep=MAX_CONDITIONING_FRAMES, keep_birth=True):
    """Drop the oldest conditioning frames past `keep`: with `keep_birth` the birth frame, the
    oldest, stays and the oldest anchor goes; without it the birth frame goes first."""
    while len(conditioned) > keep:
        del conditioned[sorted(conditioned)[1 if keep_birth else 0]]


def encode_frame_memory(tracker, current, raw, vision_feats, feat_sizes, mux, device, config, into=None):
    """Encode the spatial memory of the propagated output `current`, from the logits
    `memory_mask` names: its cleaned ones (cleaned), or `raw`, the decoder's before the cleaning
    (raw). It lands in `current`, or with `into` (a conditioning frame's output) in `into`, as a
    conditioning memory."""
    source = current if config.memory_mask == CLEANED else {"pred_masks": raw.float()}
    encode_memory_into(tracker, source, current if into is None else into, vision_feats, feat_sizes, mux, device,
                       conditioning=into is not None)


def memory_score(current):
    """easy-sam3's memory score of a propagated output (its eff_iou_score): the mean over
    objects of the normalized object score, 2 sigmoid(s) - 1 where the logit s > 0 and 0
    otherwise, times the decoder's highest predicted IoU."""
    s = current["object_score_logits"].float().flatten()
    norm = torch.where(s > 0, s.sigmoid() * 2 - 1, torch.zeros_like(s))
    return float((norm * current["iou_pred"].float().flatten()).mean())


def selected_frames(stored, frame_idx, count):
    """The propagated frames memory selection reads at `frame_idx`, newest first: the frame
    before, always, then the stored frames before `frame_idx` (frame 0 excluded, as easy-sam3's
    scan stops at 1) whose "memory_score" passes MEMORY_SELECTION_SCORE, newest first, until
    `count` of those. A frame that is not stored - an anchor, which is a conditioning frame - is
    skipped, except the frame before, which stays in the list and reads as nothing. easy-sam3's
    `frame_filter`."""
    passed = []
    for t in sorted((t for t in stored if 0 < t < frame_idx), reverse=True):
        if stored[t]["memory_score"] > MEMORY_SELECTION_SCORE:
            passed.append(t)
            if len(passed) >= count:
                break
    return passed if frame_idx - 1 in passed else [frame_idx - 1] + passed


def memory_view(output_dict, frame_idx, count):
    """`output_dict` as the tracker reads it at `frame_idx` under memory selection: the selected
    frames (selected_frames) re-keyed by rank, frame_idx - k holding the k-th newest, so core's
    lookup, which reads the frames right before by distance, reads them with slot = rank - for
    the spatial memories and the object pointers alike, as easy-sam3 does. The oldest carries no
    object pointer: easy-sam3 reads one pointer fewer than it selects frames."""
    stored = output_dict["non_cond_frame_outputs"]
    chosen = selected_frames(stored, frame_idx, count)
    view = {}
    for k, t in enumerate(chosen, start=1):
        if t in stored:
            view[frame_idx - k] = stored[t] if k < len(chosen) else {
                key: value for key, value in stored[t].items() if key != "obj_ptr"}
    return {"cond_frame_outputs": output_dict["cond_frame_outputs"], "non_cond_frame_outputs": view}


def keep_memory(stored, frame_idx, lookback, count, selection):
    """Drop the stored propagated outputs no later lookup can read: those more than `lookback`
    frames back (forget_older), except, with memory `selection`, the `count` newest ones that
    pass its score, which it reaches for however far back they are."""
    if not selection:
        forget_older(stored, frame_idx, lookback)
        return
    keep = set(selected_frames(stored, frame_idx + 1, count))
    for old in list(stored):
        if old < frame_idx - lookback and old not in keep:
            del stored[old]


# One re-anchor slot of segment_by_prompt, for the logits dump: whether it fired and why not
# ("no det": no detection on the frame; "gate": none scored recondition_score and overlapped
# recondition_iou; "track score": the tracker's own logit was at or under anchor_track_score), the
# detection it anchored with (or the one overlapping the track most), its score and IoU, the
# tracker's object-score logit, and what the tracker's mask covers that the detection does not,
# at frame size: px, and the px and inscribed radius of its largest piece.
AnchorLog = TypedDict("AnchorLog", {"frame": int, "fired": bool, "why": Optional[str], "det_score": Optional[float],
                                    "iou": Optional[float], "track_logit": float, "missed_px": Optional[int],
                                    "missed_largest_px": Optional[int], "missed_radius": Optional[float]})


def anchor_log(f, anchor, blocked, det_masks, det_scores, overlap, track, logit, H, W, config) -> AnchorLog:
    """The AnchorLog of the re-anchor slot on frame `f`: `anchor` the detection choose_anchor
    picked (None: none qualified), `blocked` whether anchor_track_score stopped it, `track` the
    tracker's propagated output on the frame and `logit` its object-score logit."""
    from scipy import ndimage
    entry: AnchorLog = {"frame": f, "fired": anchor is not None and not blocked,
                        "why": None if anchor is not None and not blocked else
                        "no det" if overlap is None else "gate" if anchor is None else "track score",
                        "det_score": None, "iou": None, "track_logit": round(logit, 4), "missed_px": None,
                        "missed_largest_px": None, "missed_radius": None}
    if overlap is None:
        return entry
    d = anchor if anchor is not None else int(overlap.argmax())
    detected = to_frame_size(clean_logits(det_masks[d:d + 1], config.fill_hole_area)[None], H, W) > 0
    missed = ((to_frame_size(track["pred_masks"], H, W) > 0) & ~detected).numpy()
    entry.update(det_score=round(float(det_scores[d]), 4), iou=round(float(overlap[d]), 4), missed_px=int(missed.sum()))
    if missed.any():
        labels, n = ndimage.label(missed)
        sizes = np.bincount(labels.ravel())[1:]
        largest = labels == int(sizes.argmax()) + 1
        entry.update(missed_largest_px=int(sizes.max()),
                     missed_radius=round(float(ndimage.distance_transform_edt(largest).max()), 1))
    return entry


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

    signed = config.input_range == SIGNED_RANGE
    frame, vision_feats, vision_pos, feat_sizes, high_res, trunk_out = backbone_frame(
        tracker, backbone_fn, frames, birth, device, dtype, size, signed)
    condition_on_mask(tracker, seed, mirror - birth, vision_feats, vision_pos, feat_sizes,
                      high_res, output_dict, N, mux, backbone, frame, trunk_out)

    selection = config.memory_selection
    count = min(N, tracker.max_obj_ptrs_in_encoder) - 1
    for f in range(birth - 1, -1, -1):
        frame, vision_feats, vision_pos, feat_sizes, high_res, _ = backbone_frame(
            tracker, backbone_fn, frames, f, device, dtype, size, signed)
        current, raw = track_and_clean(tracker, mirror - f, vision_feats, vision_pos, feat_sizes,
                                       memory_view(output_dict, mirror - f, count) if selection else output_dict, N,
                                       high_res, mux, config.fill_hole_area, config.obj_ptr_token == BEST_IOU, selection)
        if selection:
            current["memory_score"] = memory_score(current)
        encode_frame_memory(tracker, current, raw, vision_feats, feat_sizes, mux, device, config)
        output_dict["non_cond_frame_outputs"][mirror - f] = current
        keep_memory(output_dict["non_cond_frame_outputs"], mirror - f, memory_lookback(tracker), count, selection)
        emit(f, current, raw)


def _prompt_setup(model, clip, images, prompt, config):
    """What both prompt modes set up before their first frame: the model loaded and taken
    apart, the prompt encoded, the frames in the tracker's layout and the memory lookback, as
    (c, N, H, W, device, dtype, frames, detector, tracker, backbone, embedding, text_mask, size,
    backbone_fn, lookback). Raises when the checkpoint has no text encoder."""
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
    return (c, N, H, W, device, dtype, frames, detector, tracker, backbone, embedding, text_mask,
            size, backbone_fn, lookback)


# The counts segment_by_prompt reports, keyed by the label the log shows, in the order they are
# added; "tracked backwards" only when the frames before the birth were filled.
PromptCounts = TypedDict("PromptCounts", {"false starts": int, "reconditioned": int, "tracked backwards": int,
                                          "frames segmented": int, "tracked from frame": int}, total=False)


def segment_by_prompt(model, clip, images, prompt, config, result=None, logits=None):
    """[N, H, W] float masks of the person in `images` [N, H, W, 3], from the text prompt
    alone. `result`, if given, is filled with what happened for the log; `logits`, if given, a
    dict, receives each output frame's low-res logits (see `logits_record`).

    The [prompt] A/B switches of `config` pick ours or Meta's side of a policy step (A6, A7); at
    their defaults this is the policy described in the module docstring. `anchor_output` picks
    what a re-anchor frame shows (the detection, or the mask the tracker propagated onto it); the
    tracking is the same either way. `input_range`, `obj_ptr_token` and `memory_mask` pick our
    side or Meta's of how SAM 3.1 Multiplex is run; with obj_ptr_token best_iou the logits record
    also gets "mask_index", the mask the decoder selected on each output frame it propagated (None
    on the birth frame and on frames without output). `clear_on_anchor`, `anchor_mask`,
    `max_conditioning_frames`, `keep_birth_frame`, `anchor_track_score` and `memory_selection`
    pick ours or easy-sam3's side of the re-anchor and memory policy. The logits record also gets
    "anchors", one AnchorLog per re-anchor slot."""
    from comfy.utils import ProgressBar
    (c, N, H, W, device, dtype, frames, detector, tracker, backbone, embedding, text_mask,
     size, backbone_fn, lookback) = _prompt_setup(model, clip, images, prompt, config)
    best_iou = c.obj_ptr_token == BEST_IOU
    selection = c.memory_selection
    count = min(N, tracker.max_obj_ptrs_in_encoder) - 1   # the frames memory selection gathers
    record = logits_record(logits, N)
    if record is not None:
        record["raw"] = [False] * N
        record["anchors"] = []
        if best_iou:
            record["mask_index"] = [None] * N

    masks = torch.zeros(N, H, W)
    output_dict = new_output_dict()
    mux = seed = None
    birth = -1            # the frame the live track was born on, -1 while there is none
    unmatched = 0         # frames of the probation window the track went unmatched
    quiet_until = -1      # no non-conditioning memory is kept up to here, after an anchor
    pending = []          # (frame, mask, logits, raw, how, index) produced by a track still on probation
    picked = []           # with best_iou, the mask index the decoder selected on each output frame
    counts: PromptCounts = {"false starts": 0, "reconditioned": 0}
    pbar = ProgressBar(N)

    def put(f, mask, low, raw, how="prompt", index=None):
        masks[f] = mask
        if index is not None:
            picked.append(index)
        if record is not None:
            record["logits"][f], record["cut"][f], record["raw"][f] = low, how, raw
            if best_iou:
                record["mask_index"][f] = index

    with torch.inference_mode():
        for f in range(N):
            (frame, vision_feats, vision_pos, feat_sizes, high_res, trunk_out, det_masks,
             det_scores) = _frame_detections(tracker, backbone_fn, frames, f, device, dtype, size, detector, backbone,
                                             embedding, text_mask, c)
            how, index = "prompt", None

            if birth < 0:
                if det_scores.numel() == 0 or det_scores[0] < c.birth_threshold:
                    pbar.update(1)
                    continue
                mux = new_mux(tracker, device, dtype)
                seed = clean_logits(det_masks[:1], c.fill_hole_area)
                current = condition_on_mask(
                    tracker, seed, f, vision_feats, vision_pos, feat_sizes, high_res,
                    output_dict, N, mux, backbone, frame, trunk_out)
                birth, unmatched, quiet_until, how = f, 0, -1, "birth"
                # `dumped`: the logits before the output's cleaning, None when the frame shows
                # the conditioning mask itself
                shown, dumped = current["pred_masks"], None
            else:
                current, raw = track_and_clean(tracker, f, vision_feats, vision_pos, feat_sizes,
                                               memory_view(output_dict, f, count) if selection else output_dict, N,
                                               high_res, mux, c.fill_hole_area, best_iou, selection)
                shown, dumped = current["pred_masks"], raw
                if best_iou:
                    index = int(current["mask_index"][0])
                if selection:
                    current["memory_score"] = memory_score(current)

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
                        if record is not None:   # the dropped track's slots
                            record["anchors"] = [a for a in record["anchors"] if a["frame"] < birth]
                        output_dict = new_output_dict()
                        mux, birth, pending = None, -1, []
                        pbar.update(1)
                        continue

                anchor = None
                slot = f % c.recondition_every == 0
                if slot and overlap is not None:
                    anchor = choose_anchor(det_scores, overlap, c)
                if slot and (record is not None or c.anchor_track_score > 0):
                    logit = float(current["object_score_logits"].float().flatten()[0])
                    blocked = anchor is not None and c.anchor_track_score > 0 and logit <= c.anchor_track_score
                    if record is not None:
                        record["anchors"].append(anchor_log(f, anchor, blocked, det_masks, det_scores, overlap,
                                                            current, logit, H, W, c))
                    if blocked:
                        anchor = None
                if anchor is None:
                    if f > quiet_until:
                        encode_frame_memory(tracker, current, raw, vision_feats, feat_sizes, mux, device, c)
                    output_dict["non_cond_frame_outputs"][f] = current
                    keep_memory(output_dict["non_cond_frame_outputs"], f, lookback, count, selection)
                else:
                    # Our anchor only anchors if it is alone: drop the spatial memory of the
                    # frames around it, the ones carrying the drift (clear_on_anchor). Everything
                    # still in the dict is within reach of the lookup, and the frames after it are
                    # held off by quiet_until. The object pointers stay - it is the memory that is
                    # cleared, not the frame.
                    if c.clear_on_anchor:
                        clear_memory(output_dict["non_cond_frame_outputs"])
                    quiet_until = f + c.memory_gap
                    propagated, propagated_raw = current, raw
                    current = condition_on_mask(
                        tracker, clean_logits(det_masks[anchor:anchor + 1], c.fill_hole_area), f, vision_feats,
                        vision_pos, feat_sizes, high_res, output_dict, N, mux, backbone, frame, trunk_out)
                    if c.anchor_mask == PROPAGATED:
                        # the frame's spatial memory from the tracker's own mask, as a conditioning
                        # memory; the object pointer stays the detection's
                        encode_frame_memory(tracker, propagated, propagated_raw, vision_feats, feat_sizes, mux,
                                            device, c, into=current)
                    if c.anchor_output == DETECTION:
                        shown, dumped = current["pred_masks"], None
                    # else the frame keeps showing the mask the tracker propagated onto it
                    how = "anchor"
                    counts["reconditioned"] += 1
                    prune_conditioning(output_dict["cond_frame_outputs"], c.max_conditioning_frames, c.keep_birth_frame)

            mask = to_frame_size(shown, H, W)
            low = low_res_logits(shown if dumped is None else dumped) if record is not None else None
            if f - birth >= c.hotstart_frames:
                for held in pending:
                    put(*held)
                pending = []
                put(f, mask, low, dumped is not None, how, index)
            else:
                pending.append((f, mask, low, dumped is not None, how, index))
            pbar.update(1)

        for held in pending:
            put(*held)
        if birth > 0:
            log.info(f"the track starts on frame {birth}; tracking backwards to fill frames 0-{birth - 1}")
            def emit(f, current, raw):
                put(f, to_frame_size(current["pred_masks"], H, W), low_res_logits(raw) if record is not None else None,
                    True, index=int(current["mask_index"][0]) if best_iou else None)
            propagate_backwards(tracker, backbone, backbone_fn, frames, seed, birth, emit, device, dtype, c)
            counts["tracked backwards"] = birth

    segmented = count_masked_frames(masks)
    if segmented == 0:
        log.warning(f"no frame of this clip scored above {c.birth_threshold} for '{prompt}'; there is no mask")
    else:
        log.info(f"segmented {segmented} of {N} frame(s) from '{prompt}', tracked from frame {birth}"
                 + (f", reconditioned {counts['reconditioned']} time(s)" if counts["reconditioned"] else ""))
    if best_iou:
        log.info(f"obj_ptr_token best_iou: the decoder selected a mask other than token 0 on "
                 f"{sum(i != 0 for i in picked)} of {len(picked)} propagated frame(s)")
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
    masklet_confirmation_enable=True), so they would be dead code. The [prompt] A/B switches
    anchor_matching (A6) and unmatched_counting (A7) are read here as in segment_by_prompt."""
    from comfy.utils import ProgressBar
    (c, N, H, W, device, dtype, frames, detector, tracker, backbone, embedding, text_mask,
     size, backbone_fn, lookback) = _prompt_setup(model, clip, images, prompt, config)

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
                current, raw = track_and_clean(tracker, f, vision_feats, vision_pos, feat_sizes, t["output"], N,
                                               high_res, t["mux"], c.fill_hole_area, c.obj_ptr_token == BEST_IOU)
                t["current"], t["raw"] = current, raw
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
                if c.memory_mask == CLEANED:
                    memory = suppress_shrunk(stacked, c.shrink_keep)
                else:   # the decoder's logits, the suppressed tracks blanked as above
                    raw_stacked = torch.cat([live[k]["raw"][:, 0].float() for k in kept])
                    memory = suppress_shrunk(torch.where(suppressed[:, None, None], -10.0, raw_stacked), c.shrink_keep)
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
                    prune_conditioning(output["cond_frame_outputs"])
                outputs[f][t["id"]] = (t["current"]["pred_masks"].float().cpu(), t["score"])
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
                    outputs[f][track["id"]] = (current["pred_masks"].float().cpu(), object_score(current))
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
        binary = torch.stack([to_frame_size(outputs[f][i][0], H, W) > 0 for i in present])
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
