"""SAM 3.1 Multiplex on core's primitives, as both prompting modes (pipelines/sam3_1_multiplex/)
drive it: the model's parts (`multiplex_parts`, `propagation_backbone`), the prompt encoding
(`encode_prompt`), a frame's backbone features (`backbone_frame`), a conditioning frame from a
mask (`condition_on_mask`), a propagated frame (`track_frame`) and the same with its mask logits
cleaned (`track_and_clean`), one frame's mask logits from box and point prompts (`decode`), mask
propagation (`propagate`) and the helpers over the tracker's memory (`memory_lookback`,
`new_output_dict`, `new_mux`, `new_memory`, `forget_older`, `store_output`, `encode_memory`,
`encode_memory_into`, `clear_memory`, `object_score`).

Both modes drive core's primitives (`_compute_backbone_frame`, `track_step`,
`_condition_with_masks`, `_deferred_memory_encode`, `_forward_sam_heads`) directly. Core's
`SAM3Model.forward_video` / `track_video_with_detection` are not used: their detection policy
is a much cruder one - it thresholds the raw query score, its NMS measures overlap as
max(IoU, IoM), it has no false-positive guard and no reconditioning.

The model is ComfyUI's own SAM3 implementation, loaded by loader.py.
"""
import torch
import torch.nn.functional as F

from .postprocess import clean_channel_logits, low_res_logits


DEFAULT_SAM3_1_MULTIPLEX = "sam3.1_multiplex_fp16.safetensors"

SAM3_1_MULTIPLEX_SIZE = 1008
# A binary mask handed to the tracker as logits: +/- this value.
MASK_LOGIT_SCALE = 10.0


def multiplex_parts(model):
    """(sam3, detector, tracker, vision backbone) of a loaded multiplex SAM 3.1 checkpoint,
    raising when the checkpoint or this ComfyUI does not have what both modes drive."""
    sam3 = model.model.diffusion_model
    detector, tracker = getattr(sam3, "detector", None), getattr(sam3, "tracker", None)
    backbones = getattr(detector, "backbone", None)
    if not isinstance(backbones, (dict, torch.nn.ModuleDict)) or "vision_backbone" not in backbones:
        raise ValueError(f"expected a SAM3 checkpoint with a detector vision backbone, found {type(sam3).__name__}")
    backbone = backbones["vision_backbone"]
    if not getattr(backbone, "multiplex", False):
        raise ValueError("this node needs a multiplex SAM 3.1 checkpoint; a plain SAM 3 has "
                         f"no detector-driven tracker. Use {DEFAULT_SAM3_1_MULTIPLEX}")
    missing = [n for n in ("_compute_backbone_frame", "track_step", "_condition_with_masks",
                           "_deferred_memory_encode", "_forward_sam_heads") if not hasattr(tracker, n)]
    if missing:
        raise ValueError(f"this ComfyUI's SAM3 tracker has no {', '.join(missing)}; the tracking "
                         "loop is built on those primitives. Update ComfyUI")
    return sam3, detector, tracker, backbone


def propagation_backbone(backbone):
    """The backbone function the tracker's `_compute_backbone_frame` calls: the trunk once,
    then the propagation neck on the cached trunk."""
    def backbone_fn(frame, frame_idx=None):
        trunk_out = backbone.trunk(frame)
        _, _, feats, positions = backbone(frame, tracker_mode="propagation",
                                          cached_trunk=trunk_out, tracker_only=True)
        return feats, positions, trunk_out
    return backbone_fn


def backbone_frame(tracker, backbone_fn, frames, frame_idx, device, dtype, size, signed_input=False):
    """Frame `frame_idx` of `frames` [N, 3, H, W] (values in [0, 1]) as the tracker takes it, and
    its backbone features: (frame, vision_feats, vision_pos, feat_sizes, high_res, trunk_out).

    Core hands the image encoder the resized frame in [0, 1]. `signed_input` maps it to [-1, 1],
    x * 2 - 1: SAM's (x - 0.5) / 0.5, applied after the resize as Meta's SAM 3.1 preprocessing
    does. The returned frame is the one the encoder saw."""
    from comfy.ldm.sam3.tracker import _prep_frame
    frame = _prep_frame(frames, slice(frame_idx, frame_idx + 1), device, dtype, size)
    if signed_input:
        frame = frame * 2 - 1
    vision_feats, vision_pos, feat_sizes, high_res, trunk_out = tracker._compute_backbone_frame(
        backbone_fn, frame, frame_idx=frame_idx)
    return frame, vision_feats, vision_pos, feat_sizes, high_res, trunk_out


def condition_on_mask(tracker, mask, frame_idx, vision_feats, vision_pos, feat_sizes, high_res, output_dict,
                      num_frames, mux, backbone, frame, trunk_out):
    """Make frame `frame_idx` a conditioning frame of `output_dict` from the mask logits `mask`,
    cut at 0 (core's default cut is 0.5), and return the tracker's output for it."""
    return tracker._condition_with_masks(mask, frame_idx, vision_feats, vision_pos, feat_sizes, high_res,
                                         output_dict, num_frames, mux, backbone, frame, trunk_out, threshold=0.0)


def track_frame(tracker, frame_idx, vision_feats, vision_pos, feat_sizes, output_dict, num_frames, high_res, mux,
                best_iou_pointer=False, record_iou=False):
    """The tracker's output for frame `frame_idx`, propagated from the memory in `output_dict`,
    with no memory encoded for it yet (track_step, run_mem_encoder=False).

    Core builds the frame's object pointer from mask token 0 of the propagation decoder. With
    `best_iou_pointer` it is built from the output token of the mask the decoder selects - the
    highest predicted IoU, the mask the frame shows - as Meta's SAM 3.1 does
    (use_multimask_token_for_obj_ptr), and the output carries that mask's index per object,
    "mask_index" ([objects] long). With `record_iou` the output carries the decoder's highest
    predicted IoU per object, "iou_pred" ([objects])."""
    def step():
        return tracker.track_step(
            frame_idx=frame_idx, is_init_cond_frame=False, current_vision_feats=vision_feats,
            current_vision_pos_embeds=vision_pos, feat_sizes=feat_sizes, mask_inputs=None,
            output_dict=output_dict, num_frames=num_frames, propagation_high_res=high_res,
            multiplex_state=mux, run_mem_encoder=False)
    if not (best_iou_pointer or record_iou):
        return step()
    # The decoder's two-way transformer returns every output token, and its IoU head scores the
    # masks; core keeps only mask token 0 of the first. Both are read as the decoder computes them.
    decoder = tracker.sam_mask_decoder
    seen = {"tokens": [], "iou": []}
    hooks = [decoder.transformer.register_forward_hook(lambda module, args, out: seen["tokens"].append(out[0])),
             decoder.iou_prediction_head.register_forward_hook(lambda module, args, out: seen["iou"].append(out))]
    try:
        current = step()
    finally:
        for hook in hooks:
            hook.remove()
    if len(seen["tokens"]) != 1 or len(seen["iou"]) != 1:
        raise RuntimeError(f"expected one propagation decoder call for frame {frame_idx}, saw {len(seen['tokens'])}; "
                           "this ComfyUI's SAM 3.1 tracker propagates differently. Set obj_ptr_token to token_0 "
                           "and memory_selection off")
    decoder = tracker.sam_mask_decoder
    M, T = decoder.num_multiplex, decoder.num_mask_output_per_object
    ious = mux.demux(seen["iou"][0].view(-1, M, T))   # [objects, T], as core's _forward_propagation reads them
    if record_iou:
        current["iou_pred"] = ious.max(dim=-1).values
    if best_iou_pointer:
        _pointer_from_best_mask(tracker, current, seen["tokens"][0], ious, mux)
    return current


def _pointer_from_best_mask(tracker, current, tokens, ious, mux):
    """Replace the object pointer of the propagated output `current` with the one built from the
    output token of its best-IoU mask, and record that mask's index as current["mask_index"].
    `tokens` are the decoder transformer's output tokens [buckets, 2 M + M T, C] (core's order:
    M object-score tokens, M IoU tokens, M x T mask tokens), `ious` the IoU head's predictions
    per object [objects, T]. The selection and the pointer are core's `_forward_propagation`,
    token 0 replaced."""
    decoder = tracker.sam_mask_decoder
    M, T = decoder.num_multiplex, decoder.num_mask_output_per_object
    B = tokens.shape[0]
    mask_tokens = mux.demux(tokens[:, 2 * M:2 * M + M * T].view(B, M, T, -1))   # [objects, T, C]
    best = torch.argmax(ious, dim=-1)                                           # [objects]
    token = mask_tokens[torch.arange(mask_tokens.shape[0], device=best.device), best]
    obj_ptr = tracker.obj_ptr_proj(token)
    is_obj = (current["object_score_logits"] > 0).float()
    obj_ptr = is_obj * obj_ptr + (1 - is_obj) * tracker.no_obj_ptr_linear(obj_ptr)
    current["obj_ptr"] = mux.mux(obj_ptr)
    current["mask_index"] = best


def track_and_clean(tracker, frame_idx, vision_feats, vision_pos, feat_sizes, output_dict, num_frames, high_res,
                    mux, fill_hole_area, best_iou_pointer=False, record_iou=False):
    """The tracker's output for one propagated frame (track_frame), its mask logits cleaned in
    place (clean_channel_logits), and the logits before the cleaning: (current, raw)."""
    current = track_frame(tracker, frame_idx, vision_feats, vision_pos, feat_sizes, output_dict, num_frames,
                          high_res, mux, best_iou_pointer, record_iou)
    raw = current["pred_masks"]
    current["pred_masks"] = clean_channel_logits(raw, fill_hole_area)
    return current, raw


def encode_prompt(clip, detector, prompt, device, dtype):
    """`prompt` as the (embedding, attention mask) pair the detector wants, encoded the way
    CLIPTextEncode encodes it and put through the language backbone's resizer."""
    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
    embedding = conditioning[0][0].to(device=device, dtype=dtype)
    mask = conditioning[0][1].get("attention_mask")
    if mask is None:
        mask = torch.ones(embedding.shape[0], embedding.shape[1], dtype=torch.int64)
    resizer = detector.backbone["language_backbone"]["resizer"]
    return resizer(embedding), mask.to(device).bool()


def decode(sam3, frame, point_inputs, box_inputs, refine):
    """Mask logits for one 1008x1008 frame from box and point prompts, with an optional
    refinement pass that feeds the first mask back to the decoder. This is
    SAM3Model.forward_segment with the image encoder run once: its features do not depend on
    the prompt, so the refinement pass only runs the SAM heads."""
    from comfy.ops import cast_to_input
    _, _, feats, _ = sam3.detector.backbone["vision_backbone"](frame, tracker_mode="interactive")
    high_res, backbone_feat = list(feats[:-1]), feats[-1]
    tracker = sam3.tracker
    no_mem = getattr(tracker, "interactivity_no_mem_embed", None)
    if no_mem is None:
        no_mem = getattr(tracker, "no_mem_embed", None)
    if no_mem is not None:
        B, C, H, W = backbone_feat.shape
        flat = backbone_feat.flatten(2).permute(0, 2, 1)
        backbone_feat = (flat + cast_to_input(no_mem, flat)).view(B, H, W, C).permute(0, 3, 1, 2)
    num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
    _, logits, _, _ = tracker._forward_sam_heads(
        backbone_features=backbone_feat, point_inputs=point_inputs, mask_inputs=None, box_inputs=box_inputs,
        high_res_features=high_res, multimask_output=(0 < num_pts <= 1))
    if refine:
        _, logits, _, _ = tracker._forward_sam_heads(
            backbone_features=backbone_feat, point_inputs=None, mask_inputs=logits, box_inputs=None,
            high_res_features=high_res, multimask_output=False)
    return logits


def memory_lookback(tracker):
    """How many frames back the tracker's memory lookup reaches: its spatial memories or its
    object pointers, whichever reach further."""
    return max(tracker.num_maskmem, tracker.max_obj_ptrs_in_encoder)


def new_output_dict():
    """A fresh, empty output_dict of the tracker's memory: the conditioning frames' outputs and
    the others', each {frame: output}."""
    return {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}


def new_mux(tracker, device, dtype):
    """A fresh multiplex state holding one object (core's MultiplexState)."""
    from comfy.ldm.sam3.tracker import MultiplexState
    return MultiplexState(1, tracker.num_multiplex, device, dtype)


def new_memory(tracker, device, dtype):
    """A fresh memory for tracking one object: an empty output_dict and a multiplex state
    holding one object, as (output_dict, mux)."""
    output_dict = new_output_dict()
    mux = new_mux(tracker, device, dtype)
    return output_dict, mux


def forget_older(stored, frame_idx, lookback):
    """Drop the stored outputs more than `lookback` frames before `frame_idx`: the ones the
    tracker's memory lookup cannot reach any more."""
    for old in list(stored):
        if old < frame_idx - lookback:
            del stored[old]


def store_output(output_dict, frame_idx, current, lookback):
    """Store `current` as frame `frame_idx`'s non-conditioning output, and drop the stored ones
    the memory lookup can no longer reach (forget_older)."""
    output_dict["non_cond_frame_outputs"][frame_idx] = current
    forget_older(output_dict["non_cond_frame_outputs"], frame_idx, lookback)


def encode_memory(tracker, output, vision_feats, feat_sizes, mux, device, conditioning=False):
    """Encode one object's spatial memory from the mask logits of `output` with the tracker's
    deferred memory encoder; it lands in `output`'s maskmem entries. `conditioning` sets the
    memory's conditioning channel to 1.0, "trust it", as a birth or anchor frame's memory has;
    otherwise it is 0.0, a propagated frame's."""
    if conditioning:
        tracker._deferred_memory_encode(output, 1, vision_feats, feat_sizes, mux, device,
                                        cond_obj_mask=torch.ones(1, dtype=torch.bool, device=device))
    else:
        tracker._deferred_memory_encode(output, 1, vision_feats, feat_sizes, mux, device)


def encode_memory_into(tracker, source, current, vision_feats, feat_sizes, mux, device, conditioning=False):
    """Encode the spatial memory of `source` (an output, or {"pred_masks": ...}) with the
    tracker's deferred memory encoder (encode_memory), and give that memory to the output
    `current`."""
    encode_memory(tracker, source, vision_feats, feat_sizes, mux, device, conditioning)
    current["maskmem_features"] = source["maskmem_features"]
    current["maskmem_pos_enc"] = source["maskmem_pos_enc"]


def clear_memory(stored):
    """Drop the spatial memory of every stored output; the outputs, and their object pointers,
    stay."""
    for old in stored.values():
        old["maskmem_features"] = old["maskmem_pos_enc"] = None


def object_score(current):
    """The tracker's object score of one output, as a probability."""
    return float(current["object_score_logits"].float().sigmoid().flatten()[0])


def propagate(sam3, frames_chw, first_mask, device, dtype, H, W, logits_out=None):
    """Masks for frames_chw[1:], propagated by the tracker's memory from `first_mask` on
    frames_chw[0], yielded one [H, W] bool array per frame: a frame is computed only when the
    caller asks for it, so a caller that stops accepting frames stops the tracker there.

    This is exactly what core's `track_video_with_detection` computes when it is given an
    initial mask and no detector: the mask conditions frame 0, every later frame is a plain
    track_step with the pinholes filled, and the output is the tracker's high-res mask cut at 0
    and resized to the frame. Written out here
    so no part of core's detection policy runs. `logits_out`, a list, receives each returned
    frame's low-res logits before the pinholes are filled - the ones the high-res mask was
    upsampled from - before that frame is yielded."""
    from comfy import model_management as mm
    from comfy.ldm.sam3.tracker import fill_holes_in_mask_scores
    tracker, backbone = sam3.tracker, sam3.detector.backbone["vision_backbone"]
    backbone_fn = propagation_backbone(backbone)
    N = frames_chw.shape[0]
    size = tracker.image_size
    idev = mm.intermediate_device()
    initial = (first_mask[None, None].to(device, dtype) * 2 - 1) * MASK_LOGIT_SCALE
    output_dict, mux = new_memory(tracker, device, dtype)
    lookback = memory_lookback(tracker)
    for f in range(N):
        # inference mode per frame, not around the loop: a generator suspended inside it
        # would leave the caller in inference mode
        with torch.inference_mode():
            frame, vision_feats, vision_pos, feat_sizes, high_res, trunk_out = backbone_frame(
                tracker, backbone_fn, frames_chw, f, device, dtype, size)
            if f == 0:
                tracker._condition_with_masks(
                    initial, 0, vision_feats, vision_pos, feat_sizes, high_res, output_dict, N, mux,
                    backbone, frame, trunk_out)
                continue  # frame 0 is the given mask, not an output
            current = track_frame(tracker, f, vision_feats, vision_pos, feat_sizes, output_dict, N, high_res, mux)
            if logits_out is not None:
                logits_out.append(low_res_logits(current["pred_masks"]))
            current["pred_masks"] = fill_holes_in_mask_scores(current["pred_masks"], max_area=16)
            if tracker.num_maskmem > 0:
                encode_memory(tracker, current, vision_feats, feat_sizes, mux, device)
            store_output(output_dict, f, current, lookback)
            # one frame at a time: bilinear resizes every frame of a batch independently
            mask = (current["pred_masks_high_res"][0, 0] > 0).to(idev).float()[None, None]
            mask = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False)[0, 0] > 0.5
        yield mask.cpu().numpy()
