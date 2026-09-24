"""The long-video chunk loop every long-video sampler node runs (nodes/sampler.py).

Every chunk after the first is seeded with the previous
chunk's last frames (continue_motion, or previous_frames for SCAIL-2) and the driving videos are
read from the returned video_frame_offset, so they stay aligned across the whole run.
The sampling stack (ModelSamplingSD3 -> BasicScheduler -> KSamplerSelect ->
SamplerCustom -> TrimVideoLatent -> VAEDecode) is called node-by-node from
ComfyUI's own registry, so this stays in step with core (the wan_dpmpp sampler is built the
way core's SamplerDPMPP_2M_SDE builds its sampler). What differs per core
conditioning node is its animate adapter (models/common/animate.py), picked from
the registry by the node id: the core call's continuation inputs, its outputs, the videos it
seeks and the input checks. The chunk-length policy is the node's last_chunk widget
(libs/chunking.LAST_CHUNK); its tail_padding widget (libs/video.TAIL_PADDING) says how the
driving videos are extended past their end. With its color_anchor_strength widget above 0 every
chained chunk is colour-matched to the frames it was seeded with (libs/color.py), in the region
the adapter names.
"""

import logging

# torch and comfy.* are imported inside the functions that use them, so this
# module (and the package __init__) imports without a ComfyUI install.
from ..libs.chunking import LAST_CHUNK, format_plan, plan_chunks, produced_frames, snap_down
from ..libs.color import apply_transfer, feather, lab_transfer
from ..libs.log import active_bar, log_beside_bar
from ..libs.sigmas import WAN_BETA, WAN_DPMPP, wan_beta_sigmas
from ..libs.video import TAIL_PADDING
from ..models.common import registry
from ..models.common.core_nodes import call_node, node_class


class _StepLogger:
    """Wraps the sampler (KSamplerSelect's, or wan_dpmpp's) so every denoising step is logged with
    its chunk and the console bar carries the chunk label. SamplerCustom
    builds its own callback (preview + progress bar) and CFGGuider hands it to
    sampler.sample; that is the one point on the core chain where the step is
    visible without re-implementing SamplerCustom. Everything else is
    delegated to the real sampler."""

    def __init__(self, sampler, log_prefix):
        self._sampler = sampler
        self._log_prefix = log_prefix
        self.label = ""

    def __getattr__(self, name):
        return getattr(self._sampler, name)

    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        label = self.label
        prefix = self._log_prefix

        def logged(step, x0, x, total_steps):
            if step == 0:
                bar = active_bar(total_steps)
                if bar is not None:
                    bar.set_description(label, refresh=False)
            log_beside_bar("%s %s step %d/%d", prefix, label, step + 1, total_steps)
            if callback is not None:
                callback(step, x0, x, total_steps)

        return self._sampler.sample(model_wrap, sigmas, extra_args, logged, noise, latent_image, denoise_mask, disable_pbar)


def generate(
    animate_node,
    node_name,
    model,
    positive,
    negative,
    vae,
    reference_image,
    pose_video,
    width,
    height,
    frames_per_chunk,
    total_frames,
    shift,
    sampler_name,
    scheduler,
    steps,
    denoise,
    cfg,
    seed,
    seed_mode,
    last_chunk,
    tail_padding,
    sigmas_override,
    color_anchor_strength,
    animate_inputs,
):
    import torch
    import comfy.model_management
    import comfy.samplers
    import comfy.utils

    if last_chunk not in LAST_CHUNK:
        raise ValueError("last_chunk must be one of {}; found {!r}.".format(", ".join(LAST_CHUNK), last_chunk))
    chunk_length, overshoot = LAST_CHUNK[last_chunk]
    if tail_padding not in TAIL_PADDING:
        raise ValueError("tail_padding must be one of {}; found {!r}.".format(", ".join(TAIL_PADDING), tail_padding))
    pad, padded = TAIL_PADDING[tail_padding]
    adapter = registry.get("animate", animate_node).implementation(node_name, last_chunk)
    log_prefix = "[{}]".format(node_name)
    update_hint = adapter.UPDATE_HINT.format(animate_node)

    animate_cls = node_class(animate_node)
    if len(animate_cls.RETURN_TYPES) < adapter.OUTPUTS:
        raise RuntimeError("{} returns {} outputs, {} expected. {}".format(animate_node, len(animate_cls.RETURN_TYPES), adapter.OUTPUTS, update_hint))
    overlap = adapter.prepare(animate_cls, animate_inputs, reference_image, width, height, frames_per_chunk)

    pose_frames = int(pose_video.shape[0])
    if pose_frames < 1:
        raise ValueError("pose_video has no frames.")
    total = int(total_frames) if total_frames > 0 else pose_frames

    plan = plan_chunks(total, frames_per_chunk, overlap, chunk_length)
    logging.info("%s chunk plan: %s", log_prefix, format_plan(plan, produced_frames(plan, overlap), total, pose_frames, overlap))

    # Every video must reach the last frame the plan samples: past total_frames
    # (longer than the input) and past total itself when the last chunk is snapped
    # up to 4k+1 (or run at full length, as last_chunk says). Core holds
    # only the pose within a chunk; once the offset runs past a video it errors (Animate 2
    # pose) or drops it (Animate: pose, face, background, mask; SCAIL-2: pose, pose mask), and
    # inside the last chunk a short face video loses its motion, a background turns grey and
    # the mask rows turn unknown. Extend the adapter's HELD_VIDEOS up front instead, as
    # tail_padding says (the last frame held, or the video played backwards from its end). The
    # Animate character mask is not extended: past its end the character may be anywhere, so
    # core leaves those rows unknown.
    adapter.check_videos(pose_video, animate_inputs)
    reach = max(total, produced_frames(plan, overlap))
    short = {}
    for name in adapter.HELD_VIDEOS:
        video = pose_video if name == "pose_video" else animate_inputs.get(name)
        if video is None or video.shape[0] >= reach:
            continue
        short[name] = int(video.shape[0])
        if name == "pose_video":
            pose_video = pad(video, reach)
        else:
            animate_inputs[name] = pad(video, reach)
    if short:
        shorter_than_total = [name for name, frames in short.items() if frames < total]
        why = ["total_frames ({}) exceeds {}".format(total, ", ".join(shorter_than_total))] if shorter_than_total else []
        if reach > total:
            why.append("{} and runs {} frames past total_frames".format(overshoot, reach - total))
        (logging.warning if shorter_than_total else logging.info)(
            "%s %s to %d frames: %s (%s).", log_prefix, padded, reach,
            ", ".join("{} +{}".format(name, reach - frames) for name, frames in short.items()), "; ".join(why))

    patched = adapter.patch_model(call_node("ModelSamplingSD3", model=model, shift=shift)[0], animate_inputs)
    if sigmas_override is not None:
        sigmas = sigmas_override
        logging.info("%s sigmas_override connected: scheduler / steps / denoise widgets are ignored.", log_prefix)
    elif scheduler == WAN_BETA:
        sigmas = wan_beta_sigmas(steps, shift, denoise)
    else:
        sigmas = call_node("BasicScheduler", model=patched, scheduler=scheduler, steps=steps, denoise=denoise)[0]
    if sigmas.numel() < 2:
        raise ValueError("The sigma schedule is empty (denoise too low, or an empty sigmas_override).")
    logging.info("%s sigmas (%s): %s", log_prefix, "override" if sigmas_override is not None else scheduler, ", ".join("{:.4f}".format(float(v)) for v in sigmas))
    if sampler_name == WAN_DPMPP:
        # as core's SamplerDPMPP_2M_SDE builds it: eta 0 is the deterministic DPM-Solver++ 2M (libs/sigmas.py)
        inner = comfy.samplers.ksampler("dpmpp_2m_sde", {"eta": 0.0, "s_noise": 1.0, "solver_type": "midpoint"})
    else:
        inner = call_node("KSamplerSelect", sampler_name=sampler_name)[0]
    sampler = _StepLogger(inner, log_prefix)

    progress = comfy.utils.ProgressBar(len(plan))
    chunks = []
    lengths = []
    produced = 0
    anchor = None
    offset = 0
    while produced < total:
        comfy.model_management.throw_exception_if_processing_interrupted()
        index = len(lengths)
        length = chunk_length(produced, total, frames_per_chunk, overlap)
        chunk_seed = seed if seed_mode == "fixed" else (seed + index) % (1 << 64)
        pose_offset = offset
        chunk_inputs = dict(animate_inputs, **adapter.chunk_inputs(index, offset, anchor, pose_video, animate_inputs))

        animate = call_node(
            animate_node,
            positive=positive,
            negative=negative,
            vae=vae,
            width=width,
            height=height,
            length=length,
            batch_size=1,
            reference_image=reference_image,
            pose_video=pose_video,
            **adapter.continuation(anchor, offset),
            **chunk_inputs,
        )
        if len(animate) < adapter.OUTPUTS:
            raise RuntimeError("{} returned {} outputs, {} expected. {}".format(animate_node, len(animate), adapter.OUTPUTS, update_hint))
        chunk_positive, chunk_negative, latent, trim_latent, trim_image, offset = adapter.unpack(animate, anchor)
        adapter.after_animate(chunk_positive, chunk_negative, trim_image, length, pose_offset, animate_inputs)

        # 1-based frame span this chunk adds to the output, as the plan expects it.
        sampler.label = "chunk {}/{} (frames {}-{}/{})".format(index + 1, len(plan), produced + 1, min(total, produced + length - trim_image), total)
        logging.info("%s %s: length %d, pose offset %d, seed %d", log_prefix, sampler.label, length, pose_offset, chunk_seed)

        sampled = call_node(
            "SamplerCustom",
            model=patched,
            add_noise=True,
            noise_seed=chunk_seed,
            cfg=cfg,
            positive=chunk_positive,
            negative=chunk_negative,
            sampler=sampler,
            sigmas=sigmas,
            latent_image=latent,
        )[0]
        if trim_latent > 0:
            sampled = call_node("TrimVideoLatent", samples=sampled, trim_amount=trim_latent)[0]
        images = call_node("VAEDecode", vae=vae, samples=sampled)[0]
        if color_anchor_strength > 0 and anchor is not None and trim_image > 0:
            # one Lab transform from the chunk's regenerated overlap frames onto the frames it was
            # seeded with, applied to the whole chunk before it is trimmed and carried: the next
            # chunk is seeded with corrected frames, so the chain stays anchored to the first
            region = adapter.anchor_region(max(0, pose_offset - trim_image), length, images.shape[1], images.shape[2], animate_inputs)
            weight = None if region is None else feather(region).to(images.device)
            transfer = lab_transfer(images[:trim_image], anchor[-trim_image:].to(images.device), None if weight is None else weight[:trim_image])
            if transfer is None:
                logging.warning("%s %s color anchor skipped: the character region is empty in the %d overlap frames.",
                                log_prefix, sampler.label, trim_image)
            else:
                images = apply_transfer(images, transfer, color_anchor_strength, weight)
                logging.info("%s %s color anchor %.2f on the %s: mean dL %+.2f da %+.2f db %+.2f, std ratio L %.3f a %.3f b %.3f",
                             log_prefix, sampler.label, color_anchor_strength, "whole frame" if region is None else "character region",
                             *(transfer.target_mean - transfer.source_mean).tolist(), *transfer.ratio.tolist())
        if trim_image > 0:
            images = images[trim_image:]
        images = images.cpu()
        del sampled, latent
        adapter.after_chunk(index)

        if images.shape[0] == 0:
            raise RuntimeError("Chunk {} (length {}) contributed no frames after trimming {}; the overlap exceeds the chunk.".format(index, length, trim_image))
        chunks.append(images)
        lengths.append(length)
        produced += int(images.shape[0])
        # a middle chunk adds only chunk - overlap frames; keep enough for a full overlap seed
        anchor = images if anchor is None else torch.cat((anchor, images), dim=0)[-snap_down(frames_per_chunk):]
        if index > 0 and trim_image != overlap:
            logging.warning("%s %s trimmed %d frames, planner assumed %d; using %d from here on.", log_prefix, animate_node, trim_image, overlap, trim_image)
            overlap = trim_image
        logging.info("%s %s done: %d new frames, %d/%d total", log_prefix, sampler.label, int(images.shape[0]), min(produced, total), total)
        progress.update(1)

    images = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
    images = images[:total]
    plan_text = format_plan(lengths, produced, total, pose_frames, overlap)
    if lengths != plan:
        logging.info("%s ran: %s", log_prefix, plan_text)
    return (images, int(images.shape[0]), plan_text)
