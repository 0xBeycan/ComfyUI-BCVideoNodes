"""The long-video chunk loop both Wan Animate sampler nodes run (nodes/sampler.py).

Every chunk after the first is seeded with the previous
chunk's last frames (continue_motion) and the driving videos are read from
the returned video_frame_offset, so they stay aligned across the whole run.
The sampling stack (ModelSamplingSD3 -> BasicScheduler -> KSamplerSelect ->
SamplerCustom -> TrimVideoLatent -> VAEDecode) is called node-by-node from
ComfyUI's own registry, so this stays in step with core. What differs per core
conditioning node is its animate adapter (models/common/animate.py), picked from
the registry by the node id.
"""

import logging

# torch and comfy.* are imported inside the functions that use them, so this
# module (and the package __init__) imports without a ComfyUI install.
from ..libs.chunking import format_plan, next_chunk_length, plan_chunks, produced_frames, snap_down
from ..libs.log import active_bar, log_beside_bar
from ..libs.sigmas import WAN_BETA, wan_beta_sigmas
from ..libs.video import hold_last
from ..models.common import registry
from ..models.common.core_nodes import call_node, node_class

ANIMATE_OUTPUTS = 6  # positive, negative, latent, trim_latent, trim_image, video_frame_offset
UPDATE_HINT = "Update ComfyUI: this node needs the {} that returns trim_latent / trim_image / video_frame_offset."


class _StepLogger:
    """Wraps the KSamplerSelect sampler so every denoising step is logged with
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
    sigmas_override,
    animate_inputs,
):
    import torch
    import comfy.model_management
    import comfy.utils

    adapter = registry.get("animate", animate_node).implementation(node_name)
    log_prefix = "[{}]".format(node_name)
    update_hint = UPDATE_HINT.format(animate_node)

    animate_cls = node_class(animate_node)
    if len(animate_cls.RETURN_TYPES) < ANIMATE_OUTPUTS:
        raise RuntimeError("{} returns {} outputs, {} expected. {}".format(animate_node, len(animate_cls.RETURN_TYPES), ANIMATE_OUTPUTS, update_hint))
    overlap = adapter.prepare(animate_cls, animate_inputs)

    pose_frames = int(pose_video.shape[0])
    if pose_frames < 1:
        raise ValueError("pose_video has no frames.")
    total = int(total_frames) if total_frames > 0 else pose_frames

    plan = plan_chunks(total, frames_per_chunk, overlap)
    logging.info("%s chunk plan: %s", log_prefix, format_plan(plan, produced_frames(plan, overlap), total, pose_frames, overlap))

    # Every video must reach the last frame the plan samples: past total_frames
    # (longer than the input) and past total itself when the last chunk is snapped
    # up to 4k+1. Core holds only the pose within a chunk; once the offset runs past
    # a video it errors (Animate 2 pose) or drops it (Animate: pose, face,
    # background, mask), and inside the last chunk a short face video loses its
    # motion, a background turns grey and the mask rows turn unknown. Hold the last
    # frame up front instead. The mask is not held: past its end the character may be
    # anywhere, so core leaves those rows unknown. It says where the character goes in
    # each background frame, so a mask video must be as long as the background.
    character_mask, background = animate_inputs.get("character_mask"), animate_inputs.get("background_video")
    if (character_mask is not None and background is not None and character_mask.ndim >= 3
            and character_mask.shape[0] > 1 and character_mask.shape[0] != background.shape[0]):
        raise ValueError("character_mask has {} frames but background_video has {}: the mask marks where the character "
                         "goes in each background frame, so connect the two from the same video.".format(
                             int(character_mask.shape[0]), int(background.shape[0])))
    reach = max(total, produced_frames(plan, overlap))
    short = {}
    for name in ("pose_video", "face_video", "background_video"):
        video = pose_video if name == "pose_video" else animate_inputs.get(name)
        if video is None or video.shape[0] >= reach:
            continue
        short[name] = int(video.shape[0])
        if name == "pose_video":
            pose_video = hold_last(video, reach)
        else:
            animate_inputs[name] = hold_last(video, reach)
    if short:
        shorter_than_total = [name for name, frames in short.items() if frames < total]
        why = ["total_frames ({}) exceeds {}".format(total, ", ".join(shorter_than_total))] if shorter_than_total else []
        if reach > total:
            why.append("the last chunk is snapped up to 4k+1 and runs {} frames past total_frames".format(reach - total))
        (logging.warning if shorter_than_total else logging.info)(
            "%s last frame held to %d frames: %s (%s).", log_prefix, reach,
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
    sampler = _StepLogger(call_node("KSamplerSelect", sampler_name=sampler_name)[0], log_prefix)

    progress = comfy.utils.ProgressBar(len(plan))
    chunks = []
    lengths = []
    produced = 0
    anchor = None
    offset = 0
    while produced < total:
        comfy.model_management.throw_exception_if_processing_interrupted()
        index = len(lengths)
        length = next_chunk_length(produced, total, frames_per_chunk, overlap)
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
            continue_motion=anchor,
            video_frame_offset=offset,
            **chunk_inputs,
        )
        if len(animate) < ANIMATE_OUTPUTS:
            raise RuntimeError("{} returned {} outputs, {} expected. {}".format(animate_node, len(animate), ANIMATE_OUTPUTS, update_hint))
        chunk_positive, chunk_negative, latent, trim_latent, trim_image, offset = animate[:ANIMATE_OUTPUTS]
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
        if trim_image > 0:
            images = images[trim_image:]
        images = images.cpu()
        del sampled, latent

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
