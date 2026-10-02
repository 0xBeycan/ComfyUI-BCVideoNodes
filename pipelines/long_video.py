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
the adapter names. In replacement mode the adapter's PAINTED video is painted black under its mask
a chunk's window at a time (libs/mask.painted_black). A half-precision input (Load Video at
precision fp16) reaches the core node requantized to the float32 values a float32 input holds
(libs/video.requantized): an image whole, a video one chunk's window at a time. The output is then
stored in the pose video's half dtype, while every chained chunk is still seeded with its frames as
decoded.
"""

import gc
import logging

# torch and comfy.* are imported inside the functions that use them, so this
# module (and the package __init__) imports without a ComfyUI install.
from ..libs.chunking import LAST_CHUNK, format_plan, plan_chunks, produced_frames, snap_down, snap_up
from ..libs.color import apply_transfer, feather, lab_transfer
from ..libs.log import active_bar, log_beside_bar
from ..libs.sigmas import WAN_BETA, WAN_DPMPP, wan_beta_sigmas
from ..libs.video import TAIL_PADDING, is_half, requantized
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

    # Core resizes and scales what it gets in its dtype, so a half-precision input is handed over
    # requantized to float32: an image (the reference, SCAIL-2's reference mask, a one-frame
    # character mask) here, whole; a video the core node seeks in the loop below, one chunk's
    # window at a time, never as a whole.
    reference_image = requantized(reference_image)
    for name, value in list(animate_inputs.items()):
        if not isinstance(value, torch.Tensor) or name in adapter.HELD_VIDEOS:
            continue
        if name in adapter.SEEKED_VIDEOS and value.ndim >= 3 and value.shape[0] > 1:
            continue  # a video
        animate_inputs[name] = requantized(value)

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
    # the mask rows turn unknown. The adapter's HELD_VIDEOS are extended up to that frame
    # instead, as tail_padding says (the last frame held, or the video played backwards from its
    # end), but never as a whole: a chunk that reads past the end of one gets the window it reads
    # (in the loop below). The Animate character mask is not extended: past its end the
    # character may be anywhere, so core leaves those rows unknown.
    adapter.check_videos(pose_video, animate_inputs)
    reach = max(total, produced_frames(plan, overlap))
    videos = dict(animate_inputs, pose_video=pose_video)  # the core node's inputs, by their names
    short = {name: int(videos[name].shape[0]) for name in adapter.HELD_VIDEOS
             if videos.get(name) is not None and videos[name].shape[0] < reach}
    if short:
        shorter_than_total = [name for name, frames in short.items() if frames < total]
        why = ["total_frames ({}) exceeds {}".format(total, ", ".join(shorter_than_total))] if shorter_than_total else []
        if reach > total:
            why.append("{} and runs {} frames past total_frames".format(overshoot, reach - total))
        (logging.warning if shorter_than_total else logging.info)(
            "%s %s to %d frames: %s (%s).", log_prefix, padded, reach,
            ", ".join("{} +{}".format(name, reach - frames) for name, frames in short.items()), "; ".join(why))
    # the videos the core node seeks but that are not held, by their frames; a single frame is
    # not seeked (core repeats it over the chunk), so it is not among them
    seeked = {name: int(videos[name].shape[0]) for name in adapter.SEEKED_VIDEOS
              if videos.get(name) is not None and videos[name].ndim >= 3 and videos[name].shape[0] > 1}
    # the videos the core node seeks that are half precision: every chunk gets its window of them
    half = [name for name in (*adapter.HELD_VIDEOS, *seeked) if videos.get(name) is not None and is_half(videos[name])]
    if half:
        logging.info("%s %s half precision: every chunk gets its window requantized to float32.", log_prefix, ", ".join(half))
    # replacement mode (the adapter's PAINTED video and mask both connected): every chunk gets its
    # window, the video's painted black where the mask is above 0, with the mask's window taken as
    # the video's (a single-frame mask repeated over the video, as core repeats it)
    paint = None
    if adapter.PAINTED and all(videos.get(name) is not None for name in adapter.PAINTED):
        from ..libs.mask import painted_black

        video, mask = adapter.PAINTED
        paint = (video, videos[mask] if mask in seeked else videos[mask].expand(int(videos[video].shape[0]), -1, -1))
        logging.info("%s %s is painted black where %s is above 0, a chunk's window at a time.", log_prefix, video, mask)

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
    seed_frames = snap_down(frames_per_chunk)  # the last output frames every chained chunk is seeded with
    output = None  # [total, H, W, C], allocated at the first decode of a run of more than one chunk
    # the output's dtype: a half-precision pose video's (Load Video at precision fp16), else the
    # decoded frames' (core's VAE output)
    stored = pose_video.dtype if is_half(pose_video) else None
    lengths = []
    produced = 0
    anchor = None
    moved = None  # how far the core node moved the offset back by the frames it kept, once seen
    offset = 0
    while produced < total:
        comfy.model_management.throw_exception_if_processing_interrupted()
        index = len(lengths)
        length = chunk_length(produced, total, frames_per_chunk, overlap)
        chunk_seed = seed if seed_mode == "fixed" else (seed + index) % (1 << 64)
        pose_offset = offset
        start, inputs = 0, videos
        if half or paint or any(offset + length > frames for frames in short.values()):
            # The chunk reads past the end of a held video, a video is half precision, or one is
            # painted: every video the core node seeks is cut to the window it reads, a held one
            # extended there as tail_padding says, the painted one painted, a half one requantized,
            # and the core node gets the offset into that window. Before it seeks, the core node
            # moves the offset back by the frames of the anchor it keeps: as far as on the last
            # chained chunk once seen, and until then by at most the whole anchor. Nor is a seeked
            # video cut to one frame, which core would repeat over the chunk.
            if anchor is not None and moved is None:
                start, stop = max(0, offset - int(anchor.shape[0])), offset + length
            else:
                start = max(0, offset - (moved or 0))
                stop = start + length
            while any(frames - start == 1 for frames in seeked.values()):
                start -= 1
            inputs = dict(videos)
            for name in adapter.HELD_VIDEOS:
                if videos.get(name) is not None:
                    reached = min(stop, max(reach, int(videos[name].shape[0])))
                    window = pad(videos[name], start, reached)
                    if paint and name == paint[0]:
                        window = painted_black(window, pad(paint[1], start, reached))
                    inputs[name] = requantized(window)
            for name in seeked:
                inputs[name] = requantized(videos[name][start:stop])
        seek = offset - start
        chunk_inputs = dict(inputs, **adapter.chunk_inputs(index, seek, anchor, inputs["pose_video"], inputs))

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
            **adapter.continuation(anchor, seek),
            **chunk_inputs,
        )
        # core's WanAnimateToVideo leaves the Wan VAE encoder's full-resolution features in a
        # reference cycle (GiBs of VRAM at 720p): collected here, before sampling, not whenever
        # Python's own collector next runs
        gc.collect()
        if len(animate) < adapter.OUTPUTS:
            raise RuntimeError("{} returned {} outputs, {} expected. {}".format(animate_node, len(animate), adapter.OUTPUTS, update_hint))
        chunk_positive, chunk_negative, latent, trim_latent, trim_image, returned = adapter.unpack(animate, anchor)
        if anchor is not None:
            moved = seek - (returned - length)  # core returns the offset it seeked from + length
        offset = start + returned
        adapter.after_animate(chunk_positive, chunk_negative, trim_image, length, seek, inputs)

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
        del animate, chunk_positive, chunk_negative, chunk_inputs, latent  # nothing reads the conditioning after sampling
        if trim_latent > 0:
            sampled = call_node("TrimVideoLatent", samples=sampled, trim_amount=trim_latent)[0]
        decoded = 4 * int(sampled["samples"].shape[2]) - 3  # the frames the chunk decodes to: 4 per latent frame after the first
        needed = snap_up(trim_image + total - produced)
        if needed < decoded:
            # the last chunk runs past total: the Wan VAE decodes causally, so the latent frames
            # that decode only to frames past total are left out instead of decoded and cut
            sampled = dict(sampled, samples=sampled["samples"][:, :, :(needed - 1) // 4 + 1])
        images = call_node("VAEDecode", vae=vae, samples=sampled)[0]
        del sampled
        if color_anchor_strength > 0 and anchor is not None and trim_image > 0:
            # one Lab transform from the chunk's regenerated overlap frames onto the frames it was
            # seeded with, applied to the whole chunk before it is trimmed and carried: the next
            # chunk is seeded with corrected frames, so the chain stays anchored to the first
            region = adapter.anchor_region(max(0, seek - trim_image), images.shape[0], images.shape[1], images.shape[2], inputs)
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
        del inputs
        images = images[trim_image:]
        adapter.after_chunk(index)

        added = decoded - trim_image  # the frames the chunk adds, the ones past total included, as the plan counts them
        if added <= 0:
            raise RuntimeError("Chunk {} (length {}) contributed no frames after trimming {}; the overlap exceeds the chunk.".format(index, length, trim_image))
        kept = min(added, total - produced)
        if output is None and added >= total:
            # the only chunk: its frames are the output, not copied (but to a stored dtype)
            output = images[:total].to(device="cpu", dtype=stored or images.dtype)
        else:
            if output is None:
                output = torch.empty((total, *images.shape[1:]), dtype=stored or images.dtype, device="cpu")
            output[produced:produced + kept] = images[:kept]
        # the output's last frames, for the next chunk's seed: a middle chunk adds only chunk -
        # overlap frames, so this reaches back into the chunks before it. An output stored in
        # another dtype rounds the frames, so the seed is then kept apart, as decoded: the chain
        # is the one the decoded dtype gives
        if output.dtype == images.dtype:
            anchor = output[max(0, produced + added - seed_frames):produced + added]
        else:
            fresh = images[:kept].cpu()
            if anchor is None or kept >= seed_frames:
                anchor = fresh[-seed_frames:]
            else:
                anchor = torch.cat((anchor[kept - seed_frames:], fresh))
        del images
        lengths.append(length)
        produced += added
        if index > 0 and trim_image != overlap:
            logging.warning("%s %s trimmed %d frames, planner assumed %d; using %d from here on.", log_prefix, animate_node, trim_image, overlap, trim_image)
            overlap = trim_image
        logging.info("%s %s done: %d new frames, %d/%d total", log_prefix, sampler.label, added, min(produced, total), total)
        progress.update(1)

    plan_text = format_plan(lengths, produced, total, pose_frames, overlap)
    if lengths != plan:
        logging.info("%s ran: %s", log_prefix, plan_text)
    return (output, int(output.shape[0]), plan_text)
