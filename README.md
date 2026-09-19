# ComfyUI-WanAnimateLongVideoSampler

Two nodes that turn a reference image plus a pose video of any length into a
Wan Animate video of exactly that length, one for each core conditioning
node:

| Node                              | Wraps                | Model             |
|-----------------------------------|----------------------|-------------------|
| **Wan Animate Long Video Sampler** (`WanAnimateLongVideoSampler`)   | `WanAnimateToVideo`  | Wan 2.2 Animate   |
| **Wan Animate 2 Long Video Sampler** (`WanAnimate2LongVideoSampler`) | `WanAnimate2ToVideo` | Wan Animate 2     |

Both live in category `WanAnimate` and depend on ComfyUI core and torch only.

Internally each node samples fixed-size chunks and chains them: every chunk
after the first is seeded with the last frames of the previous one
(`continue_motion`) and reads the driving videos from where the previous
chunk stopped (`video_frame_offset`). This is the "original long generation"
method from the official templates, wrapped so you place one node instead of
copying the template's subgraph once per 5 seconds.

## How it differs from the official templates

The official workflows go long by copying the subgraph per segment:
`WanAnimateToVideo` / `WanAnimate2ToVideo` -> sampler -> `TrimVideoLatent` ->
`VAEDecode`, with the segment's last frames wired into the next copy's
`continue_motion`, its `video_frame_offset` output into the next copy's
input, and `ImageFromBatch` + `ImageBatch` to trim and stitch. Exact, flat
memory, but a handful of nodes per segment and a workflow that is rebuilt when
the length changes. (Wan Animate 2 additionally offers context windows: one
latent for the whole video sampled with a sliding, blended window; memory and
time grow with the whole video.)

These nodes are the per-segment method with the loop inside. Each chunk is a
complete, independent sample of `frames_per_chunk` frames; VRAM is that of one
chunk no matter how long the video is; decoded frames accumulate on CPU. The
seam between chunks is the frames the core node carries over and trims back
off (1 for Animate 2, `continue_motion_max_frames` for Animate), so there is
no cross-window blending and no re-denoising.

What they do not do, on purpose: no colour matching between chunks (it
degraded output on earlier Wan Animate models), no `continue_video` input,
no external `SAMPLER` / `total_frames` links.

## Wiring

Shared by both nodes:

| Input             | Type                | From                                                        |
|-------------------|---------------------|-------------------------------------------------------------|
| `model`           | MODEL               | The Animate model. LoRA and model patches (`WanAnimate2Cache`, context windows) pass through; do **not** add `ModelSamplingSD3`, the node applies `shift` itself. |
| `positive`        | CONDITIONING        | Character prompt (CLIP text encode).                        |
| `negative`        | CONDITIONING        | Negative prompt.                                            |
| `vae`             | VAE                 | Wan 2.1 VAE.                                                |
| `reference_image` | IMAGE               | The character.                                              |
| `pose_video`      | IMAGE               | Driving video, already preprocessed to the pose format the model expects. |
| `sigmas_override` | SIGMAS (optional)   | Replaces the internal schedule. `scheduler`, `steps`, `denoise` are then ignored (one console line says so). `shift` still applies to the model. |

Outputs: `images` (IMAGE, exactly `total_frames` frames), `frame_count`
(INT), `chunk_plan` (STRING, e.g.
`81 + 81 + 81 + 81 + 41 -> 361 produced -> 360 frames (pose 360, overlap 1)`).

Shared widgets:

| Widget                     | Default        | Meaning                                                                 |
|----------------------------|----------------|-------------------------------------------------------------------------|
| `width`, `height`          | 720 x 1280     | Output size. Multiples of 16 are ideal; the VAE crops to a multiple of 8. |
| `frames_per_chunk`         | 81             | Frames sampled per chunk. Rounded down to the 4k+1 grid, minimum 5.    |
| `total_frames`             | 81             | Exact output length. 0 = the pose video's frame count; normally linked. |
| `shift`                    | see per node   | `ModelSamplingSD3` shift, applied before the schedule is built.          |
| `sampler_name`             | euler          | Any sampler ComfyUI has; list comes from `comfy.samplers`.               |
| `scheduler`                | wan_beta       | Any scheduler ComfyUI has, plus `wan_beta` (see below).                  |
| `steps`, `denoise`         | 6, 1.0         | Schedule length (`BasicScheduler`).                                      |
| `cfg`                      | 1.0            |                                                                          |
| `seed`, `seed_mode`        | -, increment   | `increment`: chunk i uses `seed + i`. `fixed`: every chunk uses `seed`.  |

The sampling stack is built once per run in this order:
`ModelSamplingSD3(model, shift)` -> `BasicScheduler(patched, ...)` (or
`wan_beta`, or `sigmas_override`) -> `KSamplerSelect(sampler_name)`;
every chunk samples with the patched model. The sigma list is logged at the
start of every run.

`wan_beta` is the schedule WanVideoWrapper's `euler/beta` samples with:
diffusers `FlowMatchEulerDiscreteScheduler(shift, use_beta_sigmas=True)`,
reproduced exactly. It is not ComfyUI's `beta`. diffusers shifts first and
then spreads Beta(0.6, 0.6) quantiles between the shifted extremes, so the
steps stay evenly spread and `shift` only moves the last sigma; ComfyUI's
`beta` takes the quantiles on the timestep axis and reads them off the
shifted table, which bunches the steps at high noise and leaves one long
final step. At shift 5, 4 steps:

| scheduler        | sigmas                          |
|------------------|---------------------------------|
| `wan_beta`       | 1.000, 0.731, 0.293, 0.024, 0   |
| ComfyUI `beta`   | 1.000, 0.959, 0.834, 0.518, 0   |

### Wan Animate Long Video Sampler (`WanAnimateToVideo`)

Defaults: `frames_per_chunk` 81, `shift` 8, `euler` / `wan_beta`, 6
steps, cfg 1 (with the lightx2v distill LoRA on the model). Shift follows the
official Wan 2.2 Animate template; the template samples 77-frame windows with
`euler` / `simple`.

| Input / widget                | Type                | Notes                                                              |
|-------------------------------|---------------------|--------------------------------------------------------------------|
| `continue_motion_max_frames`  | INT, default 5      | Frames of the previous chunk that seed the next one and are trimmed back off: the overlap. Snapped down to the 4k+1 grid (1, 5, 9, ...); must be smaller than `frames_per_chunk`. |
| `clip_vision_output`          | CLIP_VISION_OUTPUT  | CLIP vision of the reference image.                                |
| `face_video`                  | IMAGE (optional)    | Face crops of the driving video (512 x 512), read from the same offset as the pose video. |
| `background_video`            | IMAGE (optional)    | Background to place the character into (replacement mode), same offset. |
| `character_mask`              | MASK (optional)     | Where the character goes in the background video (replacement mode). One frame is repeated; a video is read from the same offset. |

The optional videos are handed to `WanAnimateToVideo` as they are; the core
node seeks all of them by `video_frame_offset`, so they only need to be
aligned with the pose video at frame 0. A face video shorter than the pose
video is zero-padded by the model for the remaining frames.

With `character_mask` connected the node rebuilds the video part of the
concat mask that `WanAnimateToVideo` returns. The mask has 4 rows per latent
frame and pixel frame `f >= 1` belongs at row `f + 3` (frame 0 fills latent
0), which is how core's own seed-frame rows, its other Wan nodes and the
reference implementation (`get_i2v_mask`) place them; core writes the
character mask at row `f`, three rows early, so it overwrites the last three
seed rows and the seed latent is flagged "character unknown" over real
pixels. The node builds the rows from the pixel mask the way the reference
does (seed frames known, frame 0 repeated, frames past the mask unknown,
core's `nearest-exact` as the filter) and writes them over core's;
`tests/test_node_loop.py` checks the result against the reference
construction. Without a character mask, or for a window the mask does not
reach, core's rows are already right and nothing is touched.

### Wan Animate 2 Long Video Sampler (`WanAnimate2ToVideo`)

Defaults: `frames_per_chunk` 81, `shift` 5, `euler` / `wan_beta`, 10 steps,
cfg 1, `attn_log_scale` -1.3, i.e. the official distilled configuration
(`Wan-Video/Wan-Animate-2`, `infer/wan_animate_2_gradio_distillation.py`)
except for the scheduler: the official sigma list (`linspace(1, 0)` then
shift) is exactly ComfyUI's `simple` at the same shift, `wan_beta` looked
better in testing. Both are one click apart.

| Input / widget              | Type                | Notes                                                                 |
|-----------------------------|---------------------|-----------------------------------------------------------------------|
| `reference_image_strength`  | FLOAT, default 1.0  | Passed to `WanAnimate2ToVideo`.                                       |
| `pose_strength`             | FLOAT, default 1.0  | Passed to `WanAnimate2ToVideo`.                                       |
| `pose_start_percent`, `pose_end_percent` | FLOAT, 0.0 / 1.0 | Sampling window for the pose branch. start > end is an error. |
| `attn_log_scale`            | FLOAT, default -1.3 | The official `log_scale`: a logit bias on every generation self-attention's keys of latent frame 1, the seed frame. -1.3 is the distilled checkpoint's config (`infer/wan_animate_2_distillation.yaml`); set 0.0 for the base checkpoint. Core has no equivalent; 0.0 is core's behaviour. |
| `positive_pose`             | CONDITIONING (optional) | Prompt for the pose branch (motion, not character). Defaults to `positive`. The official pipeline never sends it empty; its default is `人物动作的参考视频`. |
| `clip_vision_output`        | CLIP_VISION_OUTPUT (optional) | CLIP vision of the reference image.                         |
| `clip_vision_output_pose`   | CLIP_VISION_OUTPUT (optional) | CLIP vision of the pose video's first frame, used for every chunk. Defaults to `clip_vision_output`. |
| `clip_vision`               | CLIP_VISION (optional) | When connected, the pose CLIP embedding is re-encoded from the first frame of each chunk's pose window (crop `none`), as the official pipeline does per clip; `clip_vision_output_pose` is then ignored. |

The overlap is the core node's `CONTINUE_MOTION_FRAMES` (1 in current core),
read from the class at run time.

`attn_log_scale` is applied as an attention override
(`transformer_options["optimized_attention_override"]`): generation
self-attention calls are recognised by shape and run through PyTorch SDPA
with an additive mask on the seed frame's keys; cross-attention and the pose
branch pass through untouched. Without it the distilled
model attends to the previous chunk's last frame e^1.3 = 3.7x harder than it
was trained to, which is what made chained chunks drift soft.

The base Animate 2 checkpoint with a Wan 2.1 I2V distill LoRA (lightx2v) is
not a combination the official repository runs; its fast path is the
distilled checkpoint.

## frames_per_chunk by VRAM

Per-chunk VRAM is what one plain core-node -> `SamplerCustom` run of that
many frames needs at your resolution; the chunk count does not add to it.
Starting points, not measurements, for the 14B models in bf16/fp8 (Animate 2
with `WanAnimate2Cache` on CPU):

| VRAM   | ~480 x 832 | ~720 x 1280 |
|--------|------------|-------------|
| 24 GB+ | 81         | 49 - 81     |
| 16 GB  | 49 - 65    | 33          |
| 12 GB  | 33         | 17 - 21     |

Bigger chunks mean fewer seams and better motion continuity, so use the
largest that fits. If a chunk OOMs, drop by 16 (81 -> 65 -> 49 -> 33). Every
value snaps down to the 4k+1 grid. For the Animate node remember that
`continue_motion_max_frames` of every chunk after the first are re-generated,
not new.

## Length math

- Chunk lengths are always 4k+1 and at least 5.
- The first chunk yields its full length; every later chunk yields
  `length - overlap`, where `overlap` is the span the core node trims back off
  after seeding from the previous chunk: `continue_motion_max_frames` for
  `WanAnimateToVideo` (5 by default), `CONTINUE_MOTION_FRAMES` for
  `WanAnimate2ToVideo` (1). If the node's returned `trim_image` ever differs
  from that, the loop adopts the returned value.
- The last chunk shrinks: the remaining need (plus overlap) is rounded up to
  the grid and capped at `frames_per_chunk`, so at most 3 extra frames are
  produced and cropped. Examples for 15 s at 24 fps = 360 frames:
  Animate 2, chunk 81: `81 + 81 + 81 + 81 + 41 -> 361 produced -> 360 frames`;
  Animate, chunk 77, overlap 5: `77 + 77 + 77 + 77 + 73 -> 361 produced -> 360 frames`.
- The loop is driven by the frames actually decoded, not by the plan, so the
  output is exactly `total_frames` long.
- If `total_frames` exceeds the pose video, a warning is printed and the last
  pose frame is held for the remainder.

## Install

Clone into `ComfyUI/custom_nodes/` and restart. No `requirements.txt`: torch
comes with ComfyUI and every node this package calls is core.

## Tests

```
pytest
```

`tests/test_planner.py` covers the length math and `tests/test_package.py`
the node contract, both without torch or ComfyUI. `tests/test_node_loop.py`
runs both nodes' chunk loop against stubbed core nodes with real CPU tensors;
it is skipped when torch is not installed.
