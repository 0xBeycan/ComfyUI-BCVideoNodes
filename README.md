# ComfyUI-WanAnimate2LongVideoSampler

One node that turns a reference image plus a pose video of any length into a
Wan Animate 2 video of exactly that length. Internally it samples fixed-size
chunks and chains them: every chunk after the first is seeded with the last
frame of the previous one (`continue_motion`) and reads the pose video from
where the previous chunk stopped (`video_frame_offset`). This is the
"original long generation" method, wrapped so you place one node instead of
copying the template's subgraph once per 5 seconds.

Node: **Wan Animate 2 Long Video Sampler** (`WanAnimate2LongVideoSampler`),
category `WanAnimate2`. Depends on ComfyUI core and torch only.

## How it differs from the official template

The official Wan Animate 2 workflow has two ways to go long:

1. **Context windows** (`Wan Context Windows` on the model): one latent for
   the whole video, sampled with a sliding window (e.g. 81 frames, 30
   overlap) that is blended every step. Memory and time grow with the whole
   video, and every window is re-denoised against its neighbours; with
   `WanAnimate2Cache` you must use the `static_standard` schedule or nothing
   hits the cache.
2. **Copying the subgraph per segment**: `WanAnimate2ToVideo` ->
   `SamplerCustom` -> `TrimVideoLatent` -> `VAEDecode`, with the segment's
   last frame wired into the next copy's `continue_motion` and its
   `video_frame_offset` output into the next copy's input. Exact, flat memory,
   but a handful of nodes per segment and a workflow that is rebuilt when the
   length changes.

This node is method 2 with the loop inside. Each chunk is a complete,
independent sample of `frames_per_chunk` frames; VRAM is that of one chunk no
matter how long the video is; decoded frames accumulate on CPU. The seam
between chunks is the one frame the core node carries over and trims back
off, so there is no cross-window blending and no re-denoising.

What it does not do, on purpose: no colour matching between chunks (it
degraded output on earlier Wan Animate models), no `continue_video` input,
no external `SAMPLER` / `total_frames` links.

## Wiring

Required links:

| Input             | Type                | From                                                        |
|-------------------|---------------------|-------------------------------------------------------------|
| `model`           | MODEL               | Wan Animate 2 model. LoRA, `WanAnimate2Cache` and context-window patches pass through; do **not** add `ModelSamplingSD3`, the node applies `shift` itself. |
| `positive`        | CONDITIONING        | Character prompt (CLIP text encode).                        |
| `negative`        | CONDITIONING        | Negative prompt.                                            |
| `vae`             | VAE                 | Wan 2.1 VAE.                                                |
| `reference_image` | IMAGE               | The character.                                              |
| `pose_video`      | IMAGE               | Driving video, already preprocessed to the pose format the model expects. |

Optional links:

| Input                     | Type               | Notes                                                                 |
|---------------------------|--------------------|-----------------------------------------------------------------------|
| `positive_pose`           | CONDITIONING       | Prompt for the pose branch (motion, not character). Defaults to `positive`. |
| `clip_vision_output`      | CLIP_VISION_OUTPUT | CLIP vision of the reference image.                                   |
| `clip_vision_output_pose` | CLIP_VISION_OUTPUT | CLIP vision of the pose video's first frame. Defaults to `clip_vision_output`. |
| `sigmas_override`         | SIGMAS             | Replaces the internal schedule. `scheduler`, `steps`, `denoise` are then ignored (one console line says so). `shift` still applies to the model. |

Outputs: `images` (IMAGE, exactly `total_frames` frames), `frame_count`
(INT), `chunk_plan` (STRING, e.g.
`81 + 81 + 81 + 81 + 41 -> 361 produced -> 360 frames (pose 360, overlap 1)`).

Widgets:

| Widget                     | Default     | Meaning                                                                 |
|----------------------------|-------------|-------------------------------------------------------------------------|
| `width`, `height`          | 720 x 1280  | Output size. Multiples of 16 are ideal; the VAE crops to a multiple of 8. |
| `frames_per_chunk`         | 81          | Frames sampled per chunk. Rounded down to the 4k+1 grid, minimum 5.    |
| `total_frames`             | 0           | Exact output length. 0 = the pose video's frame count.                  |
| `shift`                    | 5.0         | `ModelSamplingSD3` shift, applied before the schedule is built.          |
| `sampler_name`             | lcm         | Any sampler ComfyUI has; list comes from `comfy.samplers`.               |
| `scheduler`                | simple      | Any scheduler ComfyUI has.                                               |
| `steps`, `denoise`         | 6, 1.0      | Schedule length (`BasicScheduler`).                                      |
| `cfg`                      | 1.0         |                                                                          |
| `seed`, `seed_mode`        | -, increment| `increment`: chunk i uses `seed + i`. `fixed`: every chunk uses `seed`.  |
| `reference_image_strength` | 1.0         | Passed to `WanAnimate2ToVideo`.                                          |
| `pose_strength`            | 1.0         | Passed to `WanAnimate2ToVideo`.                                          |
| `pose_start_percent`, `pose_end_percent` | 0.0, 1.0 | Sampling window for the pose branch. start > end is an error.  |

The sampling stack is built once per run in this order:
`ModelSamplingSD3(model, shift)` -> `BasicScheduler(patched, ...)` (or
`sigmas_override`) -> `KSamplerSelect(sampler_name)`; every chunk samples with
the patched model.

## frames_per_chunk by VRAM

Per-chunk VRAM is what one plain `WanAnimate2ToVideo` -> `SamplerCustom` run
of that many frames needs at your resolution; the chunk count does not add to
it. Starting points, not measurements, for the 14B model in bf16/fp8 with
`WanAnimate2Cache` on CPU:

| VRAM   | ~480 x 832 | ~720 x 1280 |
|--------|------------|-------------|
| 24 GB+ | 81         | 49 - 81     |
| 16 GB  | 49 - 65    | 33          |
| 12 GB  | 33         | 17 - 21     |

Bigger chunks mean fewer seams and better motion continuity, so use the
largest that fits. If a chunk OOMs, drop by 16 (81 -> 65 -> 49 -> 33). Every
value snaps down to the 4k+1 grid.

## Length math

- Chunk lengths are always 4k+1 and at least 5.
- The first chunk yields its full length; every later chunk yields
  `length - overlap`, where `overlap` is the span the core node trims back off
  after seeding from the previous chunk. With the current core
  (`WanAnimate2ToVideo.CONTINUE_MOTION_FRAMES = 1`) that is 1 frame. The
  constant is read from the class at run time, and if the node's returned
  `trim_image` ever differs from it the loop adopts the returned value.
- The last chunk shrinks: the remaining need (plus overlap) is rounded up to
  the grid and capped at `frames_per_chunk`, so at most 3 extra frames are
  produced and cropped. Example, 15 s at 24 fps = 360 frames with chunk 81:
  `81 + 81 + 81 + 81 + 41 -> 361 produced -> 360 frames`.
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
runs the chunk loop against stubbed core nodes with real CPU tensors; it is
skipped when torch is not installed.
