# ComfyUI-BCVideoNodes

Video nodes for ComfyUI: a Wan Animate preprocess built from small nodes that
are usable in any video pipeline (wholebody pose, SAM 3.1 person tracking,
face crops, pose and mask checks), a SCAIL-2 preprocess, and three samplers
that turn a reference image plus a driving video of any length into a Wan
Animate or SCAIL-2 video of exactly that length.

| Node | Id | Category |
|------|----|----------|
| **Pose Detection** | `BCVPoseDetection` | `BCVideoNodes` |
| **Pose Config** | `BCVPoseConfig` | `BCVideoNodes` |
| **SAM 3.1 Multiplex Video Track** | `BCVSAM3VideoTrack` | `BCVideoNodes` |
| **SAM 3.1 Multiplex Config** | `BCVSAM3Config` | `BCVideoNodes` |
| **Face Crop** | `BCVFaceCrop` | `BCVideoNodes` |
| **Pose Guard** | `BCVPoseGuard` | `BCVideoNodes` |
| **Mask Guard** | `BCVMaskGuard` | `BCVideoNodes` |
| **WanAnimate Preprocess** | `BCVWanAnimatePreprocess` | `BCVideoNodes/Wan/Animate` |
| **WanAnimate Preprocess Guard** | `BCVWanAnimatePreprocessGuard` | `BCVideoNodes/Wan/Animate` |
| **Wan Animate Long Video Sampler** | `BCVWanAnimateLongVideoSampler` | `BCVideoNodes/Wan/Animate` |
| **Wan Animate 2 Long Video Sampler** | `BCVWanAnimate2LongVideoSampler` | `BCVideoNodes/Wan/Animate` |
| **SCAIL-2 Colored Mask** | `BCVSCAIL2ColoredMask` | `BCVideoNodes/SCAIL` |
| **SCAIL-2 Preprocess** | `BCVSCAIL2Preprocess` | `BCVideoNodes/SCAIL` |
| **SCAIL-2 Preprocess Guard** | `BCVSCAIL2PreprocessGuard` | `BCVideoNodes/SCAIL` |
| **SCAIL-2 Long Video Sampler** | `BCVSCAIL2LongVideoSampler` | `BCVideoNodes/SCAIL` |

## Preprocess nodes

Feed them frames already at the generation size; the pose images, masks and
boxes come out at the size of the frames that went in.

### Pose Detection

YOLOv10x finds the person, each frame's box is widened to the boxes of the
frames around it (`box_window`), ViTPose-H gives the 133 COCO-WholeBody
keypoints on every frame, and the pose images are drawn.

- in: `images`; optional `bboxes` (BBOX, one `(x1, y1, x2, y2)` per frame or
  one for all: the detector is then skipped), `pose_config` (POSE_CONFIG)
- widgets: `body_stick_width` -1,
  `hand_stick_width` -1 (0 leaves that part out, -1 sizes it from the frame),
  `draw_head` true, `draw_threshold` 0.5
- out: `pose_images` (IMAGE), `pose_data` (POSEDATA), `bboxes` (BBOX, the
  person box per frame as everything downstream sees it),
  `key_frame_body_points` (STRING: frame 0's confident body keypoints in the
  KJNodes PointsEditor / easy-sam3 `positive_coords` JSON format)

### Pose Config

Optional; without it Pose Detection runs with the measured defaults, which are
the values the node shows. Its widgets are generated from `PoseConfig` in
`pipelines/pose.py`: `min_keypoint_conf`, `detection_threshold`,
`box_window`. `min_keypoint_conf` travels in `pose_data` to SAM 3.1 Multiplex Video Track's
`box_keypoint` mode; the guards count what is drawn, at Pose Detection's
`draw_threshold` (also carried in `pose_data`).

### SAM 3.1 Multiplex Video Track

The person's mask on every frame, from ComfyUI's own SAM 3.1 Multiplex and its tracker
memory. The mask covers every frame, including those before the person was
first found.

- in: `images`; optional `pose_data`, `bboxes`, `positive_coords`,
  `negative_coords` (points JSON), `sam3_config` (SAM3_CONFIG)
- widgets: `mode`, `prompt`, `max_objects` 1, `object_index` -1
- out: `mask` (MASK)

`mode` stays a widget even though it could be inferred from what is
connected, so the behaviour can be switched without rewiring:

- `prompt` (default): SAM finds the person from the text `prompt` alone
  (`main person in the foreground`); nothing else goes in. `max_objects`
  lets more than one track be born; `object_index` -1 is the union of every
  tracked object, `k` is object `k`.
- `box_keypoint`: the person is described by `pose_data`'s box and body
  keypoints (required), with `bboxes` replacing the boxes and the coords
  adding hand-placed points on frame 0. `max_objects` applies to `prompt`
  mode only; `box_keypoint` mode tracks one person, ignores it and logs one
  line.

### SAM 3.1 Multiplex Config

Optional; generated from `SAM3_1MultiplexConfig` in `pipelines/sam3_1_multiplex/config.py`. Each tooltip
starts with the mode it affects.

The `[prompt]` defaults are easy-sam3's set on SAM 3.1, validated on the test
clips against the earlier defaults; every earlier value can be set back here.
What changed, and why:

- `input_range` `-1..1` (was `0..1`): the range SAM 3.1 was trained on. On
  `0..1` black reads as mid-grey and contrast is halved, so dark, blurred limbs
  dropped out.
- `obj_ptr_token` `best_iou` (was `token_0`): a propagated frame's object
  pointer, which the next frames read, is built from the mask the frame
  shows, as Meta's SAM 3.1 and easy-sam3 do.
- `anchor_mask` `propagated` (was `detection`): a re-anchor frame keeps the
  tracker's own mask as its memory; the detection only gives it its object
  pointer. A detector mask missing a limb no longer blacks it out for the
  frames that follow.
- `clear_on_anchor` off (was on) and `memory_gap` `0` (was `7`): a re-anchor no
  longer drops the memory of the frames before it or holds the frames after it
  out of memory, so those frames keep tracking from their own history.
- `max_conditioning_frames` `4` (was `2`) and `keep_birth_frame` off (was on):
  the tracker attends the newest four conditioning frames, as easy-sam3 does,
  instead of the birth frame and the newest anchor.
- `anchor_track_score` `0.8` (was `0`, off): a re-anchor fires only where the
  tracker itself is sure of the person, easy-sam3's gate.
- `memory_selection` on (was off): frames where the tracker lost the person do
  not become memory; the lookup ranks the frames that pass, anchors skipped.
- `detection_threshold` `0.50` (was `0.30`) and `birth_threshold` `0.70` (was
  `0.50`): easy-sam3's values.
- `memory_mask` stays `cleaned` (easy-sam3's side; `raw` is Meta's).

The six re-anchor and memory fields (`clear_on_anchor` to `memory_selection`)
are read at `max_objects` 1 only. With `max_objects` above 1 the shared
defaults apply (input range, pointer token, `memory_gap`, the thresholds) with
that path's own anchor policy, which is not yet measured.

`anchor_matching` and `unmatched_counting` switch one step of the tracking
policy between this pack's (`ours`, the default) and Meta's (`meta`), for the
multi-person A/B. `anchor_output` (read at `max_objects` 1) picks what a
re-anchor frame shows: the mask the tracker propagated onto that frame
(`propagated`, the default) or the detection the track is re-anchored with
(`detection`); the tracking is the same either way. `memory_mask` picks the
logits a propagated frame's memory is encoded from (`cleaned` or `raw`, the
decoder's). Prompt mode cuts every frame's
mask logits at 0; box_keypoint cuts its prompted frames at `mask_threshold`
and its propagated ones at 0.

### Input precedence

- A connected input beats the config node, and the config node beats the
  defaults: connected `bboxes` skip the detector, so
  `pose_config.detection_threshold` is not read.
- Anything the current mode or setting does not read (a widget off its
  default, a connected input, a changed config field) is ignored with one
  console line naming it; an unused setting never raises, so switching needs
  no rewiring.
- Hand-placed `positive_coords` / `negative_coords` beat automatic points: an
  automatic point of the other label within 4% of the box diagonal of a
  hand-placed one is dropped.
- In `box_keypoint` mode, connected `bboxes` replace `pose_data`'s person
  boxes and count as detections on every frame.
- `face_bboxes` on Face Crop are cut as given; `pose_data`'s face keypoints,
  `face_padding` and `face_box_smoothing` are then not used.
- A keypoint at exactly `min_keypoint_conf` counts as found.

### Face Crop

- in: `images`, `pose_data`; optional `face_bboxes` (BBOX, cut as they are)
- widget: `face_padding` 0 (pixels added around the keypoint face box)
- optional widget: `face_box_smoothing` `size` (default: the box centre
  kept, its width and height averaged over a centred Gaussian of sigma 2
  frames: stops the crop's size pulsing without letting a fast face leave
  its box), `off` (each frame's own box) or `median` (per-coordinate median
  over 5 frames)
- out: `face_images` (512 x 512, Wan Animate's `face_video`), `face_bboxes`

### Pose Guard and Mask Guard

Frame-by-frame checks of the drawn pose (incomplete skeleton, torso jump,
limb spike, subject switch) and of the mask against that pose (empty, leaking
outside the box, fragmented, keypoints outside the mask, body the pose does
not draw, unstable). The guards count what the pose images draw, at
`draw_threshold`. The detector is not judged: its box count and missed frames
are in the metrics as data. Every measurement is always reported and plotted;
with the switch (`pose_guard` / `mask_guard`) on, a failed check stops the
workflow with the report, since sampling on a wrong pose or mask is wasted.
Checks that cannot tell a defect from what the scene really does are warnings
and never stop: a limb missing for a stretch (`pose_limb_gap`), small detached
specks (`mask_specks`), background attached to the body
(`mask_attached_leak`), one limb end outside the mask (`mask_missed_limb`). The thresholds are
widgets generated from `PoseGuardConfig` / `MaskGuardConfig` in
`pipelines/guard/config.py`.

- Pose Guard: in `pose_data`; out `pose_data` (unchanged), `report`,
  `metrics` (JSON, every measurement per frame), `timeline` (IMAGE)
- Mask Guard: in `mask`, `pose_data`; out `mask` (unchanged), `report`,
  `metrics`, `timeline`

### WanAnimate Preprocess and WanAnimate Preprocess Guard

The wrappers call the individual nodes, so a wrapper produces exactly what
the chained nodes produce with the same settings.

- **WanAnimate Preprocess** = Pose Detection -> SAM 3.1 Multiplex Video Track -> Face
  Crop. Widgets: the drawing widgets, `face_padding`, `mode`,
  `prompt`; optional `pose_config`, `sam3_config`. In `box_keypoint` mode the
  mask is prompted from the pose. Outputs: `pose_images`, `face_images`,
  `mask`, `pose_data`, `bboxes`, `key_frame_body_points`, `face_bboxes`.
- **WanAnimate Preprocess Guard** = Pose Guard + Mask Guard with one combined
  report. Inputs `mask`, `pose_data`, both switches and all thresholds;
  outputs `mask`, `pose_data`, `report`, `metrics`, `timeline`.

### SCAIL-2 Colored Mask and SCAIL-2 Preprocess

The inputs of the SCAIL-2 Long Video Sampler, one person (multi-person
is a later phase).

- **SCAIL-2 Colored Mask**: `driving_mask` (MASK), `replacement_mode`,
  optional `reference_mask` (MASK) -> `pose_video_mask`,
  `reference_image_mask` (IMAGE). The person is rendered in blue, the first
  colour of the palette SCAIL-2 was trained on, on the background of the mode:
  animation mode = driving mask on black, reference mask on white;
  replacement mode = driving mask on white, reference mask on black. Masks
  are cut at 0.5; the colours are pure, as core's `SCAIL2ColoredMask` renders
  them, so `WanSCAILToVideo`'s 28-channel extraction reads them exactly.
  Without a reference mask (or with an empty one) the reference mask is the
  background alone, as in core: in animation mode that is logged, since the
  mode can then collapse into replacement behaviour (SCAIL-2 README); in
  replacement mode it is an error, since the reference would be cut out to
  black.
- **SCAIL-2 Preprocess** = SAM 3.1 Multiplex Video Track (prompt mode, one
  object) on the whole driving video once, and on the reference image unless
  `reference_mask` is connected, then SCAIL-2 Colored Mask. Tracking the
  whole clip once keeps the mask's shape and colour the same across the
  sampler's chunks (the official template re-tracks every segment). Widgets:
  `replacement_mode`, `prompt`, `black_background` (default off); optional
  `reference_mask`, `sam3_config`. Outputs: `pose_video` (the driving video,
  which SCAIL-2's end-to-end mode reads as its pose input in both modes),
  `pose_video_mask`, `reference_image_mask`, `mask`, `reference_mask`.
- `black_background` (animation mode only): `pose_video` becomes the driving
  video with every pixel outside the person's mask black, as SCAIL-2's
  training pose videos were (zai-org/SCAIL-2 issue #17; SCAIL-Pose's
  `--crop_e2e_mask`), so the driving video's background and camera do not
  reach the result. Off, `pose_video` is the driving video unchanged. In
  replacement mode the result keeps the driving video's background, so
  `black_background` on with `replacement_mode` on is an error, raised
  before any tracking.

### SCAIL-2 Preprocess Guard

Checks the two colored masks before the SCAIL-2 sampler, without a pose
(end-to-end SCAIL-2 draws none), on the person as the sampler reads it (blue
above 225/255). The mode is read from the reference mask's border, as the
sampler reads it; the driving mask is taken to be at the generation size. On
SCAIL-2's own examples blank frames and a split-up mask are normal (the person
leaves the shot, a passer-by occludes her), so only two checks stop the
workflow (with `scail2_guard` on): no driving frame has the person
(`no_driving_person`), the reference mask has no character
(`reference_empty`). Warnings, which never stop: a driving frame without the
person (`driving_empty`), a detached region at least 5% of the largest one
on a driving frame or on the reference (`driving_fragmented`,
`reference_fragmented`), more than `max_reference_cropped` (0.02) of the
reference character outside the center crop the core node cuts the
reference to (`reference_cropped`: a portrait reference in a landscape
generation loses the head or the feet), and, in replacement mode only, a
reference whose character overlaps the first driving frame's person by an
IoU below `min_reference_iou` (0.4) after that crop (`reference_misaligned`).
Both thresholds are first values, not calibrated yet. Measured as data: the
mask area, the share of the mask the sampler's half-size latent cut keeps
(`latent_kept`: a thin limb can vanish there), the mask IoU with the
previous frame, and the reference's IoU and scale against the first driving
frame.

- in: `pose_video_mask`, `reference_image_mask` (IMAGE); widgets
  `scail2_guard` (on) and the two thresholds (`SCAIL2GuardConfig` in
  `pipelines/guard/config.py`)
- out: `pose_video_mask`, `reference_image_mask` (unchanged), `report`,
  `metrics` (JSON: `"guard": "scail2"`, the driving frames, the reference
  record), `timeline` (IMAGE)

### Models

Everything is downloaded on first use; nothing has to be fetched by hand.

- Detection and pose models: from
  [huggingface.co/beycanai/BCVideoNodes-models](https://huggingface.co/beycanai/BCVideoNodes-models)
  into `ComfyUI/models/detection/` (`yolov10x_fp32.safetensors`,
  `vitpose_h_wholebody_fp16.safetensors`; a model is fetched when a node
  first needs it). They are native torch modules stored as safetensors and
  loaded and offloaded by ComfyUI's model management; `onnx` is not needed.
  `scripts/convert_models.py` rebuilds them from the upstream ONNX exports.
- SAM 3.1: ComfyUI's own `sam3.1_multiplex_fp16.safetensors`, from
  `Comfy-Org/sam3.1` into `ComfyUI/models/checkpoints/` when it is missing.

## Long video samplers

The three samplers, one for each core conditioning node:

| Node                              | Wraps                | Model             |
|-----------------------------------|----------------------|-------------------|
| **Wan Animate Long Video Sampler** (`BCVWanAnimateLongVideoSampler`)   | `WanAnimateToVideo`  | Wan 2.2 Animate   |
| **Wan Animate 2 Long Video Sampler** (`BCVWanAnimate2LongVideoSampler`) | `WanAnimate2ToVideo` | Wan Animate 2     |
| **SCAIL-2 Long Video Sampler** (`BCVSCAIL2LongVideoSampler`) | `WanSCAILToVideo` | SCAIL-2 |

All three depend on ComfyUI core and torch only.

Internally each node samples fixed-size chunks and chains them: every chunk
after the first is seeded with the last frames of the previous one
(`continue_motion`, `previous_frames` for SCAIL-2) and reads the driving
videos from where the previous chunk stopped (`video_frame_offset`). This is the "original long generation"
method from the official templates, wrapped so you place one node instead of
copying the template's subgraph once per 5 seconds.

### How it differs from the official templates

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
off (1 for Animate 2, `continue_motion_max_frames` for Animate,
`previous_frame_count` for SCAIL-2), so there is no cross-window blending and
no re-denoising.

What they do not do, on purpose: no colour matching between chunks by
default (it degraded output on earlier Wan Animate models; the SCAIL-2
reference code has none either; `color_anchor_strength` turns on an optional
anchor, see Color anchor), no `continue_video` input, no external `SAMPLER` /
`total_frames` links.

### Wiring

Shared by all three nodes:

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
| `width`, `height`          | 720 x 1280     | Output size. Multiples of 16 are ideal; the VAE crops to a multiple of 8. (SCAIL-2: 704 x 1280, multiples of 32.) |
| `frames_per_chunk`         | 81             | Frames sampled per chunk. Rounded down to the 4k+1 grid, minimum 5.    |
| `total_frames`             | 81             | Exact output length. 0 = the pose video's frame count; normally linked. |
| `shift`                    | see per node   | `ModelSamplingSD3` shift, applied before the schedule is built.          |
| `sampler_name`             | euler          | Any sampler ComfyUI has (list from `comfy.samplers`), plus `wan_dpmpp` (see below). |
| `scheduler`                | wan_beta       | Any scheduler ComfyUI has, plus `wan_beta` (see below). (SCAIL-2: simple.) |
| `steps`, `denoise`         | 6, 1.0         | Schedule length (`BasicScheduler`).                                      |
| `cfg`                      | 1.0            | Classifier-free guidance. At 1.0 the negative prompt is not evaluated (one model pass per step, the distilled setting). |
| `seed`, `seed_mode`        | -, increment   | `increment`: chunk i uses `seed + i`. `fixed`: every chunk uses `seed`.  |
| `last_chunk`               | fit (SCAIL-2: full) | How long the last chunk runs. `fit`: it shrinks to the frames still needed, snapped up to 4k+1. `full`: it runs the full `frames_per_chunk`, with the driving inputs extended past their end (`tail_padding`). `min29`: like `fit`, but never fewer than 29 frames (capped at `frames_per_chunk`), as the official Wan-Animate-2 pads its last clip. Either way the output is exactly `total_frames`; the extra frames are cut. See Length math. |
| `tail_padding`             | last_frame     | How the driving inputs are extended past their last frame when a chunk needs frames beyond them. `last_frame`: the last frame is repeated (the motion stops). `ping_pong`: the input plays backwards from its end, as the official Wan Animate code pads (the motion continues along the same path in reverse). See Tail padding. The last required widget of every sampler. |
| `color_anchor_strength`    | 0.0 (off)      | Optional. Above 0 every chunk after the first is colour-matched to the frames it was seeded with, blended in by this value (1 = fully). In replacement mode only the character. See Color anchor. The last widget of every sampler. |

The sampling stack is built once per run in this order:
`ModelSamplingSD3(model, shift)` -> `BasicScheduler(patched, ...)` (or
`wan_beta`, or `sigmas_override`) -> `KSamplerSelect(sampler_name)` (or
`wan_dpmpp`); every chunk samples with the patched model. The sigma list is
logged at the start of every run.

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

`wan_dpmpp` is the sampler the official Wan pipelines sample with:
DPM-Solver++ (2M) in its flow-matching form (Wan `fm_solvers.py`
`FlowDPMSolverMultistepScheduler`, arXiv 2211.01095). ComfyUI reproduces it as
`dpmpp_2m_sde` with eta 0 (no noise) and the midpoint solver, built the way
core's `SamplerDPMPP_2M_SDE` node builds it (a measured relative error of
4.3e-5 against the native Wan-Animate-2 solver). The listed `dpmpp_2m` is not
it (it steps in -log sigma, not the flow model's half-log-SNR: 1.3e-2 off), nor
is the listed `dpmpp_2m_sde` (eta 1, stochastic). Its official pairing is
scheduler `simple` at the same shift: the native Wan pipelines' sigmas (evenly
spaced, then shifted).

#### Wan Animate Long Video Sampler (`WanAnimateToVideo`)

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
`tests/pipelines/test_long_video.py` checks the result against the reference
construction. Without a character mask, or for a window the mask does not
reach, core's rows are already right and nothing is touched.

#### Wan Animate 2 Long Video Sampler (`WanAnimate2ToVideo`)

Defaults: `frames_per_chunk` 81, `shift` 5, `euler` / `wan_beta`, 10 steps,
cfg 1, `attn_log_scale` -1.3. Shift, steps, cfg and `attn_log_scale` follow
the official distilled configuration (`Wan-Video/Wan-Animate-2`,
`infer/wan_animate_2_gradio_distillation.py`); the sampler and scheduler do
not. The native Wan-Animate-2 pipeline samples DPM-Solver++ 2M on the
`linspace(1, 0)`-then-shift sigmas at 10 steps, here `wan_dpmpp` / `simple`;
Diffusers' distilled recipe is Euler with `FlowMatchEulerDiscreteScheduler`
shift 5, here `euler` / `normal`. The node default `euler` / `wan_beta` is
neither: it looked better in testing. All are a click apart.

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
with an additive mask on the seed frame's keys; every other call
(cross-attention, the pose branch) goes to the attention override installed
before this node, so a sage / flash / NABLA attention patch keeps working
there, or to core's attention when there is none. The biased calls always run
on PyTorch SDPA, even with a sage or flash patch connected, because those
kernels take no additive mask; set `attn_log_scale` to 0.0 to keep the patched
backend for every call. An attention patch that installs itself on top during
sampling (core's block-sparse attention does) takes the calls it handles before
this override sees them, and those get no bias. Each chunk logs how many
attention calls got the bias, and warns when a nonzero `attn_log_scale`
matched none. Without the bias the distilled
model attends to the previous chunk's last frame e^1.3 = 3.7x harder than it
was trained to, which is what made chained chunks drift soft.

The base Animate 2 checkpoint with a Wan 2.1 I2V distill LoRA (lightx2v) is
not a combination the official repository runs; its fast path is the
distilled checkpoint.

#### SCAIL-2 Long Video Sampler (`WanSCAILToVideo`)

SCAIL-2 in either of its two modes, from the same node:

- **Animation mode** (`replacement_mode` off): the reference character is
  animated by the driving video. Needs the reference image with its own
  background, the driving video as `pose_video`, and the colored masks
  rendered for animation mode (driving mask on black, reference mask on
  white). For a stable background, black out the driving video's background
  (SCAIL-2 Preprocess's `black_background`): SCAIL-2's training pose videos
  had black backgrounds (zai-org/SCAIL-2 issue #17).
- **Replacement mode** (`replacement_mode` on): the character replaces the
  person in the driving video, which keeps its background. Needs the colored
  masks rendered for replacement mode (driving mask on white, reference mask
  on black). The authors expect the reference posed like the first driving
  frame (issue #25).

SCAIL-2 Preprocess makes all three inputs. A reference or driving mask
rendered for the other mode is an error (the mode is read from each mask's
border: the reference mask is white in animation mode and black in
replacement mode, the driving mask the opposite).

Defaults: `frames_per_chunk` 81, `width` x `height` 704 x 1280, `shift` 8,
`euler` / `simple`, 6 steps, cfg 1 (for a distill LoRA), `previous_frame_count`
5. They are a starting point, not the SCAIL-2 authors' recipe. With the
distill LoRA they reproduce the sigmas the official ComfyUI SCAIL-2 templates
sample with: the templates set `ModelSamplingSD3` to 5, but their
`BasicScheduler` takes the model before that node, so their schedule runs at
the model's default shift 8 (1, 0.9757, 0.9413, 0.8889, 0.8005, 0.616, 0).
`tests/nodes/test_nodes_scail2.py` checks that list with ComfyUI's own
scheduler. The other references are in the table below.

| Input / widget              | Type                | Notes                                                                 |
|-----------------------------|---------------------|-----------------------------------------------------------------------|
| `clip_vision`               | CLIP_VISION         | `clip_vision_h`. The reference is encoded once per run, stretched (crop `none`) as SCAIL-2 was trained; in replacement mode with the character on black (pixels whose reference mask has no channel above 0.1, core's rule for the VAE reference), as the authors require (issue #30). |
| `pose_video_mask`           | IMAGE               | Colored driving mask, as long as `pose_video` (a mismatch is an error). Extended past its end like the pose (`tail_padding`). |
| `reference_image_mask`      | IMAGE               | Colored reference mask.                                               |
| `replacement_mode`          | BOOLEAN, default off | Must match the mode the masks were rendered for.                     |
| `pose_strength`             | FLOAT, default 1.0  | Passed to `WanSCAILToVideo`.                                          |
| `pose_start_percent`, `pose_end_percent` | FLOAT, 0.0 / 1.0 | Passed as `pose_start` / `pose_end`. start > end is an error. |
| `previous_frame_count`      | INT, default 5      | Frames of the previous chunk that seed the next one and are trimmed back off. SCAIL-2 was trained with 5. Snapped down to the 4k+1 grid. |

`width` and `height` must be divisible by 32 (the pose runs at half
resolution through the /16 patch grid); anything else is an error. 512 x 896
and 704 x 1280 are the sizes the authors use.

`last_chunk` defaults to `full` here: every chunk runs the full
`frames_per_chunk`, the last one too, because SCAIL-2 was trained on 65-81
frame segments (issue #16) and a short last chunk is off its training
distribution. The pose and its mask are extended past their end (`tail_padding`) up to the
end of the last chunk, and the output is cut to `total_frames`. `fit` gives
the Animate behaviour, a last chunk shortened to what is left. A
`frames_per_chunk` outside 65-81 is logged. With `color_anchor_strength` 0
(the default) the anchor is the raw decoded frames, with no colour
correction; `seed_mode` increment gives every chunk a new seed, the authors'
fix for brightness drift in loops (issue #11).

LoRAs and settings, with their sources:

| Setup | LoRAs | Sampling | Source |
|-------|-------|----------|--------|
| ComfyUI template, distill on (animation or replacement) | `lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16` @ 0.8 + `wan2.1_SCAIL_2_DPO_lora_bf16` @ 1.0 | `ModelSamplingSD3` 5, effective schedule shift 8 (the node's defaults), euler / simple, 6 steps, cfg 1 | ComfyUI SCAIL-2 templates (`video_wan21_scail2_character_replacement`, fp16 and int8) |
| ComfyUI template, distill off | `wan2.1_SCAIL_2_DPO_lora_bf16` @ 1.0 | `ModelSamplingSD3` 5, effective schedule shift 8, euler / simple, 40 steps, cfg 5 | the same templates |
| Replacement, optionally with relight | `lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16` (+ `wan2.1_SCAIL_2_relight_lora_bf16`) | not published: strengths and steps are not given; the defaults above are a starting point | SCAIL-2 authors (issues #21, #25, #30) |
| The authors' LoRA example | lightx2v I2V rank128 @ 1.0 | shift 1, 8 steps, cfg 1, `uni_pc` | SCAIL-2 README (`zai-org/SCAIL-2`, `wan-scail2` branch) |
| The authors' base setting | - | shift 3, 40 steps, cfg 5, `uni_pc` (DPM++ as an option) | SCAIL-2 README |
| The SAT config | - | shift 5, 50 steps, cfg 4 | `sat-scail2` branch, `configs/video_model/Wan2.1-i2v-14Bsc-pose-xc-latent.yaml` |

Prompt: SCAIL-2 wants a long, descriptive prompt that describes the resulting
video; in replacement mode, the video after the replacement.

### frames_per_chunk by VRAM

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

### Length math

- Chunk lengths are always 4k+1 and at least 5.
- The first chunk yields its full length; every later chunk yields
  `length - overlap`, where `overlap` is the span the core node trims back off
  after seeding from the previous chunk: `continue_motion_max_frames` for
  `WanAnimateToVideo` (5 by default), `CONTINUE_MOTION_FRAMES` for
  `WanAnimate2ToVideo` (1). If the node's returned `trim_image` ever differs
  from that, the loop adopts the returned value.
- The `last_chunk` widget sets the last chunk's length:
  - `fit` (default of both Animate nodes): the last chunk shrinks, the
    remaining need (plus overlap) rounded up to the grid and capped at
    `frames_per_chunk`, so at most 3 extra frames are produced and cut.
  - `full` (default of SCAIL-2): the last chunk runs the full
    `frames_per_chunk` like every other; the driving inputs (pose, and face
    and background for Animate, the pose mask for SCAIL-2) are extended past
    their end up to its end (see Tail padding), and everything past
    `total_frames` is cut.
  - `min29`: `fit`, but the last chunk never runs fewer than 29 frames
    (capped at `frames_per_chunk`), the official Wan-Animate-2 tail rule
    (`pipelines/utils/multiclip_utils.py` `get_padding_len` pads the last clip
    to at least 29 frames); a video shorter than 29 frames runs one 29-frame
    chunk. The extra frames are padded as `tail_padding` says and cut. 888
    frames at chunk 81, overlap 1: `fit` ends in a 9-frame chunk
    (`81 x 11 + 9 -> 889 produced`), `min29` in a 29-frame one
    (`81 x 11 + 29 -> 909 produced`); at 1110 frames the last chunk is 73
    frames and the two are the same.

  Example, 240 frames, chunk 81, overlap 5:
  `fit`: `81 + 81 + 81 + 13 -> 241 produced -> 240 frames`;
  `full`: `81 + 81 + 81 + 81 -> 309 produced -> 240 frames`.
  `fit` samples less; `full` keeps every chunk at the length the model was
  trained on. More `fit` examples for 15 s at 24 fps = 360 frames:
  Animate 2, chunk 81: `81 + 81 + 81 + 81 + 41 -> 361 produced -> 360 frames`;
  Animate, chunk 77, overlap 5: `77 + 77 + 77 + 77 + 73 -> 361 produced -> 360 frames`.
- The loop is driven by the frames actually decoded, not by the plan, so the
  output is exactly `total_frames` long.
- If `total_frames` exceeds the pose video, a warning is printed and the
  driving inputs are extended for the remainder (see Tail padding); those
  frames are part of the output.

### Tail padding

Whenever a chunk needs driving frames past the end of an input (the `fit`
snap-up of at most 3 frames, `last_chunk` `full` or `min29`, or
`total_frames` longer than the input) the loop extends the driving inputs up
front: the pose, and
face and background for Animate, the pose mask for SCAIL-2. The Animate
`character_mask` is never extended: past its end the character may be
anywhere, so core leaves those mask rows unknown. The `tail_padding` widget
picks how:

- `last_frame` (default of all three samplers): the last frame is repeated;
  the motion stops.
- `ping_pong`: the input plays backwards from its end and turns forward again
  at its first frame, without repeating the frame it turns on; the padding of
  the official Wan 2.2 Animate (`inputs_padding`) and Wan-Animate-2
  (`zigzag_padding`) code. A 10-frame input (frames 0-9) padded by 6 gets
  `8 7 6 5 4 3`; padded further it goes `... 1 0 1 2 ...`. A 1-frame input
  repeats its frame.

Padded frames past `total_frames` are cut from the output; they only affect
the real frames of the same chunk, which attend to them. When `total_frames`
is longer than the input, the output past the input's end is the padding.
The log line names the padding used (`last frame held to ...` or
`ping_pong padded to ...`).

### Color anchor

`color_anchor_strength` (optional, the last widget; default 0 = off) keeps
the colours of a long video on those of its first chunk. Every chunk after
the first regenerates the frames it was seeded with (the overlap) before its
own; the loop estimates one per-channel transform in CIE Lab (mean and std,
the std ratio clamped to 0.5-2 so a flat frame cannot blow it up) that maps
those regenerated frames onto the frames that were carried in, and applies it
to the whole decoded chunk before it is trimmed and carried:
`out = x + strength * (T(x) - x)`, clamped to 0..1. The next chunk is seeded
with the corrected frames, so the chain stays anchored to chunk 1. One line
per chunk logs the mean dL / da / db, the std ratios and the strength.

In replacement mode the background is taken from the source again every
chunk, so only the character is measured and corrected, with a soft edge:
Wan Animate with `background_video` and `character_mask` connected (the
character mask's frames of the chunk), SCAIL-2 with `replacement_mode` on
(every pixel of the driving mask that is not its white background). Wan
Animate 2 has no replacement mode. A chunk whose overlap frames have no
character is left uncorrected, with a warning. At 0 the loop is exactly the
loop without the widget.

## Roadmap

- **SCAIL-2 pose-driven mode.** Today the pack runs SCAIL-2's end-to-end mode:
  the raw driving video is the pose input. The pose-driven mode drives the
  model with a rendered 3D skeleton instead, the pipeline of
  [zai-org/SCAIL-Pose](https://github.com/zai-org/SCAIL-Pose/tree/519c7f54cb972e7f92684213b7ef6c3e05a8f3b2):
  SAM3 isolates each person, NLF (`nlf_l_multi_0.3.2`) estimates the 3D pose,
  the limbs are rendered as 3D cylinders and DWPose draws the hands and face
  in 2D on top, and the driving mask is the skeleton itself in the person's
  colour on black. It keeps the driving person's body shape and clothing
  out of the result, and the SCAIL-2 authors report it works better at 704p.
  Core's `SAM3DBody_Render` has a "scail" style as a lighter alternative;
  its parity with the NLF render is unverified.
- **Multi-person** (all nodes): multi-person pose and mask are on the roadmap, to be
  implemented and tested later. Today everything is optimised for one person; future
  multi-person changes will take effect in multi-person mode only.

## Install

Clone into `ComfyUI/custom_nodes/`, install the requirements into ComfyUI's
Python and restart:

```
pip install -r requirements.txt
```

The only runtime dependency beyond ComfyUI is `opencv-python`; torch, numpy,
scipy, safetensors and tqdm come with ComfyUI. `onnx` and `huggingface_hub`
are needed only for the offline conversion and upload scripts
(`pip install .[dev]`).

## Tests

```
python -m pytest tests
```

`tests/libs/test_chunking.py` covers the length math and
`tests/test_package.py` the node contract of all fifteen nodes, both without
torch or ComfyUI:
`python -m pytest tests/test_package.py tests/libs/test_chunking.py`.
`tests/pipelines/test_long_video*.py` run the samplers' chunk loop against
stubbed core nodes with real CPU tensors; it is skipped when torch is not
installed. The preprocess tests (`tests/nodes/test_nodes_*.py`,
`tests/pipelines/test_pose.py`, `tests/pipelines/test_sam3_1_multiplex_*.py`,
`tests/pipelines/test_guard*.py`, `tests/models/test_models_*.py`, ...) need
torch and, for most, ComfyUI on the path:
`PYTHONPATH=/path/to/ComfyUI python -m pytest tests`.

## Licences

The code is MIT (`LICENSE`), except the files vendored from the Alibaba Wan
team's WanAnimate preprocess, which are Apache-2.0 and keep their copyright
header (`libs/pose_utils/LICENSE`):

- `libs/pose_utils/pose2d_utils.py`
- `libs/pose_utils/human_visualization.py`
- `pipelines/face.py`
- `models/common/wrapper.py`
- `models/vitpose/wrapper.py`
- `models/yolo/wrapper.py`
- `models/vitpose/decode.py`

The model weights keep their own licences:

| Model | File | Licence |
|-------|------|---------|
| ViTPose-H wholebody | `vitpose_h_wholebody_fp16.safetensors` | Apache-2.0 |
| YOLOv10x | `yolov10x_fp32.safetensors` | AGPL-3.0 |
| SAM 3.1 | `sam3.1_multiplex_fp16.safetensors` | Meta's SAM License |

The person detector's weights (YOLOv10x) are AGPL-3.0. Running them locally
is unaffected; offering a service over a network that runs them (a hosted
workflow, a SaaS) brings AGPL-3.0's network clause into play, which requires
making the corresponding source available to that service's users. With
`bboxes` connected, Pose Detection neither downloads, loads nor runs the
detector.

SAM 3.1 is fetched by ComfyUI from `Comfy-Org/sam3.1` under Meta's SAM
License; this package does not redistribute it.
