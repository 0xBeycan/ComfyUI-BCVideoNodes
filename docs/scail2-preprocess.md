# SCAIL-2 Preprocess

## SCAIL-2 Colored Mask and SCAIL-2 Preprocess

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
- **SCAIL-2 Preprocess** = SAM 3.1 Multiplex Video Track (one object) in the
  chosen `mode` on the whole driving video once, and in prompt mode on the
  reference image unless `reference_mask` is connected, then SCAIL-2 Colored
  Mask. Tracking the whole clip once keeps the mask's shape and colour the
  same across the sampler's chunks (the official template re-tracks every
  segment). `mode` is the Video Track's widget (`prompt` default,
  `box_keypoint`, `prompt_pose`), so the mask comes from the chosen mode and
  switching needs no rewiring: in `box_keypoint` and `prompt_pose` the node
  first runs Pose Detection (with its `pose_model`) on the driving frames, at its default widgets
  (SCAIL-2 draws no pose; the pose only shapes the mask) with `pose_config`
  when connected, and passes its `pose_data` to the track; `prompt` runs no
  pose. The reference image is tracked in prompt mode whatever the mode: the
  pose modes are video modes, and the reference is one image, so `prompt` is
  read in every mode. Widgets: `replacement_mode`, `mode`, `prompt`,
  `black_background` (default off), `pose_model` (last; `ViTPose-H` default, or
  `Sapiens2 <model>`, read only in the pose modes); optional `reference_mask`, `pose_config`,
  `sam3_config`. Outputs: `pose_video` (the driving video, which SCAIL-2's
  end-to-end mode reads as its pose input in animation and replacement mode
  alike), `pose_video_mask`, `reference_image_mask`, `mask`,
  `reference_mask`, `replacement_mode` (the widget's value, to link to the
  SCAIL-2 Long Video Sampler's `replacement_mode`, so the two always match).
- `black_background` (animation mode only): `pose_video` becomes the driving
  video with every pixel outside the person's mask black, as SCAIL-2's
  training pose videos were (zai-org/SCAIL-2 issue #17; SCAIL-Pose's
  `--crop_e2e_mask`), so the driving video's background and camera do not
  reach the result. Off, `pose_video` is the driving video unchanged. In
  replacement mode the result keeps the driving video's background, so
  `black_background` on with `replacement_mode` on is an error, raised
  before any tracking.

## SCAIL-2 Preprocess Guard

Checks the two colored masks before the SCAIL-2 sampler, on the person as
the sampler reads it (blue above 225/255). End-to-end SCAIL-2 draws no pose,
so `pose_data` is optional: connected (Pose Detection on the driving frames
at the generation size), the driving mask also gets the Mask Guard's
pose-based checks with their levels, and a detached piece holding the
person's keypoints is her (without it, only a piece that runs off a side of
the frame she runs off is). The mode is read from the reference mask's
border, as the sampler reads it; the driving mask is taken to be at the
generation size. With `scail2_guard` on these stop the workflow: no driving
frame has the person (`no_driving_person`), a detached region at least 5% of
the largest one on a driving frame (`driving_fragmented`), the reference
mask has no character (`reference_empty`), and with `pose_data` the Mask
Guard's fails (`mask_empty`, `mask_head_out`, `mask_limb_out`, `mask_loss`).
Warnings, which never stop: without `pose_data` a driving frame without the
person (`driving_empty`: normal where the person leaves the shot, and only a
pose tells), a detached region on the reference (`reference_fragmented`), in
replacement mode a reference whose character overlaps the first driving
frame's person by an IoU below `min_reference_iou` (0.4) after the center
crop the core node cuts the reference to (`reference_misaligned`; a first
value, not calibrated yet), and with `pose_data` `mask_leak` and
`mask_attached_leak`.

The driving mask is judged the way SCAIL-2 reads it. The core node
area-resizes it to half size and cuts each colour at 225/255, then
area-pools that to the latent grid, one cell per 16 x 16 px of the
generation; it grows nothing and uses no noise mask. A cell reads as the
person when she fills at least half of it. So a region of her the model
loses (`mask_loss`) is one the mask and that reading both drop, holding a
whole cell of the grid, and a drawn keypoint the reading holds is inside the
mask (`mask_head_out`, `mask_limb_out`). `mask_loss` needs `pose_data`: on
that grid a limb in motion empties whole cells of a correct mask, which only
the pose tells from a dropped part of her. Measured as data: the mask area,
the share of the mask the sampler's half-size latent cut keeps
(`latent_kept`: a thin limb can vanish there), the share of the reference
character that crop cuts off (`cropped`: the core node crops the reference
to the generation's aspect ratio either way), and the reference's IoU and
scale against the first driving frame.

- in: `pose_video_mask`, `reference_image_mask` (IMAGE), optional
  `pose_data`; widgets `scail2_guard` (on), the reference threshold
  (`SCAIL2GuardConfig`) and the Mask Guard's thresholds (`MaskGuardConfig`,
  in `pipelines/guard/config.py`)
- out: `pose_video_mask`, `reference_image_mask` (unchanged), `report`,
  `metrics` (JSON: `"guard": "scail2"`, the driving frames, the reference
  record), `timeline` (IMAGE)
