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
  read in every mode. Inputs: `images`, `reference_image` (Load Reference
  Image's `resized_image`), optional `reference_source` (Load Reference
  Image's `source_image`, for `face_crop`), `reference_mask`, `pose_config`,
  `sam3_config`. Widgets: `replacement_mode`, `mode`, `prompt`,
  `black_background` (default off), `face_crop` (default off),
  `face_crop_upscale` (default 2, read with `face_crop` on), `pose_model`
  (last; `ViTPose-H` default, or `Sapiens2 <model>`; read in the pose modes, and
  with `face_crop` on in every mode). Outputs: `pose_video` (the driving
  video, which SCAIL-2's end-to-end mode reads as its pose input in animation
  and replacement mode alike), `pose_video_mask`, `reference_image_mask`,
  `reference_images`, `mask`, `reference_mask`, `replacement_mode` (the
  widget's value, to link to the SCAIL-2 Long Video Sampler's
  `replacement_mode`, so the two always match). Link the sampler's
  `reference_image` from `reference_images` and its `reference_image_mask`
  from `reference_image_mask`: with `face_crop` off `reference_images` is
  `reference_image` itself, with it on the two batches hold both references.
- `face_crop` (default off): a face close-up as a second reference, SCAIL-2's
  multi-reference (zai-org/SCAIL-2 README, Experimental Functions:
  Multi-Reference). It needs `reference_source`, the reference at its own
  resolution (an error says what to connect when it is missing). Pose
  Detection (its default widgets and `pose_model`, in every mode) finds the
  face on that one image; the face box is the one Face Crop cuts (the face
  keypoints grown to 1.3 times their area, Wan Animate's face and hair
  framing). No person on the image is an error; a head seen from behind still
  has its box. The close-up is a window of `reference_source` the size of
  the generation (`reference_image`'s width x height) divided by
  `face_crop_upscale` (default 2, 1 to 5), at the source's resolution, so it
  keeps the generation's aspect, resized by that factor to `reference_image`'s
  size with lanczos (Load Reference Image's resize): the face in the close-up
  is the face box's width times the factor. The face box is centred in the
  window; of the window's free height half goes above the box, but never more
  than 0.4 box heights, so the hair stays in and the rest goes below (neck,
  shoulders). A factor at which the window would be narrower than the box, or
  shorter than it, would cut the face: it is lowered to the largest factor
  that keeps the whole box (the box filling the window's width, or its height
  for a generation too wide for the box), with a log line naming both
  factors. The window is shifted inside the source; where the source is still
  smaller than the window, the part past its edges is black in the close-up
  and black (no character) in its colored mask. SAM
  3.1 Multiplex finds the character on it with `prompt` (prompt mode), and
  its colored mask is the character in blue on black in both modes, as the
  official multi-reference example renders extra references. Then
  `reference_images` = [primary, face] and `reference_image_mask` = [the
  primary's mask on the mode's background, the face's on black]. Example:
  a 1280 x 1920 source with a 180 x 200 face box at (550, 300) and a
  704 x 1280 generation: factor 1, a 704 x 1280 window at (288, 220), not
  resized, the face 180 px wide; 2, 352 x 640 at (464, 220), the face 360 px;
  3, 235 x 427 at (522, 220), the face 540 px; 4 and 5 would cut the box and
  are lowered to 704 / 180 = 3.91: a 180 x 327 window at (550, 236), the box
  filling the width, resized to
  704 x 1280. With `reference_mask` connected it must be `reference_image`'s
  size, since each batch holds one size.
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
the frame she runs off is). The mode is read from the border of the primary
reference mask (the first frame), as the sampler reads it; the driving mask
is taken to be at the generation size. Every further frame of
`reference_image_mask` is an extra reference (`face_crop`'s close-up), on
black in both modes, so it never enters the mode read: each is checked on
its own for `reference_empty` (a fail) and `reference_fragmented` (a
warning), and the report names each reference by its index (0 the
primary); `reference_misaligned` judges the primary alone. With `scail2_guard` on these stop the workflow: no driving
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
  `metrics` (JSON: `"guard": "scail2"`, the driving frames, the primary
  reference's record, and with extra references `"extra_references"`, one
  record each: `index`, `area`, `fragments`, `flags`), `timeline` (IMAGE)
