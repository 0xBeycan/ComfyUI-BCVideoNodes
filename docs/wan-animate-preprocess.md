# WanAnimate Preprocess

## WanAnimate Preprocess and WanAnimate Preprocess Guard

The wrappers call the individual nodes, so a wrapper produces exactly what
the chained nodes produce with the same settings.

- **WanAnimate Preprocess** = Pose Detection (with its `pose_model`) -> SAM 3.1
  Multiplex Video Track -> Face Crop, then the final mask. Widgets: the drawing widgets,
  `face_padding`, `mode`, `prompt`, `pose_model` (`ViTPose-H` default, or `Sapiens2 <model>`),
  then `grow` (10) and `block_size` (32); optional `pose_config`, `sam3_config`. In
  `box_keypoint` mode the
  mask is prompted from the pose; in `prompt_pose` mode the pose's drawn
  keypoints add points on the frames where the track lost a limb, and a region
  the mask drops for a few frames adds points inside it (`pose_data`
  is passed whenever the mode is not `prompt`). Outputs: `pose_images`, `face_images`,
  `mask`, `pose_data`, `bboxes`, `key_frame_body_points`, `face_bboxes`,
  `final_mask` (the mask grown by `grow` steps of the 3 x 3 cross, then cut
  into blocks of about `block_size` px: ComfyUI-BCNodes' MaskGrow with blur 0,
  then Blockify Mask) and `bg_images` (the frames with `final_mask` painted
  black: Draw Mask On Image with `0, 0, 0`), the character mask and the
  background video of a replacement run. The Wan Animate sampler paints the
  background itself, a chunk's window at a time, so its `background_video`
  can take Load Video's frames instead of `bg_images`, which then stays
  unconnected and is not computed (same result). `mask` stays the raw mask. A workflow
  saved before `grow` and `block_size` existed loads with 10 and 32.
- **WanAnimate Preprocess Guard** = Pose Guard + Mask Guard with one combined
  report. Inputs `mask`, `pose_data`, the `mask_guard` switch and all thresholds,
  optional `min_reference_iou` (0.5), `reference_image`, then `grow` (10) and
  `block_size` (32), which must equal the preprocess's;
  outputs `mask`, `pose_data`, `report`, `metrics`, `timeline`. Its `mask` is
  the preprocess's `final_mask`, the mask the sampler gets. The final's
  grid is laid from each frame's own box, so the grid moves from frame to frame and
  the final's outline is known only to within a block: the mask checks allow
  for that (`mask_attached_leak` allows the neighbouring frames' masks a
  block, the detached pieces allow for the padding, and `mask_loss` counts
  only a region holding a drawn keypoint on every frame of its run; a block
  holding a keypoint inside the raw mask is always on). The Mask Guard on the
  raw mask judges the mask at its own precision, and its `mask_loss` by what
  the final leaves out.
