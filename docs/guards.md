# Guards

## Pose Guard and Mask Guard

Frame-by-frame checks of the drawn pose and of the mask against that pose,
built to warn or fail only for what really causes problems in diffusion.
Pose errors the model tolerates - a knee the pose does not draw, a keypoint
slightly off the arm - raise nothing. The guards count the model's keypoints
at `draw_threshold`; a part a draw rule, `draw_head` off or a 0 stick width
leaves out of the pose images still counts. The detector is not judged: its
box count and missed frames are in the metrics as data. Every measurement is
always reported and plotted; with the Mask Guard's switch (`mask_guard`) on,
a failed check stops the workflow with the report.

The fails, damage the models cannot restore:

- `mask_empty`: no mask, or a mask under `min_mask_to_box` of the box, on a
  frame whose box the detector and the pose model agree on (a frame with no
  person, a black fade, is no fail).
- `mask_fragmented`: a detached region at least 5% of the largest one (a
  ghost, a second person, a limb torn off the mask). A piece holding the
  person's own drawn keypoints is her; so is a speck-sized piece most of which
  her part holds on two of the frames around. Smaller pieces are data only.
- `mask_head_out`: the drawn nose, or `head_out_eyes_ears` or more of the
  drawn eyes and ears, inside the frame and outside the (slightly grown)
  mask: a head the model redraws from nothing.
- `mask_limb_out`: a whole forearm-and-hand or lower leg outside the mask,
  judged by prompt_pose's own rule on each frame alone: 3 or more of the
  limb's drawn keypoints outside the mask, each at least 0.07 of the frame's
  shorter side from it, and the limb's distal part at least 90% outside - a
  whole limb lost, not a fingertip past the edge.
- `mask_loss`: a region of her the model loses. The mask holds it on the
  frame before a run and drops it on every frame of the run: a run of up to 8
  frames that the frame after holds again, or, with `pose_data`, a run from
  the clip's start or to its end. It counts only as the model reads it: the
  Wan Animate workflow grows the raw mask into the final mask (WanAnimate
  Preprocess's `final_mask`: grown by `grow`, blockified by `block_size`,
  which the guard's widgets of the same names must equal) before any model
  reads it, so the region has to be missing from the final of every frame of
  the run and hold a whole block of it (with the Mask Guard's `block_size` 0,
  any hand-sized part, no whole block); it is hers (the pose has her body in
  it next to the run, or the mask holds it beyond the run), not a limb that
  moved away and came back (the
  mask shows that nearby meanwhile); with `pose_data` the drawn skeleton
  crosses it on every frame of the run (or a limb that crosses it next to the
  run is lost by the pose too, so the pose cannot say where it went); and it
  is hand-sized: at least 1.5% of her mask (`LOSS_HAND` in
  `pipelines/guard/common.py`, set on the test clips: the smallest real loss
  2.1% of her, the largest piece under it that passed every other test 1.15%).

The warnings, which never stop: a torso jump (`pose_jump`), a limb spike
(`pose_spike`), a subject switch (`subject_switch`), a mask leaking outside
the box (`mask_leak`), background attached to the body
(`mask_attached_leak`), and with `reference_image` connected a reference not
placed like the first frame (`reference_misaligned`, see Reference check
below). The thresholds are widgets generated from
`PoseGuardConfig` / `MaskGuardConfig` in `pipelines/guard/config.py`; the
Pose Guard has no fail and so no switch. `head_out_eyes_ears` (INT, default
2, 1 to 4) is the last required widget of the Mask Guard and the WanAnimate
Preprocess Guard: 1 fails a single eye or ear too, also where the mask's outline at the
hair leaves one out; 4 fails only all four, or the nose. A face the pose
model draws on the back of a head lies on the head, which the mask covers.
The default is set on the test clips: a correct (prompt) mask left no head
keypoint out on any frame, and on every frame where the keypoint mask lost
the head, the nose or two or more of the eyes and ears were out.

- Pose Guard: in `pose_data`; out `pose_data` (unchanged), `report`,
  `metrics` (JSON, every measurement per frame), `timeline` (IMAGE)
- Mask Guard: in `mask`, optional `pose_data`, `min_reference_iou` (0.4),
  `reference_image`, then `grow` (10) and `block_size` (32; 0-512 on the Mask
  Guard): the final the guard models for `mask_loss`, equal to the
  preprocess's. `block_size` 0 (Mask Guard only) models a final without
  blocks, the mask grown by `grow` read a pixel at a time, so a lost region
  needs no whole block; with `grow` 0 too the raw mask is judged as it is.
  Out `mask` (unchanged), `report`, `metrics`, `timeline`. `pose_data` gives
  the best result: without it the guard runs only `mask_fragmented` and
  `mask_loss` (runs between two
  frames that hold the region), and the report names the checks it did not
  run. Without it a detached piece is the person only when it runs off a side
  of the frame her largest region also runs off (the frame edge cut it from
  her); with it, a piece holding her own drawn keypoints is her.

## Reference check

The Mask Guard and the WanAnimate Preprocess Guard check the reference of a
replacement run when `reference_image` is connected: the reference image the
Wan Animate node gets (Load Reference Image's). The guard finds the character
on it with SAM 3.1 Multiplex in prompt mode with the default prompt (`main
person in the foreground`), the call SCAIL-2 Preprocess makes for its
reference, places it as core's Wan Animate node places the reference (a
center crop to the mask's aspect ratio, resized nearest-exact to the mask's
size) and measures it against mask frame 0. The Mask Guard compares raw with
raw; the WanAnimate Preprocess Guard checks the final mask, so it grows and
blockifies the placed character the same way first (its `grow` and
`block_size`).

An IoU below `min_reference_iou` is `reference_misaligned`, a warning, which
never stops: replacement expects the reference posed and placed like the
first frame. The defaults are first values, set on a small set of clips: 0.4
on the raw mask (Mask Guard), 0.5 on the final (WanAnimate Preprocess Guard),
since the grow and the blockify raise every IoU. Wan Animate uses the mask in
replacement mode only, so the check has no mode switch.

A reference on which SAM finds no person (a back view, say) leaves the check
silent: the report says SAM 3.1 Multiplex found no person on it, and nothing
is flagged. Not connected, no SAM runs and the report and the metrics are
exactly as without the check. Connected, the report gains a line with the
measurements, and the metrics a `"reference"` record: `area` (the
character's share of the reference), `cropped` (the share of it the crop cuts
off), `iou_first_frame`, `scale_first_frame` (the placed character's height
over the person's on mask frame 0) and `flags`; `min_reference_iou` joins the
thresholds.
