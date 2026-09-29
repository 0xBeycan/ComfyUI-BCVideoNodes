"""Checks on the preprocess output: is the pose plausible and does the mask agree with it?

Every check is normalised - by the person's size where it can be - and, where possible,
crosses one signal with another that was produced independently: the SAM mask against the
pose model's keypoints, the mask's motion against the box's motion. Pure geometry on the mask
cannot tell a stable wrong mask from a right one; the pose can.

A pose check has to judge the drawn skeleton, not whether a component hiccuped. The guard
judges the pose and the mask only, never the detector: a missed box or a second person in
shot changes nothing about what comes out - the mask is carried by the tracker and the pose
model poses the foreground person either way - so those are data in the metrics, not checks.
A confidence threshold is worse than useless: it is calibrated to one model's heatmap maxima,
and on a different pose model a skeleton that visibly collapsed still read as confident. What
survives a change of model is the skeleton against its own neighbours and against the mask.

Every check counts what the pose images draw: a keypoint is drawn when it reaches the
`draw_threshold` Pose Detection drew with (carried in pose_data), and a limb when both its
ends do. Counting at any other threshold judges a skeleton the diffusion model never sees.

A missing limb is one that should be drawn and is not: the person faces the camera, the limb is
in the shot and in sight, and the pose does not draw it. A limb out of the frame (its keypoint
placed at or beyond the edge) or hidden (its keypoint placed on another part of the body: behind
the body, the hair or a hand; the person turned to profile or her back to the camera) is never
missing (common.accounted_limbs); the limb checks and body_not_drawn leave it alone.

Only damage diffusion cannot absorb stops the workflow; everything else is a warning, reported
and never stopping (common.WARNINGS).

The checks come in two groups that run on their own: `check_pose` needs only `pose_data`,
`check_mask` needs the mask and, for its pose-based checks, the `pose_data` it is checked
against (the keypoints and boxes are what the mask is judged by). `combine_guards` merges the
two results into the one report, metrics and timeline of the whole preprocess.

Pose checks, per frame, from `pose_data` (keypoints, detections):

  pose_incomplete     the frame accounts for less than `min_pose_completeness` of the limbs
                      the frames around it account for (a collapsed or half-missing
                      skeleton); the report names the limbs it lost (warning)
  pose_jump           torso keypoints moved more than `max_torso_jump` box diagonals in
                      one frame while the box hardly moved (pose glitch, not motion)
  pose_spike          a limb keypoint jumped more than `max_limb_spike` of the frame height
                      and came back within a few frames, flagged over that window; the
                      report names the keypoints (warning)
  pose_limb_gap       a limb accounted for before and after is missing for a stretch of
                      frames (warning)
  subject_switch      box IoU with the previous frame below 0.3 (the detector picked
                      someone / something else)

Mask checks, per frame, from the mask against `pose_data`:

  mask_empty          no mask, or mask area below `min_mask_to_box` of the box area - only
                      on a frame whose box the detector and the pose model agree on, since
                      an empty mask elsewhere is the pose failing, not the mask
  mask_leak           more than `max_mask_outside_box` of the mask lies outside the boxes of
                      the frames around it, grown by 10% (background or a neighbour pulled
                      in); only checked on a frame whose box the detector and the pose model
                      agree on
  mask_attached_leak  the person's region grew by more than `max_attached_leak` of its size
                      where neither the neighbouring frames' masks nor the drawn skeleton
                      are (background taken in against the body; warning)
  mask_fragmented     a second region at least 5% of the largest one (a ghost, a second
                      person, a split body); smaller detached pieces such as a shadow
                      blob are reported as mask_specks (warning only). A piece holding the
                      person's own drawn keypoints - body, hands or face - is that same
                      person, a hand the frame edge cut away from the body, and counts as
                      neither; without pose_data so is a piece that runs off a side of the
                      frame the largest region also runs off (the edge cut it away). So is a
                      speck-sized piece most of which her part holds on two of the frames
                      around (a hair tip or a shoe the mask split off; an object beside her
                      joins her for a frame at most)
  mask_missing_keypoints
                      fewer than `min_keypoint_recall` of the drawn keypoints inside the
                      frame fall inside the mask; the report names them (warning)
  mask_head_out       the head outside the mask: the drawn nose, or `head_out_eyes_ears` or more
                      of the drawn eyes and ears, inside the frame and outside the mask - a head
                      the model redraws from nothing. Fails in place of mask_missing_keypoints on
                      its frame; the report names the keypoints. A keypoint-mask defect (a
                      correct mask shows none); the SCAIL-2 guard does not run it
  mask_missed_limb    a drawn elbow, wrist, knee, ankle or foot the mask lost: a visible limb
                      end outside the mask, not a pose error, a notch, fast motion or a limb
                      leaving the shot on the frame border (mask.lost_limb_end; warning)
  body_not_drawn      more than `max_body_not_drawn` of the person's mask lies away from the
                      drawn skeleton and the limbs that run out of the shot: the mask shows a
                      body the pose image does not draw (warning)
  mask_unstable       mask IoU with the previous frame below `min_mask_iou` while the box
                      IoU is above 0.7 (the mask changed, the person did not; warning)
  mask_loss           the mask drops a region for a run of 1 to 8 frames and holds it on the
                      frames before and after, and no limb moved away to account for it; the
                      region is hers - the pose has the body in it next to the run, or the mask
                      holds it beyond the frames next to the run on both sides - not
                      background that blinks off between two frames that took it in:
                      thicker than `max_mask_loss` on a single frame, twice that over a longer
                      run, half that where the drawn skeleton crosses it on at least half the
                      frames of the run. No model reads the raw mask: the Wan Animate workflow
                      grows it into the final mask (GrowMaskWithBlur expand 10, BlockifyMask 32)
                      first, so only the part the final of every frame of the run leaves out
                      counts (warning)
  mask_loss_large     a mask_loss dropout that drops `large_loss_area` of the person's mask or
                      more (mask_loss_area): a hand or a leg the models cannot restore. Fails in
                      place of mask_loss on the frames of its run; a frame whose mask is
                      empty is mask_empty's. A keypoint-mask defect (a correct mask shows
                      none); the SCAIL-2 guard does not run it

The WanAnimate Preprocess Guard judges the final mask the sampler gets, the preprocess mask grown
(GrowMaskWithBlur expand 10) and cut into 32 px blocks (BlockifyMask), whose grid is laid from
each frame's own box and moves with it (check_mask(final=True), common.FINAL_BLOCK): its outline
is known only to within a block, so mask_attached_leak allows the neighbouring frames' masks a
block, body_not_drawn and the detached pieces allow for the padding, and mask_loss counts only a
region holding a drawn keypoint on every frame of its run.

Without pose_data the Mask Guard runs only the checks that do not read it: mask_fragmented and
mask_specks (the largest region, and the pieces the frame edge cut from it, are the person),
mask_loss and mask_loss_large.

Each group is switched on separately. Everything measured is always reported and plotted; a
failed check of an enabled group stops the workflow, since sampling on a wrong mask or pose is
wasted. Thresholds were measured on the test clips; run with the switches off on clips known to
be good and bad and read `metrics` before changing them.

SCAIL-2 has its own guard, `scail2.check_scail2`, on the colored masks its sampler reads, with
an optional pose (see pipelines/guard/scail2.py); it runs the Mask Guard's checks but
mask_head_out and mask_loss_large. It is imported from its module, not from here:
it reads the colored-mask conventions of models/scail2, which the pose and mask guards do not need.
"""
from .combine import combine_guards  # noqa: F401
from .common import (MASK_CHECKS, MASK_ROW, POSE_CHECKS, POSE_FREE_MASK_CHECKS, POSE_ROW, PREPROCESS_ROW,  # noqa: F401
                     SCAIL2_CHECKS, SCAIL2_ROW, TORSO, WARNINGS, GuardFailed)
from .config import MaskChecksConfig, MaskGuardConfig, PoseGuardConfig, SCAIL2GuardConfig  # noqa: F401
from .mask import check_mask  # noqa: F401
from .pose import check_pose  # noqa: F401
