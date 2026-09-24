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

The checks come in two groups that run on their own: `check_pose` needs only `pose_data`,
`check_mask` needs the mask and the `pose_data` it is checked against (the keypoints and
boxes are what the mask is judged by). `combine_guards` merges the two results into the one
report, metrics and timeline of the whole preprocess.

Pose checks, per frame, from `pose_data` (keypoints, detections):

  pose_incomplete     the frame draws less than `min_pose_completeness` of the limbs the
                      frames around it draw (a collapsed or half-missing skeleton); the
                      report names the limbs it lost
  pose_jump           torso keypoints moved more than `max_torso_jump` box diagonals in
                      one frame while the box hardly moved (pose glitch, not motion)
  pose_spike          a limb keypoint jumped more than `max_limb_spike` of the frame height
                      and came back within a few frames; the report names the keypoints
  pose_limb_gap       a limb drawn before and after is missing for a stretch of frames
                      (warning: a limb the body or the frame really hides looks the same)
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
                      neither
  mask_missing_keypoints
                      fewer than `min_keypoint_recall` of the drawn keypoints inside the
                      frame fall inside the mask (a missed limb, hand or foot); the report
                      names them
  mask_missed_limb    a drawn elbow, wrist, knee, ankle or foot outside the mask: either the
                      mask lost the limb or the pose put it off the body (warning)
  body_not_drawn      more than `max_body_not_drawn` of the person's mask lies away from the
                      drawn skeleton: the mask shows a body the pose image does not draw
  mask_unstable       mask IoU with the previous frame below `min_mask_iou` while the box
                      IoU is above 0.7 (the mask changed, the person did not)

Each group is switched on separately. Everything measured is always reported and plotted; a
failed check of an enabled group stops the workflow, since sampling on a wrong mask or pose is
wasted. Only a detached mask (mask_fragmented) and real pose or mask defects stop; a check that
cannot tell a defect from something the scene really does is a warning. Thresholds were
measured on the test clips; run with the switches off on clips known to be good and bad and
read `metrics` before changing them.
"""
from .combine import combine_guards  # noqa: F401
from .common import (MASK_CHECKS, MASK_ROW, POSE_CHECKS, POSE_ROW, PREPROCESS_ROW, TORSO,  # noqa: F401
                     WARNINGS, GuardFailed)
from .config import MaskGuardConfig, PoseGuardConfig  # noqa: F401
from .mask import check_mask  # noqa: F401
from .pose import check_pose  # noqa: F401
