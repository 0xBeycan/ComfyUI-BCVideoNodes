"""Checks on the preprocess output: warn or fail only for what really causes problems in diffusion.

A check fails only on damage the diffusion model cannot absorb: a person the mask leaves empty, a
region of her the model loses, a large piece torn off the mask, the head or a whole limb outside
the mask. Pose errors the model tolerates - a knee the pose does not draw, a keypoint slightly off
the arm - raise nothing; what stays a warning is reported and never stops (common.WARNINGS).

Every check is normalised - by the person's size where it can be - and, where possible, crosses
one signal with another that was produced independently: the SAM mask against the pose model's
keypoints, the mask's motion against the box's motion. The guard judges the pose and the mask
only, never the detector: a missed box or a second person in shot changes nothing about what comes
out, so those are data in the metrics, not checks. Every check counts what the pose images draw: a
keypoint is drawn when it reaches the `draw_threshold` Pose Detection drew with (carried in
pose_data), and a limb when both its ends do.

The checks come in two groups that run on their own: `check_pose` needs only `pose_data`,
`check_mask` needs the mask and, for its pose-based checks, the `pose_data` it is checked
against. `combine_guards` merges the two results into the one report, metrics and timeline of the
whole preprocess.

Pose checks, per frame, from `pose_data` (keypoints, detections), all warnings:

  pose_jump           torso keypoints moved more than `max_torso_jump` box diagonals in one frame
                      while the box hardly moved (pose glitch, not motion)
  pose_spike          a limb keypoint jumped more than `max_limb_spike` of the frame height and
                      came back within a few frames, flagged over that window; the report names
                      the keypoints
  subject_switch      box IoU with the previous frame below 0.3 (the detector picked someone or
                      something else)

Mask checks, per frame, from the mask against `pose_data`:

  mask_empty          fail: no mask, or mask area below `min_mask_to_box` of the box area - only
                      on a frame whose box the detector and the pose model agree on, since an
                      empty mask elsewhere is the pose failing, not the mask (a black fade has no
                      person to lose)
  mask_leak           warning: more than `max_mask_outside_box` of the mask lies outside the boxes
                      of the frames around it, grown by 10% (background or a neighbour pulled in);
                      only on a frame whose box the detector and the pose model agree on
  mask_attached_leak  warning: the person's region grew by more than `max_attached_leak` of its
                      size where neither the neighbouring frames' masks nor the drawn skeleton are
                      (background taken in against the body)
  mask_fragmented     fail: a detached region at least 5% of the largest one (a ghost, a second
                      person, a limb torn off the mask). A piece holding the person's own drawn
                      keypoints - body, hands or face - is that same person, a hand the frame edge
                      cut away from the body; without pose_data so is a piece that runs off a side
                      of the frame the largest region also runs off. So is a speck-sized piece most
                      of which her part holds on two of the frames around (a hair tip or a shoe the
                      mask split off)
  mask_head_out       fail: the head outside the mask - the drawn nose, or `head_out_eyes_ears` or
                      more of the drawn eyes and ears, inside the frame and outside the (slightly
                      grown) mask: a head the model redraws from nothing. The report names the
                      keypoints
  mask_limb_out       fail: a whole forearm-and-hand or lower leg outside the mask the model reads,
                      judged by prompt_pose's own rule (sam3_1_multiplex.prompt_pose.refine_points)
                      on each frame alone: 3 or more of the limb's drawn keypoints outside the mask,
                      each at least 0.07 of the frame's shorter side from it, its distal part at
                      least 90% outside - a whole limb lost, not a fingertip past the edge. The
                      report names the limbs
  mask_loss           fail: a region of her the model loses (mask.dropouts): the mask holds it
                      around a run of frames and drops it on every frame of the run - a closed run
                      of up to 8 frames, or with pose_data a run to the end of the clip (or of a
                      stretch with a mask) or from its start - and it is her (the pose has the body
                      in it next to the run, or the mask holds it beyond), not a limb that moved
                      away. What counts is what the model reads: the Wan Animate workflow grows the
                      raw mask into the final mask (GrowMaskWithBlur expand 10, BlockifyMask 32)
                      before any model reads it, so the region has to be missing from the final of
                      every frame of the run and hold a whole block of its grid. With pose_data the
                      body has to be in it on every frame of the run: the drawn skeleton crosses it,
                      or a limb that crosses it next to the run is lost by the pose too. And it has
                      to be hand-sized: common.LOSS_HAND of her mask or more

The WanAnimate Preprocess Guard judges the final mask the sampler gets, the preprocess mask grown
(GrowMaskWithBlur expand 10) and cut into 32 px blocks (BlockifyMask), whose grid is laid from
each frame's own box and moves with it (check_mask(final=True), common.FINAL_BLOCK): its outline
is known only to within a block, so mask_attached_leak allows the neighbouring frames' masks a
block, the detached pieces allow for the padding, and mask_loss counts only a block holding a drawn
keypoint on every frame of its run.

Without pose_data the Mask Guard runs only the checks that do not read it: mask_fragmented (the
largest region, and the pieces the frame edge cut from it, are the person) and mask_loss on closed
runs.

With a reference image (the node finds the character on it with SAM 3.1 Multiplex, prompt mode,
the default prompt) the Mask Guard and the WanAnimate Preprocess Guard also check the reference of
the replacement run (reference.py), a warning:

  reference_misaligned  the character on the reference, placed as core's Wan Animate node places
                        the reference (center crop to the mask's aspect ratio, resized to its size),
                        overlaps the mask on frame 0 by an IoU below `min_reference_iou`:
                        replacement expects the reference posed and placed like the first frame.
                        The Preprocess Guard compares final with final: the placed character grown
                        and blockified as the final mask first, which raises every IoU, so its
                        default is higher (ReferenceGuardConfig 0.4, FinalReferenceGuardConfig 0.5)

Each group's fails are switched on separately (the Pose Guard has none). Everything measured is
always reported and plotted; a failed check of an enabled group stops the workflow, since sampling
on a wrong mask is wasted.

SCAIL-2 has its own guard, `scail2.check_scail2`, on the colored masks its sampler reads, with
an optional pose (see pipelines/guard/scail2.py); it runs the Mask Guard's checks on its latent
grid. It is imported from its module, not from here: it reads the colored-mask conventions of
models/scail2, which the pose and mask guards do not need.
"""
from .combine import combine_guards  # noqa: F401
from .common import (MASK_CHECKS, MASK_REFERENCE, MASK_ROW, POSE_CHECKS, POSE_FREE_MASK_CHECKS, POSE_ROW,  # noqa: F401
                     PREPROCESS_ROW, SCAIL2_CHECKS, SCAIL2_ROW, TORSO, WARNINGS, GuardFailed)
from .config import (FinalReferenceGuardConfig, MaskGuardConfig, PoseGuardConfig, ReferenceGuardConfig,  # noqa: F401
                     SCAIL2GuardConfig)
from .mask import check_mask  # noqa: F401
from .pose import check_pose  # noqa: F401
