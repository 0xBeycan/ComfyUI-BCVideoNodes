# SAM 3.1 Multiplex

## SAM 3.1 Multiplex Video Track

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
  tracked object, `k` is object `k`. With one track, when the mask gains a
  large piece that lasts within 16 frames of the birth (15% of the mask or
  more, 80% of it still there on each of the next 5 frames: a limb the birth
  detection missed and the tracker found again), every frame before it is
  tracked again backwards from that frame, so the limb reaches them too.
  With one track, the mask alone then decides two repairs (below); a clip
  where neither is found gets the track's mask exactly.
- `box_keypoint`: the person is described by `pose_data`'s box and body
  keypoints (required), with `bboxes` replacing the boxes and the coords
  adding hand-placed points on frame 0. `max_objects` applies to `prompt`
  mode only; `box_keypoint` mode tracks one person, ignores it and logs one
  line.
- `prompt_pose`: `prompt` mode's track of one person, without its repairs,
  and on every frame
  where the pose (`pose_data`, required) shows the track lost a whole
  forearm-and-hand or lower leg, far outside the mask, that limb's drawn
  keypoints go onto that frame as positive points together with the mask the
  tracker had on that frame: Meta's point refine on the same object, followed
  by Meta's tracker-only re-propagation. Each frame is judged on its own, so
  a loss over several frames, or one from the birth frame on, is refined on
  every frame it qualifies on. The frames the refine can reach are tracked
  again; the rest keep the track's mask. It also refines every frame of
  a run of 1-8 frames where the mask drops a hand-sized region of her that it
  holds on both sides and the pose has her body in (the Mask Guard's
  `mask_loss` on closed runs), from up to 16 points inside that region. A
  clip where no frame needs points gets the track's mask exactly. Nothing
  is removed on pose grounds: there are no negative points. Limits: as
  `prompt`'s track; only whole-limb drops far outside the mask, and regions
  dropped for up to 8 frames between two frames that hold them, are
  recovered.

How `prompt` mode repairs its track (one track), from the mask alone, no pose:

1. The track: the detector finds her, the tracker carries her, the frames
   before the birth (or before a gain) are tracked backwards. Nothing else
   changes it.
2. A part lost for good: the largest 4-connected piece of a frame's mask that
   the next frame lacks is 8% of the mask or more, and from that next frame to
   the end of its stretch (the frames with a mask) the area never comes back
   above 92% of the frame before's. The frames from the one that lost it to
   the end of the stretch are tracked again, forwards, by the tracker alone,
   on a memory of their own that holds nothing but the mask of the frame
   before the loss (no detection, no re-anchor, no points). The track's memory
   is built with the detector's anchors, whose masks leave out a limb the
   detector does not see as her (a blurred hand reaching toward the camera),
   and there the tracker lets the part go; a memory seeded from her whole mask
   keeps it. On the test clips it fires on exactly that hand, at three of four
   widths (pieces of 12-14% of the mask, the area at most 87-89% from there on);
   no other frame keeps its area at or under 92% to the end of its stretch,
   and of the frames that lose 8% or more the area comes back to 96% at least.
   A loss the area comes back from before the end of its stretch is left
   alone, however long it lasts: on the test clips those are an arm swinging
   in front of the body, where the mask is right.
3. A part dropped for a few frames: on the masks after step 2, where the mask
   drops a part of her for 1-8 frames between two frames that hold it (each
   of the two loses a part of 1.5% of her mask or more holding a whole block
   of the Wan Animate final at its defaults, `grow` 10 and `block_size` 32;
   the two parts overlap;
   neither moved away), every frame of the run is refined from up to 16
   points where both frames hold her, together with the mask the tracker had
   on that frame (Meta's point refine, as in `prompt_pose` step 3), and shows
   its mask and the refine's together (the larger of the two logits, at the
   decoder's 288 x 288). Nothing else is tracked again: no demotion, no
   second pass.

The console names each repair: the frames tracked again, the frame before
them and the share lost, and each refined frame with its points. The repairs
keep every propagated frame's raw decoder logits on the CPU until they are
done, 162 KiB a frame (35 MiB for 233 frames, 97 MiB for 612).

How `prompt_pose` works, in Meta's order (SAM 3's video predictor, as
easy-sam3 vendors it: the text prompt with its full pass, points on the
existing object, then the re-propagation its action history asks for):

1. Pass 1 is `prompt` mode's track, unchanged, without its repairs. Where it
   tracked the frames before a gain again, the gain frame counts as the birth
   below.
2. The frames to refine are chosen from pass 1's masks and the keypoints the
   pose images draw: body and hand keypoints at `pose_data`'s
   `draw_threshold`, on the canvas (one off it is dropped, not clamped), a
   hand keypoint only at x and y of 1 px or more (the draw code's rule), and
   none an enabled Pose Config draw rule (`forearm_limit`, `limb_dedup`,
   `back_view_face`) leaves out of the images. Pose Config's
   `min_keypoint_conf` is not read. Each frame from the birth frame to the
   last, its mask not empty, is judged on its own: it is refined when a
   forearm-and-hand or a lower leg has 3 or more keypoints that lie outside
   its mask by `pose_point_distance` of the frame's shorter side or more,
   with at least 90% of their distal part outside. The frames around it do
   not decide it, so a loss over several frames refines each frame it
   qualifies on. Head, neck, shoulders and hips never trigger; the frames
   before the birth are never refined. A second trigger is the mask's own:
   the Mask Guard's `mask_loss` detection on pass 1's masks, closed runs
   only, as the Wan Animate final reads them (WanAnimate Preprocess's
   `final_mask` at its defaults, `grow` 10 and `block_size` 32): a region
   of her the final loses for 1-8
   frames and holds on the frames on both sides, a whole block of its grid,
   at least 1.5% of her mask, with the pose's body in it (a limb crosses it
   on the frames around, and crosses it or is lost by the pose too on the
   run). Every frame of such a run from the birth on, its mask not empty, is
   refined from up to 16 points inside the region, spread over it and at
   least half its depth in. The refined frames are those of both triggers.
3. Each such frame is refined from its points (in pose order, then the
   region's, capped at 16: the first 8 and the last 8) together with the mask the tracker had on that
   frame (pass 1's raw decoder logits, clamped to +/-32, as the dense prompt;
   on the birth frame, which pass 1 did not propagate, the mask it was
   conditioned with, as Meta's refine looks it up), on the interactive
   decoder with no memory, with Meta's stability fallback, and becomes a
   conditioning frame; on the birth frame or a re-anchor frame it replaces
   pass 1's conditioning. The points without the mask, as Meta decodes the
   first refine of a frame, kept the forearms and hands but dropped the head,
   torso and dress on the test clips' refined frame, and the re-tracked
   frames after it lost a forearm and hand; with the mask the refine adds the
   lost hand, keeps the body and adds no background.
4. Pass 1's conditioning frames (the birth and the fired re-anchors) within 16
   frames of any refined frame, and not refined themselves, are demoted to
   ordinary frames, and the tracker alone tracks the clip again from the
   birth: no detection, no re-anchor, no probation. Each frame reads the
   `max_conditioning_frames` conditioning frames closest to it on both sides,
   and its memory is encoded from the decoder's raw logits (Meta's
   re-propagation; `memory_mask` is read in pass 1 only).
5. The refine reaches a frame of the second pass when the conditioning frames
   it reads include a refined frame or differ from those it would read with
   no refine and no demotion, and it reaches every frame after that one,
   whose memory comes from the frames before it. The frames before the first
   one it reaches show pass 1's mask: there the second pass could only drift
   from pass 1 (no detector, raw-logit memory). The second pass still tracks
   them from the birth, for its memory.

The result: the frames before the birth, the kept conditioning frames and the
frames the refine cannot reach are pass 1's, the track's bit for bit; the
refined frames show the refine and every other frame the second pass. The
console lines name the refined frames with their points, the stability
fallback, the object score, the frames that keep the first pass, the demoted
and kept conditioning frames and the seconds of each pass. `pose_data`'s
`draw_head` and stick widths are node widgets, not in `pose_data`, so a part a
stick width of 0 leaves out still counts as drawn.

## SAM 3.1 Multiplex Config

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

The `[prompt_pose]` field comes last. `pose_point_distance` (0.07 of the
frame's shorter side, 50 px at 720) is how far outside the mask a drawn limb
keypoint must lie to become a point: on the test clips every keypoint the pose
drew on a label, toy, cabinet or floor lay within 46 px of the mask, and the
one hand the mask lost 68-126 px out. `prompt_pose` also reads every
`[prompt]` and `[prompt, max_objects 1]` field: its first pass is `prompt`
mode's track. A refine's dense prompt, `prompt` mode's and `prompt_pose`'s
alike, is the track's raw decoder logits of the frame; `prompt_pose`'s second
pass reads `input_range`, `fill_hole_area`, `obj_ptr_token`,
`memory_selection` and `max_conditioning_frames`, and `prompt` mode's re-track
after a part lost for good reads what the track's backward pass reads.

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

## Input precedence

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
- `prompt_pose` mode reads `pose_data` and the prompt; `bboxes`,
  `positive_coords`, `negative_coords`, `max_objects` and `object_index` are
  ignored with the console line.
- `face_bboxes` on Face Crop are cut as given; `pose_data`'s face keypoints,
  `face_padding` and `face_box_smoothing` are then not used.
- A keypoint at exactly `min_keypoint_conf` counts as found.
