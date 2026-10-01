# Pose and Face Crop

## Pose Detection

YOLOv10x finds the person, the pose model gives the 133 COCO-WholeBody
keypoints on every frame from the person box, and the pose images are drawn.
By default the box is the detector's raw box, as the official Wan and Kijai
preprocess crop it; Pose Config's experimental `box_window` and `edge_snap`
widen it to the boxes of the frames around it and extend it to a frame edge it
nearly touches.

- in: `images`; optional `bboxes` (BBOX, one `(x1, y1, x2, y2)` per frame or
  one for all: the detector is then skipped), `pose_config` (POSE_CONFIG),
  `width`, `height` (INT sockets, both or neither)
- widgets: `body_stick_width` -1,
  `hand_stick_width` -1 (0 leaves that part out, -1 sizes it from the size
  the pose images are drawn at), `draw_head` true, `draw_threshold` 0.5,
  `pose_model` `ViTPose-H` (below)
- out: `pose_images` (IMAGE), `pose_data` (POSEDATA), `bboxes` (BBOX, the
  person box per frame as everything downstream sees it),
  `key_frame_body_points` (STRING: frame 0's confident body keypoints in the
  KJNodes PointsEditor / easy-sam3 `positive_coords` JSON format)

With `width` and `height` connected the pose images are drawn at that size
directly (a ControlNet hint at exactly the latent's pixel size, say): the
keypoints are scaled to it, a frame of another aspect is cut centred first as
core's ControlNet cuts a hint (`center`), and a -1 stick width is picked from
that size. `pose_data`, `bboxes` and `key_frame_body_points` stay at the frame
size. Not connected, the pose images are drawn at the frame size.

`pose_model`, the last widget: `ViTPose-H` (default) gives every keypoint.
`Sapiens2 <model>` runs Meta's Sapiens2 for the body, the feet and the hands
on the same person box (YOLOv10x, or the connected `bboxes`), with the same
`pose_config`, drawing, `width` / `height` and outputs: Sapiens2 gives the
body, feet and hand keypoints on its own 1024x768 crop, mapped to
COCO-WholeBody by name, and the 68 face keypoints come from ViTPose-H on the
same box, so Face Crop, the guards and `back_view_face` read the same face as
with ViTPose-H. The models: `5b int8 convrot`, `5b bf16`, `1b int8 convrot`,
`1b bf16`, `0.8b int8 convrot`, `0.8b bf16`, `0.4b int8 convrot`, `0.4b bf16`;
`int8 convrot` is the int8 ConvRot quantized file, computing in bf16. The
model is downloaded on first use (see [Models](models.md)). A workflow saved before the
widget existed loads with `ViTPose-H` and runs as before. The separate
Sapiens2 Pose node (`BCVSapiens2Pose`) is gone: a workflow that used it shows
it as missing; replace it with Pose Detection and pick the same model in
`pose_model` (its `model` value with `Sapiens2 ` in front; its other widgets
keep their meaning).

WanAnimate Preprocess and SCAIL-2 Preprocess have the same `pose_model` widget
(SCAIL-2 Preprocess's last; on WanAnimate Preprocess `grow` and `block_size`
follow it) and hand it to Pose Detection. A workflow saved before the widget
existed loads with `ViTPose-H` and runs as before. SCAIL-2 Preprocess reads it
only in the modes that run the pose.

## Pose Config

Optional; without it Pose Detection runs with the measured defaults, which are
the values the node shows. Its widgets are generated from `PoseConfig` in
`pipelines/pose.py`: `min_keypoint_conf`, `detection_threshold`, and the
experimental `box_window`, `forearm_limit`, `limb_dedup`, `back_view_face`,
`edge_snap`. `min_keypoint_conf` travels in `pose_data` to SAM 3.1 Multiplex Video Track's
`box_keypoint` mode; the guards count the keypoints that reach Pose Detection's
`draw_threshold` (also carried in `pose_data`).

`box_window` and `edge_snap` are the box switches, both off by default and
**experimental**: the node shows them as `box_window (experimental)` and
`edge_snap (experimental)`. A five-arm pose A/B (`box_window` on and off x
`edge_snap` on and off, plus Kijai's original preprocess) found no net gain for
either on the pose: the errors moved between frames rather than going away. The
official Wan and Kijai preprocess crop the raw detector box, and with both
switches off the pose path matches theirs except for the detector model and the
ViTPose precision. The switches may still be revisited for SAM 3.1 Multiplex's
box prompt (`box_keypoint` mode), which reads the same boxes.

`box_window` (experimental; 0, off): each frame's box is widened to the union
of the detected boxes within this many frames either side, against a detector
box that shrinks to the upper body when the person comes close and around the
blur when they move fast. Supplied `bboxes` are widened too.

`edge_snap` (experimental; off): a box edge that stops within 15% of the box's
own size from a frame edge is extended to that edge. It was made for SAM 3.1
Multiplex's box prompt in `box_keypoint` mode, so clothing at the frame edge is
not cut off the mask. The same box also cuts the pose crop and goes to the
guards, and on typical clips it moves an edge on nearly every frame. Off, the
boxes are used as detected or supplied, widened by `box_window`; supplied
`bboxes` are snapped too when it is on.

`forearm_limit`, `limb_dedup` and `back_view_face` are draw rules
(`libs/draw_rules.py`), all off by default and **experimental**: the node shows
them as `forearm_limit (experimental)`, `limb_dedup (experimental)` and
`back_view_face (experimental)`. In a diffusion comparison they gave no clear
gain, and each can remove a correct part. A part any rule names is left out
of the pose images, with one console warning per side and part a rule fired on
naming the frames; `pose_data` keeps every keypoint.

`forearm_limit` (experimental; 0, off): ViTPose puts a wrist it cannot see on something
else, the leg or the frame edge, and the forearm ending there is drawn far too
long. Above 0, a drawn forearm longer than `forearm_limit` times its median
drawn length over the clip (2.0: over twice it) has its wrist and hand left out.

`limb_dedup` (experimental; off): ViTPose draws a hand it cannot see on the other hand, and
an arm it cannot see along the other arm. On, both are left out:

- a hand: two drawn hands whose matching keypoints nearly coincide are one
  hand drawn twice. When exactly one arm is intact (its elbow and wrist
  drawn), the broken arm's hand is the copy and is left out, if that arm is
  broken on the frame before or after as well (a one-frame dip of an
  otherwise intact arm is no hidden arm).
- an arm: both elbows and both wrists within 0.2 body scales of each other
  (the widest of the shoulders, the hips and 1.5 x neck-to-nose), with one
  arm's forearm drawn and something of the other arm drawn, are one arm drawn
  twice: the arm whose elbow and wrist are less confident is the copy, and its
  elbow, wrist and hand are left out.

`back_view_face` (experimental; off): ViTPose invents a nose and eyes on the back of the
head, up to 0.99 confident, and the profile drawn from them flips side from
frame to frame. On, a frame whose body is seen from behind (the model's left
shoulder on the image left of its right shoulder) and whose face is not seen
(the mean confidence of the 17 jaw-line face keypoints under 0.835) has its
nose and both eyes left out; everything else is drawn as before. The ears keep
the head's place. The jaw line tells a back of the head from a face turned
over the shoulder: the skull hides it from behind, and the model invents it
less than the nose and eyes. Each frame is read alone.

## Face Crop

- in: `images`, `pose_data`; optional `face_bboxes` (BBOX, cut as they are)
- widget: `face_padding` 0 (pixels added around the keypoint face box)
- optional widget: `face_box_smoothing` `size` (default: the box centre
  kept, its width and height averaged over a centred Gaussian of sigma 2
  frames: stops the crop's size pulsing without letting a fast face leave
  its box), `off` (each frame's own box) or `median` (per-coordinate median
  over 5 frames)
- out: `face_images` (512 x 512, Wan Animate's `face_video`), `face_bboxes`
