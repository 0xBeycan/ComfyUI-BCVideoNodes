# ComfyUI-BCVideoNodes

Video nodes for ComfyUI: a Wan Animate preprocess built from small nodes that
are usable in any video pipeline (wholebody pose with ViTPose-H or Sapiens2, SAM 3.1 person tracking,
face crops, pose and mask checks), a SCAIL-2 preprocess, three samplers
that turn a reference image plus a driving video of any length into a Wan
Animate or SCAIL-2 video of exactly that length, and the video nodes around
them: load a video at the model's generation size, fit a reference image to
it, save the result, compare two videos in the node.

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
| **Sapiens2 Pose** | `BCVSapiens2Pose` | `BCVideoNodes` |
| **Load Video** | `BCVLoadVideo` | `BCVideoNodes/Video` |
| **Get Video Info** | `BCVGetVideoInfo` | `BCVideoNodes/Video` |
| **Load Reference Image** | `BCVLoadReferenceImage` | `BCVideoNodes/Video` |
| **Conform Video** | `BCVConformVideo` | `BCVideoNodes/Video` |
| **Save Video** | `BCVSaveVideo` | `BCVideoNodes/Video` |
| **Video Comparer** | `BCVVideoComparer` | `BCVideoNodes/Video` |

## Video nodes

Load a video at the generation size of the model it is for, fit the reference
image to it, save the result, compare two videos. Decoding and encoding use
PyAV, which comes with ComfyUI. Load Video, Save Video and the Video Comparer
play in the node: the pack's player, drawn on the node, with a play button, a
seek bar and, when the clip has sound, a mute button. Nothing plays until
asked, the clip loops, and the node keeps the size you give it.

### Load Video

Loads a video one frame at a time, cropped and resized to the model's
generation size straight into the output, which is allocated once at its
final frame count: the full-resolution clip never sits in memory.

- in: `video` (a video file of ComfyUI's input folder; the node's `choose
  video to upload` button, or a video file dropped on the node, uploads one
  there and selects it)
- widgets: `model` `Wan`, `resolution` `720p`, `orientation` `auto`,
  `force_fps` (empty), `start_frame` 1, `frame_count` (empty)
- out: `images` (IMAGE), `audio` (AUDIO of the loaded range; none when the
  file has no audio), `video_info` (BCV_VIDEO_INFO, for Get Video Info and
  Load Reference Image)

The sizes are the models' generation sizes, portrait (width x height):

| `model` | `resolution` | Size | Frames |
|---------|--------------|------|--------|
| `Wan`   | `480p`, `720p` | 480 x 832, 720 x 1280 | 4n+1 |
| `SCAIL` | `512p`, `704p` | 512 x 896, 704 x 1280 | 4n+1 |

`resolution` offers the chosen model's labels; a label of the other model is
an error. `orientation` `auto` is portrait when the video is taller than wide,
otherwise landscape (a square video is landscape); `landscape` and `portrait`
force one. Landscape swaps width and height. The frame is cut centred to the
size's aspect ratio, then resized with lanczos.

Which frames are loaded:

- `force_fps`: empty keeps the video's own frame rate. A number above 0, at
  most the video's rate, keeps or drops real frames on that rate's time grid;
  it never blends or repeats a frame, so it only lowers the frame rate: a
  rate above the video's is an error, and one within 0.01% of it is the
  video's rate (29.97 on a 30000/1001 clip keeps every frame).
- `start_frame` (counted from 1) and `frame_count` (empty: every frame from
  `start_frame` on) count the frames `force_fps` kept. A `start_frame` past
  the last frame, or a `frame_count` that runs past it, is an error that says
  how many frames there are and what to set.
- The count is then cut down to the model's 4n+1 (a 100-frame range loads 97).
- A `force_fps` or a `frame_count` that is not a number is an error.

Colour: YUV is converted to RGB with the stream's own colour matrix and range,
as the file is tagged; an untagged stream is read as BT.601 limited range,
FFmpeg's default. A video stored rotated is turned upright.

Audio: the first audio stream, from `start_frame`'s time for the loaded
frames' duration.

`video_info` holds the `model`, the `resolution` and the resolved
`orientation`, then the source's `source_fps`, `source_frame_count`,
`source_duration`, `source_width` and `source_height` (as displayed), then
the loaded batch's `loaded_fps` (`force_fps`, or the source's rate),
`loaded_frame_count`, `loaded_duration`, `loaded_width` and `loaded_height`.

The preview plays the source file in the browser as the loader will take it,
with no server work: sampled at `force_fps`, looping over `start_frame` /
`frame_count`, the part the crop cuts away dimmed. The browser's frame at a
tick can be one off the loader's. Without `force_fps` the range needs the
source's frame rate, which the browser tells only while the clip plays: until
then the whole clip loops.

### Get Video Info

- in: `video_info`
- out: its 13 fields in the order above: `model`, `resolution`,
  `orientation` (STRING), `source_fps` (FLOAT), `source_frame_count` (INT),
  `source_duration` (FLOAT), `source_width`, `source_height` (INT),
  `loaded_fps` (FLOAT), `loaded_frame_count` (INT), `loaded_duration`
  (FLOAT), `loaded_width`, `loaded_height` (INT). `loaded_fps` is the rate to
  give Save Video.

### Load Reference Image

Loads an image as core's Load Image does (EXIF orientation applied, the alpha
channel as the mask, 1 - alpha; without alpha core's empty 64 x 64 mask) and
fits it to `video_info`'s `loaded_width` x `loaded_height` with the crop and
lanczos resize Load Video applies to the frames, so the reference matches the
video's model, resolution and orientation. The mask is fitted the same way.
The node shows the fitted image. With the sampler at the same size, core's
placement of the reference (a centre crop to the generation's aspect ratio,
then a resize) changes nothing.

- in: `image` (an image of the input folder, with core's upload), `video_info`
- out: `image` (IMAGE), `mask` (MASK)

### Conform Video

Fits a video to the nearest standard size: 480p (480 x 854), 720p (720 x
1280) or 1080p (1080 x 1920), portrait, landscape swapped. A pixel resize, not
diffusion.

- in: `images`
- widgets: `fit` `crop` (default: resized to cover the size, the overflow cut
  off centred) or `pad` (resized to fit inside it, centred between black
  bars); `method` `lanczos` (default), `bicubic`, `bilinear`, `area`,
  `nearest-exact`, `bislerp` (comfy.utils.common_upscale's)
- out: `images`

The target is automatic. The orientation is the frames' (portrait when taller
than wide, otherwise landscape, as Load Video's `auto`); the size is the one
whose scale on the short edge is closest to 1, measured as |log(target /
short edge)|: a short edge of 512 goes to 480p, 600 and 704 to 720p, anything
above 1080 to 1080p. The boundaries (about 588 and 882) are not whole numbers,
so no short edge ties. The frames are resized one at a time into one output;
a video already at its target size is returned untouched, with no work done.

### Save Video

Writes the frames as a video file, one frame at a time.

- in: `images`; optional `audio` (muxed in and cut to the video's length)
- widgets: `fps` 24 (wire Get Video Info's `loaded_fps` to keep the source's
  timing), `filename_prefix` `video/ComfyUI` (subfolders and ComfyUI's name
  tokens work; a counter is appended: `video/ComfyUI_00001_.mp4`), `codec`
  `h264-mp4`, `crf`, `preset`, `pix_fmt` (defaults: the codec's, below),
  `save_output` (on: the output folder; off: the
  temp folder, which ComfyUI empties when it starts), `save_metadata` (on: the
  workflow and the prompt are written into the file, so dropping it on
  ComfyUI loads the workflow; ComfyUI's `--disable-metadata` turns it off)
- out: none; the node plays the saved file

`codec` picks the encoder and the container. `crf`, `preset` and `pix_fmt`
offer that codec's values; changing the codec resets them to its defaults (a
loaded workflow keeps its values):

| `codec`    | Encoder | `crf` (default) | `preset` (default) | `pix_fmt` (default first) | Audio |
|------------|---------|-----------------|--------------------|---------------------------|-------|
| `h264-mp4` | libx264 | 0-51 (19) | `ultrafast` ... `placebo` (`medium`) | `yuv420p`, `yuv420p10le`, `yuv444p` | AAC |
| `h265-mp4` | libx265, tagged `hvc1` | 0-51 (22) | `ultrafast` ... `placebo` (`medium`) | `yuv420p10le`, `yuv420p` | AAC |
| `av1-webm` | SVT-AV1 | 1-63 (23) | `0` (slowest) ... `13` (`8`) | `yuv420p10le`, `yuv420p` | Opus |
| `vp9-webm` | libvpx-vp9 | 0-63 (20) | cpu-used `0` (slowest) ... `5` (`1`) | `yuv420p` | Opus |

- `crf`: constant quality; lower is better quality and a larger file.
- `preset`: encode speed against file size at the same `crf`; a slower
  preset gives a smaller file.
- `pix_fmt`: `yuv420p` plays everywhere; `yuv420p10le` is 10 bits, less
  banding, not every player; `yuv444p` keeps the full colour resolution, for
  editing.

h264-mp4 plays everywhere; h265-mp4 is smaller at the same quality; av1-webm
smaller still and slower to encode; vp9-webm plays in every browser.

Colour: the frames are written as BT.709 YUV in limited (tv) range and tagged
BT.709 (matrix, primaries and transfer), so players show the pixels as they
are. 8-bit formats go through FFmpeg's scaler (bicubic) from the frames
rounded to 8 bits; 10-bit ones are computed with the BT.709 formula from the
float frames (FFmpeg's scaler writes 10-bit white 0.3% too bright). Odd frame
sides are padded to even, as 4:2:0 needs, by repeating the last row or column.

Audio: the first item of the `audio` batch, AAC in mp4 (96 kbit/s per
channel), Opus in webm (64 kbit/s per channel), always cut to the video's
length. An mp4 keeps its index at the front, so the browser plays it while it
loads.

A `crf`, `preset` or `pix_fmt` the codec does not take is an error naming the
widget, before anything is written. A codec whose encoder the PyAV in
ComfyUI's Python lacks is an error naming the missing encoder and listing the
ones it has.

Browsers do not play `yuv444p` or 10-bit H.264 (`h264-mp4` with
`yuv420p10le`): the preview then says it cannot play the file, which is saved
and correct.

### Video Comparer

Two videos in one node with a divider: A fills the node; while the pointer is
over the picture, B is drawn from the left edge up to the pointer. Both are
written side by side into one temporary H.264 file (crf 18, a keyframe every
second), so they play in step from one decoder.

- in: `fps` 24; optional `video_a`, `video_b` (IMAGE), `audio` (played with
  them, cut to their length)
- out: none

Clips of different length are cut to the shorter one, and the node says so;
a smaller frame is scaled to fit the larger and letterboxed on black. Nothing
is saved into the workflow: like Preview Image, the comparison lives with the
run.

## Preprocess nodes

Feed them frames already at the generation size (Load Video loads them so);
the pose images, masks and boxes come out at the size of the frames that went
in.

### Pose Detection

YOLOv10x finds the person, ViTPose-H gives the 133 COCO-WholeBody keypoints on
every frame from the person box, and the pose images are drawn. By default the
box is the detector's raw box, as the official Wan and Kijai preprocess crop
it; Pose Config's experimental `box_window` and `edge_snap` widen it to the
boxes of the frames around it and extend it to a frame edge it nearly touches.

- in: `images`; optional `bboxes` (BBOX, one `(x1, y1, x2, y2)` per frame or
  one for all: the detector is then skipped), `pose_config` (POSE_CONFIG)
- widgets: `body_stick_width` -1,
  `hand_stick_width` -1 (0 leaves that part out, -1 sizes it from the frame),
  `draw_head` true, `draw_threshold` 0.5
- out: `pose_images` (IMAGE), `pose_data` (POSEDATA), `bboxes` (BBOX, the
  person box per frame as everything downstream sees it),
  `key_frame_body_points` (STRING: frame 0's confident body keypoints in the
  KJNodes PointsEditor / easy-sam3 `positive_coords` JSON format)

### Sapiens2 Pose

Pose Detection with Meta's Sapiens2 for the body, the feet and the hands:
the same person box (YOLOv10x, or the connected `bboxes`), the same
`pose_config`, drawing and outputs, but Sapiens2 gives the body, feet and
hand keypoints on its own 1024x768 crop, mapped to COCO-WholeBody by name.
The 68 face keypoints come from ViTPose-H on the same box, so Face Crop, the
guards and `back_view_face` read the same face as after Pose Detection.

- in, widgets and out: Pose Detection's, plus `model` after `images`:
  `5b int8 convrot` (default), `5b bf16`, `1b int8 convrot`, `1b bf16`,
  `0.8b int8 convrot`, `0.8b bf16`, `0.4b int8 convrot`, `0.4b bf16`.
  `int8 convrot` is the int8 ConvRot quantized file, computing in bf16.
- The model is downloaded on first use (see Models).

WanAnimate Preprocess and SCAIL-2 Preprocess have a `pose_model` widget, their
last: `ViTPose-H` (default) runs Pose Detection, `Sapiens2 <model>` runs
Sapiens2 Pose with that model. A workflow saved before the widget existed
loads with `ViTPose-H` and runs as before. SCAIL-2 Preprocess reads it only in
the modes that run the pose.

### Pose Config

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
   of the Wan Animate final, grow 10 and blockify 32; the two parts overlap;
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
   only, as the Wan Animate workflow's final reads them (GrowMaskWithBlur
   expand 10, BlockifyMask 32): a region of her the final loses for 1-8
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
- `prompt_pose` mode reads `pose_data` and the prompt; `bboxes`,
  `positive_coords`, `negative_coords`, `max_objects` and `object_index` are
  ignored with the console line.
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
  Wan Animate workflow grows the raw mask into the final mask (GrowMaskWithBlur
  expand 10, BlockifyMask 32) before any model reads it, so the region has to
  be missing from the final of every frame of the run and hold a whole block
  of it; it is hers (the pose has her body in it next to the run, or the mask
  holds it beyond the run), not a limb that moved away and came back (the
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
2, 1 to 4) is the last widget of the Mask Guard and the WanAnimate Preprocess
Guard: 1 fails a single eye or ear too, also where the mask's outline at the
hair leaves one out; 4 fails only all four, or the nose. A face the pose
model draws on the back of a head lies on the head, which the mask covers.
The default is set on the test clips: a correct (prompt) mask left no head
keypoint out on any frame, and on every frame where the keypoint mask lost
the head, the nose or two or more of the eyes and ears were out.

- Pose Guard: in `pose_data`; out `pose_data` (unchanged), `report`,
  `metrics` (JSON, every measurement per frame), `timeline` (IMAGE)
- Mask Guard: in `mask`, optional `pose_data`, `min_reference_iou` (0.4) and
  `reference_image` (the last input); out `mask` (unchanged),
  `report`, `metrics`, `timeline`. `pose_data` gives the best result: without
  it the guard runs only `mask_fragmented` and `mask_loss` (runs between two
  frames that hold the region), and the report names the checks it did not
  run. Without it a detached piece is the person only when it runs off a side
  of the frame her largest region also runs off (the frame edge cut it from
  her); with it, a piece holding her own drawn keypoints is her.

### WanAnimate Preprocess and WanAnimate Preprocess Guard

The wrappers call the individual nodes, so a wrapper produces exactly what
the chained nodes produce with the same settings.

- **WanAnimate Preprocess** = Pose Detection (or Sapiens2 Pose, by `pose_model`) -> SAM 3.1
  Multiplex Video Track -> Face Crop. Widgets: the drawing widgets, `face_padding`, `mode`,
  `prompt`, `pose_model` (last; `ViTPose-H` default, or `Sapiens2 <model>`); optional
  `pose_config`, `sam3_config`. In `box_keypoint` mode the
  mask is prompted from the pose; in `prompt_pose` mode the pose's drawn
  keypoints add points on the frames where the track lost a limb, and a region
  the mask drops for a few frames adds points inside it (`pose_data`
  is passed whenever the mode is not `prompt`). Outputs: `pose_images`, `face_images`,
  `mask`, `pose_data`, `bboxes`, `key_frame_body_points`, `face_bboxes`.
- **WanAnimate Preprocess Guard** = Pose Guard + Mask Guard with one combined
  report. Inputs `mask`, `pose_data`, the `mask_guard` switch and all thresholds,
  optional `min_reference_iou` (0.5) and `reference_image` (the last input);
  outputs `mask`, `pose_data`, `report`, `metrics`, `timeline`. Its `mask` is
  the final mask the sampler gets: the preprocess mask through
  GrowMaskWithBlur (expand 10) and BlockifyMask (32). BlockifyMask lays its
  grid from each frame's own box, so the grid moves from frame to frame and
  the final's outline is known only to within a block: the mask checks allow
  for that (`mask_attached_leak` allows the neighbouring frames' masks a
  block, the detached pieces allow for the padding, and `mask_loss` counts
  only a region holding a drawn keypoint on every frame of its run; a block
  holding a keypoint inside the raw mask is always on). The Mask Guard on the
  raw mask judges the mask at its own precision, and its `mask_loss` by what
  the final leaves out.

### Reference check

The Mask Guard and the WanAnimate Preprocess Guard check the reference of a
replacement run when `reference_image` is connected: the reference image the
Wan Animate node gets (Load Reference Image's). The guard finds the character
on it with SAM 3.1 Multiplex in prompt mode with the default prompt (`main
person in the foreground`), the call SCAIL-2 Preprocess makes for its
reference, places it as core's Wan Animate node places the reference (a
center crop to the mask's aspect ratio, resized nearest-exact to the mask's
size) and measures it against mask frame 0. The Mask Guard compares raw with
raw; the WanAnimate Preprocess Guard checks the final mask, so it grows and
blockifies the placed character the same way first (GrowMaskWithBlur expand
10, BlockifyMask 32).

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
- **SCAIL-2 Preprocess** = SAM 3.1 Multiplex Video Track (one object) in the
  chosen `mode` on the whole driving video once, and in prompt mode on the
  reference image unless `reference_mask` is connected, then SCAIL-2 Colored
  Mask. Tracking the whole clip once keeps the mask's shape and colour the
  same across the sampler's chunks (the official template re-tracks every
  segment). `mode` is the Video Track's widget (`prompt` default,
  `box_keypoint`, `prompt_pose`), so the mask comes from the chosen mode and
  switching needs no rewiring: in `box_keypoint` and `prompt_pose` the node
  first runs Pose Detection (or Sapiens2 Pose, by `pose_model`) on the driving frames, at its default widgets
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
  `reference_mask`.
- `black_background` (animation mode only): `pose_video` becomes the driving
  video with every pixel outside the person's mask black, as SCAIL-2's
  training pose videos were (zai-org/SCAIL-2 issue #17; SCAIL-Pose's
  `--crop_e2e_mask`), so the driving video's background and camera do not
  reach the result. Off, `pose_video` is the driving video unchanged. In
  replacement mode the result keeps the driving video's background, so
  `black_background` on with `replacement_mode` on is an error, raised
  before any tracking.

### SCAIL-2 Preprocess Guard

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
- Sapiens2 pose: from
  [huggingface.co/beycanai/sapiens2-convrot](https://huggingface.co/beycanai/sapiens2-convrot)
  into `ComfyUI/models/detection/`, the file of the chosen model only
  (`sapiens2_pose_<size>_<bf16|int8_convrot>.safetensors`).
  `scripts/convert_sapiens2.py` writes the bf16 file from the transformers
  checkpoint; the int8 ConvRot file is made from it with convert_to_quant.

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
chunk no matter how long the video is; decoded frames accumulate on CPU, in
one output allocated at `total_frames`. The
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

The optional videos are handed to `WanAnimateToVideo` as they are (a chunk
that reads past the end of a driving input gets its window of them, see Tail
padding); the core
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
- The last chunk decodes only the latent frames `total_frames` needs: the Wan
  VAE decodes causally, so the frames past `total_frames` are sampled with the
  chunk but never decoded.
- If `total_frames` exceeds the pose video, a warning is printed and the
  driving inputs are extended for the remainder (see Tail padding); those
  frames are part of the output.

### Tail padding

Whenever a chunk needs driving frames past the end of an input (the `fit`
snap-up of at most 3 frames, `last_chunk` `full` or `min29`, or
`total_frames` longer than the input) the loop extends the driving inputs: the
pose, and
face and background for Animate, the pose mask for SCAIL-2. Only the frames a
chunk reads are extended, never a copy of a whole input: a chunk that reads
past the end of an input gets its own window of each driving input, extended
past the end, and the core node the offset into that window. The Animate
`character_mask` is never extended: it is cut to the same window, and past its
end the character may be anywhere, so core leaves those mask rows unknown. The
`tail_padding` widget picks how:

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
scipy, safetensors, tqdm and PyAV (`av`, the video nodes' decoder and encoder)
come with ComfyUI, so the video nodes add no requirement. The previews are
plain JavaScript in `web/js/`, which ComfyUI serves; nothing to build. `onnx` and `huggingface_hub`
are needed only for the offline conversion and upload scripts
(`pip install .[dev]`).

## Tests

```
python -m pytest tests
```

`tests/libs/test_chunking.py` covers the length math and
`tests/test_package.py` the node contract of all twenty-two nodes, both without
torch or ComfyUI:
`python -m pytest tests/test_package.py tests/libs/test_chunking.py`.
`tests/pipelines/test_long_video*.py` run the samplers' chunk loop against
stubbed core nodes with real CPU tensors; it is skipped when torch is not
installed. The preprocess tests (`tests/nodes/test_nodes_*.py`,
`tests/pipelines/test_pose.py`, `tests/pipelines/test_sam3_1_multiplex_*.py`,
`tests/pipelines/test_guard*.py`, `tests/models/test_models_*.py`, ...) need
torch and, for most, ComfyUI on the path:
`PYTHONPATH=/path/to/ComfyUI python -m pytest tests`. The video node tests
(`tests/libs/test_video_*.py`, `tests/libs/test_resize.py`,
`tests/pipelines/test_video_input.py`, `tests/nodes/test_video_*.py`) write
small synthetic clips with PyAV; none reads a real clip.

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
| Sapiens2 pose | `sapiens2_pose_*.safetensors` | Sapiens2 License (Meta) |

The person detector's weights (YOLOv10x) are AGPL-3.0. Running them locally
is unaffected; offering a service over a network that runs them (a hosted
workflow, a SaaS) brings AGPL-3.0's network clause into play, which requires
making the corresponding source available to that service's users. With
`bboxes` connected, Pose Detection neither downloads, loads nor runs the
detector.

SAM 3.1 is fetched by ComfyUI from `Comfy-Org/sam3.1` under Meta's SAM
License; this package does not redistribute it.

The Sapiens2 pose weights are Meta's, under the Sapiens2 License; the
converted files are fetched from `beycanai/sapiens2-convrot`, whose
`LICENSE.md` carries it.
