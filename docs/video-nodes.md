# Video nodes

Load a video at the generation size of the model it is for, fit the reference
image to it, save the result, compare two videos. Decoding and encoding use
PyAV, which comes with ComfyUI. Load Video, Save Video and the Video Comparer
play in the node: the pack's player, drawn on the node, with a play button, a
seek bar and, when the clip has sound, a mute button. Nothing plays until
asked, the clip loops, and the node keeps the size you give it. A press on
the seek bar pauses the clip and seeks, a drag that starts on it scrubs
(the picture stays on screen while each seek lands), and the clip plays on
from there on release when it was playing; the seek bar stops
short of the node's bottom corners, which resize it. Labels are cut to the
node's width.

## Load Video

Loads a video one frame at a time, cropped and resized to the model's
generation size (or, at `resolution` `source`, cut at the video's own size)
straight into the output, which is allocated once at its final frame count:
the full-resolution clip never sits in memory.

- in: `video` (a video file of ComfyUI's input folder; the node's `choose
  video to upload` button, or a video file dropped on the node, uploads one
  there and selects it; a video dragged from the queue or the media assets
  panel is selected where it is, as `name [output]` or `name [temp]`)
- widgets: `model` `Wan` (or `SCAIL`, `None`), `resolution` `720p`, `orientation` `auto`,
  `force_fps` (empty), `start_frame` 1, `frame_count` (empty), `precision`
  `fp16` (or `fp32`)
- out: `images` (IMAGE), `audio` (AUDIO of the loaded range; none when the
  file has no audio), `video_info` (BCV_VIDEO_INFO, for Get Video Info and
  Load Reference Image)

The sizes are the models' generation sizes, portrait (width x height); `None`
is no model, with Conform Video's sizes:

| `model` | `resolution` | Size | Frames |
|---------|--------------|------|--------|
| `Wan`   | `480p`, `720p`, `source` | 480 x 832, 720 x 1280, the video's own cut to /16 | 4n+1 |
| `SCAIL` | `512p`, `704p`, `source` | 512 x 896, 704 x 1280, the video's own cut to /32 | 4n+1 |
| `None`  | `480p`, `720p`, `1080p`, `source` | 480 x 854, 720 x 1280, 1080 x 1920, the video's own | every frame |

`resolution` offers the chosen model's labels; a label of another model is
an error. `orientation` `auto` is portrait when the video is taller than wide,
otherwise landscape (a square video is landscape); `landscape` and `portrait`
force one. Landscape swaps width and height. The frame is cut centred to the
size's aspect ratio, then resized with lanczos.

`source` keeps the video's own pixels, never resized. With an `orientation`
opposite to the video's, the frame is cut centred to that orientation's
aspect, keeping the short side (1920 x 1080 as portrait: 608 x 1080);
orientation never rotates the picture. With `Wan` or `SCAIL` each side is
then cut, centred, down to the model's grid (Wan 16, SCAIL 32, the steps
their core nodes take: 720 x 405 is 720 x 400 for Wan, 704 x 384 for SCAIL);
a video smaller than the grid is an error. `None` has no grid, so its
`source` is the video's own size, and no frame rule: every frame of the range
is loaded.

Which frames are loaded:

- `force_fps`: empty keeps the video's own frame rate as it is (a 29.97 fps
  video loads at 29.97). A number above 0 is the loaded frame rate exactly as
  typed (`30` loads at 30, also from a 29.97 fps video), and the frames are
  the video's real frames on that rate's time grid: output frame m is the
  first frame at or after m / `force_fps` seconds. Below the video's rate
  frames are dropped (30 to 24 drops one frame in five), above it frames are
  repeated (16 to 32: the first frame once, then every frame twice; 29.97 to
  30: one frame in a thousand shows twice); a frame is never blended or
  interpolated. A rate a hair below the video's drops a frame early: 29.97 on
  a 30000/1001 clip skips the second frame, then keeps every frame for about
  ten thousand. Leave `force_fps` empty to keep every frame at the video's
  own rate.
- `start_frame` (counted from 1) and `frame_count` (empty: every frame from
  `start_frame` on) count the frames `force_fps` kept. A `start_frame` past
  the last frame, or a `frame_count` that runs past it, is an error that says
  how many frames there are and what to set.
- The count is then cut down to the model's 4n+1 (a 100-frame range loads 97;
  with `None`, 100): the range's last frames are dropped, the audio ends with
  the loaded frames, and the console names the cut ("100 frames -> 97 for
  Wan's 4n+1: the last 3 dropped, audio cut to match").
- A `force_fps` or a `frame_count` that is not a number is an error.

`precision`: `fp16`, the default, stores the frames as float16, half the RAM
of an `fp32` load (901 frames at 720 x 1280: 4.98 GB instead of 9.96 GB).
`fp32` stores them as float32, as every IMAGE. float16 keeps every 8-bit
level of the video exactly (its step near 1.0 is an eighth of a level).
Core resizes and scales a clip in the dtype it gets, and OpenCV refuses
float16, so the pack's nodes read an `fp16` clip back as the float32 values
an `fp32` load holds, a frame at a time (the samplers a chunk's window at a
time, before core sees it), never the whole clip at once, and compute what
they compute on `fp32`. What they make from an `fp16` clip is float16 too:
the pose images, the SAM 3.1 Multiplex mask, the final mask, `bg_images` and
SCAIL-2's colored driving mask (8-bit levels or 0 / 1, exact in float16), and
the samplers' output (core's decoded frames rounded to float16; every chunk
is still seeded with them as decoded, so the generation is `fp32`'s). Face
Crop's crops stay float32: resized, they are not 8-bit levels. Save Video
writes the frames as they come: from Load Video's `fp16` clip 8-bit output is
the same and 10-bit output can be one 10-bit step off; from the samplers'
float16 output about 2% of the values sit close enough to a rounding edge to
come out one 8-bit level off. Nodes of other packs, core's included, get the
float16 clips as they are: pick `fp32` when such a node needs float32 frames.

Colour: YUV is converted to RGB with the stream's own colour matrix and range,
as the file is tagged; an untagged stream is read as BT.601 limited range,
FFmpeg's default. A video stored rotated is turned upright.

Audio: the first audio stream, from `start_frame`'s time for the loaded
frames' duration, both on the loaded rate's time grid: (`start_frame` - 1) /
`loaded_fps` seconds on, for `loaded_frame_count` / `loaded_fps` seconds. The
audio is the video's own, never stretched, so it stays in sync whether
`force_fps` drops or repeats frames and after the 4n+1 cut.

`video_info` holds `audio`, the `audio` output itself (not a copy), then the
`model` (`None` for no model), the `resolution` and the resolved
`orientation`, then the source's `source_fps`, `source_frame_count`,
`source_duration`, `source_width` and `source_height` (as displayed), then
the loaded batch's `loaded_fps` (`force_fps` as typed, or the source's rate),
`loaded_frame_count`, `loaded_duration`, `loaded_width` and `loaded_height`.

The preview plays the source file in the browser as the loader will take it:
sampled at `force_fps`, looping over `start_frame` / `frame_count`, the part
the crop cuts away dimmed. The browser's frame at a tick can be one off the
loader's. What the loader will load comes from the server, which answers from
the loader's own code (a probe of the file, no full decode): the label shows
the exact frame count, rate and size, and a value the loader would reject
shows the loader's own error over the picture. An empty `force_fps` shows the
source's frame rate greyed in the widget, an empty `frame_count` the frames
from `start_frame` on at the kept rate (before the 4n+1 cut); the widget's
value stays empty.

## Get Video Info

- in: `video_info`
- out: its 14 fields in the order above: `audio` (AUDIO), `model`,
  `resolution`, `orientation` (STRING), `source_fps` (FLOAT),
  `source_frame_count` (INT), `source_duration` (FLOAT), `source_width`,
  `source_height` (INT), `loaded_fps` (FLOAT), `loaded_frame_count` (INT),
  `loaded_duration` (FLOAT), `loaded_width`, `loaded_height` (INT).
  `loaded_fps` is the rate to give Save Video.

Wire Save Video's and the Video Comparer's `audio` from here, not from Load
Video: it is Load Video's audio itself, and ComfyUI keeps every output of a
node, the frames included, until all the nodes linked to it have run. Save
Video and the Video Comparer run last, so a link from Load Video keeps its
frames in memory to the end of the run; Get Video Info runs right after Load
Video, and Load Video's frames go as soon as their last reader is done (when
ComfyUI's cache lets them go: under RAM pressure, or with `--cache-none`).

The `source_` and `loaded_` outputs are drawn in a colour per group (source
amber, loaded sky blue), with a thin line and the group's name above each
group.

## Load Reference Image

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

## Conform Video

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

## Save Video

Writes the frames as a video file, one frame at a time.

- in: `images`; optional `audio` (muxed in and cut to the video's length;
  from Get Video Info, see there)
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

## Video Comparer

Two videos in one node with a divider: A fills the node; while the pointer is
over the picture, B is drawn from the left edge up to the pointer. Both are
written side by side into one temporary H.264 file (crf 18, a keyframe every
second), so they play in step from one decoder.

- in: `fps` 24; optional `video_a`, `video_b` (IMAGE), `audio` (played with
  them, cut to their length; from Get Video Info, see there)
- out: none

Clips of different length are cut to the shorter one, and the node says so;
a smaller frame is scaled to fit the larger and letterboxed on black. Nothing
is saved into the workflow: like Preview Image, the comparison lives with the
run.
