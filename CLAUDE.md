# ComfyUI-BCVideoNodes

## What this is

ComfyUI custom nodes for Wan Animate and SCAIL-2: the preprocess (pose, SAM 3.1 Multiplex person
mask, face crops, pose and mask guards, SCAIL-2 colored masks and their guard), three
long-video samplers, and the video nodes (Load Video, Get Video Info, Load Reference Image, Conform
Video, Save Video, Video Comparer, with their player in `web/js/`). The 22 node keys are locked,
and so is everything ComfyUI reads from a node (inputs, types, order, defaults, ranges, return
types, categories, display names), because saved workflows depend on it: a change to it needs
the owner. `tests/test_package.py` and the gate's
`NODE_KEYS` pin the keys, their order, display names and categories.

SAM naming: the SAM nodes and their code are named after the one model they run, SAM 3.1
Multiplex. Display names read "SAM 3.1 Multiplex ...", modules `sam3_1_multiplex`, constants
`SAM3_1_MULTIPLEX_*`, classes `SAM3_1Multiplex*` (in identifiers the dot of 3.1 is an underscore).
The node keys `BCVSAM3*`, the `SAM3_CONFIG` type and the input names keep the old spelling,
because saved workflows use them.

## Layout

Four layers, `nodes -> pipelines -> models -> libs`:

- `nodes/`: the ComfyUI surface only (INPUT_TYPES, tooltips, DESCRIPTION, ComfyUI types in and
  out, one pipeline call). One file per domain.
- `pipelines/`: flows that combine models and libs, and their config dataclasses.
- `models/`: one package per model (architecture, wrapper, model-specific pre/post-processing,
  adapters over ComfyUI core models), plus `models/common/`.
- `libs/`: model-independent code.

`web/js/` is the frontend, outside the layers: plain ES modules ComfyUI serves from the root
`WEB_DIRECTORY`, no build step.

```
__init__.py          registration only: the 22 node classes, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS,
                     WEB_DIRECTORY = "./web", the link stamp of the unused heavy outputs (register_link_stamp),
                     Load Video's plan route (register_plan_route)
nodes/               common (the PREPROCESS and VIDEO categories, _config, _ConfigNode, prompt_server), unused_outputs (the
                     unused-heavy-outputs helper: LinkStamp, register_link_stamp, heavy_wanted, wants,
                     drop_unwanted, drop_unlinked_heavy), sampler, pose (Pose Detection, Pose Config,
                     Sapiens2 Pose), sam3_1_multiplex (and track_reference: the character on a reference
                     image), face, guard,
                     preprocess (the two WanAnimate wrappers, composed of the nodes above),
                     scail2 (SCAIL-2 Colored Mask, the SCAIL-2 Preprocess wrapper, SCAIL-2 Preprocess Guard),
                     video_input (Load Video, Get Video Info, Load Reference Image, Conform Video),
                     video_output (Save Video, Video Comparer)
pipelines/           long_video (the chunk loop), pose, sapiens2_pose (Sapiens2 Pose: Sapiens2 body, feet and
                     hands, ViTPose-H face), face, scail2 (the colored masks, the driving video
                     on black), guard/ (config, common, pose, mask, reference, report, timeline, combine,
                     scail2), sam3_1_multiplex/ (config, prompt, pose, prompt_pose, refine, track: the entry
                     the node calls), video_input (Load Video, Load Reference Image, Conform Video)
models/              __init__ (imports the model packages in registration order),
                     common/ (registry, interfaces, checkpoint, download, loader, wrapper, blocks, pose_input,
                     core_nodes, animate), vitpose/, yolo/, sapiens2/ (net, wrapper, decode, keypoints),
                     sam3_1_multiplex/ (adapter, loader,
                     postprocess), wan_animate/, wan_animate2/, scail2/
libs/                log, bbox, keypoints, mask, chunking, sigmas, video (tail padding), color, config_widgets,
                     pose_data, draw_rules (the Pose Config draw rules: parts left out of the pose images),
                     video_sizes (the model table: sizes, frame rule; the orientation rule; the Conform
                     Video ladder), video_info (the VideoInfo TypedDict), resize (the one fit function: crop
                     or pad, a frame into a preallocated output), video_decode (PyAV decode, frame
                     selection, audio), video_encode (the codec table, the writer), video_compare (the
                     Video Comparer's side by side), pose_utils/ (vendored, with its LICENSE)
web/js/              player.js (the player the previews share), load_video.js, save_video.js, video_comparer.js
scripts/             offline model conversion and upload (ComfyUI-free)
web/js/              unused_outputs.js (the toast when the unused-outputs saving is off for a run)
tests/               tests/{nodes,pipelines,models,libs}/ mirror the layers; the gate, the layer test
```

## The layer rule

- Imports go one way: `nodes -> pipelines -> models -> libs`, and within a layer. `nodes -> nodes`
  is allowed (`nodes/common.py`, the WanAnimate and SCAIL-2 wrappers composing the other nodes,
  and the Mask Guard and the WanAnimate Preprocess Guard calling
  `nodes/sam3_1_multiplex.track_reference` for their reference check, as SCAIL-2 Preprocess does
  for its reference mask).
- A model package imports only itself, `models/common/` and `libs/`. Model packages never import
  each other, and `models/common/` never imports a model package. Inside `models/`, only
  `models/__init__.py` imports the model packages: that is the registration list.
- The layers above `models/`, and `scripts/`, may import a model package directly: the SAM 3.1
  Multiplex pipeline imports `models/sam3_1_multiplex/` (SAM is not registered), and
  `scripts/convert_models.py` imports the ViTPose decoder. No pipeline imports an
  animate adapter package but the SCAIL-2 guard (`pipelines/guard/scail2.py`), which reads the
  colored-mask conventions (`ON`, `mask_convention`) of `models/scail2/adapter.py`; the sampler
  pipeline reaches the adapters through the registry.
- From the pack, `scripts/` import only `models/` and `libs/`.
- All imports between pack modules are relative. There are exactly two absolute `import nodes`,
  in `nodes/sampler.py` and `models/common/core_nodes.py`, and both mean ComfyUI core.
- `tests/test_layers.py` walks every import, lazy ones included, and fails on a forbidden edge, a
  sideways model import, an absolute import of `pipelines`, `models` or `libs`, an absolute
  `nodes` outside the two core sites, an import through a test alias (`bcvideonodes`, `walong`)
  outside `scripts/`, an import by string, or a module-level import cycle.

Underscore names are package-private: a sibling module of the same package may import them (for
example `_config` and `_ConfigNode` from `nodes/common.py`, or `_thresholds` and `_finish` inside
`pipelines/guard/`). A name that another package imports has no underscore. Tests may still read
and patch underscore names through the `Names` tables.

## Where things go

| what | where |
|---|---|
| node surface: inputs, tooltips, DESCRIPTION, type conversion | `nodes/<domain>.py` |
| a flow that combines models | `pipelines/` |
| an nn.Module, its wrapper, its pre/post-processing | `models/<name>/` |
| model-independent code | `libs/` |
| a config dataclass | next to its pipeline; the node generates its widgets from it (`libs/config_widgets.py`) |
| a node's frontend: its preview, widgets that follow another widget | `web/js/<node>.js`, on `web/js/player.js` |

## Unused heavy outputs

A whole-clip IMAGE or MASK output that no node consumes is not kept in ComfyUI's output cache.
ComfyUI's cache key holds a node's own inputs and its ancestors only, so an on_prompt handler
(`LinkStamp` in `nodes/unused_outputs.py`, registered by the root `__init__`) writes the linked
heavy outputs of each heavy node into its inputs as `bcv_linked_heavy` before validation; every
input key is part of the cache key, and an undeclared one never reaches the function. No stamp (no
server, a direct executor call, a node a wrapper calls): every output full. A link the stamp
missed: full, with a warning. Another pack's on_prompt handler registered after ours turns the
saving off for that prompt, with a console line and a toast; the owner chose that over reordering
the handlers.

- A whole-clip IMAGE or MASK output the node makes, among two or more outputs (or on an output
  node), is a heavy output: list it in `HEAVY_OUTPUTS` (names from `RETURN_NAMES`), add
  `"hidden": dict(LINK_INPUTS)` to `INPUT_TYPES` (`prompt_graph`, `unique_id`; not `prompt`, which
  the preprocess wrappers' SAM widget is called), read `wanted = heavy_wanted(type(self),
  prompt_graph, unique_id)` and return through `drop_unwanted(type(self), outputs, wanted)`.
- When the output is a step of its own that no other output reads, pass `wants(wanted, name)` down
  so the step does not run (the pose images, the face crops, the driving colored mask, the driving
  video on black, the WanAnimate SAM track); when another output needs the step (SCAIL-2
  Preprocess `mask`, Load Video `images`, the samplers' `images`), it is only dropped at return.
- A pass-through (the input tensor itself, the guards' masks), a one-frame output (a reference
  mask, a timeline) and the output of a single-output node that is not an output node (it runs
  only when that output is linked) are not heavy.
- A wrapper passes its own `wanted` to the nodes it calls, as the keyword-only `wanted=` of their
  methods (its outputs carry their names); ComfyUI never passes it. A wrapper also asks for none of
  an inner output it discards (SCAIL-2 Preprocess: `wanted=set()` to Pose Detection, so no pose
  images are drawn).
- Tests: `tests/nodes/test_unused_outputs.py` (its `HEAVY` table, the dropped output empty with the
  full one's dtype and trailing shape, the linked ones equal, the skipped step not run).

## How to add a model

- Create `models/<name>/` with `net.py` (the nn.Module), `wrapper.py` (a `NativeModel` subclass
  implementing `PoseEstimator` or `PersonDetector` of `models/common/interfaces.py`) and
  `decode.py` if it needs one.
- Register it in its `__init__.py`, after all of its imports: its architecture
  (`registry.register("architecture", ...)`) and its entry in the `pose_estimator` or
  `person_detector` family with its model file (see `models/vitpose/__init__.py`).
- Add the package to the import list in `models/__init__.py`. The loader loads the pose
  estimator and the person detector by their registry names (`models/common/loader.py`:
  `POSE_ESTIMATOR`, `DETECTOR`); offering a choice between models means a node widget, which
  changes the node surface and needs the owner's word.
- Keep its module-level imports to torch, numpy and the standard library.
- List its modules in `CHECK3_MODULES` of `tests/test_import_time.py` (with an `ALLOWED` row if
  one may pull in a heavy module): the gate fails on a layer module that is not listed there.
- Add seeded tiny-config tests (see `tests/models/`).
- A new Animate conditioning node means an `AnimateAdapter` subclass (`models/common/animate.py`)
  in its own model package, registered in the `animate` family under its core node id (and the
  package added to `models/__init__.py`), plus a `_LongVideoSampler` subclass in
  `nodes/sampler.py`, registered in the root `__init__.py`. A new node key changes the surface,
  so it needs the owner's word, and a row in `tests/test_package.py` and in the gate's
  `NODE_KEYS`.
- The chunk loop (`pipelines/long_video.py`) knows the core node only through the adapter. The
  base class is the Wan Animate contract; a node with another contract overrides what differs
  (`models/scail2/adapter.py` overrides all of them):
  - `OUTPUTS` / `UPDATE_HINT`: the fewest outputs the core node must return, and the error hint;
  - `HELD_VIDEOS`: the videos the core node seeks by offset, extended past their end up to the
    plan's reach as the samplers' `tail_padding` widget says (below), but never as a whole: a
    chunk that reads past the end of one gets the window it reads, the held videos extended
    there, and the core node the offset into that window;
  - `SEEKED_VIDEOS`: the other videos the core node seeks by offset, never extended: on such a
    chunk they are cut to the same window, never padded (Wan Animate's `character_mask`; a
    single frame is not seeked, core repeats it over the chunk);
  - the per-chunk hooks below (`continuation`, `chunk_inputs`, `after_animate`, `anchor_region`)
    get the chunk's inputs and its offset as the core node gets them: on a windowed chunk the
    window and the offset into it, so a hook indexes what it is handed;
  - `prepare(animate_cls, animate_inputs, reference_image, width, height, frames_per_chunk)`:
    validate, rename or pop the node's own inputs, encode what is encoded once per run; returns
    the overlap;
  - `check_videos(pose_video, animate_inputs)`: checks between the videos, before any is held;
  - `patch_model`: model patches, once per run;
  - `continuation(anchor, offset)`: the core call's chaining inputs (default `continue_motion`,
    `video_frame_offset`; `offset` is the chunk's video_frame_offset), spliced before
    `chunk_inputs`;
  - `chunk_inputs`: per-chunk inputs; `after_animate`: conditioning repairs before sampling;
  - `unpack(outputs, anchor)`: the outputs as (positive, negative, latent, trim_latent,
    trim_image, video_frame_offset);
  - `anchor_region(first, length, height, width, animate_inputs)`: where the colour anchor
    measures and corrects a chunk whose decoded frames show driving frames `first` onwards,
    [length, height, width] weights, or None (the default) for the whole frame; the character
    only where core re-feeds the background every chunk (replacement mode);
  - `after_chunk(index)`: called once per chunk after it is sampled and decoded (no-op default).
  A change to a default is a change to both Wan Animate samplers.
- The chunk length policy is not the adapter's: it is the samplers' `last_chunk` widget,
  `libs/chunking.LAST_CHUNK` (`fit`: the last chunk fitted to what is left; `full`: every chunk
  full length, the output cut to `total_frames`; `min29`: `fit`, but the last chunk never under 29
  frames, capped at `frames_per_chunk`, the official Wan-Animate-2 tail rule), with the reason the hold log line gives. The node's default is its `DEFAULT_LAST_CHUNK` (`fit` for both Wan
  Animate samplers, `full` for SCAIL-2). The adapter gets the value (`self.last_chunk`) for its
  log lines only.
- How `HELD_VIDEOS` are extended is not the adapter's either: it is the samplers'
  `tail_padding` widget (the last required widget, after `last_chunk`), `libs/video.TAIL_PADDING`
  (`last_frame`: `hold_last`, the default of all three; `ping_pong`: the official Wan Animate
  padding, backwards from the end), with the words the hold log line names it by. The functions
  take `(video, start, stop)` and give frames `start` to `stop` of the video extended past its
  end: a view when they lie inside it, else one gather of just those frames. The Wan Animate
  `character_mask` is never extended.
- The loop writes each chunk's frames into one output on the CPU, allocated once at
  `total_frames` (no list joined at the end), and seeds the next chunk from that output's last
  frames. The last chunk decodes only the latent frames `total_frames` needs: the Wan VAE
  decodes causally, so the latent frames that decode only to frames past it are left out
  instead of decoded and cut.
- The colour anchor is not the adapter's either, beyond its region: it is the samplers'
  `color_anchor_strength` widget (optional, the last widget, after the `sigmas_override` link;
  default 0 = off, and 0 skips the code path), `libs/color.py`.

## Coding style

- Before writing new code, search the pack for code that already does the same work (grep for
  the operation, not only the name). If it exists, call it. If the same code would end up in
  two places, move it into one function and call that from both. Never write a second copy.
- No spaghetti, no duplication; clean, readable, debuggable. Small functions with one job.
- Explicit data contracts as TypedDict annotations (`libs/pose_data.py`, the guard rows and
  reference records in `pipelines/guard/common.py`, the SAM counts in
  `pipelines/sam3_1_multiplex/{prompt,pose,prompt_pose,refine}.py`, `libs/video_info.py`); the
  values stay plain dicts.
- The owner's extraction rule: a new function only if (a) identical code already lives in 2+
  places (reduce it to one) or (b) it is likely (~70-80%) to be reused by future nodes of this
  repo's kind. Otherwise keep it inline. Near-copies that differ in any detail stay separate.
- Errors say what to do. No silent defaults.
- Behaviour changes need the owner.

## Optimization principles

- Speed and RAM are equal priorities. No RAM saving that makes generation slower.
- Bit-exact output is not required, but the output never drifts from the origin. A departure
  from the origin is a widget setting, never hidden behaviour.
- A port of a third-party node does the same job in our style: preallocated outputs, no
  list -> stack/cat, no clones of read-only inputs, no leaks. Read the original first and never
  reproduce its bugs. Take the logic only: no import of, dependency on or reference to the
  original.
- Precision is decided per tensor, by measurement, never globally.
- Unused heavy outputs are not kept: see Unused heavy outputs above.
- A node never resizes itself to its content; previews and widgets scale to the node.
- Values that can differ between uses are widgets, not constants.

## Imports

- The lazy rule: module level imports only torch, numpy and the standard library. cv2, scipy, PIL,
  safetensors, torchvision, tqdm, `av` (PyAV), `folder_paths`, `node_helpers`, `comfy_api` and
  `comfy.*` are imported inside the function that uses them. Node modules import their pipeline
  or model module inside the method.
- Exactly three exceptions:
  - E1: `models/common/download.py` imports `folder_paths` and registers ComfyUI's `detection`
    model folder at import. The two loaders (`models/common/loader.py`,
    `models/sam3_1_multiplex/loader.py`) import it at module level, so the registration happens
    when a SAM node's INPUT_TYPES or Pose Detection first runs.
  - E2: the vendored `libs/pose_utils/*` import cv2 at module level. Every consumer imports them
    inside functions.
  - E3: `pipelines/sam3_1_multiplex/track.py` imports core's SAM 3.1 tracker names at module level,
    above every relative import, so the SAM nodes' INPUT_TYPES fail on a ComfyUI without the
    tracker, before the detection folder is registered.
  - `pipelines/sam3_1_multiplex/__init__.py` and `models/sam3_1_multiplex/__init__.py` stay empty,
    so importing either package triggers neither E1 nor E3.
- The gate, with the ComfyUI venv's Python: `PYTHONPATH=/path/to/ComfyUI python
  tests/test_import_time.py`. A standalone script (pytest does not collect it). It checks the
  package import (under 0.1 s, no heavy module, the 22 keys in order), each node module, each
  module against its allowed heavy set, and the ComfyUI-free set.
- Code outside the pack binds the repo root as a package and imports through it: tests and
  `scripts/` as `bcvideonodes` (`tests/conftest.py` runs the root `__init__` as ComfyUI does; the
  scripts do not), the sampler tests' `node_module` fixture as `walong`, and the owner's A/B test
  scripts the same way. Nobody puts the repo root on `sys.path` to import pack modules.

## Tests and gates

- The suite, with the ComfyUI venv's Python: `PYTHONPATH=/path/to/ComfyUI python -m pytest tests`.
  Expect every test to pass with 0 skipped; a skip means ComfyUI or a dependency is not on the
  path. A bare `pytest` from the repo root fails by design: `tests/pytest.ini` anchors pytest's
  rootdir at `tests/`, so pytest never imports the root `__init__`.
- Without torch or ComfyUI: `python -m pytest tests/test_package.py tests/libs/test_chunking.py`.
- Then the gate (above). The layer test, `tests/test_layers.py`, runs in the suite.
- The tests state behaviour: hand-written expected values, and comparisons with an external
  reference (the vendored or official implementation, core ComfyUI, a formula written out in the
  test). None compares the pack with a recorded fingerprint of its own earlier output, so an
  intended behaviour change updates the tests that state the old behaviour, in the same commit.
- Test bodies reach pack names through the `Names` tables (`tests/names.py`, each domain's in
  `tests/*_fakes.py`). Moving code changes table rows, never test bodies.
- `tests/conftest.py` imports PyAV before anything imports cv2, as ComfyUI does (its `nodes.py`
  imports `comfy_api`, which imports `av`, before any custom node). The opencv-python of the
  ComfyUI venv bundles its own FFmpeg; loaded first, it makes PyAV's x265 segfault at the slow
  presets.
- `tests/video_input_fakes.py` and `tests/video_output_fakes.py` write small synthetic clips with
  PyAV into the test's tmp dir (lossless FFV1 with PCM audio for the loader, so a decoded pixel is
  the pixel written; colour patches on a gradient that shifts a pixel per frame, for the
  encoder). No test reads a real clip.
- `tests/nodes/test_unused_outputs_runtime.py` runs ComfyUI's PromptServer, validate_prompt and
  PromptExecutor in a process of its own (`tests/unused_outputs_runtime.py`): under `python -m
  pytest` the repo root is on sys.path and the pack's `nodes/` hides ComfyUI's `nodes.py`, which
  the executor imports. The pack is bound there from its `__init__` after the server exists, as
  ComfyUI loads it, so its root registers the link stamp.

- `tests/nodes/test_video_input_route.py` states Load Video's plan route: its answer equals what
  `load_video` loads and raises, and the file stays inside the input folder.

## Contracts at the boundary

- POSEDATA is a plain dict at runtime everywhere. Its TypedDicts (`PoseData`, `PoseMeta`,
  `Detection`) are annotations only, and `tests/pipelines/test_pose_data.py` ties them to the keys
  the pose pipeline writes. The guard rows and the SAM counts work the same way
  (`test_guard_rows.py`, `test_sam3_1_multiplex_counts.py`).
- BCV_VIDEO_INFO is a plain dict at runtime too: `libs/video_info.VideoInfo` is its annotation
  and fixes its key order, which is Get Video Info's output order.
- What the frontend reads from the node definitions: Load Video's `resolution` input carries
  `bcv_sizes` (model -> label -> [width, height], portrait), from `libs/video_sizes.MODELS`, so
  the table has no second copy in JS;
  Save Video's `codec` input carries `bcv_codecs` (codec -> the values of its crf, preset and
  pix_fmt, `libs/video_encode.widget_values`).
- Load Video's plan route, `GET /bcvideonodes/load_video/plan` (`nodes/video_input.py`): the seven
  widget values in, `pipelines/video_input.LoadPreview` out ({source {fps, frames, width,
  height, start, audio}, info (VideoInfo, what the loader would output), available, error (the
  loader's own message or null)}). It answers from the loader's own functions, so the preview
  and the loader cannot drift; the file is resolved only inside ComfyUI's input folder.
- The codec names (`h264-mp4`, `h265-mp4`, `av1-webm`, `vp9-webm`) are stored in saved
  workflows: add codecs, never rename or remove one.
- The ui payload the players read, `bcv_video` (`UI_KEY` in `nodes/video_output.py`): a list of
  one entry {filename, subfolder, type, fps, frames, audio, then `format` for Save Video or
  `sides` for the Video Comparer, whose `frames` is per side}.
- Locked, because code outside the repo reads them: the guard metrics JSON (the SCAIL-2 guard
  writes a record of its own, `"guard": "scail2"`, beside the pose and mask records; with a
  `reference_image` connected, the mask record and the combined record also carry `"reference"`,
  before `"frames"` (area, cropped, iou_first_frame, scale_first_frame, flags), and
  `min_reference_iou` in their thresholds: the Mask Guard's default 0.4 on the raw mask, the
  WanAnimate Preprocess Guard's 0.5 on the final mask; added keys, nothing renamed); the log
  format the owner's A/B test scripts parse (the `BCVideoNodes` logger, its `[BCVideoNodes]`
  prefix and the " done in " / " failed after " step lines in `libs/log.py`); the model file format
  (`models/common/checkpoint.py`); and `LOGITS_SINK` in `pipelines/sam3_1_multiplex/track.py`,
  where the A/B dump node installs its sink. The A/B test scripts and the A/B dump node are both
  outside the repo.
- Locked for the unused heavy outputs: the stamp key `bcv_linked_heavy` (part of every heavy node's
  cache key), the `bcvideonodes.unused_outputs` event `web/js/unused_outputs.js` listens to, and the
  `bc_link_stamp` marker on the stamping handler, which ComfyUI-BCNodes' handler reads to leave ours
  out of "another pack" (and ours reads on its).

## Closed decisions

Do not reopen or "improve" them. They live in the owner's closed-decision documents (outside the
repo: the plan, the guard, pose-process and mask-process specs, and the review report whose open
items are decided elsewhere). In short: the guard judges pose and mask only and counts at
`draw_threshold`; the guard's warning set (the owner reopened both once, for one warning:
`reference_misaligned` of the Mask Guard and the WanAnimate Preprocess Guard, with their
optional `reference_image`); the pose models are ViTPose-H and Sapiens2; the two remaining SAM A/B switches (`anchor_matching`, `unmatched_counting`, for
multi-person) default to "ours" and are bit-identical there; features the owner did not adopt
(the ViTPose flip test, the other SAM A/B switches, the M4 mask-threshold options) are removed;
multi-person is phase 2; a failed download keeps its `.part` file; Load Video decodes with PyAV,
YUV -> RGB with the stream's own colour tags (cv2 ignores them); `force_fps` only lowers the
frame rate (real frames kept or dropped, never repeated); the video player is not a node (it lives
in the Load Video, Save Video and Video Comparer previews).

## Roadmap

- SCAIL-2 pose-driven mode (planned, not started): the SCAIL-Pose pipeline
  (https://github.com/zai-org/SCAIL-Pose/tree/519c7f54cb972e7f92684213b7ef6c3e05a8f3b2): SAM3
  per person, NLF `nlf_l_multi_0.3.2` 3D pose, 3D cylinder render plus DWPose hands and face in
  2D, and the skeleton in the person's colour on black as the driving mask. The pack runs
  end-to-end mode only (the raw driving video as the pose input). When it is built it follows the
  layers: the NLF model as its own `models/` package, the render in `pipelines/`, new preprocess
  nodes; the SCAIL-2 sampler needs no change (it takes any pose video and colored mask).
- Multi-person: phase 2 (see Closed decisions).

## Git and process

- The owner's global rules (his user-level Claude configuration) apply: no commits on `main`,
  work on `local-*` worker branches, no push.
- The A/B test scripts are outside the repo and not under git: back them up before editing them.

## Licences

- The code is MIT (`LICENSE`).
- The files vendored from the Alibaba Wan team's WanAnimate preprocess are Apache-2.0
  (`libs/pose_utils/LICENSE`) and keep `# Copyright 2024-2025 The Alibaba Wan Team Authors. All
  rights reserved.` as their first line: `libs/pose_utils/pose2d_utils.py`,
  `libs/pose_utils/human_visualization.py`, `pipelines/face.py`, `models/common/wrapper.py`,
  `models/{vitpose,yolo}/wrapper.py`, `models/vitpose/decode.py`.
- The model weights keep their own licences, listed in the README.
