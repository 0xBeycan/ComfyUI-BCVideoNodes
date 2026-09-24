# ComfyUI-BCVideoNodes

## What this is

ComfyUI custom nodes for Wan Animate and SCAIL-2: the preprocess (pose, SAM 3.1 Multiplex person
mask, face crops, pose and mask guards, SCAIL-2 colored masks) and three long-video samplers. The
14 node keys are locked, and so is everything ComfyUI reads from a node (inputs, types, order,
defaults, ranges, return types, categories, display names): `tests/nodes/test_surface_golden.py`
(G2) pins that surface.

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

```
__init__.py          registration only: the 14 node classes, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
nodes/               common (category, _config, _ConfigNode), sampler, pose, sam3_1_multiplex, face, guard,
                     preprocess (the two WanAnimate wrappers, composed of the nodes above),
                     scail2 (SCAIL-2 Colored Mask, and the SCAIL-2 Preprocess wrapper)
pipelines/           long_video (the chunk loop), pose, face, scail2 (the colored masks),
                     guard/ (config, common, pose, mask, report, timeline, combine),
                     sam3_1_multiplex/ (config, prompt, pose, track: the entry the node calls)
models/              __init__ (imports the model packages in registration order),
                     common/ (registry, interfaces, checkpoint, download, loader, wrapper, blocks, pose_input,
                     core_nodes, animate), vitpose/, rtmw/, yolo/, sam3_1_multiplex/ (adapter, loader,
                     postprocess), wan_animate/, wan_animate2/, scail2/
libs/                log, bbox, keypoints, temporal, mask, chunking, sigmas, video, config_widgets, pose_data,
                     pose_utils/ (vendored, with its LICENSE)
scripts/             offline model conversion and upload (ComfyUI-free)
tests/               tests/{nodes,pipelines,models,libs}/ mirror the layers; the gate, the layer test, goldens/
```

## The layer rule

- Imports go one way: `nodes -> pipelines -> models -> libs`, and within a layer. `nodes -> nodes`
  is allowed (`nodes/common.py`, and the WanAnimate and SCAIL-2 wrappers composing the other nodes).
- A model package imports only itself, `models/common/` and `libs/`. Model packages never import
  each other, and `models/common/` never imports a model package. Inside `models/`, only
  `models/__init__.py` imports the model packages: that is the registration list.
- The layers above `models/`, and `scripts/`, may import a model package directly: the SAM 3.1
  Multiplex pipeline imports `models/sam3_1_multiplex/` (SAM is not registered), and
  `scripts/convert_models.py` imports the ViTPose and RTMW decoders. No pipeline imports a Wan
  package; the sampler pipeline reaches it through the registry.
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

## How to add a model

- Create `models/<name>/` with `net.py` (the nn.Module), `wrapper.py` (a `NativeModel` subclass
  implementing `PoseEstimator` or `PersonDetector` of `models/common/interfaces.py`) and
  `decode.py` if it needs one.
- Register it in its `__init__.py`, after all of its imports: its architecture
  (`registry.register("architecture", ...)`) and its entry in the `pose_estimator` or
  `person_detector` family with its model file (see `models/vitpose/__init__.py`).
- Add the package to the import list in `models/__init__.py`. Its position there is its position
  in the combo list, so G2 fails until the new surface is re-recorded with the owner's word (the
  re-record procedure under Tests).
- Keep its module-level imports to torch, numpy and the standard library.
- List its modules in `CHECK3_MODULES` of `tests/test_import_time.py` (with an `ALLOWED` row if
  one may pull in a heavy module): the gate fails on a layer module that is not listed there.
- Add seeded tiny-config tests (see `tests/models/`).
- A new Animate conditioning node means an `AnimateAdapter` subclass (`models/common/animate.py`)
  in its own model package, registered in the `animate` family under its core node id (and the
  package added to `models/__init__.py`), plus a `_LongVideoSampler` subclass in
  `nodes/sampler.py`, registered in the root `__init__.py`. A new node key changes the G2
  surface, so it needs the same re-record with the owner's word, and a row in
  `tests/test_package.py` and in the gate's `NODE_KEYS`.
- The chunk loop (`pipelines/long_video.py`) knows the core node only through the adapter. The
  base class is the Wan Animate contract; a node with another contract overrides what differs
  (`models/scail2/adapter.py` overrides all of them):
  - `OUTPUTS` / `UPDATE_HINT`: the fewest outputs the core node must return, and the error hint;
  - `HELD_VIDEOS`: the videos the core node seeks by offset, held on their last frame up to the
    plan's reach; `OVERSHOOT`: why the plan runs past `total_frames`, for that log line;
  - `prepare(animate_cls, animate_inputs, reference_image, width, height, frames_per_chunk)`:
    validate, rename or pop the node's own inputs, encode what is encoded once per run; returns
    the overlap;
  - `chunk_length`: the length policy the plan and the loop share (default: the last chunk fitted;
    SCAIL-2: every chunk full length, `libs/chunking.full_chunk_length`);
  - `check_videos(pose_video, animate_inputs)`: checks between the videos, before any is held;
  - `patch_model`: model patches, once per run;
  - `continuation(anchor, offset)`: the core call's chaining inputs (default `continue_motion`,
    `video_frame_offset`), spliced after `pose_video`, before `chunk_inputs`;
  - `chunk_inputs`: per-chunk inputs; `after_animate`: conditioning repairs before sampling;
  - `unpack(outputs, anchor)`: the outputs as (positive, negative, latent, trim_latent,
    trim_image, video_frame_offset).
  A change to a default is a change to both Wan Animate samplers, which G1 pins.

## Coding style

- Before writing new code, search the pack for code that already does the same work (grep for
  the operation, not only the name). If it exists, call it. If the same code would end up in
  two places, move it into one function and call that from both. Never write a second copy.
- No spaghetti, no duplication; clean, readable, debuggable. Small functions with one job.
- Explicit data contracts as TypedDict annotations (`libs/pose_data.py`, the guard rows in
  `pipelines/guard/common.py`, the SAM counts in `pipelines/sam3_1_multiplex/{prompt,pose}.py`);
  the values stay plain dicts.
- The owner's extraction rule: a new function only if (a) identical code already lives in 2+
  places (reduce it to one) or (b) it is likely (~70-80%) to be reused by future nodes of this
  repo's kind. Otherwise keep it inline. Near-copies that differ in any detail stay separate.
- Errors say what to do. No silent defaults.
- Behaviour changes need the owner. A golden that changes means the change is reverted.

## Imports

- The lazy rule: module level imports only torch, numpy and the standard library. cv2, scipy, PIL,
  safetensors, torchvision, tqdm, `folder_paths` and `comfy.*` are imported inside the function
  that uses them. Node modules import their pipeline or model module inside the method.
- Exactly three exceptions:
  - E1: `models/common/download.py` imports `folder_paths` and registers ComfyUI's `detection`
    model folder at import. The two loaders (`models/common/loader.py`,
    `models/sam3_1_multiplex/loader.py`) import it at module level, so the registration happens
    when Pose Detection's or a SAM node's INPUT_TYPES first runs.
  - E2: the vendored `libs/pose_utils/*` import cv2 at module level. Every consumer imports them
    inside functions.
  - E3: `pipelines/sam3_1_multiplex/track.py` imports core's SAM 3.1 tracker names at module level,
    above every relative import, so the SAM nodes' INPUT_TYPES fail on a ComfyUI without the
    tracker, before the detection folder is registered.
  - `pipelines/sam3_1_multiplex/__init__.py` and `models/sam3_1_multiplex/__init__.py` stay empty,
    so importing either package triggers neither E1 nor E3.
- The gate, with the ComfyUI venv's Python: `PYTHONPATH=/path/to/ComfyUI python
  tests/test_import_time.py`. A standalone script (pytest does not collect it). It checks the
  package import (under 0.1 s, no heavy module, the 14 keys in order), each node module, each
  module against its allowed heavy set, the ComfyUI-free set, and the E1/E3 trigger points of each
  node key against `tests/goldens/import_gate.json`.
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
- Goldens (`tests/golden.py`, `tests/goldens/*.json`) change only by the owner-approved re-record:
  delete the affected keys from the JSON file, run the test with `BCV_GOLDEN_RECORD=1` (it writes
  missing keys only and never overwrites one), and show the owner `git diff tests/goldens` with
  the reason. A refactor or cleanup never re-records.
- Test bodies reach pack names through the `Names` tables (`tests/names.py`, each domain's in
  `tests/*_fakes.py`). Moving code changes table rows, never test bodies.

## Contracts at the boundary

- POSEDATA is a plain dict at runtime everywhere. Its TypedDicts (`PoseData`, `PoseMeta`,
  `Detection`) are annotations only, and `tests/pipelines/test_pose_data.py` ties them to the keys
  the pose pipeline writes. The guard rows and the SAM counts work the same way
  (`test_guard_rows.py`, `test_sam3_1_multiplex_counts.py`).
- Locked, because code outside the repo reads them: the guard metrics JSON; the log format the
  owner's A/B test scripts parse (the `BCVideoNodes` logger, its `[BCVideoNodes]` prefix and the
  " done in " / " failed after " step lines in `libs/log.py`); the model file format
  (`models/common/checkpoint.py`); and `LOGITS_SINK` in `pipelines/sam3_1_multiplex/track.py`,
  where the A/B dump node installs its sink. The A/B test scripts and the A/B dump node are both
  outside the repo.

## Closed decisions

Do not reopen or "improve" them. They live in the owner's closed-decision documents (outside the
repo: the plan, the guard, pose-process and mask-process specs, and the review report whose open
items are decided elsewhere). In short: the guard judges pose and mask only and counts at
`draw_threshold`; the guard's warning set; ViTPose-H is the default pose model and RTMW-l
optional; the two remaining SAM A/B switches (`anchor_matching`, `unmatched_counting`, for
multi-person) default to "ours" and are bit-identical there; features the owner did not adopt
(the ViTPose flip test, the other SAM A/B switches, the M4 mask-threshold options) are removed;
multi-person is phase 2; a failed download keeps its `.part` file.

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
  `models/{vitpose,rtmw,yolo}/wrapper.py`, `models/{vitpose,rtmw}/decode.py`.
- The model weights keep their own licences, listed in the README.
