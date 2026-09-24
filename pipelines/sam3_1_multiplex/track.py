"""Person segmentation with SAM 3.1: one entry point, `track`, and two prompting modes."""
import json
from typing import Optional

import numpy as np

# At module level and above every pack import on purpose: the SAM 3.1 Multiplex nodes'
# INPUT_TYPES import this module, so a ComfyUI without the tracker primitives fails there, before
# the detection folder is registered. Nothing here reads these names; their users import them.
from comfy.ldm.sam3.tracker import MultiplexState, _prep_frame, fill_holes_in_mask_scores

from ...libs import log
from ...libs.bbox import point_in_frame, supplied_boxes
from ...libs.pose_data import PoseData
# re-exported: the node loads the model through this module
from ...models.sam3_1_multiplex.loader import load_sam3_1_multiplex
from .config import SINGLE_OBJECT_FIELDS, TRACKER_FIELDS, SAM3_1MultiplexConfig, changed_fields, output_cut
from .pose import segment_by_pose
from .prompt import PROMPT, segment_by_prompt, segment_by_prompt_multi


MODE_PROMPT = "prompt"
MODE_BOX_KEYPOINT = "box_keypoint"
# prompt-only produces the better mask; box+keypoint is the v1 behaviour and the fallback
MODES = (MODE_PROMPT, MODE_BOX_KEYPOINT)


# --- inputs --------------------------------------------------------------------------------

def parse_coords(coords, W, H, name="coords"):
    """Hand-placed points as a list of (x, y) in pixels, from the JSON string KJNodes'
    PointsEditor and easy-sam3 pass around: either `[{"x": 50, "y": 120}, ...]` in pixels, or
    easy-sam3's `{"points": [[x, y], ...]}` normalised to the frame. None or an empty string
    is no points. A point outside the frame raises."""
    if coords is None or not str(coords).strip():
        return []
    try:
        data = json.loads(coords)
    except json.JSONDecodeError as e:
        raise ValueError(f"{name} must be a JSON string, found {coords!r} ({e})") from e
    if isinstance(data, dict) and "points" in data:
        raw = data["points"] or []
        if not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in raw):
            raise ValueError(f'{name}: expected {{"points": [[x, y], ...]}} normalised, found {coords!r}')
        points = [(float(x) * W, float(y) * H) for x, y in raw]
    elif isinstance(data, list):
        if not all(isinstance(p, dict) and "x" in p and "y" in p for p in data):
            raise ValueError(f'{name}: expected [{{"x": x, "y": y}}, ...] in pixels, found {coords!r}')
        points = [(float(p["x"]), float(p["y"])) for p in data]
    else:
        raise ValueError(f'{name}: expected a JSON list of {{"x", "y"}} or {{"points": [...]}}, found {coords!r}')
    for x, y in points:
        if not point_in_frame(x, y, W, H):
            raise ValueError(f"{name}: point ({x}, {y}) is outside the {W}x{H} frame")
    return points


def parse_bboxes(bboxes, N):
    """Boxes as the per-frame list `segment_by_pose` reads: N arrays (x1, y1, x2, y2, 1.0) in
    pixels, from anything the shared `bbox.parse_bboxes` accepts. A supplied box counts as a
    detection (score 1.0)."""
    return supplied_boxes(bboxes, N)


def pose_keypoint_conf(pose_data: PoseData) -> float:
    """The keypoint confidence the pose was made with (PoseConfig.min_keypoint_conf, carried in
    pose_data["pose_config"]): one threshold for every node that reads pose_data, so the mask is
    prompted from the same keypoints the pose images draw and the guards check."""
    pose_config = pose_data.get("pose_config") if isinstance(pose_data, dict) else None
    if not isinstance(pose_config, dict) or "min_keypoint_conf" not in pose_config:
        raise ValueError("pose_data has no pose_config.min_keypoint_conf; it must come from Pose Detection "
                         "or WanAnimate Preprocess")
    return pose_config["min_keypoint_conf"]


def pose_inputs(pose_data: PoseData, N):
    """(bboxes, pose_metas, min_keypoint_conf) from pose_data the way segment_by_pose reads them:
    each frame's detections box as (x1, y1, x2, y2, score), the `pose_metas_original` dicts and
    the pose's keypoint confidence threshold."""
    if not isinstance(pose_data, dict) or "pose_metas_original" not in pose_data or "detections" not in pose_data:
        found = sorted(pose_data) if isinstance(pose_data, dict) else type(pose_data).__name__
        raise ValueError(f"box_keypoint mode needs pose_data with 'pose_metas_original' and 'detections', found {found}")
    metas, detections = pose_data["pose_metas_original"], pose_data["detections"]
    if len(metas) != N or len(detections) != N:
        raise ValueError(f"pose_data covers {len(metas)} pose frames and {len(detections)} detections, "
                         f"the images are {N} frames")
    bboxes = [np.array([*d["bbox"], d["score"]], dtype=np.float64) for d in detections]
    return bboxes, metas, pose_keypoint_conf(pose_data)


# --- entry point ---------------------------------------------------------------------------

# Set to a callable to receive every run's low-res logits (see `track`); the test dump sets it,
# since the node's outputs are masks only. None collects nothing.
LOGITS_SINK = None


def track(sam3_model, images, pose_data: Optional[PoseData] = None, bboxes=None, positive_coords=None,
          negative_coords=None, mode=MODE_PROMPT, prompt=PROMPT, max_objects=1, object_index=-1, config=None,
          logits_sink=None):
    """[N, H, W] float mask of the person (or people) in `images` [N, H, W, 3], the one function
    the SAM3 node calls. `sam3_model` is the (model, clip) pair from `load_sam3_1_multiplex`.

    `mode` picks how the person is described to SAM:
    - "prompt": the text `prompt` alone, under the tracking policy in `config` (SAM3_1MultiplexConfig).
      pose_data, bboxes and the coords are not used.
    - "box_keypoint": pose_data (required) gives every frame's box and body keypoints; the
      positive points are computed from them frame by frame and the negatives from the running
      mask. `bboxes`, when given, replace pose_data's detection boxes. `positive_coords` /
      `negative_coords` are hand-placed points on frame 0 (KJNodes / easy-sam3 JSON), added to
      frame 0's prompt; a derived point of the opposite label near one is dropped (see
      HAND_POINT_CLEARANCE). `prompt`, `max_objects`, `object_index` and the [prompt] config
      fields are not used: one person, the one the pose describes.

    `max_objects` is how many tracks may be born and kept. 1 is the single-person policy
    (`segment_by_prompt`); above 1, prompt mode only, `segment_by_prompt_multi` runs instead and
    nothing of it runs at 1. `object_index` -1 is the union of every tracked object, k is object
    k alone, objects numbered from 0 in birth order; asking for an object that was not tracked
    raises with the count found.

    Whatever the mode does not read is ignored, never raised on, so the mode can be switched
    without rewiring: a connected input, a widget or a config field changed from its default
    that the run ignores is named in one console line.

    `logits_sink` (or, when it is None, the module's LOGITS_SINK) is called once when the run is
    done, with `(logits, info)`: `logits` is a list of N [h, w] float16 CPU tensors - the low-res
    mask logits each output frame was cut from, None for a frame with no output - and `info` a
    dict {"mode", "cut", "size", "threshold", "mask_threshold"}, in prompt mode also "raw" and
    "fill_hole_area". "cut" says per frame how its mask came from its logits, so a threshold can
    be swept offline:
    - "prompt": bilinear to `size` (H, W), then > threshold; where "raw" is true for the frame,
      the logits are the ones before the output's speck and pinhole cleaning, cleaned relative
      to the threshold by `shown_logits(logits, threshold, fill_hole_area)` first
    - "birth" / "anchor": the same as "prompt", on the frame the track was born on and on the
      frames it was re-anchored on (where "raw" is false, the logits are the conditioning mask)
    - "prompted" (box_keypoint): bilinear to `size`, > threshold, then clean_mask
    - "propagated" (box_keypoint): bilinear to the tracker's 1008 x 1008, > threshold, bilinear
      to `size` as 0/1, > 0.5, then clean_mask
    "threshold" is the cut this run used on "prompt" / "propagated" frames and "mask_threshold"
    the one on "prompted" frames. Not collected with max_objects above 1."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, found {mode!r}")
    if not isinstance(max_objects, int) or max_objects < 1:
        raise ValueError(f"max_objects must be an integer of at least 1, found {max_objects!r}")
    if not isinstance(object_index, int) or object_index < -1:
        raise ValueError(f"object_index must be -1 (every object) or an object number, found {object_index!r}")
    if mode == MODE_PROMPT and object_index >= max_objects:
        raise ValueError(f"object_index {object_index}: at most {max_objects} object(s) are tracked "
                         f"(max_objects), numbered from 0")
    if config is None:
        config = SAM3_1MultiplexConfig()
    if not isinstance(config, SAM3_1MultiplexConfig):
        raise TypeError(f"config must be a SAM3_1MultiplexConfig, found {type(config).__name__}")
    if images.dim() != 4 or images.shape[-1] < 3:
        raise ValueError(f"images must be [N, H, W, 3], found {list(images.shape)}")
    model, clip = sam3_model
    N, H, W = images.shape[:3]
    sink = logits_sink if logits_sink is not None else LOGITS_SINK
    collected = {} if sink is not None and not (mode == MODE_PROMPT and max_objects > 1) else None
    dump = {"logits": collected} if collected is not None else {}

    result = {}
    with log.step(f"segmenting the person with SAM3 ({mode}) on {N} frames", result):
        if mode == MODE_PROMPT:
            unused = [n for n, v in (("pose_data", pose_data), ("bboxes", bboxes), ("positive_coords", positive_coords),
                                     ("negative_coords", negative_coords)) if v is not None]
            unused += [f"sam3_config.{n}" for n in changed_fields(config, "[box_keypoint]")
                       if not (n == "mask_threshold" and config.uniform_mask_threshold)]
            if max_objects == 1 and not config.uniform_mask_threshold:
                unused += [f"sam3_config.{n} (uniform_mask_threshold off)" for n in changed_fields(config, names=("m4_anchor_frames",))]
            if max_objects == 1:
                unused += [f"sam3_config.{n} (max_objects 1)" for n in changed_fields(config, "[prompt, max_objects > 1]")]
            else:
                unused += [f"sam3_config.{n} (max_objects > 1)" for n in changed_fields(config, names=SINGLE_OBJECT_FIELDS)]
                if sink is not None:
                    unused.append("the logits sink (max_objects > 1)")
            if unused:
                log.info(f"prompt mode segments from the text alone; {', '.join(unused)} not used")
            if not prompt or not prompt.strip():
                raise ValueError("prompt mode needs a text prompt, found an empty one")
            if max_objects == 1:
                mask = segment_by_prompt(model, clip, images, prompt, config, result=result, **dump)
            else:
                mask = segment_by_prompt_multi(model, clip, images, prompt, config, max_objects,
                                               object_index, result=result)
        else:
            if pose_data is None:
                raise ValueError("box_keypoint mode prompts from the pose; connect pose_data")
            frame_boxes, pose_metas, keypoint_conf = pose_inputs(pose_data, N)
            unused = [f"{n} {v!r}" for n, v, default in (("prompt", prompt, PROMPT), ("max_objects", max_objects, 1),
                                                         ("object_index", object_index, -1)) if v != default]
            if bboxes is not None:
                unused.append("pose_data's person boxes (bboxes replace them)")
            unused += [f"sam3_config.{n}" for n in changed_fields(config, "[prompt]") + changed_fields(config, "[prompt, max_objects > 1]")]
            if not config.temporal:
                unused += [f"sam3_config.{n} (temporal off)" for n in changed_fields(config, names=TRACKER_FIELDS)]
            if unused:
                log.info(f"box_keypoint mode prompts the one person the pose describes; {', '.join(unused)} not used")
            if bboxes is not None:
                frame_boxes = parse_bboxes(bboxes, N)
            mask = segment_by_pose(model, images, frame_boxes, pose_metas, config, keypoint_conf,
                                   extra_positive=parse_coords(positive_coords, W, H, "positive_coords"),
                                   extra_negative=parse_coords(negative_coords, W, H, "negative_coords"),
                                   result=result, **dump)
        if mode == MODE_PROMPT and object_index == 0 and max_objects == 1 and not bool(mask.any()):
            raise ValueError("object_index 0: 0 objects were tracked (numbered from 0)")
        coverage = mask.mean(dim=(1, 2))
        result["frames without a mask"] = int((coverage == 0).sum())
        result["mask coverage"] = f"{coverage.min() * 100:.1f}-{coverage.max() * 100:.1f}%"
    if collected is not None:
        info = {"mode": mode, "cut": collected["cut"], "size": (H, W),
                "threshold": output_cut(config), "mask_threshold": config.mask_threshold}
        if "raw" in collected:
            info.update(raw=collected["raw"], fill_hole_area=config.fill_hole_area)
        sink(collected["logits"], info)
    return mask
