"""Pose detection over a batch of frames: the person box (YOLO, or boxes the caller supplies),
the 133 COCO-WholeBody keypoints (ViTPose or RTMW), read against the frames around them, and
the pose images drawn from them.

The detector and the pose model are passed in as the wrapper objects the models package
builds; nothing here loads a model, and nothing here calls SAM3 - the mask is its own node
and reads the pose_data this module produces.
"""
import json
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field, fields

import numpy as np
import torch

from ..libs import log
from ..libs.bbox import box_corners, point_in_frame, supplied_boxes, whole_frame_box
from ..libs.pose_data import PoseData
from ..libs.temporal import (BOX_WINDOW, DROPPED, MAX_GAP, MAX_RESIDUAL, MAX_STEP, MEASURED, RECOVERED, REPLACED,
                             all_measured, boxes_over_time, keypoints_over_time, widen_over_time)
from ..libs.video import as_numpy
from ..models.common.pose_input import pose_crop
from ..models.common.wrapper import load_models as _to_device

DETECTOR_INPUT_SIZE = (640, 640)
# A detector box smaller than this on either side is no detection.
MIN_BOX_SIDE = 10
# A detector box that stops within this fraction of its own size from a frame edge belongs
# to a person the frame cuts off; it is extended to that edge before it prompts the mask,
# otherwise the decoder stops at the box and the clothing below it stays unmasked.
EDGE_SNAP = 0.15
# key_frame_body_points: the frame it is taken from (easy-sam3 prompts frame_index 0 by
# default) and the body keypoints it exports, in the AAPose body layout - nose, neck, the
# shoulders, the hips and the knees, the set the upstream node exported.
KEY_FRAME = 0
KEY_FRAME_BODY_POINTS = (0, 1, 2, 5, 8, 11, 10, 13)


@dataclass
class PoseConfig:
    """The Pose Detection tunables. Defaults are the measured values; the Pose Config node
    only overrides them. Each field's metadata holds its range and a one-line description. A
    field changed from its default that the run does not read (detection_threshold with
    supplied boxes, the temporal_* fields with temporal off) is named in one console line."""

    confidence_scale: float = field(default=0.0, metadata={
        "min": 0.0, "max": 20.0, "step": 0.1,
        "doc": "RTMW only: divisor of its raw SimCC keypoint score; 0 keeps the default 4.6. Ignored with ViTPose, whose confidences are its heatmap maxima"})
    min_keypoint_conf: float = field(default=0.3, metadata={
        "min": 0.0, "max": 1.0, "step": 0.05,
        "doc": "Keypoints below this confidence count as not found by SAM 3.1 Multiplex box_keypoint mode, which reads it from pose_data. The drawing and the guards use Pose Detection's draw_threshold"})
    detection_threshold: float = field(default=0.05, metadata={
        "min": 0.0, "max": 1.0, "step": 0.01,
        "doc": "Person detector (YOLO) score below which a box is discarded. Ignored when bboxes is connected (YOLO does not run)"})
    temporal: bool = field(default=True, metadata={
        "doc": "Read boxes and keypoints against the frames around them (fill short gaps, replace glitches); off ignores the temporal_* fields (not SAM 3.1 Multiplex Config's temporal)"})
    temporal_max_gap: int = field(default=MAX_GAP, metadata={
        "min": 0, "max": 30, "step": 1,
        "doc": "Longest run of unconfident frames a keypoint is bridged over. Ignored with temporal off"})
    temporal_max_step: float = field(default=MAX_STEP, metadata={
        "min": 0.0, "max": 1.0, "step": 0.01,
        "doc": "Fastest motion, in box diagonals per frame, a bridged gap may span. Ignored with temporal off"})
    temporal_max_residual: float = field(default=MAX_RESIDUAL, metadata={
        "min": 0.0, "max": 2.0, "step": 0.01,
        "doc": "Distance, in box diagonals, from its neighbours' median past which a keypoint is a glitch. Ignored with temporal off"})
    box_window: int = field(default=BOX_WINDOW, metadata={
        "min": 0, "max": 30, "step": 1,
        "doc": "Frames either side whose person boxes each frame's box is widened to; supplied bboxes are widened too"})

    def __post_init__(self):
        for f in fields(self):
            value, meta = getattr(self, f.name), f.metadata
            if "min" in meta and not meta["min"] <= value <= meta["max"]:
                raise ValueError(f"PoseConfig.{f.name} is {value}; it must be within {meta['min']}..{meta['max']}")


# The fields only the temporal layer reads.
TEMPORAL_FIELDS = ("temporal_max_gap", "temporal_max_step", "temporal_max_residual")


def unused_config_fields(config, supplied_boxes):
    """(name, why) for every PoseConfig field changed from its default that this run does not read."""
    why = {"detection_threshold": "bboxes connected, the detector does not run"} if supplied_boxes else {}
    if not config.temporal:
        why.update(dict.fromkeys(TEMPORAL_FIELDS, "temporal off"))
    return [(f.name, why[f.name]) for f in fields(config) if f.name in why and getattr(config, f.name) != f.default]


def snap_to_frame(bbox, W, H):
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    bw, bh = x2 - x1, y2 - y1
    return np.array([0.0 if x1 < EDGE_SNAP * bw else x1, 0.0 if y1 < EDGE_SNAP * bh else y1,
                     float(W) if W - x2 < EDGE_SNAP * bw else x2, float(H) if H - y2 < EDGE_SNAP * bh else y2,
                     float(bbox[4])])


@contextmanager
def _overridden(model, attribute, value):
    """`model.attribute` set to `value` for the duration, and restored after."""
    if not hasattr(model, attribute):
        raise ValueError(f"{type(model).__name__} has no '{attribute}' to override; the pose config needs a model "
                         f"wrapper that exposes it")
    before = getattr(model, attribute)
    setattr(model, attribute, value)
    try:
        yield
    finally:
        setattr(model, attribute, before)


def _input_resolution(pose_model):
    """(height, width) of the pose model's input, from its [N, C, H, W] `input_shape`."""
    shape = getattr(pose_model, "input_shape", None)
    if shape is None or len(shape) != 4 or not all(isinstance(v, (int, np.integer)) for v in shape[2:]):
        raise ValueError(f"{type(pose_model).__name__}.input_shape is {shape}; expected [N, C, H, W] with integer H and W")
    return int(shape[2]), int(shape[3])


def _detected_boxes(detector, images_np, W, H, threshold, pbar):
    """YOLO's person box on every frame as (x1, y1, x2, y2, score), and the number of people
    it was fairly sure of. A frame with nothing usable gets the whole frame, score -1."""
    import cv2
    from tqdm import tqdm
    shape = np.array([H, W])[None]
    bboxes, person_counts = [], []
    with _overridden(detector, "threshold_conf", threshold):
        for i, img in enumerate(tqdm(images_np, desc="Detecting bboxes")):
            detection = detector(cv2.resize(img, DETECTOR_INPUT_SIZE).transpose(2, 0, 1)[None], shape)[0][0]
            bbox, count = detection["bbox"], detection.get("person_count", 0)
            if bbox[-1] <= 0 or (bbox[2] - bbox[0]) < MIN_BOX_SIDE or (bbox[3] - bbox[1]) < MIN_BOX_SIDE:
                # nothing usable detected: the pose, the mask prompt and the guard all see the
                # whole frame as the box, marked undetected
                bbox, count = whole_frame_box(W, H), 0
            bboxes.append(bbox)
            person_counts.append(count)
            pbar.update_absolute(i + 1)
    return bboxes, person_counts


def _supplied_boxes(bboxes, frames):
    """The caller's boxes, (x1, y1, x2, y2) per frame or a single one for every frame, as the
    detector would have handed them over: score 1 and one person each."""
    return supplied_boxes(bboxes, frames), [1] * frames


def detect(detector, pose_model, images, bboxes=None, config=None
           ) -> tuple[PoseData, list[tuple[float, float, float, float]]]:
    """Runs the detector and the pose model on every frame of `images` [B, H, W, 3].

    `detector` is not called when `bboxes` is given (one (x1, y1, x2, y2) per frame, or one
    for all) and may then be None. `config` is a PoseConfig, the defaults when None.

    Returns (pose_data, boxes). pose_data carries the per-frame pose metas (`pose_metas` as
    AAPoseMeta for drawing, `pose_metas_original` as dicts with the normalised keypoints),
    `detections` (the person box as everything downstream sees it: interpolated from the
    neighbouring detections where the detector found nobody when the temporal layer is on,
    widened to the neighbouring frames' boxes and extended to frame edges it nearly touches,
    its score, -1 when nothing was detected, and the number of people the detector was fairly
    sure of), `keypoint_source`, which names every keypoint the temporal layer wrote rather
    than the model, so a filled value is never read as a measurement (all MEASURED with the
    layer off), and `pose_config`, the config the pose was made with. boxes are the same
    boxes as (x1, y1, x2, y2) tuples, one per frame.
    """
    from comfy.utils import ProgressBar
    from tqdm import tqdm

    from ..libs.pose_utils.pose2d_utils import AAPoseMeta, load_pose_metas_from_kp2ds_seq
    config = config or PoseConfig()
    B, H, W, C = images.shape
    images_np = as_numpy(images)
    resolution = _input_resolution(pose_model)
    log.info(f"preprocessing {B} frames of {W}x{H}")
    unused = unused_config_fields(config, bboxes is not None)
    if unused:
        log.info(", ".join(f"pose_config.{name} ({why})" for name, why in unused) + " not used")
    with log.step("loading the detection models"):
        _to_device(*((pose_model,) if bboxes is not None else (detector, pose_model)))

    pbar = ProgressBar(B * 2)
    result = {}
    if bboxes is not None:
        with log.step(f"using the {len(bboxes)} supplied person boxes for {B} frames"):
            raw, person_counts = _supplied_boxes(bboxes, B)
            pbar.update_absolute(B)
    else:
        with log.step(f"detecting the person on {B} frames", result):
            raw, person_counts = _detected_boxes(detector, images_np, W, H, config.detection_threshold, pbar)
            result["frames without a person"] = sum(1 for b in raw if b[-1] <= 0)
            result["frames with several people"] = sum(1 for n in person_counts if n > 1)
    # one set of boxes for everything downstream: the pose crop, the mask prompt and the guard
    if config.temporal:
        boxes = [snap_to_frame(b, W, H) for b in boxes_over_time(raw, config.box_window)]
    else:
        boxes = [snap_to_frame(b, W, H) for b in widen_over_time(raw, config.box_window)]

    kp2ds = []
    scale_override = nullcontext()
    if config.confidence_scale > 0:
        if getattr(pose_model, "conf_scale", 0) is None:
            # a model whose confidences are not a scaled score (ViTPose's heatmap maxima)
            log.info(f"confidence_scale {config.confidence_scale:g} ignored: it applies to RTMW only, "
                     f"{type(pose_model).__name__} confidences are used as they are")
        else:
            scale_override = _overridden(pose_model, "conf_scale", config.confidence_scale)
    with log.step(f"extracting keypoints on {B} frames"), scale_override:
        for i, (img, bbox) in enumerate(tqdm(zip(images_np, boxes), total=B, desc="Extracting keypoints")):
            img_norm, center, scale = pose_crop(img, bbox, resolution)
            kp2ds.append(pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None]))
            pbar.update_absolute(B + i + 1)
    kp2ds = np.concatenate(kp2ds, 0)

    if config.temporal:
        result = {}
        with log.step("reading the keypoints against the frames around them", result):
            kp2ds, source = keypoints_over_time(kp2ds, boxes, max_gap=config.temporal_max_gap,
                                                max_step=config.temporal_max_step,
                                                max_residual=config.temporal_max_residual)
            for name, code in (("recovered", RECOVERED), ("replaced", REPLACED), ("dropped", DROPPED)):
                hit = source == code
                result[f"keypoints {name}"] = f"{int(hit.sum())} on {int(hit.any(axis=1).sum())} frames"
            result["keypoints the model gave"] = f"{100 * (source == MEASURED).mean():.1f}%"
    else:
        source = all_measured(kp2ds)
    pose_metas = load_pose_metas_from_kp2ds_seq(kp2ds, width=W, height=H)

    pose_data = {
        "pose_metas": [AAPoseMeta.from_humanapi_meta(meta) for meta in pose_metas],
        "pose_metas_original": pose_metas,
        "detections": [
            {"bbox": [float(v) for v in box[:4]], "score": float(box[4]), "persons": int(count)}
            for box, count in zip(boxes, person_counts)
        ],
        "keypoint_source": source.tolist(),
        "pose_config": asdict(config),
    }
    return pose_data, [box_corners(box) for box in boxes]


def draw(pose_data: PoseData, body_stick_width=-1, hand_stick_width=-1, draw_head=True, draw_threshold=0.5):
    """The pose images [B, H, W, 3] drawn from pose_data at the size of the frames the pose
    was found on, so they line up with the frames and the mask. A stick width of 0 leaves
    that part out; a limb is drawn when both its ends reach `draw_threshold`."""
    from comfy.utils import ProgressBar
    from tqdm import tqdm

    from ..libs.pose_utils.human_visualization import draw_aapose_by_meta_new
    pose_metas = pose_data["pose_metas"]
    pbar = ProgressBar(len(pose_metas))
    pose_images = []
    with log.step(f"drawing {len(pose_metas)} pose images"):
        for i, meta in enumerate(tqdm(pose_metas, desc="Drawing pose images")):
            canvas = np.zeros((meta.height, meta.width, 3), dtype=np.uint8)
            image = draw_aapose_by_meta_new(canvas, meta, threshold=draw_threshold, draw_body=body_stick_width != 0,
                                            draw_hand=hand_stick_width != 0, draw_head=draw_head,
                                            body_stick_width=body_stick_width, hand_stick_width=hand_stick_width)
            pose_images.append(image)
            pbar.update_absolute(i + 1)
    return torch.from_numpy(np.stack(pose_images, 0)).float() / 255.0


def key_frame_body_points(pose_data: PoseData, threshold=0.5):
    """KEY_FRAME's body keypoints in KEY_FRAME_BODY_POINTS that reach `threshold` and lie in the
    frame, as the JSON string KJNodes PointsEditor emits and easy-sam3 reads as
    `positive_coords`: '[{"x": 50, "y": 120}, ...]' in frame pixels. '[]' when none does. A
    keypoint the pose model puts outside the frame is left out: it is no point on the frame,
    and `positive_coords` refuses one."""
    meta = pose_data["pose_metas_original"][KEY_FRAME]
    W, H = meta["width"], meta["height"]
    body = meta["keypoints_body"][list(KEY_FRAME_BODY_POINTS)]
    confident = body[body[:, 2] >= threshold]
    points = (confident[:, :2] * np.array([[W, H]])).astype(np.int32)
    return json.dumps([{"x": int(x), "y": int(y)} for x, y in points if point_in_frame(x, y, W, H)])


def pose_detection(images, detector, pose_model, bboxes=None, config=None, body_stick_width=-1,
                   hand_stick_width=-1, draw_head=True, draw_threshold=0.5
                   ) -> tuple[torch.Tensor, PoseData, list[tuple[float, float, float, float]], str]:
    """The Pose Detection node: `images` [B, H, W, 3] in, and out
    (pose_images [B, H, W, 3], pose_data, bboxes, key_frame_body_points) - see `detect`,
    `draw` and `key_frame_body_points`. pose_data also carries `draw_threshold`: the guards
    count the keypoints and limbs the pose images draw."""
    pose_data, boxes = detect(detector, pose_model, images, bboxes=bboxes, config=config)
    pose_data["draw_threshold"] = draw_threshold
    pose_images = draw(pose_data, body_stick_width, hand_stick_width, draw_head, draw_threshold)
    return pose_images, pose_data, boxes, key_frame_body_points(pose_data, draw_threshold)
