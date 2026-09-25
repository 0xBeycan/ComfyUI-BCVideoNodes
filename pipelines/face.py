# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""The face crops: a box around the face keypoints of every frame of pose_data, or boxes
supplied by the caller, cut out of the frames and resized to FACE_SIZE squares."""
import numpy as np
import torch

from ..libs import log
from ..libs.bbox import parse_bboxes
from ..libs.pose_data import PoseData
from ..libs.video import as_numpy

# the face keypoints' box is grown to this multiple of its area before it is cut
FACE_CROP_SCALE = 1.3
FACE_SIZE = 512

# Temporal smoothing of the computed face boxes (the Face Crop node's face_box_smoothing).
# Each frame's box comes from that frame's face keypoints alone, so its size pulses with keypoint
# noise and the 512x512 crop "breathes". Measured on eight clips (face-box A/B, 2026-09-25):
#   off     today's boxes, untouched
#   median  per-coordinate median of the boxes over a centred MEDIAN_WINDOW-frame window (the
#           person-box smoothing of the WanAnimatePreprocess hand-fix fork, applied to the face box)
#   size    the box centre untouched; its width and height (in log) averaged over a centred
#           Gaussian of SIZE_SIGMA frames. A centred window has no lag, and leaving the centre alone
#           keeps a fast-moving face inside its box.
FACE_BOX_SMOOTHING = ("off", "median", "size")
DEFAULT_FACE_BOX_SMOOTHING = "size"  # owner's call after the A/B (2026-09-25)
MEDIAN_WINDOW = 5
SIZE_SIGMA = 2.0


def get_face_bboxes(kp2ds, scale, image_shape):
    h, w = image_shape
    kp2ds_face = kp2ds.copy()[1:] * (w, h)

    # Drop NaN/inf keypoints (undetected face); fall back to image center if none remain.
    kp2ds_face = kp2ds_face[np.isfinite(kp2ds_face).all(axis=1)]
    if len(kp2ds_face) == 0:
        kp2ds_face = np.array([[w / 2.0, h / 2.0]])

    min_x, min_y = np.min(kp2ds_face, axis=0)
    max_x, max_y = np.max(kp2ds_face, axis=0)

    initial_width = max_x - min_x
    initial_height = max_y - min_y

    # Degenerate (collapsed) axis gives 0/0 -> NaN below; clamp to a minimum box around its center.
    min_side = max(8.0, 0.05 * min(h, w))
    if not np.isfinite(initial_width) or initial_width < min_side:
        cx = (min_x + max_x) / 2
        min_x, max_x = cx - min_side / 2, cx + min_side / 2
        initial_width = max_x - min_x
    if not np.isfinite(initial_height) or initial_height < min_side:
        cy = (min_y + max_y) / 2
        min_y, max_y = cy - min_side / 2, cy + min_side / 2
        initial_height = max_y - min_y

    initial_area = initial_width * initial_height

    expanded_area = initial_area * scale

    new_width = np.sqrt(expanded_area * (initial_width / initial_height))
    new_height = np.sqrt(expanded_area * (initial_height / initial_width))

    delta_width = (new_width - initial_width) / 2
    delta_height = (new_height - initial_height) / 4

    expanded_min_x = max(min_x - delta_width, 0)
    expanded_max_x = min(max_x + delta_width, w)
    expanded_min_y = max(min_y - 3 * delta_height, 0)
    expanded_max_y = min(max_y + delta_height, h)

    return [int(expanded_min_x), int(expanded_max_x), int(expanded_min_y), int(expanded_max_y)]


def _smooth_median(boxes, valid):
    """Per-coordinate median of the valid boxes in a centred MEDIAN_WINDOW-frame window; a frame
    with no valid neighbour keeps its box."""
    half = MEDIAN_WINDOW // 2
    out = boxes.copy()
    for i in range(len(boxes)):
        near = boxes[max(0, i - half):i + half + 1][valid[max(0, i - half):i + half + 1]]
        if len(near):
            out[i] = np.median(near, axis=0)
    return out


def _smooth_size(boxes, valid):
    """The box centres kept; log width and log height averaged over a centred Gaussian of
    SIZE_SIGMA frames, over the valid boxes only (the weights renormalised at the clip's ends)."""
    radius = int(np.ceil(3 * SIZE_SIGMA))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / SIZE_SIGMA) ** 2)
    weight = valid.astype(np.float64)
    log_wh = np.log(np.maximum(boxes[:, 2:] - boxes[:, :2], 1.0))
    def blur(x):  # zero-padded so a clip shorter than the kernel keeps its length
        return np.convolve(np.pad(x, radius), kernel, "valid")
    total = blur(weight)
    smooth = np.stack([blur(log_wh[:, k] * weight) for k in range(2)], 1)
    smooth = np.where(total[:, None] > 0, smooth / np.maximum(total, 1e-12)[:, None], log_wh)
    half = np.exp(smooth) / 2
    centre = (boxes[:, :2] + boxes[:, 2:]) / 2
    return np.concatenate([centre - half, centre + half], 1)


def _smooth_face_boxes(boxes, valid, smoothing, width, height):
    """`boxes` [(x1, y1, x2, y2)] smoothed over time as `smoothing` names, back inside the frame and
    as ints. Frames without face keypoints (`valid` False) keep their fallback box and are left
    out of their neighbours' smoothing."""
    b = np.asarray(boxes, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    b = _smooth_median(b, valid) if smoothing == "median" else _smooth_size(b, valid)
    b[~valid] = np.asarray(boxes, dtype=np.float64)[~valid]
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, width)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, height)
    return [tuple(int(v) for v in row) for row in b]


def face_bboxes_from_pose(pose_data: PoseData, width, height, face_padding=0, smoothing=DEFAULT_FACE_BOX_SMOOTHING):
    """One (x1, y1, x2, y2) face box per frame of `pose_data`, from its face keypoints, smoothed
    over time as `smoothing` (one of FACE_BOX_SMOOTHING) says, grown by `face_padding` pixels on
    every side and kept inside the frame."""
    if smoothing not in FACE_BOX_SMOOTHING:
        raise ValueError(f"face_box_smoothing must be one of {FACE_BOX_SMOOTHING}, got {smoothing!r}")
    metas = pose_data.get("pose_metas_original") if isinstance(pose_data, dict) else None
    if metas is None:
        raise ValueError("pose_data has no 'pose_metas_original'; it must come from Pose Detection")
    boxes = []
    for meta in metas:
        x1, x2, y1, y2 = get_face_bboxes(meta["keypoints_face"][:, :2], scale=FACE_CROP_SCALE, image_shape=(height, width))
        boxes.append((x1, y1, x2, y2))
    if smoothing != "off" and boxes:
        valid = [np.isfinite(meta["keypoints_face"][1:, :2]).all(axis=1).any() for meta in metas]
        boxes = _smooth_face_boxes(boxes, valid, smoothing, width, height)
    padded = []
    for x1, y1, x2, y2 in boxes:
        if face_padding > 0:
            x1, y1 = max(0, x1 - face_padding), max(0, y1 - face_padding)
            x2, y2 = min(width, x2 + face_padding), min(height, y2 + face_padding)
        padded.append((x1, y1, x2, y2))
    return padded


def _supplied_face_bboxes(face_bboxes, frames):
    """The caller's face boxes (any BBOX value parse_bboxes reads) as one (x1, y1, x2, y2) of
    ints per frame, inside the frame's top-left corner: a crop cannot start at a negative pixel."""
    boxes = [tuple(int(v) for v in b) for b in parse_bboxes(face_bboxes, frames)]
    bad = [b for b in boxes if b[0] < 0 or b[1] < 0 or b[2] <= b[0] or b[3] <= b[1]]
    if bad:
        raise ValueError(f"face_bboxes must be (x1, y1, x2, y2) with 0 <= x1 < x2 and 0 <= y1 < y2, got {bad[:3]}")
    return boxes


def crop_faces(images, pose_data: PoseData, face_padding=0, face_bboxes=None, smoothing="off"):
    """The Face Crop node: `images` [B, H, W, C] and the pose_data Pose Detection made from
    them. Returns (face_images [B, FACE_SIZE, FACE_SIZE, C], face_bboxes), the boxes one
    (x1, y1, x2, y2) per frame. `face_bboxes`, when given (one per frame, or one for all), is
    cut as it is instead of the boxes the face keypoints give; `face_padding` and `smoothing`
    apply only to the computed boxes. What face_bboxes overrides is named in one console line."""
    import cv2

    B, H, W, C = images.shape
    if face_bboxes is not None:
        boxes = _supplied_face_bboxes(face_bboxes, B)
        unused = [f"face_padding {face_padding}"] if face_padding else []
        unused += [f"face_box_smoothing {smoothing}"] if smoothing != "off" else []
        log.info("face_bboxes connected: cut as given; pose_data's face keypoints"
                 + "".join(f" and {u}" for u in unused) + " not used")
    else:
        boxes = face_bboxes_from_pose(pose_data, W, H, face_padding, smoothing)
        if len(boxes) != B:
            raise ValueError(f"pose_data holds {len(boxes)} frames and images {B}; they must be the same frames")
    images_np = as_numpy(images)
    face_images = []
    result = {"fallback crops": 0}
    with log.step(f"cropping the faces on {B} frames", result):
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            face = images_np[i][y1:y2, x1:x2]
            if face.size == 0:
                # no usable face box on this frame: fall back to the upper centre of the frame
                log.warning(f"empty face crop on frame {i}, using a centre crop instead")
                result["fallback crops"] += 1
                size = int(min(H, W) * 0.3)
                fx, fy = (W - size) // 2, int(H * 0.1)
                face = images_np[i][fy:fy + size, fx:fx + size]
                if face.size == 0:
                    face = np.zeros((size, size, C), dtype=images_np.dtype)
            face_images.append(cv2.resize(face, (FACE_SIZE, FACE_SIZE)))
    return torch.from_numpy(np.stack(face_images, 0)), boxes
