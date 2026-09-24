"""The mask checks: the per-frame measurements of the mask against the pose it belongs to, the
flags they raise, and check_mask."""
import numpy as np

from ...libs import log
from ...libs.keypoints import BODY_NAMES, LIMBS
from ...libs.pose_data import Detection, PoseData, PoseMeta
from .common import (BOX_MARGIN, BOX_WINDOW, FRAGMENT_FRACTION, LEAK_REACH, LEAK_WINDOW, LIMB_ENDS, MASK_CHECKS,
                     RELIABLE_CONF, RELIABLE_KEYPOINTS, SKELETON_REACH, SPECK_FRACTION, WHOLE_BODY, MaskRow,
                     _box_iou_prev, _flag, _frame_pose, _in_frame, _pose_inputs, _thresholds, box_sides)
from .config import MaskGuardConfig, _config
from .report import _finish
from .timeline import MASK_PANELS


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


def _mask_inputs(mask, pose_data: PoseData):
    """The mask as [frames, height, width] booleans and the pose it is checked against, checked
    against each other."""
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.dim() != 3:
        raise ValueError(f"mask must be [frames, height, width], got a tensor of shape {tuple(mask.shape)}")
    masks = (mask.cpu().numpy() > 0.5)
    pose_metas, detections = _pose_inputs(pose_data)
    N, H, W = masks.shape
    if len(pose_metas) != N:
        raise ValueError(f"mask has {N} frames, pose_data {len(pose_metas)}; connect the outputs of the same clip")
    if pose_metas and (pose_metas[0]["height"], pose_metas[0]["width"]) != (H, W):
        raise ValueError(f"mask is {W}x{H} but the pose was found on {pose_metas[0]['width']}x{pose_metas[0]['height']} "
                         "frames; connect the mask straight from the tracker, before any resize")
    return masks, pose_metas, detections


def _keypoint_rows(meta: PoseMeta, key):
    """The keypoint set `key` of `meta` as [K, 3] float64 rows (x, y, confidence)."""
    return np.asarray(meta[key], dtype=np.float64).reshape(-1, 3)


def _whole_body(meta: PoseMeta, W, H, draw_threshold):
    """Every drawn keypoint of the person inside the frame - body, hands and face - in pixels."""
    parts = [_keypoint_rows(meta, key) for key in WHOLE_BODY if key in meta]
    kps = np.concatenate(parts) * np.array([W, H, 1.0])
    return kps[(kps[:, 2] >= draw_threshold) & _in_frame(kps, W, H)]


def box_envelopes(detections: list[Detection], N, W, H):
    """Per frame, the union of the detector boxes within BOX_WINDOW frames either side,
    grown by BOX_MARGIN and clipped to the frame."""
    grown = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        if det["score"] <= 0:
            grown.append(None)
            continue
        bw, bh = box_sides(x1, y1, x2, y2)
        grown.append((max(0.0, x1 - BOX_MARGIN * bw), max(0.0, y1 - BOX_MARGIN * bh),
                      min(float(W), x2 + BOX_MARGIN * bw), min(float(H), y2 + BOX_MARGIN * bh)))
    out = []
    for i in range(N):
        near = [b for b in grown[max(0, i - BOX_WINDOW):i + BOX_WINDOW + 1] if b is not None]
        if not near:
            out.append((0.0, 0.0, float(W), float(H)))
        else:
            out.append((min(b[0] for b in near), min(b[1] for b in near),
                        max(b[2] for b in near), max(b[3] for b in near)))
    return out


def mask_regions(mask, person_kps):
    """The person's part of `mask` and the regions detached from it, as fractions of its
    largest region.

    The person is the largest region and every region holding one of her own drawn keypoints
    (`person_kps`, [K, 2+] pixels inside the frame): a piece the frame edge or a gap in the
    mask has split off - a hand that leaves the shot and comes back at the corner, a hand the
    burned-in text cuts off the arm - is still her. Nothing else is excused. Asking only
    whether a piece touches the frame border is not enough to say the border is what
    separates it: on a clip where the body runs off the bottom of every frame, that excuses
    anything at any edge, including the objects beside her that the decoder took in."""
    import cv2

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 2:
        return mask, []
    areas = stats[1:, cv2.CC_STAT_AREA]
    main = int(np.argmax(areas)) + 1
    xs = person_kps[:, 0].astype(int)
    ys = person_kps[:, 1].astype(int)
    person = (set(labels[ys, xs].tolist()) - {0}) | {main}
    out = [round(float(stats[i, cv2.CC_STAT_AREA] / areas[main - 1]), 4)
           for i in range(1, count) if i not in person]
    return np.isin(labels, list(person)), sorted((f for f in out if f >= SPECK_FRACTION), reverse=True)


def detached_fractions(mask, person_kps):
    """The regions of `mask` detached from the person, as fractions of its largest region
    (see `mask_regions`)."""
    return mask_regions(mask, person_kps)[1]


def skeleton_zone(meta: PoseMeta, shape, draw_threshold, reach_scales=None):
    """The pixels the drawn skeleton accounts for: `reach_scales` body scales (SKELETON_REACH
    when None) around every drawn body limb and keypoint and every drawn hand keypoint."""
    import cv2

    H, W = shape
    zone = np.zeros(shape, np.uint8)
    kps = np.asarray(meta["keypoints_body"], dtype=np.float64) * np.array([W, H, 1.0])
    drawn = kps[:, 2] >= draw_threshold
    if not drawn.any():
        return zone.astype(bool)

    def length(a, b):
        return float(np.hypot(*(kps[a, :2] - kps[b, :2]))) if drawn[a] and drawn[b] else 0.0

    scale = max(length(2, 5), length(8, 11), 1.5 * length(1, 0), 1.0)
    reach = max(3, int((SKELETON_REACH if reach_scales is None else reach_scales) * scale))
    points = [kps[j, :2] for j in np.flatnonzero(drawn)]
    for key in ("keypoints_left_hand", "keypoints_right_hand"):
        if key in meta:
            hand = _keypoint_rows(meta, key) * np.array([W, H, 1.0])
            points += [p[:2] for p in hand if p[2] >= draw_threshold]
    for a, b in LIMBS:
        if drawn[a] and drawn[b]:
            cv2.line(zone, tuple(int(v) for v in kps[a, :2]), tuple(int(v) for v in kps[b, :2]), 1, 2 * reach)
    for x, y in points:
        cv2.circle(zone, (int(x), int(y)), reach, 1, -1)
    return zone.astype(bool)


def _grown(mask, size):
    """`mask` dilated by a square kernel of 2% of `size`, odd and at least 3 pixels."""
    import cv2

    k = max(3, int(0.02 * size) | 1)
    return cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)


def mask_frame_metrics(masks, pose_metas: list[PoseMeta], detections: list[Detection],
                       draw_threshold) -> list[MaskRow]:
    """One dict of raw mask measurements per frame (keys MASK_ROW); thresholds are applied
    afterwards. `masks` is [N, H, W] booleans on the frames the pose was found on."""
    N, H, W = masks.shape
    envelopes = box_envelopes(detections, N, W, H)
    areas = masks.reshape(N, H * W).sum(axis=1)
    rows = []
    prev_mask = None
    for i in range(N):
        det, meta = detections[i], pose_metas[i]
        bw, bh, diag, kps, drawn, visible = _frame_pose(meta, det, W, H, draw_threshold)
        mask = masks[i]
        area = int(areas[i])

        m = {"frame": i, "mask_area": area / (H * W), "mask_to_box": area / (bw * bh)}
        m["box_reliable"] = bool(det["score"] > 0 and drawn.sum() >= RELIABLE_KEYPOINTS
                                 and (kps[drawn, 2].mean() if drawn.any() else 0) >= RELIABLE_CONF)

        gx1, gy1, gx2, gy2 = envelopes[i]
        inside = int(mask[int(gy1):int(gy2), int(gx1):int(gx2)].sum())
        m["mask_outside_box"] = (area - inside) / area if area else 0.0

        # the person's part of the mask and the regions detached from it
        person, fragments = mask_regions(mask, _whole_body(meta, W, H, draw_threshold)) if area else (mask, [])

        # the person's region grown where neither the neighbours' masks nor the skeleton are
        near = [j for j in range(max(0, i - LEAK_WINDOW), min(N, i + LEAK_WINDOW + 1)) if j != i]
        if area and near:
            reference = masks[near].any(axis=0)
            grown = _grown(reference, np.sqrt(np.median(areas[near])))
            excess = person & ~grown & ~skeleton_zone(meta, (H, W), draw_threshold, LEAK_REACH)
            m["attached_leak"] = float(excess.sum() / max(np.median(areas[near]), 1))
        else:
            m["attached_leak"] = 0.0
        m["fragments"] = fragments

        # drawn keypoints inside the (slightly grown) mask
        if area and visible.any():
            grown = _grown(mask, diag)
            xs = np.clip(kps[:, 0].astype(int), 0, W - 1)
            ys = np.clip(kps[:, 1].astype(int), 0, H - 1)
            hit = grown[ys, xs] & visible
            m["keypoint_recall"] = float(hit.sum() / visible.sum())
            m["missed_keypoints"] = [BODY_NAMES[j] for j in np.flatnonzero(visible & ~hit)]
        else:
            m["keypoint_recall"] = 0.0 if visible.any() else 1.0
            m["missed_keypoints"] = [BODY_NAMES[j] for j in np.flatnonzero(visible)] if area == 0 else []
        m["missed_limbs"] = [name for name in m["missed_keypoints"] if BODY_NAMES.index(name) in LIMB_ENDS]

        # the person's mask the drawn skeleton does not account for
        person_area = int(person.sum())
        m["body_not_drawn"] = (float((person & ~skeleton_zone(meta, (H, W), draw_threshold)).sum() / person_area)
                               if person_area else 0.0)

        # motion against the previous frame
        m["box_iou_prev"] = _box_iou_prev(detections, i)
        m["mask_iou_prev"] = _iou(mask, prev_mask) if prev_mask is not None else None

        rows.append(m)
        prev_mask = mask
    return rows


def mask_flags(rows: list[MaskRow], t):
    """The mask checks: check name -> frames it fired on, in the order the checks first fired."""
    flags = {}

    for m in rows:
        i = m["frame"]
        if m["mask_area"] == 0 and m["box_reliable"]:
            # an empty mask on a frame the pose pipeline could not describe is a pose
            # failure, left to the pose checks
            _flag(flags, "mask_empty", i)
        elif m["box_reliable"]:
            if m["mask_to_box"] < t["min_mask_to_box"]:
                _flag(flags, "mask_empty", i)
            elif m["mask_outside_box"] > t["max_mask_outside_box"]:
                _flag(flags, "mask_leak", i)
        if m["attached_leak"] > t["max_attached_leak"]:
            _flag(flags, "mask_attached_leak", i)
        if any(f >= FRAGMENT_FRACTION for f in m["fragments"]):
            _flag(flags, "mask_fragmented", i)
        elif m["fragments"]:
            _flag(flags, "mask_specks", i)
        if m["keypoint_recall"] < t["min_keypoint_recall"] and m["mask_area"] > 0:
            _flag(flags, "mask_missing_keypoints", i)
        if m["missed_limbs"]:
            _flag(flags, "mask_missed_limb", i)
        if m["body_not_drawn"] > t["max_body_not_drawn"]:
            _flag(flags, "body_not_drawn", i)
        if (m["mask_iou_prev"] is not None and m["box_iou_prev"] is not None
                and m["box_iou_prev"] > 0.7 and m["mask_iou_prev"] < t["min_mask_iou"]):
            _flag(flags, "mask_unstable", i)
    return flags


def check_mask(mask, pose_data: PoseData, config=None, enabled=True, stop_on_fail=True):
    """The mask checks on `mask` [frames, H, W] (or one [H, W] frame) against the keypoints and
    boxes in `pose_data`, which must be of the same frames at the same size.

    `config` is a MaskGuardConfig (None = defaults); `enabled` and `stop_on_fail` as in
    `check_pose`. Returns (mask unchanged, report, metrics JSON, timeline IMAGE)."""
    config = _config(config, MaskGuardConfig)
    masks, pose_metas, detections = _mask_inputs(mask, pose_data)
    thresholds = _thresholds(pose_data, config)
    with log.step(f"mask guard: checking {len(masks)} frames ({'on' if enabled else 'off'})"):
        rows = mask_frame_metrics(masks, pose_metas, detections, thresholds["draw_threshold"])
        flags = mask_flags(rows, thresholds)
    report, metrics, timeline = _finish("Mask guard", "mask", rows, flags, thresholds,
                                        set(MASK_CHECKS if enabled else ()), MASK_PANELS, stop_on_fail)
    return mask, report, metrics, timeline
