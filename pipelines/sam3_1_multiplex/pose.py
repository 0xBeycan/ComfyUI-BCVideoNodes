"""`box_keypoint` - there is no text prompt. The person is described to SAM by what the pose
pipeline already knows about each frame (`pose_data`): the detector's box and the body
keypoints, as a box and positive points on the joints, along the limbs and down the torso,
plus negative points on the previous frame's background and on whatever the mask is known
to have annexed. That mask then propagates through the tracker's memory, which is what
carries the mask through motion blur, and the tracker is re-seeded from a fresh prompt every
`reseed_interval` frames and whenever its mask stops covering the keypoints - so the memory
cannot drift away from the person the way a tracker prompted only once does.
"""
from typing import TypedDict

import numpy as np
import torch
import torch.nn.functional as F

from ...libs import log
from ...libs.keypoints import L_ANKLE, L_FOOT, L_HIP, L_SHOULDER, R_ANKLE, R_FOOT, R_HIP, R_SHOULDER
from ...libs.mask import clean_mask
from ...libs.pose_data import PoseMeta
from ...models.sam3_1_multiplex.adapter import SAM3_1_MULTIPLEX_SIZE, decode, multiplex_parts, propagate
from ...models.sam3_1_multiplex.postprocess import low_res_logits
from .config import logits_record, output_cut, report_counts


# Body keypoints used as positive points: nose, neck, both shoulders, both hips, both ankles
# and both feet in the body layout (see keypoints.BODY_NAMES). The feet matter because the ankle
# is at the top of the foot: prompted from the ankle alone the decoder cut the shoe off.
PROMPT_KEYPOINTS = (0, 1, 2, 5, 8, 11, 10, 13, 18, 19)
# Keypoints sit on joints, so the body between them carries no positive evidence and the
# decoder drops parts of it: a close-up lost the lower half of a jacket, and a dancer's
# legs dropped out of the mask on the frames they moved fastest, while the knees the pose
# model was sure of sat outside it. These fractions along a limb or the torso put points
# on the body itself, halfway between the joints.
TORSO_FRACTIONS = (0.35, 0.65)
LIMB_FRACTIONS = (0.5,)
# Limbs to put a point on, as (from, to) keypoints: thighs, shins, upper arms and feet.
LIMBS = ((R_HIP, 9), (L_HIP, 12), (9, R_ANKLE), (12, L_ANKLE), (R_SHOULDER, 3), (L_SHOULDER, 6),
         (R_ANKLE, R_FOOT), (L_ANKLE, L_FOOT))
# A hand-placed point is the user's word on that spot and wins over the points derived from
# the pose and the previous mask: a derived point of the opposite label closer to it than
# this share of the box diagonal would contradict it in the same prompt, so it is dropped.
# Hand-placed points themselves are always passed as given.
HAND_POINT_CLEARANCE = 0.04


# --- box_keypoint mode: prompt points ------------------------------------------------------

def body_points(kps, threshold, aspect=1.0):
    """Points on the body between the joints, in normalised coordinates: down the torso and
    along every limb whose two ends the pose model is sure of. `aspect` is W / H, needed
    where an x distance has to be stepped along y."""
    points = []
    if kps[R_SHOULDER][2] >= threshold and kps[L_SHOULDER][2] >= threshold:
        shoulder = ((kps[R_SHOULDER][0] + kps[L_SHOULDER][0]) / 2, (kps[R_SHOULDER][1] + kps[L_SHOULDER][1]) / 2)
        hips = [kps[i] for i in (R_HIP, L_HIP) if kps[i][2] >= threshold]
        if hips:
            hip = (sum(h[0] for h in hips) / len(hips), sum(h[1] for h in hips) / len(hips))
            points += [(shoulder[0] + t * (hip[0] - shoulder[0]), shoulder[1] + t * (hip[1] - shoulder[1])) for t in TORSO_FRACTIONS]
        else:
            # the hips are out of frame: step down from the shoulders by their own width, in
            # frame units - the coordinates are normalised per axis, so the x span has to be
            # scaled by the aspect ratio before it can be added to y
            span = (abs(kps[R_SHOULDER][0] - kps[L_SHOULDER][0]) or 0.15) * aspect
            points += [(shoulder[0], min(shoulder[1] + t * span, 0.99)) for t in (1.0, 2.0)]
    for a, b in LIMBS:
        if kps[a][2] >= threshold and kps[b][2] >= threshold:
            points += [(kps[a][0] + t * (kps[b][0] - kps[a][0]), kps[a][1] + t * (kps[b][1] - kps[a][1])) for t in LIMB_FRACTIONS]
    return points


def box_bounds(bbox, W, H):
    """The box clipped to the frame as integers, or None when nothing usable is left."""
    x1, y1, x2, y2 = (int(max(0, min(v, limit))) for v, limit in zip(bbox[:4], (W, H, W, H)))
    return (x1, y1, x2, y2) if x2 - x1 >= 8 and y2 - y1 >= 8 else None


def spread_points(free, x1, y1, count):
    """Up to `count` points taken from a boolean map, one per cell of a grid over it so they
    are spread rather than clustered, offset to (x1, y1) and returned in pixels."""
    if not free.any():
        return []
    rows = cols = int(np.ceil(np.sqrt(count)))
    h, w = free.shape
    points = []
    for r in range(rows):
        for c in range(cols):
            cell = free[r * h // rows:(r + 1) * h // rows, c * w // cols:(c + 1) * w // cols]
            ys, xs = np.nonzero(cell)
            if len(ys):
                middle = len(ys) // 2
                points.append((x1 + c * w // cols + int(xs[middle]), y1 + r * h // rows + int(ys[middle])))
    return points[:count]


def background_points(previous_mask, bbox, W, H, count, margin):
    """Up to `count` points inside the box that the previous frame's mask says are background,
    kept clear of the mask by `margin` of the box diagonal and spread over the box, in pixels."""
    import cv2
    bounds = box_bounds(bbox, W, H)
    if previous_mask is None or not previous_mask.any() or bounds is None:
        return []
    x1, y1, x2, y2 = bounds
    inside = previous_mask[y1:y2, x1:x2]
    if not inside.any():
        return []
    clearance = int(margin * float(np.hypot(x2 - x1, y2 - y1)))
    free = cv2.distanceTransform((~inside).astype(np.uint8), cv2.DIST_L2, 3) > clearance
    return spread_points(free, x1, y1, count)


def annexed_points(annexed, bbox, W, H, count):
    """Up to `count` points spread over the background the decoder is already known to have annexed, inside
    the box, in pixels."""
    bounds = box_bounds(bbox, W, H)
    if annexed is None or bounds is None:
        return []
    x1, y1, x2, y2 = bounds
    return spread_points(annexed[y1:y2, x1:x2], x1, y1, count)


def confident_pixels(kps, W, H, threshold):
    """The frame's body keypoints at or above `threshold` as integer pixel coordinates."""
    return [(int(min(max(x * W, 0), W - 1)), int(min(max(y * H, 0), H - 1)))
            for x, y, c in kps if c >= threshold]


def _regions(mask, points):
    """The 8-connected regions of `mask` as cv2 labels them (count, labels, stats), and the
    labels the pixel `points` fall on."""
    import cv2
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    occupied = {labels[y, x] for x, y in points}
    return count, labels, stats, occupied


def remember_annexed(annexed, mask, previous, kps, W, H, min_fraction, threshold):
    """The running map of background the decoder has annexed.

    A region the mask gained since the previous frame that is large against the mask and holds
    no confident keypoint is not the person appearing, it is an adjacent object being taken
    in; it joins the map. A region already on the map that a keypoint has since landed in is
    the person having moved there, and the whole of it leaves - otherwise a negative point
    would be put on the body itself. Clearing only a radius around the keypoints instead was
    tried and is worse: the stale remainder ate into the person on a clip where they fill the
    frame. `min_fraction` is the gained-region size against the mask, `threshold` the keypoint
    confidence."""
    import cv2
    area = int(mask.sum())
    gained = mask & ~previous
    points = confident_pixels(kps, W, H, threshold)
    if area and gained.any():
        count, labels, stats, occupied = _regions(gained, points)
        keep = [i for i in range(1, count)
                if stats[i, cv2.CC_STAT_AREA] >= min_fraction * area and i not in occupied]
        if keep:
            fresh = np.isin(labels, keep)
            annexed = fresh if annexed is None else (annexed | fresh)
    if annexed is None:
        return None
    count, labels, _, occupied = _regions(annexed, points)
    keep = [i for i in range(1, count) if i not in occupied]
    return np.isin(labels, keep) if keep else None


def _detected(bbox):
    """Whether a frame's box is a detection: there is one, and it scores above 0."""
    return bbox is not None and bbox[-1] > 0


def prompt_for(frame_index, bboxes, pose_metas: list[PoseMeta], W, H, device, dtype, config, min_keypoint_conf,
               previous_mask=None, annexed=None, extra_positive=(), extra_negative=()):
    """(box_inputs, point_inputs) for the SAM decoder in its 1008x1008 space, or (None, None)
    when the frame has neither a detection nor a confident keypoint. The points are the
    keypoints and the body points as positives, plus background points from the previous
    frame's mask and points from whatever the decoder has already annexed as negatives.
    `extra_positive` / `extra_negative` are hand-placed points in pixels, added as given; a
    derived point of the opposite label within HAND_POINT_CLEARANCE of one is dropped."""
    sx, sy = SAM3_1_MULTIPLEX_SIZE / W, SAM3_1_MULTIPLEX_SIZE / H
    bbox = bboxes[frame_index]
    box_inputs = None
    if _detected(bbox):
        box_inputs = torch.tensor([[[bbox[0] * sx, bbox[1] * sy], [bbox[2] * sx, bbox[3] * sy]]], device=device, dtype=dtype)
    kps = pose_metas[frame_index]["keypoints_body"]
    threshold = min_keypoint_conf
    positive = [(kps[k][0] * W, kps[k][1] * H) for k in PROMPT_KEYPOINTS if kps[k][2] >= threshold]
    positive += [(x * W, y * H) for x, y in body_points(kps, threshold, W / H)]
    negative = []
    if _detected(bbox):
        negative = (background_points(previous_mask, bbox, W, H, config.negative_points, config.negative_margin)
                    + annexed_points(annexed, bbox, W, H, config.annexed_points))
    if extra_positive or extra_negative:
        positive, negative = _clear_of_hand_points(positive, negative, extra_positive, extra_negative,
                                                   bbox if _detected(bbox) else (0, 0, W, H))
    positive += list(extra_positive)
    negative += list(extra_negative)
    point_inputs = None
    if positive:
        coords = [(x * sx, y * sy) for x, y in positive + negative]
        labels = [1] * len(positive) + [0] * len(negative)
        point_inputs = {"point_coords": torch.tensor([coords], device=device, dtype=dtype),
                        "point_labels": torch.tensor([labels], dtype=torch.int32, device=device)}
    return box_inputs, point_inputs


def _clear_of_hand_points(positive, negative, extra_positive, extra_negative, bbox):
    """The derived `positive` / `negative` points without those that land within
    HAND_POINT_CLEARANCE of the box diagonal of a hand-placed point of the opposite label, in
    their order; how many were dropped is logged."""
    reach = HAND_POINT_CLEARANCE * float(np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))

    def clear(points, against):
        if not against:
            return points
        hand = np.asarray(against, dtype=np.float64)
        return [p for p in points if np.hypot(hand[:, 0] - p[0], hand[:, 1] - p[1]).min() > reach]

    kept_positive, kept_negative = clear(positive, extra_negative), clear(negative, extra_positive)
    dropped = len(positive) - len(kept_positive), len(negative) - len(kept_negative)
    if any(dropped):
        log.info(f"hand-placed points win: {dropped[0]} derived positive and {dropped[1]} derived negative point(s) "
                 f"within {reach:.0f} px of a hand-placed point of the other label dropped")
    return kept_positive, kept_negative


def keypoint_recall(mask, kps, W, H, threshold):
    """Fraction of the frame's body keypoints at or above `threshold` that fall inside `mask`, 1.0
    when the pose model is sure of none of them."""
    confident = confident_pixels(kps, W, H, threshold)
    if not confident:
        return 1.0
    return sum(1 for x, y in confident if mask[y, x]) / len(confident)


def confident_count(pose_meta: PoseMeta, threshold):
    return sum(1 for _, _, c in pose_meta["keypoints_body"] if c >= threshold)


def is_anchor(frame_index, bboxes, pose_metas: list[PoseMeta], config, min_keypoint_conf, reference_count=0):
    """Whether this frame's detection and pose are good enough to re-seed the tracker from.
    `reference_count` is how many keypoints the running segment was seeded with: a frame
    that sees fewer of them is a worse view of the person, not a better anchor."""
    bbox = bboxes[frame_index]
    if bbox is None or bbox[-1] <= 0:
        return False
    conf = [c for _, _, c in pose_metas[frame_index]["keypoints_body"] if c >= min_keypoint_conf]
    return (len(conf) >= max(config.min_anchor_keypoints, config.min_anchor_completeness * reference_count)
            and sum(conf) / len(conf) >= config.min_anchor_conf)


def _previous_mask(masks, i):
    """Frame i - 1's mask as a bool array, or None on frame 0 and when that frame has no mask."""
    return masks[i - 1].numpy() > 0.5 if i > 0 and masks[i - 1].any() else None


# The counts segment_by_pose reports, keyed by the label the log shows; all six are there from
# the start.
PoseCounts = TypedDict("PoseCounts", {"prompted": int, "propagated": int, "re-seeded early": int,
                                      "kept at low recall": int, "no prompt": int, "empty prompt": int})


def segment_by_pose(model, images, bboxes, pose_metas: list[PoseMeta], config, min_keypoint_conf, extra_positive=(),
                    extra_negative=(), result=None, logits=None):
    """[N, H, W] float masks of the detected person in `images` [N, H, W, 3].

    `bboxes[i]` is frame i's detector box as (x1, y1, x2, y2, score); a score of -1 means
    nothing was detected. `pose_metas[i]["keypoints_body"]` are its body keypoints,
    normalised to the frame, with confidence; `min_keypoint_conf` is the confidence the pose was
    made with, at or above which a keypoint is prompted, counted and checked. `extra_positive` / `extra_negative` are
    hand-placed points in pixels of frame 0, added to frame 0's prompt.

    A frame the detector and the pose model agree on is prompted with that box and those
    keypoints; the tracker's memory then propagates the mask, which is what carries it
    through frames the pose model loses to motion blur. The tracker is re-seeded from a
    fresh prompt every `config.reseed_interval` frames - but only on a frame that is an anchor,
    and immediately when the propagated mask stops covering the keypoints. Whatever the mask
    annexes along the way is remembered and prompted against from then on. With
    `config.temporal` off, every frame is prompted on its own, without the tracker. `result`, if given, is
    filled with what happened for the log; `logits`, if given, a dict, receives each output
    frame's low-res logits (see `logits_record`)."""
    from comfy import model_management as mm
    from comfy.utils import ProgressBar, common_upscale
    c = config
    refine, temporal = c.refine, c.temporal
    N, H, W, _ = images.shape
    mm.load_model_gpu(model)
    device, dtype = mm.get_torch_device(), model.model.get_dtype()
    sam3 = multiplex_parts(model)[0]
    frames_chw = images[..., :3].movedim(-1, 1)
    masks = torch.zeros(N, H, W)
    pbar = ProgressBar(N)
    counts: PoseCounts = {"prompted": 0, "propagated": 0, "re-seeded early": 0, "kept at low recall": 0,
                          "no prompt": 0, "empty prompt": 0}
    i = 0
    seed_frame, seed_count = -1, 0
    annexed = None
    cut = output_cut(c)
    record = logits_record(logits, N)

    def keep(index, mask, low=None, how=None):
        """Store frame `index`'s mask and update what the decoder is known to have annexed.
        The annexation shows itself on the frame it happens, which is almost never a frame
        that is prompted, so it has to be carried to the next prompt."""
        nonlocal annexed
        was = _previous_mask(masks, index)
        masks[index] = torch.from_numpy(mask).float()
        if record is not None:
            record["logits"][index], record["cut"][index] = low, how
        if was is not None:
            annexed = remember_annexed(annexed, mask.astype(bool), was,
                                       pose_metas[index]["keypoints_body"], W, H,
                                       c.min_annexed_fraction, min_keypoint_conf)

    while i < N:
        carried = False  # this frame's prompt failed and the track is carried on from i - 1
        previous = _previous_mask(masks, i)
        box_inputs, point_inputs = prompt_for(i, bboxes, pose_metas, W, H, device, dtype, c, min_keypoint_conf,
                                              previous, annexed,
                                              *((extra_positive, extra_negative) if i == 0 else ()))
        if i == 0 and (extra_positive or extra_negative) and (box_inputs is None or point_inputs is None):
            log.warning("the hand-placed points are for frame 0, which has no box or no positive point to prompt "
                        "with; they are not used")
        if box_inputs is None or point_inputs is None:
            # nothing to describe the person with: carry the track on if there is one, else
            # leave this frame without a mask (the guard reports it as pose, not mask)
            counts["no prompt"] += 1
            if not (temporal and i > 0 and masks[i - 1].any()):
                pbar.update(1)
                i += 1
                continue
            carried = True
        else:
            frame = common_upscale(frames_chw[i:i + 1], SAM3_1_MULTIPLEX_SIZE, SAM3_1_MULTIPLEX_SIZE, "bilinear", crop="disabled").to(device, dtype)
            with torch.inference_mode():
                decoded = decode(sam3, frame, point_inputs, box_inputs, refine)
                seed = (F.interpolate(decoded.float(), size=(H, W), mode="bilinear",
                                      align_corners=False)[0, 0] > c.mask_threshold).cpu().numpy()
            if seed.any():
                keep(i, clean_mask(seed, c), low_res_logits(decoded) if record is not None else None, "prompted")
                counts["prompted"] += 1
                seed_frame, seed_count = i, confident_count(pose_metas[i], min_keypoint_conf)
                pbar.update(1)
                i += 1
            else:
                # the decoder found nothing here; propagating from an empty mask is wasted work
                counts["empty prompt"] += 1
                if not (temporal and i > 0 and masks[i - 1].any()):
                    pbar.update(1)
                    i += 1
                    continue
                carried = True
        if not temporal or i >= N:
            continue
        # propagate from the last frame that has a mask; accept frames while the mask still
        # covers the keypoints, and stop at the first anchor once the interval is up
        while i < N:
            end = min(i + c.reseed_interval, N)
            lows = [] if record is not None else None
            tracked = propagate(sam3, frames_chw[i - 1:end], masks[i - 1], device, dtype, H, W, cut, lows)
            stop = False
            start = i
            for k, mask in enumerate(tracked, start=start):
                if keypoint_recall(mask, pose_metas[k]["keypoints_body"], W, H, min_keypoint_conf) < c.min_tracked_recall:
                    if carried and k == start:
                        # the prompt for this frame just failed, so re-seeding it would only
                        # repeat the same failure forever: keep the tracked mask and move on
                        counts["kept at low recall"] += 1
                    else:
                        counts["re-seeded early"] += 1
                        stop = True
                        break
                keep(k, clean_mask(mask, c), lows[k - start] if lows is not None else None, "propagated")
                counts["propagated"] += 1
                pbar.update(1)
                i = k + 1
                since_seed = i - seed_frame if seed_frame >= 0 else c.reseed_interval
                if since_seed >= c.reseed_interval and i < N and is_anchor(i, bboxes, pose_metas, c, min_keypoint_conf, seed_count):
                    stop = True
                    break
                if since_seed >= c.max_propagate:
                    stop = True
                    break
            carried = False  # only the frame whose prompt just failed is exempt, not later windows
            if stop:
                break
    report_counts(result, counts)
    log.info("segmented " + ", ".join(f"{v} frame(s) {k}" for k, v in counts.items() if v))
    return masks
