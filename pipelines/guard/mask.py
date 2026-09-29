"""The mask checks: the per-frame measurements of the mask against the pose it belongs to (or
alone, without pose_data), the flags they raise, and check_mask."""
import numpy as np

from ...libs import log
from ...libs.keypoints import BODY_NAMES, LIMBS, in_frame
from ...libs.pose_data import Detection, PoseData, PoseMeta
from .common import (BOX_MARGIN, BOX_WINDOW, CARRIES, FINAL_BLOCK, FINAL_GROW, FINAL_ON_BODY, FINAL_PAD, FRAGMENT_FRACTION, HANDS,
                     HEAD_OUT_KEYPOINT, HEAD_OUT_SIDES, LEAK_REACH, LEAK_WINDOW, LIMB_ENDS,
                     LIMB_WIDTH, LOSS_DRAWN, LOSS_GAIN, LOSS_ON_BODY, LOSS_REACH, LOSS_REACH_FRAMES,
                     LOSS_RUN, LOSS_WINDOW, MASK_CHECKS,
                     POSE_FREE_MASK_CHECKS, RELIABLE_CONF, RELIABLE_KEYPOINTS,
                     SKELETON_REACH, SPECK_FRACTION, WHOLE_BODY, MaskRow, _body, _box_iou_prev, _flag, _frame_pose,
                     _hand, _keypoint_rows, _pose_inputs, _thresholds, body_scale, box_sides,
                     out_of_shot_limbs)
from .config import LIMB_SPIKE, MaskGuardConfig, _config
from .pose import limb_spikes
from .report import _finish
from .timeline import MASK_PANELS

# A limb end's parent joint: the limb that reaches it runs from there.
PARENT = {3: 2, 4: 3, 6: 5, 7: 6, 9: 8, 10: 9, 12: 11, 13: 12, 18: 13, 19: 10}


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


def _mask_inputs(mask, pose_data: PoseData, final=False):
    """The mask as [frames, height, width] booleans and the pose it is checked against (None,
    None without pose_data), checked against each other; `final` says it is the final mask."""
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.dim() != 3:
        raise ValueError(f"mask must be [frames, height, width], got a tensor of shape {tuple(mask.shape)}")
    masks = (mask.cpu().numpy() > 0.5)
    hint = ("connect the final mask of the same frames (GrowMaskWithBlur and BlockifyMask keep the size), before any "
            "resize" if final else "connect the mask straight from the tracker, before any resize")
    return (masks, *pose_of(masks, pose_data, hint))


def pose_of(masks, pose_data: PoseData, hint):
    """The per-frame keypoints and detections of `pose_data` (None, None without it), checked
    against the [N, H, W] masks they judge; `hint` says what to connect when the sizes differ."""
    if pose_data is None:
        return None, None
    pose_metas, detections = _pose_inputs(pose_data)
    N, H, W = masks.shape
    if len(pose_metas) != N:
        raise ValueError(f"mask has {N} frames, pose_data {len(pose_metas)}; connect the outputs of the same clip")
    if pose_metas and (pose_metas[0]["height"], pose_metas[0]["width"]) != (H, W):
        raise ValueError(f"mask is {W}x{H} but the pose was found on {pose_metas[0]['width']}x{pose_metas[0]['height']} "
                         f"frames; {hint}")
    return pose_metas, detections


def _whole_body(meta: PoseMeta, W, H, draw_threshold):
    """Every drawn keypoint of the person inside the frame - body, hands and face - in pixels."""
    parts = [_keypoint_rows(meta, key) for key in WHOLE_BODY if key in meta]
    kps = np.concatenate(parts) * np.array([W, H, 1.0])
    return kps[(kps[:, 2] >= draw_threshold) & in_frame(kps, W, H)]


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


def _sides(stats, i, H, W):
    """The sides of the frame region `i` of cv2 connected-component `stats` reaches."""
    import cv2

    x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
    x2, y2 = x + stats[i, cv2.CC_STAT_WIDTH], y + stats[i, cv2.CC_STAT_HEIGHT]
    return {side for side, on in (("left", x == 0), ("top", y == 0), ("right", x2 == W), ("bottom", y2 == H)) if on}


def mask_regions(mask, person_kps, pad=0, around=()):
    """The person's part of `mask` and the regions detached from it, as fractions of its
    largest region, each region measured with `pad` px taken off its outline (the final mask's
    padding, FINAL_PAD; 0 on a raw mask).

    The person is the largest region and every region holding one of her own drawn keypoints
    (`person_kps`, [K, 2+] pixels inside the frame): a piece the frame edge or a gap in the
    mask has split off - a hand that leaves the shot and comes back at the corner, a hand the
    burned-in text cuts off the arm - is still her. Asking only whether a piece touches the
    frame border is not enough to say the border is what separates it: on a clip where the body
    runs off the bottom of every frame, that excuses anything at any edge, including the objects
    beside her that the decoder took in.

    Without a pose (`person_kps` None) nothing on the frame itself says which piece is hers but
    the frame edge: a piece that runs off a side of the frame the largest region also runs off is
    a part of her that side cut away (an arm that leaves the shot and comes back further along
    it). A piece off the edges, or at a side she does not reach, is not.

    `around` holds the person's parts of the frames around this one (LEAK_WINDOW either side,
    each by its own frame alone). A speck-sized piece (under FRAGMENT_FRACTION) most of which
    the person's part holds on at least two of them is hers, split off on this frame: a hair tip
    or a shoe the mask cut from the body. A leak island is not: an object beside her joins her
    part for a frame at most before it is detached or gone. A bigger piece is the mask torn in
    two, whoever it belongs to."""
    import cv2

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 2:
        return mask, []
    areas = stats[1:, cv2.CC_STAT_AREA]
    main = int(np.argmax(areas)) + 1
    if person_kps is None:
        H, W = mask.shape
        edges = _sides(stats, main, H, W)
        person = {i for i in range(1, count) if _sides(stats, i, H, W) & edges} | {main}
    else:
        xs = person_kps[:, 0].astype(int)
        ys = person_kps[:, 1].astype(int)
        person = (set(labels[ys, xs].tolist()) - {0}) | {main}
    if pad:
        core = cv2.erode(mask.astype(np.uint8), np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)).astype(bool)
        areas = np.bincount(labels[core], minlength=count)[1:]
    fraction = {i: round(float(areas[i - 1] / max(areas[main - 1], 1)), 4) for i in range(1, count) if i not in person}
    for i, f in fraction.items():
        if SPECK_FRACTION <= f < FRAGMENT_FRACTION and around:
            piece = labels == i
            if sum(2 * int((piece & part).sum()) > stats[i, cv2.CC_STAT_AREA] for part in around) >= 2:
                person.add(i)
    out = [f for i, f in fraction.items() if i not in person]
    return np.isin(labels, list(person)), sorted((f for f in out if f >= SPECK_FRACTION), reverse=True)


def detached_fractions(mask, person_kps):
    """The regions of `mask` detached from the person, as fractions of its largest region
    (see `mask_regions`)."""
    return mask_regions(mask, person_kps)[1]


def skeleton_zone(meta: PoseMeta, shape, draw_threshold, reach_scales=None, pad=0):
    """The pixels the drawn skeleton accounts for: `reach_scales` body scales (SKELETON_REACH
    when None), and `pad` px more (the final mask's padding), around every drawn body limb and
    keypoint and every drawn hand keypoint, and along every limb that runs out of the shot from
    a drawn end (a thigh whose knee is below the frame is body the pose image cannot draw)."""
    import cv2

    H, W = shape
    zone = np.zeros(shape, np.uint8)
    kps, drawn = _body(meta, W, H, draw_threshold)
    if not drawn.any():
        return zone.astype(bool)
    reach = max(3, int((SKELETON_REACH if reach_scales is None else reach_scales) * body_scale(kps, drawn))) + pad
    points = [kps[j, :2] for j in np.flatnonzero(drawn)]
    for arm in HANDS:
        points += list(_hand(meta, arm, W, H, draw_threshold))
    segments = [(kps[a, :2], kps[b, :2]) for a, b in LIMBS if drawn[a] and drawn[b]]
    segments += out_of_shot_limbs(meta, W, H, draw_threshold)
    for a, b in segments:
        cv2.line(zone, tuple(int(v) for v in a), tuple(int(v) for v in b), 1, 2 * reach)
    for x, y in points:
        cv2.circle(zone, (int(x), int(y)), reach, 1, -1)
    return zone.astype(bool)


def _dilated(mask, k):
    """`mask` dilated by a k x k square kernel."""
    import cv2

    return cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)


def _grown(mask, size):
    """`mask` dilated by a square kernel of 2% of `size`, odd and at least 3 pixels."""
    return _dilated(mask, max(3, int(0.02 * size) | 1))


def _final_mask(mask):
    """The final mask the Wan Animate workflow makes of one raw [H, W] boolean frame, the one the
    sampler and every model after the preprocess read: GrowMaskWithBlur (expand FINAL_GROW,
    tapered corners: FINAL_GROW dilations by a 3 x 3 cross; an empty frame stays empty), then
    BlockifyMask (FINAL_BLOCK): the grown mask's box cut into side // FINAL_BLOCK blocks along
    each side (at least one, the remainder joining the last block), a block on wherever it holds
    a grown pixel, nothing outside the box."""
    import cv2

    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    grown = cv2.dilate(mask.astype(np.uint8), cross, iterations=FINAL_GROW).astype(bool)
    out = np.zeros_like(grown)
    rows, cols = np.flatnonzero(grown.any(axis=1)), np.flatnonzero(grown.any(axis=0))
    if not len(rows):
        return out
    y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    hd, wd = max(1, (y1 - y0) // FINAL_BLOCK), max(1, (x1 - x0) // FINAL_BLOCK)
    by = np.minimum(np.arange(y1 - y0) // ((y1 - y0) // hd), hd - 1)
    bx = np.minimum(np.arange(x1 - x0) // ((x1 - x0) // wd), wd - 1)
    on = np.zeros((hd, wd), bool)
    ys, xs = np.nonzero(grown[y0:y1, x0:x1])
    on[by[ys], bx[xs]] = True
    out[y0:y1, x0:x1] = on[np.ix_(by, bx)]
    return out


def _thickest(lost):
    """The radius of the largest disc inside the boolean map `lost`, its piece (the connected
    region holding that disc) and the piece's top-left corner (y, x). The frame border is not
    the region's edge: a piece the frame cuts continues beyond it."""
    import cv2

    ys, xs = np.nonzero(lost)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    H, W = lost.shape
    if y0 == 0 or x0 == 0 or y1 == H or x1 == W:
        dt = cv2.distanceTransform(lost.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[y0:y1, x0:x1]
    else:
        dt = cv2.distanceTransform(np.pad(lost[y0:y1, x0:x1], 1).astype(np.uint8), cv2.DIST_L2,
                                   cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    _, labels = cv2.connectedComponents(lost[y0:y1, x0:x1].astype(np.uint8), connectivity=8)
    piece = labels == labels[np.unravel_index(int(dt.argmax()), dt.shape)]
    py, px = np.nonzero(piece)
    return float(dt.max()), piece[py.min():py.max() + 1, px.min():px.max() + 1], (y0 + py.min(), x0 + px.min())


def _crosses(limbs):
    """The body test of a raw mask: frame f's drawn body limbs (`limbs(f)`, pixel point pairs)
    cross the piece `P`, whose top-left pixel is at `origin` (y, x)."""
    def on_body(f, P, origin):
        import cv2

        (y0, x0), drawn = origin, np.zeros(P.shape, np.uint8)
        for a, b in limbs(f):
            cv2.line(drawn, (int(a[0]) - x0, int(a[1]) - y0), (int(b[0]) - x0, int(b[1]) - y0), 1, 1)
        return bool((drawn.astype(bool) & P).any())
    return on_body


def _holds(keypoints):
    """The body test of the final mask: the piece `P` holds one of frame f's drawn keypoints
    (`keypoints(f)`, [K, 2+] pixels inside the frame)."""
    def on_body(f, P, origin):
        (y0, x0), kps = origin, keypoints(f)
        ys, xs = kps[:, 1].astype(int) - y0, kps[:, 0].astype(int) - x0
        inside = (ys >= 0) & (ys < P.shape[0]) & (xs >= 0) & (xs < P.shape[1])
        return bool(P[ys[inside], xs[inside]].any())
    return on_body


def _run_evidence(masks, t, g, piece, corner, reach, on_body):
    """Around the dropped piece of the run t..t+g-1: the mask that turns up within `reach`
    pixels of it on the run's frames and on neither end, as a share of its area on average (a
    limb that moved away), and the share of the run's frames the pose puts the body in it
    (`on_body`, `_crosses` or `_holds`; None without a pose)."""
    import cv2

    N, H, W = masks.shape
    (py, px), (ph, pw), r = corner, piece.shape, int(reach) + 1
    y0, y1, x0, x1 = max(0, py - r), min(H, py + ph + r), max(0, px - r), min(W, px + pw + r)
    P = np.zeros((y1 - y0, x1 - x0), bool)
    P[py - y0:py - y0 + ph, px - x0:px - x0 + pw] = piece
    near = cv2.distanceTransform((~P).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) <= reach
    ends = masks[t - 1, y0:y1, x0:x1] | masks[t + g, y0:y1, x0:x1]
    gain = float(np.mean([(masks[f, y0:y1, x0:x1] & ~ends & near).sum() for f in range(t, t + g)])) / int(piece.sum())
    if on_body is None:
        return gain, None
    return gain, float(np.mean([on_body(f, P, (y0, x0)) for f in range(t, t + g)]))


def _loss_warns(single, run, drawn, max_mask_loss, final):
    """Whether dropout thicknesses reach the mask_loss warning: a single-frame one thicker than
    `max_mask_loss`, a run of two frames or more LOSS_RUN times that, one the pose puts the body in
    LOSS_DRAWN of it (each None where there is none). On the final mask (`final`) only the last
    counts: its outline moves by a block with no change in the raw mask."""
    return ((drawn is not None and drawn > LOSS_DRAWN * max_mask_loss)
            or (not final and single is not None and single > max_mask_loss)
            or (not final and run is not None and run > LOSS_RUN * max_mask_loss))


def dropouts(masks, max_mask_loss, on_body=None, share=LOSS_ON_BODY, reading=None, final=False):
    """mask_loss: per frame, the thickest region the mask drops on a run of up to LOSS_WINDOW
    frames covering it while holding it on the frame before the run and the frame after it - a
    dropout, out and back, the mask's counterpart of pose_spike - that is not a limb which moved
    away and came back (LOSS_GAIN), and that is a part of her rather than background blinking
    off: the pose puts the body in it on the frame before or the frame after the run, or the mask
    holds most of it on another frame within LOSS_WINDOW before the run and on another within
    LOSS_WINDOW after it (at the clip's first or last frame there is no frame beyond to ask).
    Background that joins the mask on the frames around a run and is gone beyond them is a leak
    blinking off, not a region the mask dropped; a hand that leaves the shot right after the run
    is gone beyond it too, but the pose has it on the frame next to the run. Thickness is the
    radius of the largest disc inside the region, as a fraction of the frame's shorter side: a
    dropped hand or foot holds a disc of its own width, the slivers a mask's outline jitters by
    are only a few pixels thick however long they are. Regions thinner than LOSS_DRAWN of
    `max_mask_loss` (the same fraction) are not measured.

    `reading(f)`, when given, is what the model downstream reads of frame f's mask, [H, W]
    booleans: the final the Wan Animate workflow grows the raw mask into (`_final_mask`), or the
    latent grid SCAIL-2 reads its colored driving mask on (`scail2.latent_reading`). A region is
    dropped only where the mask and its reading both drop it: both hold it on the frames around
    the run and neither holds it on any frame of the run - what the reading of a run frame holds
    reaches the model, and what the reading of the frames around it leaves out the model never
    had. A final holds all of its raw mask, so on the raw mask only the finals of the run count.

    `masks` is [N, H, W] booleans; `on_body(f, piece, origin)` says whether the pose puts the
    body in the piece on frame f (None without a pose): the drawn skeleton crosses it on a raw
    mask, it holds a drawn keypoint on the final mask. `final` says `masks` are final masks
    (`_loss_warns`). Returns (loss, loss_run, loss_drawn, loss_area), per frame: the thickness of
    the single-frame dropout on it and of the thickest run of two frames or more over it (0.0
    without one, None on the first and the last frame), the thickest dropout over it the pose puts
    the body in on at least `share` of the frames of its run (None without `on_body`), and the
    largest region a dropout over it that reaches the mask_loss warning drops, as a share of the
    mask on the frame before its run (0.0 without one, None on the first and the last frame)."""
    N, H, W = masks.shape
    S = min(H, W)
    floor = LOSS_DRAWN * max_mask_loss
    loss = [None if i in (0, N - 1) else 0.0 for i in range(N)]
    loss_run = list(loss)
    loss_area = list(loss)
    loss_drawn = [None] * N if on_body is None else list(loss)
    min_area = np.pi * (floor * S) ** 2    # a region holding a disc of the floor's radius is at least this big
    views = {}                             # frame -> (what the mask and its reading both hold, what either holds)

    def view(f):
        if f not in views:
            read = masks[f] if reading is None else reading(f)
            views[f] = (masks[f] & read, masks[f] | read)
        return views[f]

    def held(f, piece, corner):
        (py, px), (ph, pw) = corner, piece.shape
        return 2 * int((view(f)[0][py:py + ph, px:px + pw] & piece).sum()) > int(piece.sum())

    def hers(t, g, piece, corner):
        if on_body is not None and any(on_body(f, piece, corner) for f in (t - 1, t + g)):
            return True
        beyond = (range(max(0, t - LOSS_WINDOW), t - 1), range(t + g + 1, min(N, t + g + LOSS_WINDOW)))
        return all(any(held(f, piece, corner) for f in frames) for frames in beyond if len(frames))

    for t in range(1, N - 1):
        for f in [f for f in views if f < t - LOSS_WINDOW]:
            del views[f]
        before = view(t - 1)[0]
        gone = np.zeros((H, W), bool)
        for g in range(1, LOSS_WINDOW + 1):
            if t + g >= N:
                break
            gone |= view(t + g - 1)[1]
            if (before & ~gone).sum() < min_area:
                break                                  # nothing held before the run is left to drop
            lost = before & view(t + g)[0] & ~gone
            if lost.sum() < min_area:
                continue
            radius, piece, corner = _thickest(lost)
            if radius < floor * S:
                continue
            if not hers(t, g, piece, corner):
                continue                               # background blinking off
            gain, body = _run_evidence(masks, t, g, piece, corner, min(g, LOSS_REACH_FRAMES) * LOSS_REACH * S, on_body)
            if gain >= LOSS_GAIN:
                continue                               # a limb that moved away and came back
            thickness, drawn = radius / S, body is not None and body >= share
            warns = _loss_warns(thickness if g == 1 else None, thickness if g > 1 else None,
                                thickness if drawn else None, max_mask_loss, final)
            area = int(piece.sum()) / max(int(masks[t - 1].sum()), 1)
            for f in range(t, t + g):
                if g == 1:
                    loss[f] = max(loss[f], thickness)
                else:
                    loss_run[f] = max(loss_run[f], thickness)
                if drawn:
                    loss_drawn[f] = max(loss_drawn[f], thickness)
                if warns:
                    loss_area[f] = max(loss_area[f], area)
    return loss, loss_run, loss_drawn, loss_area


def lost_limb_end(j, kps, drawn, mask, grown, meta: PoseMeta, draw_threshold, spikes, scale):
    """Whether the drawn limb end `j`, outside the grown mask, is a limb the mask lost - a
    visible limb end outside it (motion blur, a burned-in label over it) - and not one of:

    - the limb leaving the shot: the keypoint lies on the frame's outermost row or column, where
      the pose model places a joint at the edge of what it sees; the limb runs out of the frame
      there and its end is beyond it;
    - a pose error: the pose checks call the keypoint a spike on this frame (`spikes`), or it
      is a wrist whose own drawn hand points back along the forearm, more than a right angle
      from it - no wrist bends that far, so the wrist keypoint is not where the hand's wrist is;
    - a notch: less than half of the limb's end, the disc of LIMB_WIDTH body scales around the
      keypoint, is outside the mask - a keypoint in a small cut of an otherwise whole mask;
    - fast motion: the limb from its drawn parent joint runs outside the grown mask for more
      than LIMB_WIDTH body scales beyond the keypoint's own distance from it - the pose drew the
      limb beside the body, following the blur the mask leaves out, rather than out of it."""
    import cv2

    H, W = mask.shape
    x, y = kps[j, :2]
    if int(x) in (0, W - 1) or int(y) in (0, H - 1) or BODY_NAMES[j] in spikes:
        return False
    arm = CARRIES[j]
    if j == arm[-1] and arm in HANDS and drawn[PARENT[j]]:
        fingers = _hand(meta, arm, W, H, draw_threshold, fingers_only=True)
        if len(fingers) and np.dot(fingers.mean(axis=0) - kps[j, :2], kps[j, :2] - kps[PARENT[j], :2]) < 0:
            return False
    r = max(1, int(LIMB_WIDTH * scale))
    x0, x1, y0, y1 = max(0, int(x) - r), min(W, int(x) + r + 1), max(0, int(y) - r), min(H, int(y) + r + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disc = (xx - int(x)) ** 2 + (yy - int(y)) ** 2 <= r * r
    if disc.any() and (~mask[y0:y1, x0:x1])[disc].mean() < 0.5:
        return False
    parent = PARENT[j]
    if drawn[parent]:
        a, b = kps[parent, :2], kps[j, :2]
        length = float(np.hypot(*(b - a)))
        t = np.linspace(0.0, 1.0, max(2, int(length)))[:, None]
        pts = a + t * (b - a)
        xs = np.clip(pts[:, 0].astype(int), 0, W - 1)
        ys = np.clip(pts[:, 1].astype(int), 0, H - 1)
        outside = float((~grown[ys, xs]).mean()) * length
        away = cv2.distanceTransform((~grown).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[int(y), int(x)]
        if outside - float(away) > LIMB_WIDTH * scale:
            return False
    return True


def mask_frame_metrics(masks, pose_metas: list[PoseMeta], detections: list[Detection], draw_threshold,
                       max_mask_loss, max_limb_spike=LIMB_SPIKE, final=False, reading=None,
                       read_keypoints=False) -> list[MaskRow]:
    """One dict of raw mask measurements per frame (keys MASK_ROW); thresholds are applied
    afterwards, except where a measurement starts: `max_mask_loss` sets the thinnest dropout
    measured (LOSS_DRAWN of it) and the dropouts whose area mask_loss_area measures (those that
    reach the mask_loss warning), and `max_limb_spike` is pose_spike's jump (a limb end the pose
    checks call a spike is not a limb the mask lost). `masks` is [N, H, W] booleans on the
    frames the pose was found on; without a pose (`pose_metas` None) the pose-based
    measurements are None and the lists empty. `final` says the masks are the final masks of
    the Wan Animate workflow, measured as common.FINAL_BLOCK describes (it needs a pose).
    `reading(f)` is what the model downstream reads of frame f's mask, for mask_loss (None: the
    mask as it is; see `dropouts`); with `read_keypoints` the keypoint tests read it too: a drawn
    keypoint the reading holds is inside the mask, and a limb end is measured against the mask
    and its reading together."""
    N, H, W = masks.shape
    posed = pose_metas is not None
    areas = masks.reshape(N, H * W).sum(axis=1)
    pad = FINAL_PAD if final else 0
    if posed:
        envelopes = box_envelopes(detections, N, W, H)
        body = [_body(meta, W, H, draw_threshold) for meta in pose_metas]
        whole = [_whole_body(meta, W, H, draw_threshold) for meta in pose_metas]
        spikes = limb_spikes(np.array([kps[:, :2] for kps, _ in body]).reshape(N, len(BODY_NAMES), 2),
                             np.array([on for _, on in body]).reshape(N, len(BODY_NAMES)), H, max_limb_spike)
    def drawn_limbs(f):
        kps, drawn = body[f]
        return [(kps[a, :2], kps[b, :2]) for a, b in LIMBS if drawn[a] and drawn[b]]

    if not posed:
        on_body, share = None, LOSS_ON_BODY
    elif final:
        on_body, share = _holds(lambda f: whole[f]), FINAL_ON_BODY
    else:
        on_body, share = _crosses(drawn_limbs), LOSS_ON_BODY
    loss, loss_run, loss_drawn, loss_area = dropouts(masks, max_mask_loss, on_body, share, reading, final)
    alone = {}                             # frame -> its mask_regions by that frame alone

    def regions(f):
        if f not in alone:
            alone[f] = mask_regions(masks[f], whole[f] if posed else None, pad)
        return alone[f]

    rows = []
    prev_mask = None
    for i in range(N):
        mask = masks[i]
        area = int(areas[i])
        m = {"frame": i, "mask_area": area / (H * W), "mask_to_box": None, "box_reliable": False,
             "mask_outside_box": None, "attached_leak": None}
        near = [j for j in range(max(0, i - LEAK_WINDOW), min(N, i + LEAK_WINDOW + 1)) if j != i]
        for f in [f for f in alone if f < i - LEAK_WINDOW]:
            del alone[f]

        # the person's part of the mask and the regions detached from it; a speck-sized piece that
        # is hers on the frames around it is hers on this one too
        person, fragments = regions(i) if area else (mask, [])
        if any(f < FRAGMENT_FRACTION for f in fragments):
            person, fragments = mask_regions(mask, whole[i] if posed else None, pad, [regions(j)[0] for j in near])
        m["fragments"] = fragments
        if not posed:
            m.update({"keypoint_recall": None, "missed_keypoints": [], "missed_limbs": [], "body_not_drawn": None,
                      "box_iou_prev": None})
        else:
            det, meta = detections[i], pose_metas[i]
            bw, bh, diag, kps, drawn, visible = _frame_pose(meta, det, W, H, draw_threshold)
            m["mask_to_box"] = area / (bw * bh)
            m["box_reliable"] = bool(det["score"] > 0 and drawn.sum() >= RELIABLE_KEYPOINTS
                                     and (kps[drawn, 2].mean() if drawn.any() else 0) >= RELIABLE_CONF)

            gx1, gy1, gx2, gy2 = envelopes[i]
            inside = int(mask[int(gy1):int(gy2), int(gx1):int(gx2)].sum())
            m["mask_outside_box"] = (area - inside) / area if area else 0.0

            # the person's region grown where neither the neighbours' masks nor the skeleton are
            if area and near:
                reference = masks[near].any(axis=0)
                grown = (_dilated(reference, 2 * FINAL_BLOCK + 1) if final
                         else _grown(reference, np.sqrt(np.median(areas[near]))))
                excess = person & ~grown & ~skeleton_zone(meta, (H, W), draw_threshold, LEAK_REACH)
                m["attached_leak"] = float(excess.sum() / max(np.median(areas[near]), 1))
            else:
                m["attached_leak"] = 0.0

            # drawn keypoints inside the (slightly grown) mask, or its reading; the limb ends outside
            # both the mask lost
            if area and visible.any():
                seen = mask | reading(i) if read_keypoints else mask
                grown = _grown(mask, diag) | seen
                xs = np.clip(kps[:, 0].astype(int), 0, W - 1)
                ys = np.clip(kps[:, 1].astype(int), 0, H - 1)
                hit = grown[ys, xs] & visible
                m["keypoint_recall"] = float(hit.sum() / visible.sum())
                m["missed_keypoints"] = [BODY_NAMES[j] for j in np.flatnonzero(visible & ~hit)]
                scale = body_scale(kps, drawn)
                m["missed_limbs"] = [BODY_NAMES[j] for j in np.flatnonzero(visible & ~hit) if j in LIMB_ENDS and
                                     lost_limb_end(j, kps, drawn, seen, grown, meta, draw_threshold, spikes[i], scale)]
            else:
                m["keypoint_recall"] = 0.0 if visible.any() else 1.0
                m["missed_keypoints"] = [BODY_NAMES[j] for j in np.flatnonzero(visible)] if area == 0 else []
                m["missed_limbs"] = [name for name in m["missed_keypoints"] if BODY_NAMES.index(name) in LIMB_ENDS]

            # the person's mask the drawn skeleton does not account for
            person_area = int(person.sum())
            m["body_not_drawn"] = (float((person & ~skeleton_zone(meta, (H, W), draw_threshold, pad=pad)).sum()
                                         / person_area) if person_area else 0.0)
            m["box_iou_prev"] = _box_iou_prev(detections, i)

        # motion against the previous frame, and a region the mask drops on a run through this frame
        m["mask_iou_prev"] = _iou(mask, prev_mask) if prev_mask is not None else None
        m["mask_loss"], m["mask_loss_run"], m["mask_loss_drawn"] = loss[i], loss_run[i], loss_drawn[i]
        m["mask_loss_area"] = loss_area[i]
        rows.append(m)
        prev_mask = mask
    return rows


def mask_flags(rows: list[MaskRow], t, final=False, checks=MASK_CHECKS):
    """The mask checks: check name -> frames it fired on, in the order the checks first fired.
    A check whose measurement is None on a frame (no pose_data, no neighbour) does not run there.
    On the final mask (`final`) mask_loss counts only a dropout the pose puts the body in
    (mask_loss_drawn): its outline moves by a block with no change in the raw mask.

    Two fails take the place of a warning on their frame, where the caller runs them (`checks`;
    the SCAIL-2 guard does not, and its `t` has no thresholds for them): mask_head_out that of
    mask_missing_keypoints - the drawn nose, or `head_out_eyes_ears` of the drawn eyes and ears,
    outside the mask - and mask_loss_large that of mask_loss - the dropout drops
    `large_loss_area` of the person or more (mask_loss_area). A frame
    whose mask is empty is mask_empty's (a fail where the detector and the pose agree on the frame,
    the pose's failure elsewhere), as it is mask_missing_keypoints'."""
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
        if m["attached_leak"] is not None and m["attached_leak"] > t["max_attached_leak"]:
            _flag(flags, "mask_attached_leak", i)
        if any(f >= FRAGMENT_FRACTION for f in m["fragments"]):
            _flag(flags, "mask_fragmented", i)
        elif m["fragments"]:
            _flag(flags, "mask_specks", i)
        missed = set(m["missed_keypoints"])
        if ("mask_head_out" in checks and m["mask_area"] > 0 and
                (HEAD_OUT_KEYPOINT in missed or len(missed & set(HEAD_OUT_SIDES)) >= t["head_out_eyes_ears"])):
            _flag(flags, "mask_head_out", i)
        elif m["keypoint_recall"] is not None and m["keypoint_recall"] < t["min_keypoint_recall"] and m["mask_area"] > 0:
            _flag(flags, "mask_missing_keypoints", i)
        if m["missed_limbs"]:
            _flag(flags, "mask_missed_limb", i)
        if m["body_not_drawn"] is not None and m["body_not_drawn"] > t["max_body_not_drawn"]:
            _flag(flags, "body_not_drawn", i)
        if (m["mask_iou_prev"] is not None and m["box_iou_prev"] is not None
                and m["box_iou_prev"] > 0.7 and m["mask_iou_prev"] < t["min_mask_iou"]):
            _flag(flags, "mask_unstable", i)
        if _loss_warns(m["mask_loss"], m["mask_loss_run"], m["mask_loss_drawn"], t["max_mask_loss"], final):
            large = "mask_loss_large" in checks and m["mask_area"] > 0 and m["mask_loss_area"] >= t["large_loss_area"]
            _flag(flags, "mask_loss_large" if large else "mask_loss", i)
    return flags


def check_mask(mask, pose_data: PoseData = None, config=None, enabled=True, stop_on_fail=True, max_limb_spike=LIMB_SPIKE,
               final=False):
    """The mask checks on `mask` [frames, H, W] (or one [H, W] frame) against the keypoints and
    boxes in `pose_data`, which must be of the same frames at the same size. Without pose_data
    only the checks that do not read it run (POSE_FREE_MASK_CHECKS), and the report says so.

    `config` is a MaskGuardConfig (None = defaults); `enabled` and `stop_on_fail` as in
    `check_pose`; `max_limb_spike` is pose_spike's jump (the Pose Guard's widget in the wrapper
    that runs both groups). `final` says `mask` is the final mask of the Wan Animate workflow,
    grown and blockified (the WanAnimate Preprocess Guard's), whose measures allow for its blocks
    (common.FINAL_BLOCK); it needs pose_data. Otherwise `mask` is the raw mask, which the
    workflow grows into that final before any model reads it: a region it drops counts only
    where the final leaves it out (`_final_mask`). Returns (mask unchanged, report, metrics
    JSON, timeline IMAGE)."""
    config = _config(config, MaskGuardConfig)
    if final and pose_data is None:
        raise ValueError("the final mask is judged against its pose; connect pose_data")
    masks, pose_metas, detections = _mask_inputs(mask, pose_data, final)
    thresholds = _thresholds(pose_data, config)
    with log.step(f"mask guard: checking {len(masks)} frames ({'on' if enabled else 'off'})"):
        rows = mask_frame_metrics(masks, pose_metas, detections, thresholds["draw_threshold"], config.max_mask_loss,
                                  max_limb_spike, final, None if final else lambda f: _final_mask(masks[f]))
        flags = mask_flags(rows, thresholds, final)
    note = None if pose_data is not None else (
        "without pose_data, not checked: " + ", ".join(name for name in MASK_CHECKS if name not in POSE_FREE_MASK_CHECKS))
    report, metrics, timeline = _finish("Mask guard", "mask", rows, flags, thresholds,
                                        set(MASK_CHECKS if enabled else ()), MASK_PANELS, stop_on_fail, note)
    return mask, report, metrics, timeline
