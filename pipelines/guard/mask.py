"""The mask checks: the per-frame measurements of the mask against the pose it belongs to (or
alone, without pose_data), the flags they raise, and check_mask."""
from dataclasses import asdict
from functools import reduce

import numpy as np

from ...libs import log
from ...libs.keypoints import BODY_NAMES, LIMBS, in_frame
from ...libs.pose_data import Detection, PoseData, PoseMeta
from .common import (BOX_MARGIN, BOX_WINDOW, FINAL_BLOCK, FINAL_GROW, FINAL_PAD, FRAGMENT_FRACTION, HANDS,
                     HEAD_OUT_KEYPOINT, HEAD_OUT_SIDES, LEAK_REACH, LEAK_WINDOW, LOSS_GAIN, LOSS_HAND, LOSS_REACH,
                     LOSS_REACH_FRAMES, LOSS_WINDOW, MASK_CHECKS, POSE_FREE_MASK_CHECKS, RELIABLE_CONF,
                     RELIABLE_KEYPOINTS, SPECK_FRACTION, WHOLE_BODY, MaskRow, _body, _box_iou_prev, _flag,
                     _frame_pose, _hand, _keypoint_rows, _pose_inputs, _thresholds, body_scale, box_sides, out_of_shot,
                     out_of_shot_limbs)
from .config import FinalReferenceGuardConfig, MaskGuardConfig, ReferenceGuardConfig, _config
from .reference import mask_reference
from .report import _finish
from .timeline import MASK_PANELS

# The head keypoints mask_head_out reads, as body keypoint indices.
HEAD = [BODY_NAMES.index(name) for name in (HEAD_OUT_KEYPOINT,) + HEAD_OUT_SIDES]
# No part of the pose left out: the guards count the keypoints the pose model draws at the draw
# threshold whatever the Pose Config draw rules leave out of the images (common._draw_threshold).
NOTHING_HIDDEN = {"body": [], "hands": []}


class _Booleans:
    """The [frames, height, width] float mask as the checks read it, a frame's booleans (above 0.5)
    at a time - masks[f], masks[f, rows, columns], its `shape` and len() - so the clip is never
    held as booleans whole (0.56 GB over 609 frames at 720p). A frame is cut when it is read and
    kept while it is one of the WINDOW frames read last: the frame the per-frame measurements are
    on and LEAK_WINDOW frames either side of it, each cut once as they move along the clip."""

    WINDOW = 2 * LEAK_WINDOW + 1

    def __init__(self, mask):
        self.mask, self.shape, self.kept = mask, tuple(mask.shape), {}

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        f, box = (key[0], key[1:]) if isinstance(key, tuple) else (key, ())
        frame = self.kept.pop(f, None)
        if frame is None:
            frame = (self.mask[f] > 0.5).numpy()
            if len(self.kept) == self.WINDOW:
                del self.kept[next(iter(self.kept))]         # the frame read longest ago
        self.kept[f] = frame
        return frame[box] if box else frame


def _mask_inputs(mask, pose_data: PoseData, final=False):
    """The mask as [frames, height, width] booleans (_Booleans) and the pose it is checked against
    (None, None without pose_data), checked against each other; `final` says it is the final mask."""
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.dim() != 3:
        raise ValueError(f"mask must be [frames, height, width], got a tensor of shape {tuple(mask.shape)}")
    masks = _Booleans(mask.cpu())
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


def skeleton_zone(meta: PoseMeta, shape, draw_threshold, reach_scales):
    """The pixels the drawn skeleton accounts for: `reach_scales` body scales around every drawn
    body limb and keypoint and every drawn hand keypoint, and along every limb that runs out of
    the shot from a drawn end (a thigh whose knee is below the frame is body the pose image
    cannot draw)."""
    import cv2

    H, W = shape
    zone = np.zeros(shape, np.uint8)
    kps, drawn = _body(meta, W, H, draw_threshold)
    if not drawn.any():
        return zone.astype(bool)
    reach = max(3, int(reach_scales * body_scale(kps, drawn)))
    points = [kps[j, :2] for j in np.flatnonzero(drawn)]
    for key in HANDS:
        points += list(_hand(meta, key, W, H, draw_threshold))
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


def _grow_side(size):
    """The side of _grown's square kernel: 2% of `size`, odd and at least 3 pixels."""
    return max(3, int(0.02 * size) | 1)


def _grown(mask, size):
    """`mask` dilated by a square kernel of 2% of `size`, odd and at least 3 pixels."""
    return _dilated(mask, _grow_side(size))


def _box(region):
    """The box of the [H, W] booleans `region` as (row slice, column slice), None when it is empty."""
    rows, cols = np.flatnonzero(region.any(axis=1)), np.flatnonzero(region.any(axis=0))
    if not len(rows):
        return None
    return slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)


# --- the final mask and the grids the models read ----------------------------------------------

def _block_edges(start, end):
    """BlockifyMask's cuts of one side of the box, start to end: side // FINAL_BLOCK blocks (at
    least one), the remainder joining the last, as the edges from `start` to `end`."""
    n = max(1, (end - start) // FINAL_BLOCK)
    size = (end - start) // n
    return np.array([start + i * size for i in range(n)] + [end])


def block_grid(mask):
    """BlockifyMask's grid over the box of `mask` ([H, W] booleans, the grown mask; a final mask
    has the same box): (row edges, column edges), each block from one edge to the next along
    both sides; None on an empty frame. The grid is laid from each frame's own box."""
    box = _box(mask)
    if box is None:
        return None
    rows, cols = box
    return _block_edges(rows.start, rows.stop), _block_edges(cols.start, cols.stop)


def _final_blocks(mask):
    """The final mask the Wan Animate workflow makes of one raw [H, W] boolean frame, the one the
    sampler and every model after the preprocess read, as its block grid and its blocks:
    GrowMaskWithBlur (expand FINAL_GROW, tapered corners: FINAL_GROW dilations by a 3 x 3 cross; an
    empty frame stays empty), then BlockifyMask (FINAL_BLOCK, block_grid): a block on wherever it
    holds a grown pixel, nothing outside the box. Returns (grid, [rows, columns] booleans, the
    blocks that are on), None on an empty frame. The grown mask lies within FINAL_GROW px of the
    mask's box, so only that part of the frame is grown."""
    import cv2

    box = _box(mask)
    if box is None:
        return None
    H, W = mask.shape
    y0, x0 = max(0, box[0].start - FINAL_GROW), max(0, box[1].start - FINAL_GROW)
    y1, x1 = min(H, box[0].stop + FINAL_GROW), min(W, box[1].stop + FINAL_GROW)
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    grown = cv2.dilate(mask[y0:y1, x0:x1].astype(np.uint8), cross, iterations=FINAL_GROW).astype(bool)
    rows, cols = block_grid(grown)
    grown = grown[rows[0]:rows[-1], cols[0]:cols[-1]]
    on = np.logical_or.reduceat(np.logical_or.reduceat(grown, rows[:-1] - rows[0], axis=0), cols[:-1] - cols[0], axis=1)
    return (rows + y0, cols + x0), on


def _final_reading(masks):
    """`read(f)` (see lost_regions) of the raw [N, H, W] boolean `masks`: frame f's final mask, [H, W]
    booleans laid out from its blocks, and its grid (_final_blocks). Each frame is grown and
    blockified once, however often it is read."""
    shape, blocks = masks.shape[1:], {}

    def read(f):
        if f not in blocks:
            blocks[f] = _final_blocks(masks[f])
        final = np.zeros(shape, bool)
        if blocks[f] is None:
            return final, None
        (rows, cols), on = blocks[f]
        on = np.repeat(np.repeat(on, np.diff(rows), axis=0), np.diff(cols), axis=1)
        final[rows[0]:rows[-1], cols[0]:cols[-1]] = on
        return final, (rows, cols)
    return read


def _holds_unit(grid, piece, corner):
    """Whether the piece `piece` (top-left pixel at `corner`, (y, x)) holds a whole unit of `grid`
    ((row edges, column edges): a block of the final, a cell of SCAIL-2's latent grid)."""
    if grid is None:
        return False
    rows, cols = grid
    (py, px), (ph, pw) = corner, piece.shape
    r0, r1 = np.searchsorted(rows, py, "left"), np.searchsorted(rows, py + ph, "right") - 1
    c0, c1 = np.searchsorted(cols, px, "left"), np.searchsorted(cols, px + pw, "right") - 1
    for a, b in zip(rows[r0:r1], rows[r0 + 1:r1 + 1]):
        for c, d in zip(cols[c0:c1], cols[c0 + 1:c1 + 1]):
            if piece[a - py:b - py, c - px:d - px].all():
                return True
    return False


# --- mask_loss: a region of her the model loses ------------------------------------------------

def _piece(labels, stats, i):
    """Piece `i` of cv2 connected-component `labels` and `stats`, as (piece, top-left corner (y, x))."""
    x, y, w, h = (int(v) for v in stats[i, :4])
    return labels[y:y + h, x:x + w] == i, (y, x)


def _pieces(lost):
    """The connected pieces of the boolean map `lost`, each as (piece, top-left corner (y, x))."""
    import cv2

    count, labels, stats, _ = cv2.connectedComponentsWithStats(lost.astype(np.uint8), connectivity=8)
    for i in range(1, count):
        yield _piece(labels, stats, i)


def _gain(masks, anchors, run, piece, corner, reach):
    """Around the piece: the mask that turns up within `reach` pixels of it on the frames of the
    `run` and on none of the `anchors`, as a share of its area on average - a limb that moved away
    is somewhere while it is away, a dropped part is nowhere."""
    import cv2

    N, H, W = masks.shape
    (py, px), (ph, pw), r = corner, piece.shape, int(reach) + 1
    y0, y1, x0, x1 = max(0, py - r), min(H, py + ph + r), max(0, px - r), min(W, px + pw + r)
    P = np.zeros((y1 - y0, x1 - x0), bool)
    P[py - y0:py - y0 + ph, px - x0:px - x0 + pw] = piece
    near = cv2.distanceTransform((~P).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) <= reach
    ends = np.zeros_like(P)
    for a in anchors:
        ends |= masks[a, y0:y1, x0:x1]
    return float(np.mean([(masks[f, y0:y1, x0:x1] & ~ends & near).sum() for f in run])) / int(piece.sum())


class _LimbEvidence:
    """The body test of a raw or a SCAIL-2 driving mask: whether the pose puts her body in a piece
    the mask drops. On a frame at the run's ends her drawn limbs cross it (`at_end`); on a frame of
    the run (`on_run`) they cross it too, or a limb that crosses it on an end frame is lost by the
    pose as well - not drawn, and no undrawn end of it out of the shot - so the pose cannot say
    where the limb went (a limb that moved away is drawn elsewhere; one that left the shot is
    placed at the edge). The frames' body keypoints are those of `pose_metas` drawn at
    `draw_threshold`, in pixels (common._body)."""

    def __init__(self, pose_metas: list[PoseMeta], W, H, draw_threshold):
        self.body, self.W, self.H = [_body(meta, W, H, draw_threshold) for meta in pose_metas], W, H

    def crossing(self, f, piece, corner):
        import cv2

        kps, drawn = self.body[f]
        (y0, x0), out = corner, set()
        for j, (a, b) in enumerate(LIMBS):
            if drawn[a] and drawn[b]:
                line = np.zeros(piece.shape, np.uint8)
                cv2.line(line, (int(kps[a, 0]) - x0, int(kps[a, 1]) - y0), (int(kps[b, 0]) - x0, int(kps[b, 1]) - y0), 1, 1)
                if (line.astype(bool) & piece).any():
                    out.add(j)
        return out

    def lost(self, f, j):
        kps, drawn = self.body[f]
        a, b = LIMBS[j]
        if drawn[a] and drawn[b]:
            return False
        scale = body_scale(kps, drawn)
        return not any(out_of_shot(kps[e, :2], self.W, self.H, scale) for e in (a, b) if not drawn[e])

    def at_end(self, f, piece, corner):
        return bool(self.crossing(f, piece, corner))

    def on_run(self, run, anchors, piece, corner):
        crossing = set().union(*(self.crossing(a, piece, corner) for a in anchors))
        return all(self.crossing(f, piece, corner) or any(self.lost(f, j) for j in crossing) for f in run)


class _KeypointEvidence:
    """The body test of the final mask: the piece holds one of the frame's drawn keypoints (its
    outline moves by a block with no change in the raw mask, while the block of a keypoint inside
    the raw mask is always on). `keypoints` is the frames' drawn keypoints in pixels."""

    def __init__(self, keypoints):
        self.keypoints = keypoints

    def at_end(self, f, piece, corner):
        (y0, x0), kps = corner, self.keypoints[f]
        ys, xs = kps[:, 1].astype(int) - y0, kps[:, 0].astype(int) - x0
        inside = (ys >= 0) & (ys < piece.shape[0]) & (xs >= 0) & (xs < piece.shape[1])
        return bool(piece[ys[inside], xs[inside]].any())

    def on_run(self, run, anchors, piece, corner):
        return all(self.at_end(f, piece, corner) for f in run)


class _Reading:
    """What the model reads of each frame of `masks` (`read`, see lost_regions), frame by frame and kept
    in `kept` while a run may ask for it: reading(f) is (where the mask and its reading both hold her,
    where either does, the grid the reading is laid on, the box of where either does: _box). `areas`
    is each frame's mask area in pixels, at least 1; `present` says which frames have a mask."""

    def __init__(self, masks, read):
        self.masks, self.read, self.kept = masks, read, {}
        counts = [int(np.count_nonzero(masks[f])) for f in range(masks.shape[0])]
        self.present, self.areas = np.array(counts, dtype=bool), [max(count, 1) for count in counts]

    def __call__(self, f):
        if f not in self.kept:
            reading, grid = self.read(f)
            frame = self.masks[f]
            both, either = (frame, frame) if reading is None else (frame & reading, frame | reading)
            self.kept[f] = (both, either, grid, _box(either))
        return self.kept[f]

    def whole_unit(self, frames, piece, corner):
        """Whether the piece holds a whole unit of the grid of one of `frames` (_holds_unit)."""
        return any(_holds_unit(self(f)[2], piece, corner) for f in frames)

    def hand_sized(self, region, f):
        """Whether the [h, w] booleans `region` can hold a hand-sized piece of frame f's mask: LOSS_HAND of its area."""
        return np.count_nonzero(region) / self.areas[f] >= LOSS_HAND


def _closed_runs(view):
    """The closed runs mask_loss asks about, as (run, anchors, beyond, box, gone): every run t..t+g-1 of
    1 to LOSS_WINDOW frames after a frame with a mask, the frames around it (t - 1, t + g), the frames
    within LOSS_WINDOW beyond each of those, the box of what the frame before holds (its _Reading box),
    and what the run's frames hold, raw or read, inside that box ([h, w] booleans, grown in place for
    the next run: read it before asking for that one). Whatever the run drops of the frame before lies
    in the box. A run stops growing once what the frame before holds and the run does not is no longer
    hand-sized (view.hand_sized): no region of it is. `view` is a _Reading; the frames no later run
    reads are let go."""
    N = view.masks.shape[0]
    for t in range(1, N - 1):
        for f in [f for f in view.kept if f < t - LOSS_WINDOW - 1]:
            del view.kept[f]
        if not view.present[t - 1]:
            continue
        before, _, _, box = view(t - 1)
        before = before[box]
        gone = np.zeros(before.shape, bool)
        for g in range(1, LOSS_WINDOW + 1):
            if t + g >= N:
                break
            gone |= view(t + g - 1)[1][box]
            if not view.hand_sized(before & ~gone, t - 1):
                break                                  # too little held before the run is left to drop
            yield (range(t, t + g), (t - 1, t + g),
                   (range(max(0, t - LOSS_WINDOW), t - 1), range(t + g + 1, min(N, t + g + LOSS_WINDOW))), box, gone)


def lost_regions(masks, read, evidence=None, open_runs=True):
    """mask_loss's detection: every region of her the model loses, as (run, anchors, piece, corner,
    share) - the frames of the run (a range), the frames that hold the region around it (the frame
    before and the frame after a closed run; the one end an open run has), the region as a connected
    piece ([h, w] booleans) with its top-left pixel `corner` (y, x), and its share of her mask.

    A dropout is a region the mask holds on the frame before a run and drops on every frame of the
    run: a closed run of up to LOSS_WINDOW frames the frame after holds again, or, with a pose and
    `open_runs`, an open run to the last frame of a stretch of frames with a mask (the clip's end, or
    before an empty stretch) or from its first. Each connected piece of it is her body the model
    loses when:

    - the model reads it as lost: `read(f)` is (what the model reads of frame f's mask, [H, W]
      booleans or None for the mask itself; the grid it reads it on, (row edges, column edges)):
      the final and its block grid on a raw mask (the Wan Animate workflow grows it into the final
      before any model reads it), the final itself and its own grid, or SCAIL-2's latent reading
      and its cell grid. The piece is where the mask and its reading both hold the region on the
      run's ends and neither holds it on any frame of the run, and it holds a whole unit of the
      grid: the model reads the mask a block or a cell at a time, so what it loses is a whole one,
      one it read as her on a frame at the run's ends or one the run's own grid leaves off (a
      block grid is laid from each frame's own box, so a unit of either counts);
    - it is hers: `evidence` (a _LimbEvidence or _KeypointEvidence; None without a pose) puts the
      body in it on a frame at the run's ends, or the mask holds it on another frame within
      LOSS_WINDOW beyond each end - not background blinking off (at the clip's first or last frame
      there is no frame beyond to ask);
    - it is not a limb that moved away and came back (`_gain` under LOSS_GAIN);
    - with a pose, the body is in it on every frame of the run (`evidence.on_run`). Without one,
      only closed runs count: an open run is a limb leaving the shot as often as a lost one;
    - it is hand-sized or bigger: LOSS_HAND of her mask or more.

    `masks` is [N, H, W] booleans, or anything that gives a frame's booleans by the same indexing
    (masks[f], masks[f, y0:y1, x0:x1]) and has their `shape`. The share is the piece's area over the
    mask of the frame before the run (after, for a run from a stretch's first frame). A generator:
    the regions come run by run, the closed runs first."""
    N, H, W = masks.shape
    S = min(H, W)
    view = _Reading(masks, read)

    def held(f, piece, corner):
        (py, px), (ph, pw) = corner, piece.shape
        return 2 * int((view(f)[0][py:py + ph, px:px + pw] & piece).sum()) > int(piece.sum())

    def hers(anchors, beyond, piece, corner):
        if evidence is not None and any(evidence.at_end(f, piece, corner) for f in anchors):
            return True
        return all(any(held(f, piece, corner) for f in frames) for frames in beyond if len(frames))

    def judge(lost, box, run, anchors, beyond):
        # `lost` is inside `box`; a region too small to hold a hand-sized piece has none to judge
        if not view.hand_sized(lost, anchors[0]):
            return
        region = np.zeros((H, W), bool)
        region[box] = lost
        reach = min(len(run), LOSS_REACH_FRAMES) * LOSS_REACH * S
        area = view.areas[anchors[0]]
        for piece, corner in _pieces(region):
            share = int(piece.sum()) / area
            if share < LOSS_HAND or not view.whole_unit((*anchors, *run), piece, corner):
                continue
            if evidence is not None and not evidence.on_run(run, anchors, piece, corner):
                continue
            if not hers(anchors, beyond, piece, corner) or _gain(masks, anchors, run, piece, corner, reach) >= LOSS_GAIN:
                continue
            yield run, anchors, piece, corner, share

    # closed runs: t..t+g-1, held on t-1 and on t+g
    for run, anchors, beyond, box, gone in _closed_runs(view):
        lost = view(anchors[0])[0][box] & view(anchors[1])[0][box] & ~gone
        yield from judge(lost, box, run, anchors, beyond)
    if evidence is None or not open_runs:
        return
    # open runs, with a pose: within each stretch a..b of frames with a mask, t..b held on t-1, and a..e held on e+1;
    # what the run holds is grown inside each frame's box, and what the anchor loses read inside its own
    views = view.kept
    views.clear()
    f = 0
    while f < N:
        if not view.present[f]:
            f += 1
            continue
        a = f
        while f < N and view.present[f]:
            f += 1
        b = f - 1
        gone = np.zeros((H, W), bool)
        for t in range(b, a, -1):
            _, either, _, box = view(t)
            gone[box] |= either[box]
            before, _, _, box = view(t - 1)
            yield from judge(before[box] & ~gone[box], box, range(t, b + 1), (t - 1,),
                             (range(max(0, t - LOSS_WINDOW), t - 1),))
            views.pop(t, None)
        views.clear()
        gone = np.zeros((H, W), bool)
        for e in range(a, b):
            _, either, _, box = view(e)
            gone[box] |= either[box]
            before, _, _, box = view(e + 1)
            yield from judge(before[box] & ~gone[box], box, range(a, e + 1), (e + 1,),
                             (range(e + 2, min(N, e + 1 + LOSS_WINDOW)),))
            views.pop(e, None)
        views.clear()


def dropouts(masks, read, evidence=None):
    """mask_loss: per frame, the largest share of her mask the model loses on it (0.0 without one),
    over every region `lost_regions` finds."""
    loss = [0.0] * masks.shape[0]
    for run, _, _, _, share in lost_regions(masks, read, evidence):
        for f in run:
            loss[f] = max(loss[f], share)
    return loss


def closed_losses(masks, pose_metas: list[PoseMeta], draw_threshold):
    """prompt_pose's mask-driven trigger (sam3_1_multiplex.prompt_pose): `lost_regions` of the raw
    `masks` as the Wan Animate workflow's final reads them (GrowMaskWithBlur expand FINAL_GROW,
    BlockifyMask FINAL_BLOCK), the body evidence of `pose_metas` at `draw_threshold` (a raw mask's),
    closed runs only - a region of her the mask drops for up to LOSS_WINDOW frames and holds on both
    sides. A generator, as lost_regions."""
    _, H, W = masks.shape
    evidence = _LimbEvidence(pose_metas, W, H, draw_threshold)
    return lost_regions(masks, _final_reading(masks), evidence, open_runs=False)


def dropped_parts(masks):
    """prompt mode's trigger (sam3_1_multiplex.refine.segment_by_prompt_repaired), from the mask alone:
    a part of her the raw `masks` hold on both sides of a closed run of up to LOSS_WINDOW frames and
    drop on every frame of it, as the Wan Animate workflow's final reads them (GrowMaskWithBlur expand
    FINAL_GROW, BlockifyMask FINAL_BLOCK). The runs are lost_regions' closed runs; the part is judged on
    each side:

    - each of the two frames around the run loses a part: a connected piece of what it holds (the mask
      and the final both) that no frame of the run holds (the mask or the final), and the two parts
      overlap - the region both frames hold and the run drops;
    - each part is hand-sized or bigger (LOSS_HAND of its frame's mask) and holds a whole block of the
      final, as lost_regions asks of a region: the final reads the mask a block at a time;
    - neither is a limb that moved away and came back (_gain under LOSS_GAIN, as lost_regions).

    Each part is measured whole, not their overlap alone: a hand that moves between the two frames
    leaves only a sliver both of them hold. For the same reason nothing is asked of the frames further
    beyond the run: a moving hand is held there nowhere. `masks` is as lost_regions'. A generator of
    (run, anchors, region, corner, share) as closed_losses': the region is the overlap of the two
    parts, the only place a part moving from one to the other covers all the way, as [h, w] booleans
    with its top-left pixel `corner` (y, x), and the share the smaller part's."""
    import cv2

    N, H, W = masks.shape
    view = _Reading(masks, _final_reading(masks))
    for run, anchors, _, box, gone in _closed_runs(view):
        if not (view(anchors[0])[0][box] & view(anchors[1])[0][box] & ~gone).any():
            continue                               # the two frames hold nothing the run drops
        # the part of the frame after reaches beyond the box of the frame before: what the run holds, whole
        held = np.zeros((H, W), bool)
        for f in run:
            held |= view(f)[1]
        sides = [view(a)[0] & ~held for a in anchors]
        both = sides[0] & sides[1]
        if not all(view.hand_sized(side, a) for side, a in zip(sides, anchors)):
            continue                               # no part of that side can be hand-sized
        areas = [view.areas[a] for a in anchors]
        labelled = [cv2.connectedComponentsWithStats(side.astype(np.uint8), connectivity=8)[1:3] for side in sides]
        reach = min(len(run), LOSS_REACH_FRAMES) * LOSS_REACH * min(H, W)
        ys, xs = np.nonzero(both)
        for pair in sorted(set(zip(labelled[0][0][ys, xs].tolist(), labelled[1][0][ys, xs].tolist()))):
            shares = []
            for (labels, stats), i, area in zip(labelled, pair, areas):
                part, corner = _piece(labels, stats, i)
                share = int(part.sum()) / area
                if (share < LOSS_HAND or not view.whole_unit((*anchors, *run), part, corner)
                        or _gain(masks, anchors, run, part, corner, reach) >= LOSS_GAIN):
                    break
                shares.append(share)
            else:
                region = (labelled[0][0] == pair[0]) & (labelled[1][0] == pair[1])
                box = _box(region)
                yield run, anchors, region[box], (int(box[0].start), int(box[1].start)), min(shares)


# --- the head and a whole limb outside the mask -------------------------------------------------

def head_out(kps, visible, seen, diag):
    """The drawn head keypoints inside the frame (nose, eyes, ears) outside `seen` (the mask, and
    what the model reads of it) grown by 2% of the box diagonal `diag` (_grown), by name: those with
    no pixel of `seen` in the grow kernel around them."""
    H, W = seen.shape
    r = _grow_side(diag) // 2
    xs = np.clip(kps[HEAD, 0].astype(int), 0, W - 1)
    ys = np.clip(kps[HEAD, 1].astype(int), 0, H - 1)
    return [BODY_NAMES[j] for j, x, y in zip(HEAD, xs, ys)
            if visible[j] and not seen[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].any()]


def keypoints_out(seen, shape, poses, areas, pose_metas: list[PoseMeta], draw_threshold, chunk=32):
    """Per frame, the head keypoints and the limbs outside `seen(f)` ([H, W] booleans, the mask the
    model reads of frame f; `shape` is (N, H, W)), `chunk` frames at a time, each frame's `seen` built
    once. Returns (head_out of each frame with a mask - `areas` its pixels, `poses` the frames'
    common._frame_pose - and [] on an empty one, the limbs wholly outside on each frame). The limbs are
    prompt_pose's own rule (sam3_1_multiplex.prompt_pose.refine_points) judging each frame on its
    own - a forearm-and-hand or a lower leg with LIMB_POINTS drawn keypoints or more that lie
    outside the mask, POSE_POINT_DISTANCE of the shorter side or more from it, its distal part at
    least OUTSIDE_TENTHS in ten outside: a whole limb lost, not a fingertip past the edge. The
    keypoints are the ones drawn at the draw threshold, no draw rule left out (NOTHING_HIDDEN)."""
    import torch

    from ..sam3_1_multiplex.config import POSE_POINT_DISTANCE
    from ..sam3_1_multiplex.prompt_pose import LIMBS as POSE_LIMBS
    from ..sam3_1_multiplex.prompt_pose import drawn_keypoints, refine_points
    N, H, W = shape
    heads, limbs = [], [[] for _ in range(N)]
    for s in range(0, N, chunk):
        metas = pose_metas[s:s + chunk]
        frames = np.stack([seen(f) for f in range(s, s + len(metas))])
        for f, frame in enumerate(frames, s):
            _, _, diag, kps, _, visible = poses[f]
            heads.append(head_out(kps, visible, frame, diag) if areas[f] else [])
        xy, drawn = drawn_keypoints(metas, draw_threshold, [NOTHING_HIDDEN] * len(metas), H, W)
        fired = refine_points(torch.from_numpy(frames), xy, drawn, 0, POSE_POINT_DISTANCE * min(H, W))
        for f, keypoints in fired.items():
            limbs[s + f] = [name for name, limb in POSE_LIMBS.items() if set(limb) & set(keypoints)]
    return heads, limbs


# --- the measurements, the flags, check_mask ----------------------------------------------------

def mask_frame_metrics(masks, pose_metas: list[PoseMeta], detections: list[Detection], draw_threshold, read,
                       final=False, read_keypoints=None, loss_without_pose=True) -> list[MaskRow]:
    """One dict of raw mask measurements per frame (keys MASK_ROW); thresholds are applied
    afterwards. `masks` is [N, H, W] booleans, or reads as them (lost_regions), on the frames the
    pose was found on; without a pose (`pose_metas` None) the pose-based measurements are None
    and the lists empty. `read(f)` is what the model reads of frame f's mask and the grid it reads
    it on (see `dropouts`). `final` says the masks are the final masks of the Wan Animate
    workflow, measured as common.FINAL_BLOCK describes (it needs a pose). `read_keypoints(f)`, when
    given, is a reading the keypoint tests count as the mask too: a drawn keypoint SCAIL-2's latent
    grid reads as her is inside the mask.
    With `loss_without_pose` False, mask_loss is not measured without a pose (None)."""
    N, H, W = masks.shape
    posed = pose_metas is not None
    areas = np.array([np.count_nonzero(masks[f]) for f in range(N)], dtype=np.int64)
    pad = FINAL_PAD if final else 0

    def seen(f):
        return masks[f] | read_keypoints(f) if read_keypoints is not None else masks[f]

    if posed:
        envelopes = box_envelopes(detections, N, W, H)
        whole = [_whole_body(meta, W, H, draw_threshold) for meta in pose_metas]
        poses = [_frame_pose(meta, det, W, H, draw_threshold) for meta, det in zip(pose_metas, detections)]
        evidence = _KeypointEvidence(whole) if final else _LimbEvidence(pose_metas, W, H, draw_threshold)
        heads, limbs = keypoints_out(seen, masks.shape, poses, areas, pose_metas, draw_threshold)
    loss = dropouts(masks, read, evidence) if posed else (dropouts(masks, read) if loss_without_pose else [None] * N)
    alone = {}                             # frame -> its mask_regions by that frame alone

    def regions(f):
        if f not in alone:
            alone[f] = mask_regions(masks[f], whole[f] if posed else None, pad)
        return alone[f]

    rows = []
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
            m.update({"head_out": [], "limbs_out": [], "box_iou_prev": None})
        else:
            det, meta = detections[i], pose_metas[i]
            bw, bh, _, kps, drawn, _ = poses[i]
            m["mask_to_box"] = area / (bw * bh)
            m["box_reliable"] = bool(det["score"] > 0 and drawn.sum() >= RELIABLE_KEYPOINTS
                                     and (kps[drawn, 2].mean() if drawn.any() else 0) >= RELIABLE_CONF)

            gx1, gy1, gx2, gy2 = envelopes[i]
            inside = int(mask[int(gy1):int(gy2), int(gx1):int(gx2)].sum())
            m["mask_outside_box"] = (area - inside) / area if area else 0.0

            # the person's region grown where neither the neighbours' masks nor the skeleton are
            if area and near:
                reference = reduce(np.logical_or, (masks[j] for j in near))
                grown = (_dilated(reference, 2 * FINAL_BLOCK + 1) if final
                         else _grown(reference, np.sqrt(np.median(areas[near]))))
                excess = person & ~grown & ~skeleton_zone(meta, (H, W), draw_threshold, LEAK_REACH)
                m["attached_leak"] = float(excess.sum() / max(np.median(areas[near]), 1))
            else:
                m["attached_leak"] = 0.0

            # the head keypoints outside the mask, and the limbs wholly outside it
            m["head_out"] = heads[i]
            m["limbs_out"] = limbs[i]
            m["box_iou_prev"] = _box_iou_prev(detections, i)
        m["mask_loss"] = loss[i]
        rows.append(m)
    return rows


def mask_flags(rows: list[MaskRow], t, checks=MASK_CHECKS):
    """The mask checks the caller runs (`checks`): check name -> frames it fired on, in the order
    the checks first fired. A check whose measurement is None on a frame (no pose_data, no
    neighbour) does not run there. A frame whose mask is empty is mask_empty's (a fail where the
    detector and the pose agree on the frame, the pose's failure elsewhere): the head, a limb or a
    region the mask leaves out there are no checks of their own."""
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
        if "mask_fragmented" in checks and any(f >= FRAGMENT_FRACTION for f in m["fragments"]):
            _flag(flags, "mask_fragmented", i)
        if m["mask_area"] == 0:
            continue
        head = set(m["head_out"])
        if HEAD_OUT_KEYPOINT in head or len(head & set(HEAD_OUT_SIDES)) >= t["head_out_eyes_ears"]:
            _flag(flags, "mask_head_out", i)
        if m["limbs_out"]:
            _flag(flags, "mask_limb_out", i)
        if m["mask_loss"]:
            _flag(flags, "mask_loss", i)
    return flags


def check_mask(mask, pose_data: PoseData = None, config=None, enabled=True, stop_on_fail=True, final=False,
               reference=None, reference_config=None):
    """The mask checks on `mask` [frames, H, W] (or one [H, W] frame) against the keypoints and
    boxes in `pose_data`, which must be of the same frames at the same size. Without pose_data
    only the checks that do not read it run (POSE_FREE_MASK_CHECKS), and the report says so.

    `config` is a MaskGuardConfig (None = defaults). `enabled` False still measures and reports
    every check but marks them off, so none can fail. With `stop_on_fail` a failed enabled check
    raises GuardFailed with the report; the wrapper that combines both groups passes False and lets
    `combine_guards` decide. `final` says `mask` is the final mask of the Wan Animate workflow,
    grown and blockified (the WanAnimate Preprocess Guard's), whose measures allow for its blocks
    (common.FINAL_BLOCK); it needs pose_data. Otherwise `mask` is the raw mask, which the workflow
    grows into that final before any model reads it: what the model loses is judged on the final
    (`_final_blocks`).

    `reference`, when given, is the character on the reference image as a MASK (SAM 3.1 Multiplex
    on it, [frames, H', W'], its first frame read): it is placed on the mask's frames as core places
    the reference and measured against mask frame 0 (reference.mask_reference), grown and
    blockified as the final first when `final`; an IoU below `reference_config`'s
    min_reference_iou (a ReferenceGuardConfig, with `final` a FinalReferenceGuardConfig; None =
    defaults) is reference_misaligned, a warning.
    The threshold joins the metrics' thresholds, the record their "reference"; without a reference
    both are as they were. Returns (mask unchanged, report, metrics JSON, timeline IMAGE)."""
    config = _config(config, MaskGuardConfig)
    if final and pose_data is None:
        raise ValueError("the final mask is judged against its pose; connect pose_data")
    masks, pose_metas, detections = _mask_inputs(mask, pose_data, final)
    thresholds = _thresholds(pose_data, config)
    if reference is not None:
        thresholds.update(asdict(_config(reference_config, FinalReferenceGuardConfig if final else ReferenceGuardConfig)))
    read = (lambda f: (None, block_grid(masks[f]))) if final else _final_reading(masks)
    with log.step(f"mask guard: checking {len(masks)} frames ({'on' if enabled else 'off'})"):
        rows = mask_frame_metrics(masks, pose_metas, detections, thresholds["draw_threshold"], read, final)
        flags = mask_flags(rows, thresholds)
        # the final grows and blockifies the placed character as the workflow does the raw mask
        grown = (lambda placed: _final_reading(placed[None])(0)[0]) if final else None
        record = None if reference is None else mask_reference(reference, masks.shape[1:], masks[0] if len(masks) else None,
                                                               thresholds, grown)
    note = None if pose_data is not None else (
        "without pose_data, not checked: " + ", ".join(name for name in MASK_CHECKS if name not in POSE_FREE_MASK_CHECKS))
    report, metrics, timeline = _finish("Mask guard", "mask", rows, flags, thresholds,
                                        set(MASK_CHECKS if enabled else ()), MASK_PANELS, stop_on_fail, note, record)
    return mask, report, metrics, timeline
