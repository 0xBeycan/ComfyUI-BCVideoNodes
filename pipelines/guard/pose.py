"""The pose checks: the per-frame pose measurements, the flags they raise, and check_pose. All of
them are warnings: a pose error the diffusion model does not absorb shows in the mask checks."""
import numpy as np

from ...libs import log
from ...libs.keypoints import BODY_NAMES, LIMBS
from ...libs.pose_data import Detection, PoseData, PoseMeta
from .common import LIMB_ENDS, SPIKE_RETURN, TORSO, PoseRow, _box_iou_prev, _flag, _frame_pose, _pose_inputs, _thresholds
from .config import PoseGuardConfig, _config
from .report import _finish
from .timeline import POSE_PANELS


def limb_spikes(kps, drawn, H, max_jump):
    """Per frame, the limb keypoints away from where they were: jumped more than `max_jump`
    frame heights from the previous frame and back within SPIKE_RETURN frames to within half
    of that. `kps` is [N, 20, 2] pixels, `drawn` [N, 20]."""
    N = len(kps)
    out = [[] for _ in range(N)]
    for j in LIMB_ENDS:
        for i in range(1, N):
            if not (drawn[i - 1, j] and drawn[i, j]) or np.hypot(*(kps[i, j] - kps[i - 1, j])) <= max_jump * H:
                continue
            for k in range(1, SPIKE_RETURN + 1):
                if i + k < N and drawn[i + k, j] and np.hypot(*(kps[i + k, j] - kps[i - 1, j])) < max_jump * H / 2:
                    for f in range(i, i + k):
                        if BODY_NAMES[j] not in out[f]:
                            out[f].append(BODY_NAMES[j])
                    break
    return out


def pose_frame_metrics(pose_metas: list[PoseMeta], detections: list[Detection], W, H, draw_threshold,
                       max_limb_spike) -> list[PoseRow]:
    """One dict of raw pose measurements per frame (keys POSE_ROW); thresholds are applied
    afterwards, except the spike's, which decides what counts as one. W and H are the size of
    the frames the pose was found on."""
    N = len(pose_metas)
    rows, all_kps, all_drawn = [], [], []
    prev = None
    for i in range(N):
        det = detections[i]
        _, _, diag, kps, on, _ = _frame_pose(pose_metas[i], det, W, H, draw_threshold)
        all_kps.append(kps[:, :2])
        all_drawn.append(on)
        m = {"frame": i, "detected": det["score"] > 0, "persons": det["persons"],
             "pose_conf": float(kps[:, 2].mean()), "drawn_keypoints": int(on.sum()),
             "drawn_limbs": sum(bool(on[a] and on[b]) for a, b in LIMBS), "box_iou_prev": _box_iou_prev(detections, i)}
        # torso motion against the previous frame
        if prev is not None:
            both = on & prev["drawn"]
            torso = [j for j in TORSO if both[j]]
            m["torso_jump"] = float(np.linalg.norm(kps[torso, :2] - prev["kps"][torso, :2], axis=1).max() / diag) if torso else 0.0
        else:
            m["torso_jump"] = 0.0
        rows.append(m)
        prev = {"kps": kps, "drawn": on}
    spikes = limb_spikes(np.array(all_kps).reshape(N, len(BODY_NAMES), 2),
                         np.array(all_drawn).reshape(N, len(BODY_NAMES)), H, max_limb_spike)
    for m, spike in zip(rows, spikes):
        m["limb_spikes"] = spike
    return rows


def pose_flags(rows: list[PoseRow], t):
    """The pose checks: check name -> frames it fired on, in the order the checks first fired."""
    flags = {}

    for m in rows:
        i = m["frame"]
        if m["box_iou_prev"] is not None and m["box_iou_prev"] > 0.5 and m["torso_jump"] > t["max_torso_jump"]:
            _flag(flags, "pose_jump", i)
        if m["limb_spikes"]:
            _flag(flags, "pose_spike", i)
        if m["box_iou_prev"] is not None and m["box_iou_prev"] < 0.3:
            _flag(flags, "subject_switch", i)
    return flags


def check_pose(pose_data: PoseData, config=None, stop_on_fail=True):
    """The pose checks on `pose_data` alone, all warnings: they are reported and never fail.

    `config` is a PoseGuardConfig (None = defaults); the keypoints and limbs counted are the
    ones drawn, at `pose_data["draw_threshold"]`. With `stop_on_fail` the report is logged here;
    the wrapper that combines both groups passes False and lets `combine_guards` log it.

    Returns (pose_data unchanged, report, metrics JSON, timeline IMAGE)."""
    config = _config(config, PoseGuardConfig)
    pose_metas, detections = _pose_inputs(pose_data)
    W, H = (pose_metas[0]["width"], pose_metas[0]["height"]) if pose_metas else (0, 0)
    thresholds = _thresholds(pose_data, config)
    with log.step(f"pose guard: checking {len(pose_metas)} frames"):
        rows = pose_frame_metrics(pose_metas, detections, W, H, thresholds["draw_threshold"], config.max_limb_spike)
        flags = pose_flags(rows, thresholds)
    report, metrics, timeline = _finish("Pose guard", "pose", rows, flags, thresholds, set(), POSE_PANELS, stop_on_fail)
    return pose_data, report, metrics, timeline
