"""ViTPose's test-time flip (pose-process P3) on fake models: the COCO-WholeBody mirror map,
the flip-back of a mirrored heatmap, and the switch in PoseConfig. No ComfyUI, no real model:

    python -m pytest tests/test_pose_flip.py
"""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("cv2")

import test_pose  # noqa: E402,F401  (installs the ProgressBar stub when ComfyUI is absent)
from test_pose import FakeDetector, FakePose, frames, no_device  # noqa: E402,F401
from preprocess import pose  # noqa: E402
from preprocess.models import vitpose  # noqa: E402
from preprocess.models.decode import decode_heatmaps  # noqa: E402

K, h, w = 133, 64, 48


def test_the_mirror_map_is_its_own_inverse():
    idx = np.array(vitpose.FLIP_INDEX)
    assert sorted(idx) == list(range(K))
    assert (idx[idx] == np.arange(K)).all()


@pytest.mark.parametrize("a, b", [(0, 0), (1, 2), (3, 4), (5, 6), (9, 10), (15, 16),   # nose, eyes, ears, shoulders, wrists, ankles
                                  (17, 20), (19, 22),                                   # big toes, heels
                                  (23, 39), (31, 31), (53, 53), (40, 49),               # jaw ends, chin, nose bridge, brows
                                  (91, 112), (111, 132)])                               # hand roots, little fingertips
def test_the_mirror_map_pairs_the_named_keypoints(a, b):
    assert vitpose.FLIP_INDEX[a] == b and vitpose.FLIP_INDEX[b] == a


def peaks():
    """Heatmaps with one Gaussian per keypoint, each at its own place, clear of the borders."""
    rng = np.random.default_rng(0)
    ys, xs = np.mgrid[0:h, 0:w]
    out = np.zeros((1, K, h, w), np.float32)
    for k in range(K):
        cy, cx = rng.uniform(8, h - 8), rng.uniform(8, w - 8)
        out[0, k] = np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / 8.0) * rng.uniform(0.5, 0.95)
    return out


def mirrored(heatmaps):
    """What a model sees in the mirrored crop, as flip_back undoes it: the map a column left,
    then mirrored with partners swapped."""
    out = heatmaps.copy()
    out[..., :-1] = heatmaps[..., 1:]
    return out[:, list(vitpose.FLIP_INDEX), :, ::-1].copy()


def test_flip_back_undoes_the_mirror():
    maps = peaks()
    assert np.allclose(vitpose.flip_back(mirrored(maps))[..., 1:-1], maps[..., 1:-1], atol=1e-6)


def test_flip_back_gives_each_keypoint_its_partners_map():
    maps = np.zeros((1, K, h, w), np.float32)
    maps[0, 5, 10, 30] = 1.0   # left shoulder, seen in the mirrored crop
    back = vitpose.flip_back(maps)
    assert back[0, 6].max() == 1.0 and back[0, 5].max() == 0.0          # it is the right shoulder
    assert np.unravel_index(back[0, 6].argmax(), (h, w)) == (10, w - 1 - 30 + 1)   # mirrored, shifted one right


class FakeViTPose:
    """A heatmap model that sees the mirrored crop as the mirror of what it sees in the crop."""

    architecture = "vitpose"
    input_shape = [1, 3, 256, 192]

    def __init__(self):
        self.maps = peaks()
        self.first = None
        self.runs = self.calls = 0

    def run(self, x):
        self.runs += 1
        if self.first is None or not np.array_equal(x, self.first[..., ::-1]):
            self.first = x.copy()
            return self.maps.copy()
        return mirrored(self.maps)

    def __call__(self, img, center, scale):
        self.calls += 1
        return decode_heatmaps(self.run(img), center, scale)


def test_flip_on_a_consistent_model_gives_the_one_pass_keypoints():
    model, x = FakeViTPose(), np.random.default_rng(1).random((1, 3, 256, 192), dtype=np.float32)
    center, scale = np.array([[60.0, 80.0]]), np.array([[0.5, 0.66]])
    one_pass = model(x, center, scale)
    both = pose.flip_test_keypoints(model, x, center, scale)
    assert model.runs == 3
    assert np.allclose(both[..., :2], one_pass[..., :2], atol=1e-3)
    assert np.allclose(both[..., 2], one_pass[..., 2], atol=1e-5)


def test_flip_test_runs_the_mirror_and_default_off_does_not():
    model = FakeViTPose()
    pose.detect(FakeDetector(), model, frames(), config=pose.PoseConfig(temporal=False))
    assert model.calls == len(frames()) and model.runs == len(frames())
    model = FakeViTPose()
    pose.detect(FakeDetector(), model, frames(), config=pose.PoseConfig(temporal=False, flip_test=True))
    assert model.calls == 0 and model.runs == 2 * len(frames())


def test_rtmw_ignores_flip_test_in_one_line(caplog):
    model = FakePose()   # no heatmaps: stands in for RTMW
    model.architecture = "rtmw"
    with caplog.at_level("INFO"):
        plain, _ = pose.detect(FakeDetector(), FakePose(), frames(), config=pose.PoseConfig(temporal=False))
        flipped, _ = pose.detect(FakeDetector(), model, frames(), config=pose.PoseConfig(temporal=False, flip_test=True))
    lines = [r.getMessage() for r in caplog.records if "flip_test" in r.getMessage()]
    assert len(lines) == 1 and "ViTPose only" in lines[0]
    assert model.calls == len(frames())
    for a, b in zip(plain["pose_metas_original"], flipped["pose_metas_original"]):
        assert np.array_equal(a["keypoints_body"], b["keypoints_body"])
