"""ViTPose's test-time flip (pose-process P3) on fake models: the COCO-WholeBody mirror map,
the flip-back of a mirrored heatmap, and the flip of a consistent model. No ComfyUI, no real
model:

    python -m pytest tests/models/test_vitpose_flip.py
"""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("cv2")

from pose_fakes import FakeViTPose, K, h, mirrored, no_device, peaks, pose, vitpose, w  # noqa: E402,F401


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


def test_flip_back_undoes_the_mirror():
    maps = peaks()
    assert np.allclose(vitpose.flip_back(mirrored(maps))[..., 1:-1], maps[..., 1:-1], atol=1e-6)


def test_flip_back_gives_each_keypoint_its_partners_map():
    maps = np.zeros((1, K, h, w), np.float32)
    maps[0, 5, 10, 30] = 1.0   # left shoulder, seen in the mirrored crop
    back = vitpose.flip_back(maps)
    assert back[0, 6].max() == 1.0 and back[0, 5].max() == 0.0          # it is the right shoulder
    assert np.unravel_index(back[0, 6].argmax(), (h, w)) == (10, w - 1 - 30 + 1)   # mirrored, shifted one right


def test_flip_on_a_consistent_model_gives_the_one_pass_keypoints():
    model, x = FakeViTPose(), np.random.default_rng(1).random((1, 3, 256, 192), dtype=np.float32)
    center, scale = np.array([[60.0, 80.0]]), np.array([[0.5, 0.66]])
    one_pass = model(x, center, scale)
    both = pose.flip_test_keypoints(model, x, center, scale)
    assert model.runs == 3
    assert np.allclose(both[..., :2], one_pass[..., :2], atol=1e-3)
    assert np.allclose(both[..., 2], one_pass[..., 2], atol=1e-5)
