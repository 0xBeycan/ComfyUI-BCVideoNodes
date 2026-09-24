"""SAM 3.1 Multiplex prompt mode's multi-object helpers on synthetic mask logits. No model is
loaded.

The prompt module imports without ComfyUI, but this file imports comfy.cli_args first, so it runs
where ComfyUI is importable (the pod, with the ComfyUI root on PYTHONPATH) and is skipped
elsewhere."""
import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
cli_args = pytest.importorskip("comfy.cli_args")
cli_args.args.disable_xformers = True

from sam3_1_multiplex_fakes import sam3  # noqa: E402


# --- multi-object helpers ------------------------------------------------------------------

def logits(*boxes, size=20):
    """[K, size, size] mask logits, +5 inside each (y1, y2, x1, x2) box, -5 elsewhere."""
    out = torch.full((len(boxes), size, size), -5.0)
    for k, (y1, y2, x1, x2) in enumerate(boxes):
        out[k, y1:y2, x1:x2] = 5.0
    return out


def test_suppress_recently_occluded_blanks_the_later_occluded_of_an_overlapping_pair():
    masks = logits((0, 10, 0, 10), (0, 10, 0, 9), (12, 20, 12, 20))
    # tracks 0 and 1 overlap by IoU 0.9; 1 came out of occlusion more recently than 0
    got = sam3.suppress_recently_occluded(masks, torch.tensor([3, 7, -1]), 0.7)
    assert got.tolist() == [False, True, False]
    # never-occluded tracks are never suppressed by each other
    assert not sam3.suppress_recently_occluded(masks, torch.tensor([-1, -1, -1]), 0.7).any()
    # below the IoU threshold nothing happens
    assert not sam3.suppress_recently_occluded(masks, torch.tensor([3, 7, -1]), 0.95).any()
    assert sam3.suppress_recently_occluded(masks[:1], torch.tensor([5]), 0.7).tolist() == [False]


def test_suppress_shrunk_blanks_a_track_buried_under_another():
    masks = logits((0, 20, 0, 20), (5, 10, 5, 10), (0, 4, 0, 4))
    masks[0, 5:10, 5:10] = 9.0    # track 0 outscores track 1 everywhere track 1 is
    masks[2] = masks[2] * 2       # track 2 outscores track 0 on its own patch
    out = sam3.suppress_shrunk(masks, 0.3)
    assert (out[0] > 0).sum() == 400 and (out[2] > 0).sum() == 16
    assert not (out[1] > 0).any()
    single = logits((0, 5, 0, 5))
    assert torch.equal(sam3.suppress_shrunk(single, 0.3), single)


def test_non_overlapping_gives_shared_pixels_to_the_higher_score():
    binary = logits((0, 10, 0, 10), (5, 15, 5, 15)) > 0
    out = sam3.non_overlapping(binary, torch.tensor([0.6, 0.9]))
    assert not (out[0] & out[1]).any()
    assert out[1, 5:10, 5:10].all() and not out[0, 5:10, 5:10].any()
    assert (out[0] | out[1]).sum() == (binary[0] | binary[1]).sum()
    tie = sam3.non_overlapping(binary, torch.tensor([0.5, 0.5]))
    assert tie[0, 5:10, 5:10].all()   # a tie goes to the earlier-born object
