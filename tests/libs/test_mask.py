"""The mask helpers of libs/mask.py - hole capping and island handling - on synthetic masks. No
model is loaded.

libs/mask.py imports without ComfyUI, but this file imports comfy.cli_args first, so it runs
where ComfyUI is importable (the pod, with the ComfyUI root on PYTHONPATH) and is skipped
elsewhere."""
import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
cli_args = pytest.importorskip("comfy.cli_args")
cli_args.args.disable_xformers = True

from sam3_1_multiplex_fakes import C, sam3  # noqa: E402


# --- hole capping --------------------------------------------------------------------------

def test_fill_holes_fills_a_small_enclosed_hole():
    mask = np.zeros((100, 100), np.uint8)
    mask[10:90, 10:90] = 1          # 6400 px, 1% is 64
    mask[40:45, 40:45] = 0          # 25 px hole
    assert sam3.fill_holes(mask, C.max_hole_fraction)[40:45, 40:45].all()


def test_fill_holes_leaves_a_hole_above_the_cap():
    mask = np.zeros((100, 100), np.uint8)
    mask[10:90, 10:90] = 1
    mask[30:40, 30:40] = 0          # 100 px against 6300 px of mask: 1.6%, a limb gap
    out = sam3.fill_holes(mask, C.max_hole_fraction)
    assert not out[30:40, 30:40].any()


def test_fill_holes_cap_is_a_fraction_of_the_mask_itself():
    mask = np.zeros((100, 100), np.uint8)
    mask[10:90, 10:90] = 1
    mask[40:48, 40:48] = 0          # 64 px against 6336 px: 1.01% - just over
    assert not sam3.fill_holes(mask, C.max_hole_fraction)[40:48, 40:48].any()
    mask[40:48, 40:47] = 1          # now 8 px against 6392 px
    assert sam3.fill_holes(mask, C.max_hole_fraction)[40:48, 47].all()


def test_fill_holes_ignores_background_touching_the_border():
    mask = np.zeros((100, 100), np.uint8)
    mask[0:50, 0:50] = 1
    mask[0:3, 20:23] = 0            # a notch open to the top edge is not enclosed
    out = sam3.fill_holes(mask, C.max_hole_fraction)
    assert not out[0:3, 20:23].any()


def test_fill_holes_without_holes_is_unchanged():
    mask = np.zeros((50, 50), np.uint8)
    mask[10:40, 10:40] = 1
    assert np.array_equal(sam3.fill_holes(mask, C.max_hole_fraction), mask)


# --- island handling -----------------------------------------------------------------------

def test_drop_islands_removes_specks_and_keeps_real_pieces():
    mask = np.zeros((200, 200), np.uint8)
    mask[20:120, 20:120] = 1        # 10000 px body
    mask[150:153, 150:153] = 1      # 9 px speck: under 1%
    mask[150:162, 20:32] = 1        # 144 px piece: over 1%
    out = sam3.drop_islands(mask, C.min_island_fraction)
    assert out[20:120, 20:120].all()
    assert not out[150:153, 150:153].any()
    assert out[150:162, 20:32].all()


def test_drop_islands_single_region_is_returned_as_is():
    mask = np.zeros((50, 50), np.uint8)
    mask[5:10, 5:10] = 1
    assert sam3.drop_islands(mask, C.min_island_fraction) is mask


def test_clean_mask_fills_then_drops():
    mask = np.zeros((200, 200), bool)
    mask[20:120, 20:120] = True
    mask[60:62, 60:62] = False      # pinhole
    mask[180:182, 180:182] = True   # speck
    out = sam3.clean_mask(mask, C)
    assert out.dtype == np.uint8
    assert out[60:62, 60:62].all() and not out[180:182, 180:182].any()
