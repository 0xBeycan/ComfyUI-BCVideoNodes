"""The mask helpers of libs/mask.py - hole capping, island handling, the largest piece, the frames
that hold a mask, and the final mask (grown and blockified) with the frames painted black under it -
on synthetic masks. No model is loaded.

libs/mask.py imports without ComfyUI, but this file imports comfy.cli_args first, so it runs
where ComfyUI is importable (the pod, with the ComfyUI root on PYTHONPATH) and is skipped
elsewhere."""
import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
cli_args = pytest.importorskip("comfy.cli_args")
cli_args.args.disable_xformers = True

from guard_fakes import final, grown_and_blockified, painted  # noqa: E402
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


# --- the largest piece ---------------------------------------------------------------------

def test_largest_piece_is_4_connected_and_the_first_of_a_tie():
    mask = np.zeros((10, 10), bool)
    mask[0:3, 0:3] = True           # 9 px
    mask[3:5, 3:5] = True           # 4 px touching it at one corner only: another piece
    mask[6:10, 6:9] = True          # 12 px
    assert np.array_equal(sam3.largest_piece(mask), np.pad(np.ones((4, 3), bool), ((6, 0), (6, 1))))
    mask[3, 2] = True               # an edge joins the first two: 14 px
    assert sam3.largest_piece(mask).sum() == 14 and sam3.largest_piece(mask)[4, 4]
    tie = np.zeros((4, 9), bool)
    tie[:, 0:2] = tie[:, 7:9] = True
    assert np.array_equal(sam3.largest_piece(tie), np.pad(np.ones((4, 2), bool), ((0, 0), (0, 7))))
    assert sam3.largest_piece(np.zeros((4, 4), bool)) is None


# --- the frames that hold a mask -----------------------------------------------------------

@pytest.mark.parametrize("chunk", [1, 16, 64])
def test_masked_frames_are_the_frames_with_a_pixel_set(chunk):
    masks = torch.zeros(37, 6, 5)
    masks[0, 2, 3] = 1.0
    masks[15, 5, 4] = 0.5
    masks[16, 0, 0] = -1.0          # any value but 0 is set
    masks[36] = 1.0
    assert sam3.masked_frames(masks, chunk).tolist() == [f in (0, 15, 16, 36) for f in range(37)]
    assert sam3.count_masked_frames(masks) == 4
    assert sam3.count_masked_frames(torch.zeros(0, 6, 5)) == 0


# --- the final mask and the frames painted black under it ----------------------------------

def test_the_final_mask_of_a_pixel_is_its_grown_diamond_in_its_blocks():
    mask = torch.zeros(1, 20, 24)
    mask[0, 10, 12] = 1.0
    # grown twice by the cross: the diamond of radius 2, rows 8-12 and columns 10-14; a 5 px box is
    # one block of any block size from 6 on, so the whole box is on
    expected = torch.zeros(1, 20, 24)
    expected[0, 8:13, 10:15] = 1.0
    assert torch.equal(final.final_mask(mask, 2, 8), expected)
    # not grown, blockified by 8: an 8 x 8 L cut into four 4 x 4 blocks, three of them holding it
    mask = torch.zeros(1, 20, 24)
    mask[0, 4:12, 6:8] = 1.0
    mask[0, 10:12, 6:14] = 1.0
    expected = torch.zeros(1, 20, 24)
    expected[0, 4:12, 6:10] = 1.0
    expected[0, 8:12, 10:14] = 1.0
    assert torch.equal(final.final_mask(mask, 0, 4), expected)


def test_the_final_mask_follows_mask_grow_s_8_bit_levels_and_keeps_an_empty_frame_empty():
    # MaskGrow quantises to 8 bits first: a value under 1/255 is no level, so no pixel
    mask = torch.zeros(3, 16, 16)
    mask[0, 5, 5] = 0.003
    mask[1, 5, 5] = 0.004
    out = final.final_mask(mask, 1, 8)
    assert not out[0].any() and not out[2].any()
    assert out[1, 4:7, 4:7].all() and out[1].sum() == 9


@pytest.mark.parametrize("grow, block_size", [(10, 32), (0, 8), (3, 16), (25, 9), (10, 0), (0, 0)])
def test_the_final_mask_is_mask_grow_then_blockify(grow, block_size):
    # block_size 0 cuts no blocks: the grown mask is the final
    # random masks with pieces at the frame edges, an empty frame and soft values, against the final
    # written out from its definition over the whole frame
    rng = np.random.default_rng(grow * 100 + block_size)
    masks = torch.zeros(6, 90, 70)
    for f in range(5):
        for _ in range(4):
            y, x, h, w = rng.integers(-10, 90), rng.integers(-10, 70), rng.integers(1, 40), rng.integers(1, 30)
            masks[f, max(0, y):max(0, y + h), max(0, x):max(0, x + w)] = 1.0
    masks[4] *= torch.from_numpy(rng.random((90, 70)).astype(np.float32))
    out = final.final_mask(masks, grow, block_size)
    assert out.dtype == torch.float32 and out.shape == masks.shape
    assert torch.equal(out, grown_and_blockified(masks, grow, block_size))
    assert not out[5].any()


def test_block_size_0_cuts_no_blocks():
    mask = torch.zeros(1, 20, 24)
    mask[0, 4:12, 6:8] = 1.0
    mask[0, 10:12, 6:14] = 1.0
    # not grown: the 28 px L itself; grown once by the cross: 31 px around it (2 above, 8 left, 6
    # right of the upright, 5 above and 2 right of the foot, 8 below), no corner
    assert torch.equal(final.final_mask(mask, 0, 0), mask)
    grown = final.final_mask(mask, 1, 0)
    assert mask.sum() == 28 and grown.sum() == 28 + 31 and not grown[0, 3, 5] and grown[0, 3, 6]


def test_the_defaults_are_the_wan_animate_workflow_s():
    assert (final.GROW, final.BLOCK_SIZE) == (10, 32)
    masks = (torch.rand(2, 64, 48) > 0.7).float()
    assert torch.equal(final.final_mask(masks), grown_and_blockified(masks, 10, 32))


def test_painted_black_is_draw_mask_on_image_in_black():
    images = torch.rand(4, 30, 20, 3)
    mask = (torch.rand(4, 30, 20) > 0.5).float()
    out = final.painted_black(images, mask)
    assert out.dtype == torch.float32 and out.shape == images.shape and out.data_ptr() != images.data_ptr()
    assert torch.equal(out, painted(images, mask))
    assert torch.equal(out[mask.bool()], torch.zeros(int(mask.sum()), 3))
    assert torch.equal(out[~mask.bool()], images[~mask.bool()])
