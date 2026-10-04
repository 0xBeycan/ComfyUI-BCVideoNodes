"""SCAIL-2's colored masks against core's own reading of them, the SAM 3.1 Multiplex track on
the one-frame reference SCAIL-2 Preprocess hands it, and the face close-up of its face_crop (the
crop's geometry on synthetic boxes, its resize, its mask). Core's comfy_extras/nodes_scail.py is
imported, so this runs where ComfyUI is importable (with the ComfyUI root on PYTHONPATH):

    PYTHONPATH=/path/to/ComfyUI python -m pytest tests/pipelines/test_scail2.py
"""
import logging

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
cli_args = pytest.importorskip("comfy.cli_args")
cli_args.args.disable_xformers = True

from scail2_fakes import scail2  # noqa: E402

# channels of core's 7-colour extraction, per 4-frame group: white, red, green, blue, yellow, magenta, cyan
WHITE_CHANNEL, BLUE_CHANNEL = 0, 3


def person(frames=5, height=64, width=32):
    """A MASK clip: a box that moves one pixel right per frame, 1.0 inside."""
    mask = torch.zeros(frames, height, width)
    for f in range(frames):
        mask[f, 16:48, 8 + f:20 + f] = 1.0
    return mask


def core():
    return pytest.importorskip("comfy_extras.nodes_scail")


@pytest.mark.parametrize("replacement_mode", [False, True])
def test_the_masks_are_pure_colours_on_the_mode_background(replacement_mode):
    driving, reference = person(), person(1)
    pose_video_mask, reference_image_mask = scail2.colored_masks(driving, replacement_mode, reference)
    assert pose_video_mask.shape == (5, 64, 32, 3) and reference_image_mask.shape == (1, 64, 32, 3)
    assert pose_video_mask.dtype == reference_image_mask.dtype == torch.float32
    blue = torch.tensor(scail2.PALETTE[0])
    assert scail2.PALETTE[0] == (0.0, 0.0, 1.0)
    driving_bg, reference_bg = (scail2.WHITE, scail2.BLACK) if replacement_mode else (scail2.BLACK, scail2.WHITE)
    for image, mask, background in ((pose_video_mask, driving, driving_bg), (reference_image_mask, reference, reference_bg)):
        on = mask > 0.5
        assert (image[on] == blue).all()
        assert (image[~on] == torch.tensor(background)).all()


def test_the_mask_threshold_is_one_half():
    mask = torch.tensor([[[0.5, 0.51]]])
    pose_video_mask, _ = scail2.colored_masks(mask, False, person(1))
    assert pose_video_mask[0, 0].tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]


def test_a_single_frame_mask_without_a_batch_axis(caplog):
    pose_video_mask, reference_image_mask = scail2.colored_masks(person(1)[0], False, person(1)[0])
    assert pose_video_mask.shape == (1, 64, 32, 3) and reference_image_mask.shape == (1, 64, 32, 3)


@pytest.mark.parametrize("reference", [None, torch.zeros(1, 64, 32)])
def test_animation_mode_without_a_reference_identity_is_logged(caplog, reference):
    caplog.set_level(logging.WARNING, logger="BCVideoNodes")
    _, reference_image_mask = scail2.colored_masks(person(), False, reference)
    assert reference_image_mask.shape == (1, 64, 32, 3) and (reference_image_mask == 1.0).all()
    assert "can collapse into replacement behaviour" in caplog.text


@pytest.mark.parametrize("reference", [None, torch.zeros(1, 64, 32)])
def test_replacement_mode_without_a_reference_identity_is_an_error(reference):
    with pytest.raises(ValueError, match="Connect a MASK of the character on the reference image"):
        scail2.colored_masks(person(), True, reference)


@pytest.mark.parametrize("replacement_mode", [False, True])
def test_core_reads_the_identity_and_the_background(replacement_mode):
    driving = person()
    pose_video_mask, reference_image_mask = scail2.colored_masks(driving, replacement_mode, person(1))
    extracted = core()._extract_mask_to_28ch(pose_video_mask)  # (1, T_lat, 28, h, w)
    groups = extracted[0].view(extracted.shape[1], 4, 7, *extracted.shape[-2:])
    person_channel = groups[:, :, BLUE_CHANNEL]
    background_channel = groups[:, :, WHITE_CHANNEL]
    others = [c for c in range(7) if c not in (WHITE_CHANNEL, BLUE_CHANNEL)]
    assert (groups[:, :, others] == 0).all()  # no other colour anywhere
    assert float(person_channel.sum()) > 0
    if replacement_mode:  # white background: the white channel covers everything the person does not
        assert torch.allclose(person_channel + background_channel, torch.ones_like(person_channel))
    else:  # black background: nothing but the person
        assert (background_channel == 0).all()
    reference = core()._extract_mask_to_28ch(reference_image_mask)[0, 0].view(4, 7, *extracted.shape[-2:])[0]
    assert float(reference[BLUE_CHANNEL].sum()) > 0
    assert float(reference[WHITE_CHANNEL].sum()) == (0.0 if replacement_mode else float((1 - reference[BLUE_CHANNEL]).sum()))


@pytest.mark.parametrize("replacement_mode", [False, True])
def test_the_reference_mask_is_what_core_renders_for_a_plain_mask(replacement_mode):
    reference = person(2)
    _, reference_image_mask = scail2.colored_masks(person(), replacement_mode, reference)
    rendered = core()._render_mask_as_identity(reference, "black" if replacement_mode else "white")
    assert torch.equal(reference_image_mask, rendered.to(device="cpu", dtype=torch.float32))


@pytest.mark.parametrize("replacement_mode", [False, True])
def test_the_adapter_reads_the_mode_the_masks_were_rendered_for(replacement_mode):
    _, reference_image_mask = scail2.colored_masks(person(), replacement_mode, person(1))
    assert scail2.mask_convention(reference_image_mask) == (scail2.REPLACEMENT if replacement_mode else scail2.ANIMATION)


# --- the driving video on black ------------------------------------------------------------

def test_the_driving_video_on_black_keeps_the_person_and_blacks_out_the_rest():
    images = torch.rand(5, 64, 32, 3, generator=torch.Generator().manual_seed(0)) + 0.01  # no pixel black already
    mask = person() * 0.8  # on above 0.5, whatever the value
    mask[2, 0, 0] = 0.5    # the cut: 0.5 is off
    out = scail2.driving_on_black(images, mask)
    assert out.shape == images.shape and out.dtype == images.dtype
    on = mask > 0.5
    assert torch.equal(out[on], images[on])
    assert (out[~on] == 0).all()


def test_the_driving_video_on_black_takes_a_single_frame_mask():
    images = torch.rand(1, 64, 32, 3, generator=torch.Generator().manual_seed(0))
    assert torch.equal(scail2.driving_on_black(images, person(1)[0]), scail2.driving_on_black(images, person(1)))


def clip(frames=37, height=12, width=10):
    """A MASK clip over two chunks and a part of a third of what the fills cut at once (16 frames),
    with values at the 0.5 cut on every fifth frame."""
    mask = torch.rand(frames, height, width, generator=torch.Generator().manual_seed(3))
    mask[::5, 0, 0] = 0.5
    return mask


@pytest.mark.parametrize("replacement_mode", [False, True])
def test_the_colored_driving_mask_is_the_whole_clip_cut_and_coloured_at_once(replacement_mode):
    driving = clip()
    pose_video_mask, _ = scail2.colored_masks(driving, replacement_mode, driving[:1])
    background = scail2.backgrounds(replacement_mode)[0]
    expected = torch.where((driving > 0.5).unsqueeze(-1), torch.tensor(scail2.PALETTE[0]), torch.tensor(background))
    assert pose_video_mask.dtype == torch.float32 and torch.equal(pose_video_mask, expected)


def test_the_driving_video_on_black_is_the_whole_clip_cut_at_once():
    images = torch.rand(37, 12, 10, 3, generator=torch.Generator().manual_seed(4))
    driving = clip()
    black = torch.zeros(())
    assert torch.equal(scail2.driving_on_black(images, driving), torch.where((driving > 0.5).unsqueeze(-1), images, black))
    # a single mask is every frame's, as the whole-clip select broadcasts it
    assert torch.equal(scail2.driving_on_black(images, driving[0]), torch.where((driving[0] > 0.5).unsqueeze(-1), images, black))


@pytest.mark.parametrize("black_background, replacement_mode", [(False, False), (False, True), (True, False)])
def test_black_background_is_allowed_outside_replacement_mode(black_background, replacement_mode):
    scail2.check_black_background(black_background, replacement_mode)


def test_black_background_in_replacement_mode_is_an_error():
    with pytest.raises(ValueError, match="black_background is for animation mode: in replacement mode the result keeps "
                                         "the driving video's background.*Turn black_background off, or turn "
                                         "replacement_mode off"):
        scail2.check_black_background(True, True)


# --- the one-frame SAM 3.1 Multiplex track ---------------------------------------------------

from sam3_1_multiplex_fakes import sam3  # noqa: E402
from test_sam3_1_multiplex_ab import LOW, FakeModel, FakeSam3, FakeTracker, person_detection  # noqa: E402


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_the_sam_track_runs_on_a_one_frame_reference(monkeypatch, caplog, dtype):
    # a float16 image (Load Video at precision fp16) gets a float16 mask, its coverage counted in float32
    tracker = FakeTracker()

    def detect(detector, backbone, trunk_out, embedding, text_mask, config):
        mask, score = person_detection(trunk_out)
        return mask[None], torch.tensor([score])

    class ProgressBar:
        def __init__(self, total):
            pass

        def update(self, value):
            pass

    monkeypatch.setattr(sam3.mm, "load_model_gpu", lambda model: None)
    monkeypatch.setattr(sam3.mm, "get_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(sam3.mm, "intermediate_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(sam3, "_multiplex_parts", lambda model: (FakeSam3(tracker), "detector", tracker, "backbone"))
    monkeypatch.setattr(sam3, "MultiplexState", lambda *args: object())
    monkeypatch.setattr(sam3, "_prep_frame", lambda frames, idx, device, dtype, size: idx.start)
    monkeypatch.setattr(sam3, "encode_prompt", lambda *args: (("embedding", None), "text_mask"))
    monkeypatch.setattr(sam3, "detect_person", detect)
    monkeypatch.setattr(sam3, "ProgressBar", ProgressBar)
    caplog.set_level(logging.INFO)
    caplog.set_level(logging.INFO, logger="BCVideoNodes")
    images = torch.rand(1, 2 * LOW, 2 * LOW, 3, generator=torch.Generator().manual_seed(1)).to(dtype)
    mask = sam3.track((FakeModel(), object()), images)
    closing = [r for r in caplog.records if r.name in ("BCVideoNodes", "root")][-1].getMessage()
    assert "frames without a mask 0" in closing and mask.dtype == dtype
    covered = float((mask > 0).float().mean() * 100)
    assert f"mask coverage {covered:.1f}-{covered:.1f}%" in closing


# --- the face close-up: SCAIL-2 Preprocess's extra reference (face_crop) --------------------------

# the worked example: a 180 x 200 face box at (550, 300) in a 1280 x 1920 source, a 704 x 1280 generation
EXAMPLE = ((550, 300, 730, 500), (1280, 1920), (704, 1280))


@pytest.mark.parametrize("factor, expected", [
    # the generation's size over the factor; centred on the box (x = 640 - w / 2); of the free height
    # half, but no more than 0.4 * 200 = 80, above the box: y = 300 - 80 = 220. The face is 180 * factor px
    (1, (288, 220, 704, 1280, 1)),
    (2, (464, 220, 352, 640, 2)),
    # 704 / 3 = 234.67 -> 235, 1280 / 3 = 426.67 -> 427; x = 640 - 117.5 = 522.5, rounded to even 522
    (3, (522, 220, 235, 427, 3)),
    # 4 and 5 would make the window narrower than the box: lowered to 704 / 180 = 3.91, the box filling
    # the width (180), 1280 / 3.91 = 327 high; 127 px free, half (63.5 < 80) above: y = 236.5 -> 236
    (4, (550, 236, 180, 327, 704 / 180)),
    (5, (550, 236, 180, 327, 704 / 180)),
])
def test_the_face_crop_window_of_the_example(factor, expected):
    box, image, generation = EXAMPLE
    *window, used = scail2.face_crop_box(box, *image, *generation, factor)
    assert tuple(window) == expected[:4] and used == pytest.approx(expected[4])
    assert scail2.FACE_HEADROOM == 0.4


@pytest.mark.parametrize("box, image, generation, factor, expected", [
    # a box at the top-left corner: x 50 - 176 = -126 and y 10 - 40 = -30 shifted inside to 0
    ((0, 10, 100, 110), (1000, 1000), (704, 1280), 2, (0, 0, 352, 640, 2)),
    # a box at the bottom-right corner: x 950 - 176 = 774 and y 850 - 40 = 810 shifted to 1000 - 352
    # and 1000 - 640
    ((900, 850, 1000, 950), (1000, 1000), (704, 1280), 2, (648, 360, 352, 640, 2)),
    # a landscape generation: 4 lowered to 704 / 200 = 3.52, the box's height filling the window,
    # 1280 / 3.52 = 364 wide, centred: x = 640 - 182; no free height, so y = 300
    ((550, 300, 730, 500), (1920, 1280), (1280, 704), 4, (458, 300, 364, 200, 3.52)),
    # a source smaller than the 352 x 640 window: it stays around the box, past the source's edges
    # (x 100 - 176 = -76; y 20 - min(260, 48) = -28), the part face_reference pads with black
    ((40, 20, 160, 140), (200, 200), (704, 1280), 2, (-76, -28, 352, 640, 2)),
])
def test_the_face_crop_window_at_the_edges(box, image, generation, factor, expected):
    *window, used = scail2.face_crop_box(box, *image, *generation, factor)
    assert tuple(window) == expected[:4] and used == pytest.approx(expected[4])


def test_the_face_close_up_at_factor_1_is_the_window_itself(caplog):
    caplog.set_level(logging.INFO)
    source = torch.randint(0, 256, (1, 400, 300, 3), generator=torch.Generator().manual_seed(0)) / 255.0
    # a 10 x 10 box: a 32 x 64 window at (145 - 16, 100 - min(27, 4)), not resized
    face, covered = scail2.face_reference(source, (140, 100, 150, 110), 32, 64, 1)
    assert face.shape == (1, 64, 32, 3) and face.dtype == torch.float32
    assert torch.equal(face[0], source[0, 96:160, 129:161]) and bool(covered.all())
    assert "would cut the face box" not in caplog.text


def test_the_face_close_up_is_the_window_resized_by_lanczos_and_a_cut_factor_is_logged(caplog):
    caplog.set_level(logging.INFO)
    source = torch.randint(0, 256, (1, 400, 300, 3), generator=torch.Generator().manual_seed(0)) / 255.0
    face, covered = scail2.face_reference(source, (100, 80, 160, 140), 32, 64, 2)
    # a 16 x 32 window would cut the 60 x 60 box: lowered to 32 / 60, the box filling 60 px of width,
    # 120 high; 60 px free, half of it capped at 0.4 * 60 = 24 above: y = 56
    *window, used = scail2.face_crop_box((100, 80, 160, 140), 300, 400, 32, 64, 2)
    assert tuple(window) == (100, 56, 60, 120) and used == pytest.approx(32 / 60)
    expected = torch.empty(64, 32, 3)
    scail2.fit(source[0, 56:176, 100:160], expected)
    assert torch.equal(face[0], expected) and bool(covered.all())
    assert "face_crop: face_crop_upscale 2 would cut the face box (100, 80, 160, 140); used 0.53" in caplog.text


def test_a_source_smaller_than_the_window_is_padded_black_in_the_image_and_the_mask():
    source = torch.rand(1, 20, 10, 3, generator=torch.Generator().manual_seed(1))
    source = torch.round(source * 255) / 255
    # a 6 x 6 box, factor 2: a 16 x 32 window; x 5 - 8 = -3 (the source is 10 wide: 3 px black left,
    # 3 right), y 4 - 2.4 = 1.6 -> 2, shifted to 0 (the source is 20 high: 12 px black below)
    assert scail2.face_crop_box((2, 4, 8, 10), 10, 20, 32, 64, 2)[:4] == (-3, 0, 16, 32)
    face, covered = scail2.face_reference(source, (2, 4, 8, 10), 32, 64, 2)
    window = torch.zeros(32, 16, 3)
    window[:20, 3:13] = source[0]
    expected = torch.empty(64, 32, 3)
    scail2.fit(window, expected)
    assert torch.equal(face[0], expected)
    # the source covers rows 0..40 and columns 6..26 of the 32 x 64 close-up (twice the window's)
    inside = torch.zeros(1, 64, 32, dtype=torch.bool)
    inside[0, :40, 6:26] = True
    assert torch.equal(covered, inside)
    # black past the lanczos kernel's reach (3 window px, 6 close-up px) from the source's edge
    assert torch.equal(face[0][46:], torch.zeros(18, 32, 3))
    # a character mask over the whole close-up: blue only where the source is, black on the padding
    colored = scail2.extra_reference_mask(torch.ones(1, 64, 32), covered)
    blue = torch.tensor(scail2.PALETTE[0])
    assert (colored[inside] == blue).all() and (colored[~inside] == torch.tensor(scail2.BLACK)).all()


def face_pose_data(score=0.9, conf=0.9):
    """The pose_data of one 1000 x 1000 image: 68 face keypoints spanning x 0.2..0.3, y 0.3..0.4;
    row 0 (the right heel, not a face point) far off."""
    face = np.zeros((69, 3), dtype=np.float32)
    face[0] = (0.9, 0.9, conf)
    face[1:, 0], face[1:, 1], face[1:, 2] = np.linspace(0.2, 0.3, 68), np.linspace(0.3, 0.4, 68), conf
    return {"pose_metas_original": [{"width": 1000, "height": 1000, "keypoints_face": face}],
            "detections": [{"bbox": [100.0, 100.0, 600.0, 900.0], "score": score, "persons": 1}]}


@pytest.mark.parametrize("conf", [0.9, 0.0])
def test_the_face_box_is_the_face_crops(conf):
    # the 100 x 100 keypoint box grown to 1.3 times its area: sqrt(13000) = 114.02 a side, the width
    # 7.01 each side, the height 3.50 below and 10.51 above (Wan Animate's hair framing), then int.
    # A face seen from behind (confidence 0) still has its box.
    assert scail2.FACE_CROP_SCALE == 1.3
    assert scail2.face_box(face_pose_data(conf=conf), 1000, 1000) == (192, 289, 307, 403)


def test_no_person_on_the_source_is_an_error():
    with pytest.raises(ValueError, match="face_crop: Pose Detection found no person on reference_source"):
        scail2.face_box(face_pose_data(score=-1.0), 1000, 1000)


@pytest.mark.parametrize("replacement_mode", [False, True])
def test_an_extra_reference_mask_is_blue_on_black_in_both_modes(replacement_mode, caplog):
    mask = person(1)
    colored = scail2.extra_reference_mask(mask)
    # the mode does not enter: the official multi-reference example keeps the extra references on
    # black in animation mode too
    expected = torch.zeros(1, 64, 32, 3)
    expected[0, 16:48, 8:20] = torch.tensor([0.0, 0.0, 1.0])
    assert torch.equal(colored, expected)
    _, primary = scail2.colored_masks(person(), replacement_mode, mask)
    assert torch.equal(primary[0, 0, 0], torch.tensor(scail2.BLACK if replacement_mode else scail2.WHITE))
    assert "marks no pixel" not in caplog.text


def test_an_empty_extra_reference_mask_is_logged(caplog):
    caplog.set_level(logging.WARNING)
    assert torch.equal(scail2.extra_reference_mask(torch.zeros(1, 8, 4)), torch.zeros(1, 8, 4, 3))
    assert "the extra reference's mask marks no pixel" in caplog.text


def test_the_extra_reference_is_appended_after_the_primary():
    references, masks = torch.rand(1, 8, 4, 3), torch.rand(1, 8, 4, 3)
    image, image_mask = torch.rand(1, 8, 4, 3), torch.rand(1, 8, 4, 3)
    out_images, out_masks = scail2.with_extra_reference(references, masks, image, image_mask)
    assert torch.equal(out_images, torch.cat([references, image])) and torch.equal(out_masks, torch.cat([masks, image_mask]))


def test_face_crop_off_leaves_a_connected_source_unused(caplog):
    caplog.set_level(logging.INFO)
    scail2.check_face_crop(False, None, torch.zeros(1, 8, 4, 3))
    assert "reference_source not used" not in caplog.text
    scail2.check_face_crop(False, torch.zeros(1, 20, 10, 3), torch.zeros(1, 8, 4, 3))
    assert "face_crop is off; reference_source not used" in caplog.text
    caplog.clear()
    scail2.check_face_crop(False, None, torch.zeros(1, 8, 4, 3), upscale_changed=True)
    assert "face_crop is off; face_crop_upscale not used" in caplog.text
    caplog.clear()
    scail2.check_face_crop(False, torch.zeros(1, 20, 10, 3), torch.zeros(1, 8, 4, 3), upscale_changed=True)
    assert "face_crop is off; reference_source and face_crop_upscale not used" in caplog.text


@pytest.mark.parametrize("source, mask, message", [
    (None, None, "face_crop is on but reference_source is not connected: connect Load Reference Image's source_image"),
    (torch.zeros(2, 20, 10, 3), None, "reference_source holds 2 images"),
    (torch.zeros(1, 20, 10, 3), torch.zeros(1, 16, 16), "reference_mask is 16x16 but reference_image 4x8"),
])
def test_face_crop_on_checks_its_inputs(source, mask, message):
    with pytest.raises(ValueError, match=message):
        scail2.check_face_crop(True, source, torch.zeros(1, 8, 4, 3), mask)
    # a reference_mask of reference_image's size passes
    scail2.check_face_crop(True, torch.zeros(1, 20, 10, 3), torch.zeros(1, 8, 4, 3), torch.zeros(1, 8, 4))
