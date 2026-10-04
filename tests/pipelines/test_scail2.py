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

@pytest.mark.parametrize("box, image, generation, expected", [
    # the worked example: a 180 x 200 face box in a 1280 x 1920 source, a 704 x 1280 generation. The
    # box fills the width (180), the height follows the aspect (180 / 0.55 = 327); of the 127 px
    # free height half (63.5 < 0.4 * 200) goes above the box: y = 300 - 63.5 = 236.5, rounded to 236
    ((550, 300, 730, 500), (1280, 1920), (704, 1280), (550, 236, 180, 327)),
    # a tall crop: 200 px free, but no more than 0.4 box heights (40) above the box; the rest below
    ((500, 500, 600, 600), (2000, 2000), (512, 1536), (500, 460, 100, 300)),
    # a box at the top-left corner: shifted inside the image (y -30 -> 0)
    ((0, 10, 100, 110), (1000, 1000), (704, 1280), (0, 0, 100, 182)),
    # a box at the right edge, too tall for the aspect at its width (50 * 1.818 < 100): the crop is
    # 55 wide so the box's height fits; x 948 shifted to 1000 - 55
    ((950, 500, 1000, 600), (1000, 1000), (704, 1280), (945, 500, 55, 100)),
    # the image is too small for 120 x 218: the largest crop of the aspect it holds, 110 x 200,
    # centred on the box and shifted inside (y 20 - 40 -> 0)
    ((40, 20, 160, 140), (200, 200), (704, 1280), (45, 0, 110, 200)),
    # a landscape generation: the box's height (200) fills the crop, 364 wide, nothing of the box cut
    ((550, 300, 730, 500), (1920, 1280), (1280, 704), (458, 300, 364, 200)),
])
def test_the_face_crop_box(box, image, generation, expected):
    assert scail2.face_crop_box(box, *image, *generation) == expected
    assert scail2.FACE_HEADROOM == 0.4


def test_the_face_close_up_is_the_crop_resized_by_lanczos():
    source = torch.randint(0, 256, (1, 400, 300, 3), generator=torch.Generator().manual_seed(0)) / 255.0
    face = scail2.face_reference(source, (100, 80, 160, 140), 32, 64)
    # the box fills 60 px of width, 60 / 0.5 = 120 high; 60 px free, half of it capped at 0.4 * 60 = 24
    # above: y = 56
    assert scail2.face_crop_box((100, 80, 160, 140), 300, 400, 32, 64) == (100, 56, 60, 120)
    expected = torch.empty(64, 32, 3)
    scail2.fit(source[0, 56:176, 100:160], expected)
    assert face.shape == (1, 64, 32, 3) and face.dtype == torch.float32 and torch.equal(face[0], expected)


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
