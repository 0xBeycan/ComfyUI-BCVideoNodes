"""SCAIL-2's colored masks against core's own reading of them, and the SAM 3.1 Multiplex track on
the one-frame reference SCAIL-2 Preprocess hands it. Core's comfy_extras/nodes_scail.py is
imported, so this runs where ComfyUI is importable (with the ComfyUI root on PYTHONPATH):

    PYTHONPATH=/path/to/ComfyUI python -m pytest tests/pipelines/test_scail2.py
"""
import logging

import pytest

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


@pytest.mark.parametrize("black_background, replacement_mode", [(False, False), (False, True), (True, False)])
def test_black_background_is_allowed_outside_replacement_mode(black_background, replacement_mode):
    scail2.check_black_background(black_background, replacement_mode)


def test_black_background_in_replacement_mode_is_an_error():
    with pytest.raises(ValueError, match="black_background is for animation mode: in replacement mode the result keeps "
                                         "the driving video's background.*Turn black_background off, or turn "
                                         "replacement_mode off"):
        scail2.check_black_background(True, True)


# --- the one-frame SAM 3.1 Multiplex track ---------------------------------------------------

from test_sam3_1_multiplex_ab import FakeTracker, person_detection  # noqa: E402
from test_sam3_1_multiplex_golden import clip, rig  # noqa: E402,F401


def test_the_sam_track_runs_on_a_one_frame_reference(rig):
    out = rig.track(FakeTracker(), clip(n=1, seed=1), detections=lambda real: [person_detection(real)])
    assert out["error"] is None
    assert "frames without a mask 0" in out["closing"]
