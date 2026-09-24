"""The face crop on synthetic frames and pose_data. No ComfyUI and no model:

    python -m pytest tests/pipelines/test_face.py
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

from bcvideonodes.pipelines import face  # noqa: E402

B, H, W = 4, 200, 160


def pose_data(face_xy=(0.5, 0.3), spread=0.05):
    """Face keypoints (normalised) in a small cloud around `face_xy` on every frame."""
    metas = []
    for _ in range(B):
        kp = np.zeros((69, 3))
        kp[:, 0] = face_xy[0] + np.linspace(-spread, spread, 69)
        kp[:, 1] = face_xy[1] + np.linspace(-spread, spread, 69)
        kp[:, 2] = 0.9
        metas.append({"width": W, "height": H, "keypoints_face": kp})
    return {"pose_metas_original": metas}


def frames():
    return torch.rand(B, H, W, 3)


def test_the_boxes_come_from_the_face_keypoints():
    images = frames()
    crops, boxes = face.crop_faces(images, pose_data())
    assert crops.shape == (B, face.FACE_SIZE, face.FACE_SIZE, 3) and len(boxes) == B
    x1, y1, x2, y2 = boxes[0]
    assert 0 <= x1 < 0.5 * W < x2 <= W and 0 <= y1 < 0.3 * H < y2 <= H


def test_padding_grows_the_computed_boxes_inside_the_frame():
    _, plain = face.crop_faces(frames(), pose_data())
    _, padded = face.crop_faces(frames(), pose_data(), face_padding=10)
    assert padded[0] == (plain[0][0] - 10, plain[0][1] - 10, plain[0][2] + 10, plain[0][3] + 10)
    _, huge = face.crop_faces(frames(), pose_data(), face_padding=500)
    assert huge[0] == (0, 0, W, H)


def test_supplied_face_boxes_override_the_computed_ones():
    images = frames()
    given = [(10, 20, 60, 90)] * B
    crops, boxes = face.crop_faces(images, pose_data(), face_padding=10, face_bboxes=given)
    # used as given: the padding belongs to the computed boxes only
    assert boxes == given
    import cv2
    expected = cv2.resize(images[0].numpy()[20:90, 10:60], (face.FACE_SIZE, face.FACE_SIZE))
    assert np.array_equal(crops[0].numpy(), expected)


def test_a_single_supplied_face_box_is_used_on_every_frame():
    _, boxes = face.crop_faces(frames(), pose_data(), face_bboxes=[(10, 20, 60, 90)])
    assert boxes == [(10, 20, 60, 90)] * B


def test_the_wrong_number_of_face_boxes_raises():
    with pytest.raises(ValueError, match="2 boxes for 4 frames"):
        face.crop_faces(frames(), pose_data(), face_bboxes=[(10, 20, 60, 90)] * 2)


def test_a_negative_face_box_raises():
    with pytest.raises(ValueError, match="0 <= x1"):
        face.crop_faces(frames(), pose_data(), face_bboxes=[(-5, 20, 60, 90)])


def test_pose_data_for_other_frames_raises():
    with pytest.raises(ValueError, match="pose_data holds 4 frames and images 2"):
        face.crop_faces(torch.rand(2, H, W, 3), pose_data())


def test_pose_data_without_keypoints_raises():
    with pytest.raises(ValueError, match="pose_metas_original"):
        face.crop_faces(frames(), {"pose_metas": []})


def test_supplied_face_boxes_ignore_the_padding_and_say_so(caplog):
    images = frames()
    given = [(10, 20, 60, 90)]
    plain = face.crop_faces(images, pose_data(), face_bboxes=given)
    with caplog.at_level("INFO"):
        padded = face.crop_faces(images, pose_data(), face_padding=10, face_bboxes=given)
    assert torch.equal(plain[0], padded[0]) and plain[1] == padded[1]
    lines = [r.getMessage() for r in caplog.records if "not used" in r.getMessage()]
    assert lines == ["[BCVideoNodes] face_bboxes connected: cut as given; pose_data's face keypoints and face_padding 10 not used"]
