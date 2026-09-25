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


def test_an_empty_crop_on_a_tiny_frame_fails_in_cv2():
    # the face box lies off a 2x2 frame, so the crop is empty; the centre crop is empty too
    # (its side is int(0.3 * 2) = 0), and the zeros that stand in for it are 0x0, which cv2.resize refuses
    cv2 = pytest.importorskip("cv2")
    kp = np.zeros((69, 3))
    kp[:, 0], kp[:, 1], kp[:, 2] = 10.0 + np.linspace(-0.1, 0.1, 69), 0.5, 0.9
    with pytest.raises(cv2.error):
        face.crop_faces(torch.rand(1, 2, 2, 3), {"pose_metas_original": [{"width": 2, "height": 2, "keypoints_face": kp}]})


def test_supplied_face_boxes_ignore_the_padding_and_say_so(caplog):
    images = frames()
    given = [(10, 20, 60, 90)]
    plain = face.crop_faces(images, pose_data(), face_bboxes=given)
    with caplog.at_level("INFO"):
        padded = face.crop_faces(images, pose_data(), face_padding=10, face_bboxes=given)
    assert torch.equal(plain[0], padded[0]) and plain[1] == padded[1]
    lines = [r.getMessage() for r in caplog.records if "not used" in r.getMessage()]
    assert lines == ["[BCVideoNodes] face_bboxes connected: cut as given; pose_data's face keypoints and face_padding 10 not used"]


def moving_pose_data(spreads, centres=None, width=W, height=H):
    """One frame per entry of `spreads`: a face keypoint cloud of that half-size (normalised)
    around `centres[i]` (default the upper middle)."""
    centres = centres or [(0.5, 0.3)] * len(spreads)
    metas = []
    for (cx, cy), spread in zip(centres, spreads):
        kp = np.zeros((69, 3))
        kp[:, 0] = cx + np.linspace(-spread, spread, 69)
        kp[:, 1] = cy + np.linspace(-spread, spread, 69)
        kp[:, 2] = 0.9
        metas.append({"width": width, "height": height, "keypoints_face": kp})
    return {"pose_metas_original": metas}


def side(box):
    return np.sqrt((box[2] - box[0]) * (box[3] - box[1]))


def test_size_is_the_default_smoothing():
    pd = moving_pose_data([0.05, 0.08, 0.05, 0.06, 0.05])
    assert face.DEFAULT_FACE_BOX_SMOOTHING == "size"
    assert face.face_bboxes_from_pose(pd, W, H) == face.face_bboxes_from_pose(pd, W, H, smoothing="size")


def test_smoothing_off_leaves_the_boxes_as_they_were():
    pd = moving_pose_data([0.05, 0.08, 0.05, 0.06, 0.05])
    plain = face.face_bboxes_from_pose(pd, W, H, smoothing="off")
    # each frame on its own, as before the switch existed
    assert plain == [face.face_bboxes_from_pose({"pose_metas_original": [m]}, W, H, smoothing="off")[0]
                     for m in pd["pose_metas_original"]]


def test_median_smoothing_is_the_per_coordinate_median_of_a_centred_window():
    pd = moving_pose_data([0.05, 0.08, 0.04, 0.09, 0.05, 0.06, 0.07])
    raw = np.array(face.face_bboxes_from_pose(pd, W, H, smoothing="off"), dtype=np.float64)
    smoothed = face.face_bboxes_from_pose(pd, W, H, smoothing="median")
    half = face.MEDIAN_WINDOW // 2
    expected = [tuple(int(v) for v in np.median(raw[max(0, i - half):i + half + 1], axis=0)) for i in range(len(raw))]
    assert smoothed == expected


def test_size_smoothing_keeps_the_centre_and_flattens_a_size_pulse():
    spreads = [0.05] * 6 + [0.07] + [0.05] * 6
    pd = moving_pose_data(spreads)
    raw = face.face_bboxes_from_pose(pd, W, H, smoothing="off")
    smoothed = face.face_bboxes_from_pose(pd, W, H, smoothing="size")
    for r, s in zip(raw, smoothed):
        assert abs((r[0] + r[2]) / 2 - (s[0] + s[2]) / 2) <= 1 and abs((r[1] + r[3]) / 2 - (s[1] + s[3]) / 2) <= 1
    pulse = side(raw[6]) / side(raw[0]) - 1
    assert side(smoothed[6]) / side(smoothed[0]) - 1 < pulse / 2
    # the frames far from the pulse keep (almost) their size
    assert abs(side(smoothed[0]) - side(raw[0])) <= 2


def test_size_smoothing_on_a_clip_shorter_than_its_kernel():
    pd = moving_pose_data([0.05, 0.07])
    assert len(face.face_bboxes_from_pose(pd, W, H, smoothing="size")) == 2


def test_a_frame_without_face_keypoints_keeps_its_box_and_is_left_out_of_the_smoothing():
    pd = moving_pose_data([0.05] * 5)
    pd["pose_metas_original"][2]["keypoints_face"][:, :2] = np.nan
    raw = face.face_bboxes_from_pose(pd, W, H, smoothing="off")
    for mode in ("median", "size"):
        smoothed = face.face_bboxes_from_pose(pd, W, H, smoothing=mode)
        assert smoothed[2] == raw[2]
        assert smoothed[1] == raw[1] and smoothed[3] == raw[3]


def test_smoothing_comes_before_the_padding():
    pd = moving_pose_data([0.05, 0.07, 0.05])
    smoothed = face.face_bboxes_from_pose(pd, W, H, smoothing="size")
    padded = face.face_bboxes_from_pose(pd, W, H, face_padding=4, smoothing="size")
    assert padded == [(x1 - 4, y1 - 4, x2 + 4, y2 + 4) for x1, y1, x2, y2 in smoothed]


def test_an_unknown_smoothing_raises():
    with pytest.raises(ValueError, match="face_box_smoothing must be one of"):
        face.face_bboxes_from_pose(pose_data(), W, H, smoothing="gauss")


def test_supplied_face_boxes_ignore_the_smoothing_and_say_so(caplog):
    given = [(10, 20, 60, 90)]
    with caplog.at_level("INFO"):
        _, boxes = face.crop_faces(frames(), pose_data(), face_padding=10, face_bboxes=given, smoothing="size")
    assert boxes == given * B
    lines = [r.getMessage() for r in caplog.records if "not used" in r.getMessage()]
    assert lines == ["[BCVideoNodes] face_bboxes connected: cut as given; pose_data's face keypoints and face_padding 10"
                     " and face_box_smoothing size not used"]
