"""The pose module on synthetic frames with fake models: the supplied-box path that skips the
detector, the key_frame_body_points string, the config, the raw box by default, edge_snap on the
detected and the supplied boxes, draw_head off and 0 stick widths at draw threshold 0, and the pose
images drawn at another size. No ComfyUI and no real model:

    python -m pytest tests/pipelines/test_pose.py
"""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

from pose_fakes import (B, DETECTOR_ROWS, H, W, FakeDetector, FakePose, RecordingPose,  # noqa: E402,F401
                        ScriptedDetector, frames, no_device, pose)


class NoDetector:
    def __call__(self, *args, **kwargs):
        raise AssertionError("the detector must not run when boxes are supplied")


def test_supplied_boxes_skip_the_detector():
    boxes = [(30.0 + i, 20.0, 90.0 + i, 140.0) for i in range(B)]
    model = FakePose()
    images, pose_data, out_boxes, _ = pose.pose_detection(frames(), NoDetector(), model, bboxes=boxes)
    assert model.calls == B
    assert images.shape == (B, H, W, 3)
    assert len(out_boxes) == B and len(pose_data["detections"]) == B
    assert all(d["score"] == 1.0 and d["persons"] == 1 for d in pose_data["detections"])


def test_supplied_boxes_need_no_detector_object():
    pose_data, _ = pose.detect(None, FakePose(), frames(), bboxes=[(30.0, 20.0, 90.0, 140.0)])
    assert len(pose_data["detections"]) == B


def test_a_single_supplied_box_is_used_on_every_frame():
    _, boxes = pose.detect(NoDetector(), FakePose(), frames(), bboxes=[(30.0, 20.0, 90.0, 140.0, 0.5)])
    assert boxes == [(30.0, 20.0, 90.0, 140.0)] * B


def test_supplied_boxes_go_through_the_same_box_logic_as_detections():
    """Supplying the detector's own boxes gives what the detector gives."""
    detected, detected_boxes = pose.detect(FakeDetector(), FakePose(), frames())
    supplied, supplied_boxes = pose.detect(NoDetector(), FakePose(), frames(), bboxes=[(30.0, 20.0, 90.0, 140.0)])
    assert supplied_boxes == detected_boxes


def test_the_wrong_number_of_supplied_boxes_raises():
    with pytest.raises(ValueError, match="3 boxes for 12 frames"):
        pose.detect(NoDetector(), FakePose(), frames(), bboxes=[(0, 0, 10, 10)] * 3)


def test_an_inverted_supplied_box_raises():
    with pytest.raises(ValueError, match="x1 < x2"):
        pose.detect(NoDetector(), FakePose(), frames(), bboxes=[(90, 20, 30, 140)])


def test_the_detection_threshold_reaches_the_detector_and_is_restored():
    detector = FakeDetector()
    pose.detect(detector, FakePose(), frames(), config=pose.PoseConfig(detection_threshold=0.2))
    assert detector.seen_threshold == 0.2 and detector.threshold_conf == 0.05


def test_pose_data_keys():
    pose_data, _ = pose.detect(FakeDetector(), FakePose(), frames())
    assert set(pose_data) == {"pose_metas", "pose_metas_original", "detections", "pose_config"}
    assert pose_data["pose_config"] == pose.asdict(pose.PoseConfig())


def test_the_keypoints_are_the_pose_model_output_unchanged():
    conf = np.full(133, 0.9, dtype=np.float32)
    conf[7] = 0.1
    pose_data, _ = pose.detect(FakeDetector(), FakePose(conf), frames())
    for meta in pose_data["pose_metas_original"]:
        assert meta["keypoints_body"].dtype == np.float32
        # keypoint 7 (the left elbow, AAPose body 6) keeps the model's 0.1 on every frame
        assert meta["keypoints_body"][6, 2] == np.float32(0.1)


def test_config_values_out_of_range_raise():
    with pytest.raises(ValueError, match="detection_threshold"):
        pose.PoseConfig(detection_threshold=1.5)
    with pytest.raises(ValueError, match="box_window"):
        pose.PoseConfig(box_window=-1)


def test_key_frame_body_points_is_the_points_editor_string():
    pose_data, _ = pose.detect(FakeDetector(), FakePose(), frames())
    text = pose.key_frame_body_points(pose_data, 0.5)
    points = json.loads(text)
    assert isinstance(points, list) and len(points) == len(pose.KEY_FRAME_BODY_POINTS)
    assert all(set(p) == {"x", "y"} and type(p["x"]) is int and type(p["y"]) is int for p in points)
    # frame 0's body keypoints in frame pixels, in the exported order
    meta = pose_data["pose_metas_original"][0]
    body = meta["keypoints_body"][list(pose.KEY_FRAME_BODY_POINTS)]
    assert points == [{"x": int(x * W), "y": int(y * H)} for x, y in body[:, :2]]
    # what easy-sam3 accepts: pixel coordinates inside the frame
    assert all(0 <= p["x"] < W and 0 <= p["y"] < H for p in points)


def test_key_frame_body_points_keeps_only_confident_points():
    conf = np.full(133, 0.9, dtype=np.float32)
    conf[[5, 6]] = 0.2  # the shoulders; the neck is their mean, so it drops too
    pose_data, _ = pose.detect(FakeDetector(), FakePose(conf), frames())
    points = json.loads(pose.key_frame_body_points(pose_data, 0.5))
    body = pose_data["pose_metas_original"][0]["keypoints_body"]
    kept = [i for i in pose.KEY_FRAME_BODY_POINTS if body[i, 2] >= 0.5]
    assert len(points) == len(kept) == len(pose.KEY_FRAME_BODY_POINTS) - 3


def test_key_frame_body_points_is_an_empty_list_without_a_confident_point():
    pose_data, _ = pose.detect(FakeDetector(), FakePose(0.1), frames())
    assert pose.key_frame_body_points(pose_data, 0.5) == "[]"


def test_key_frame_body_points_leaves_out_keypoints_outside_the_frame():
    body = np.zeros((20, 3))
    body[:, 2] = 0.9
    body[list(pose.KEY_FRAME_BODY_POINTS), :2] = [(0.5, -0.02), (1.02, 0.5), (-0.004, 0.5), (0.999, 0.999),
                                                   (1.0, 0.25), (0.25, 0.75), (-0.5, 2.0), (0.5, 0.5)]
    pose_data = {"pose_metas_original": [{"width": W, "height": H, "keypoints_body": body}]}
    # y -3.2, x 122.4, x 120 (= W) and (-60, 320) lie outside the 120x160 frame and are left out;
    # x -0.48 truncates to 0 and (119.88, 159.84) to (119, 159), inside, and stay as they were
    assert pose.key_frame_body_points(pose_data, 0.5) == \
        '[{"x": 0, "y": 80}, {"x": 119, "y": 159}, {"x": 30, "y": 120}, {"x": 60, "y": 80}]'


def test_draw_threshold_decides_what_is_drawn():
    pose_data, _ = pose.detect(FakeDetector(), FakePose(0.6), frames())
    assert pose.draw(pose_data, draw_threshold=0.5).sum() > 0
    assert pose.draw(pose_data, draw_threshold=0.7, draw_head=False).sum() == 0


def test_at_draw_threshold_0_draw_head_off_and_0_stick_widths_still_leave_their_parts_out():
    pose_data, _ = pose.detect(FakeDetector(), FakePose(0.6), frames())
    assert pose.draw(pose_data, draw_threshold=0.0).sum() > 0
    assert pose.draw(pose_data, draw_threshold=0.0, body_stick_width=0, hand_stick_width=0).sum() == 0
    assert not torch.equal(pose.draw(pose_data, draw_threshold=0.0, draw_head=False),
                           pose.draw(pose_data, draw_threshold=0.0))


def test_the_pose_images_are_the_uint8_drawings_scaled_to_0_1():
    # each frame drawn on a black uint8 canvas, the frames stacked, converted to float32 and
    # divided by 255: the same bits
    pose_data, _ = pose.detect(FakeDetector(), FakePose(0.6), frames())
    drawn = [pose.draw_aapose_by_meta_new(np.zeros((meta.height, meta.width, 3), dtype=np.uint8), meta, threshold=0.5,
                                          draw_body=True, draw_hand=True, draw_head=True, body_stick_width=-1,
                                          hand_stick_width=-1)
             for meta in pose_data["pose_metas"]]
    expected = torch.from_numpy(np.stack(drawn, 0)).float() / 255.0
    images = pose.draw(pose_data, draw_threshold=0.5)
    assert images.dtype == torch.float32 and images.shape == (B, H, W, 3)
    assert expected.sum() > 0 and torch.equal(images, expected)


def placed(meta, size, x0, y0):
    """The AAPoseMeta `meta` as on a `size` (width, height) canvas, the formula written out: its
    keypoints moved by the x0 columns and y0 rows cut off each side, then scaled by the size over
    what is left."""
    width, height = size
    placed = pose.AAPoseMeta.from_humanapi_meta(meta)
    scale = (width / (meta["width"] - 2 * x0), height / (meta["height"] - 2 * y0))
    for name in ("kps_body", "kps_lhand", "kps_rhand"):
        kps = getattr(placed, name)
        kps -= (x0, y0)
        kps *= scale
    placed.width, placed.height = width, height
    return placed


# the cut of core's center crop (comfy.utils.common_upscale) from the 120 x 160 frames to each size, by
# hand: (60, 60) keeps rows 20-139 (round((160 - 160 x 0.75) / 2) = 20), (90, 40) rows 53-106
# (round(53.33)), (30, 80) columns 30-89; the same aspect cuts nothing
@pytest.mark.parametrize("size, cut", [((W, H), (0, 0)), ((2 * W, 2 * H), (0, 0)), ((60, 60), (0, 20)),
                                       ((90, 40), (0, 53)), ((30, 80), (30, 0))])
def test_drawn_at_a_size_the_pose_is_cut_as_core_cuts_a_hint_and_scaled_to_it(size, cut):
    pose_data, _ = pose.detect(FakeDetector(), FakePose(0.6), frames())
    width, height = size
    drawn = [pose.draw_aapose_by_meta_new(np.zeros((height, width, 3), dtype=np.uint8), placed(meta, size, *cut),
                                          threshold=0.5, draw_body=True, draw_hand=True, draw_head=True,
                                          body_stick_width=-1, hand_stick_width=-1)
             for meta in pose_data["pose_metas_original"]]
    expected = torch.from_numpy(np.stack(drawn, 0)).float() / 255.0
    images = pose.draw(pose_data, draw_threshold=0.5, size=size)
    assert images.dtype == torch.float32 and images.shape == (B, height, width, 3)
    assert expected.sum() > 0 and torch.equal(images, expected)
    if size == (W, H):
        assert torch.equal(images, pose.draw(pose_data, draw_threshold=0.5))
    # pose_data is untouched: drawing again at the frame size gives the frame-size images
    assert torch.equal(pose.draw(pose_data, draw_threshold=0.5), pose.draw(pose_data, draw_threshold=0.5, size=(W, H)))


def test_pose_data_without_frames_raises_what_stacking_no_drawings_raised():
    with pytest.raises(ValueError, match="^need at least one array to stack$"):
        pose.draw({"pose_metas": [], "pose_metas_original": []})


def test_the_draw_threshold_travels_in_pose_data():
    # the guards count the keypoints at the threshold the pose images were drawn at
    _, pose_data, _, _ = pose.pose_detection(frames(), FakeDetector(), FakePose(0.6), draw_threshold=0.65)
    assert pose_data["draw_threshold"] == 0.65


# --- precedence: config fields a run does not read ------------------------------------------

def not_used_lines(caplog):
    return [r.getMessage() for r in caplog.records if "not used" in r.getMessage()]


def test_supplied_boxes_ignore_the_detection_threshold_in_one_line(caplog):
    box = [(30.0, 20.0, 90.0, 140.0)]
    plain, _ = pose.detect(NoDetector(), FakePose(), frames(), bboxes=box)
    with caplog.at_level("INFO"):
        changed, _ = pose.detect(NoDetector(), FakePose(), frames(), bboxes=box,
                                 config=pose.PoseConfig(detection_threshold=0.5))
    assert not_used_lines(caplog) == ["[BCVideoNodes] pose_config.detection_threshold "
                                      "(bboxes connected, the detector does not run) not used"]
    assert plain["detections"] == changed["detections"]


def test_the_default_config_ignores_nothing(caplog):
    with caplog.at_level("INFO"):
        pose.detect(NoDetector(), FakePose(), frames(), bboxes=[(30.0, 20.0, 90.0, 140.0)])
        pose.detect(FakeDetector(), FakePose(), frames(), config=pose.PoseConfig(detection_threshold=0.2))
    assert not_used_lines(caplog) == []


# --- edge_snap ------------------------------------------------------------------------------

def seeded_frames():
    return torch.from_numpy(np.random.default_rng(0).random((B, H, W, 3), dtype=np.float32))


def raw_boxes():
    """DETECTOR_ROWS as the detector hands them over, (x1, y1, x2, y2, score): a row that is None,
    or under 10 px on a side (row 5, 6 px wide), is the whole frame, score -1."""
    return [np.array([0.0, 0.0, W, H, -1.0]) if row is None or row[2] - row[0] < 10 else np.array(row[:5])
            for row in DETECTOR_ROWS]


def corners(boxes):
    return [tuple(float(v) for v in box[:4]) for box in boxes]


def snapped(box):
    """`box` with each edge that stops within 0.15 of the box's width or height from the frame edge
    moved onto that edge."""
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    bw, bh = x2 - x1, y2 - y1
    return (0.0 if x1 < 0.15 * bw else x1, 0.0 if y1 < 0.15 * bh else y1,
            float(W) if W - x2 < 0.15 * bw else x2, float(H) if H - y2 < 0.15 * bh else y2)


def centres(model):
    """The crop centre of each of `model`'s calls."""
    return [tuple(call[1][0].tolist()) for call in model.calls]


def test_box_window_and_edge_snap_are_off_by_default_so_the_pose_is_cropped_from_the_raw_box():
    # the official Wan and Kijai preprocess crop the raw detector box
    config = pose.PoseConfig()
    assert (config.box_window, config.edge_snap) == (0, False)
    model = RecordingPose()
    pose_data, boxes = pose.detect(ScriptedDetector(), model, seeded_frames())
    assert boxes == corners(raw_boxes())
    assert [tuple(d["bbox"]) for d in pose_data["detections"]] == boxes
    assert centres(model) == [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in boxes]


def test_edge_snap_on_moves_only_an_edge_near_the_frame():
    model = RecordingPose()
    pose_data, boxes = pose.detect(ScriptedDetector(), model, seeded_frames(), config=pose.PoseConfig(edge_snap=True))
    expected = corners(raw_boxes())
    expected[7] = (0.0, 21.0, 65.0, 128.0)  # its left edge 5 px from the frame, under 0.15 of its 60 px width
    assert boxes == expected == [snapped(box) for box in raw_boxes()]
    assert [tuple(d["bbox"]) for d in pose_data["detections"]] == boxes
    # the pose is cropped from the snapped box
    assert centres(model) == [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in boxes]


def test_edge_snap_on_snaps_the_widened_boxes():
    model = RecordingPose()
    config = pose.PoseConfig(box_window=4, edge_snap=True)
    pose_data, boxes = pose.detect(ScriptedDetector(), model, seeded_frames(), config=config)
    assert boxes == [snapped(box) for box in pose.widen_over_time(raw_boxes(), 4)]
    assert [tuple(d["bbox"]) for d in pose_data["detections"]] == boxes
    assert centres(model) == [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in boxes]


def test_edge_snap_off_uses_the_widened_boxes_as_they_are():
    model = RecordingPose()
    config = pose.PoseConfig(box_window=4)
    pose_data, boxes = pose.detect(ScriptedDetector(), model, seeded_frames(), config=config)
    widened = corners(pose.widen_over_time(raw_boxes(), 4))
    assert boxes == widened
    assert [tuple(d["bbox"]) for d in pose_data["detections"]] == widened
    assert pose_data["pose_config"]["edge_snap"] is False
    assert centres(model) == [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in widened]
    # on, the snap moves the left edge of every box frame 7's 5 px reaches through the widening
    _, on = pose.detect(ScriptedDetector(), RecordingPose(), seeded_frames(),
                        config=pose.PoseConfig(box_window=4, edge_snap=True))
    assert [i for i in range(B) if on[i] != widened[i]] == [4, 6, 7, 8, 9, 10, 11]


def test_edge_snap_acts_on_supplied_boxes():
    box = [(5.0, 20.0, 65.0, 140.0)]  # 5 px from the left edge, under 0.15 of its 60 px width
    _, on = pose.detect(NoDetector(), FakePose(), frames(), bboxes=box, config=pose.PoseConfig(edge_snap=True))
    _, off = pose.detect(NoDetector(), FakePose(), frames(), bboxes=box)
    assert on == [(0.0, 20.0, 65.0, 140.0)] * B
    assert off == [(5.0, 20.0, 65.0, 140.0)] * B


# --- a half-precision clip -------------------------------------------------------------------

def test_a_half_clip_is_read_frame_by_frame_as_its_float32_levels(monkeypatch):
    # Load Video at precision fp16: the detector and the pose model get the inputs the float32
    # clip gives them, each frame read as float32 on its own (once per pass), never the whole clip
    from video_input_fakes import levels, record_reads

    def run(images):
        detector, model = ScriptedDetector(), RecordingPose()
        pose_data, boxes = pose.detect(detector, model, images)
        return detector.calls, model.calls, pose_data["detections"], boxes

    exact = levels(B, H, W, 3)
    expected = run(exact)
    reads = record_reads(monkeypatch)
    detections, crops, data, boxes = run(exact.half())
    assert reads == [(H, W, 3)] * (2 * B)  # the detector's pass, then the keypoints'
    assert all(np.array_equal(a[0], b[0]) and a[0].dtype == np.float32 for a, b in zip(detections, expected[0]))
    assert all(np.array_equal(a[0], b[0]) and a[0].dtype == np.float32 for a, b in zip(crops, expected[1]))
    assert (data, boxes) == (expected[2], expected[3])


def test_a_half_clip_s_pose_images_are_half_with_the_same_levels():
    # every drawn level k / 255 is exact in float16: the float16 of the float32 clip's pose images
    from video_input_fakes import levels

    exact = levels(B, H, W, 3)
    images = pose.pose_detection(exact, FakeDetector(), FakePose())[0]
    half = pose.pose_detection(exact.half(), FakeDetector(), FakePose())[0]
    assert images.dtype == torch.float32 and half.dtype == torch.float16 and torch.equal(half, images.half())
    assert images.max() == 1.0  # something is drawn
    empty = pose.pose_detection(exact.half(), FakeDetector(), FakePose(), draw_images=False)[0]
    assert empty.shape == (0, H, W, 3) and empty.dtype == torch.float16
