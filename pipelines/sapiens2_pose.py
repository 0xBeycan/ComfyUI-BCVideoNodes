"""Sapiens2 Pose over a batch of frames: Pose Detection's flow (pipelines/pose.py: the person box from
YOLO or the supplied bboxes, the Pose Config box switches and draw rules, the drawing,
key_frame_body_points) with Sapiens2 in place of ViTPose-H for the body, the feet and the hands.

The keypoints are a hybrid, the owner's choice from the pose bake-off: COCO-WholeBody rows 23-90 (the
68 face points) are ViTPose-H's on the same boxes, exactly what Pose Detection gives there; every other
row is Sapiens2's, from its own crop and decode (models/sapiens2/decode.py, Sapiens2's UDP pipeline)
and its 70 -> 133 table (models/sapiens2/keypoints.py). So Face Crop, the guards' face checks and the
back_view_face draw rule read the same face as they do after Pose Detection. Raw per frame, as Pose
Detection: no flip test, no temporal step, the confidences the raw heatmap peaks.
"""
import numpy as np

from .pose import PoseConfig, _input_resolution, crop_keypoints, detect_with, pose_outputs

# person crops (1024x768) per Sapiens2 forward pass
BATCH_SIZE = 4


def _frames_uint8(images_np):
    """Frames 0..1 float -> uint8 RGB: Sapiens2's crop warps uint8 frames, as its official pipeline does."""
    return (np.clip(images_np, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def hybrid_keypoints(sapiens2, vitpose, images_np, boxes, progress=None):
    """[B, 133, 3] COCO-WholeBody keypoints in frame pixels: the face rows 23-90 from `vitpose` (a
    PoseEstimator, on Pose Detection's crop of each box), every other row from `sapiens2`
    (models/sapiens2/wrapper.Sapiens2Pose, on its own crop of the same box, BATCH_SIZE crops a pass).
    `progress(done)` after each Sapiens2 batch."""
    from tqdm import tqdm

    from ..models.sapiens2.decode import crop_input
    from ..models.sapiens2.keypoints import FACE

    face = crop_keypoints(vitpose, images_np, boxes, _input_resolution(vitpose))[:, FACE]
    out = []
    for start in tqdm(range(0, len(boxes), BATCH_SIZE), desc="Extracting Sapiens2 keypoints"):
        stop = min(start + BATCH_SIZE, len(boxes))
        crops, centers, scales = zip(*(crop_input(frame, box) for frame, box in
                                       zip(_frames_uint8(images_np[start:stop]), boxes[start:stop])))
        out.append(sapiens2(np.stack(crops), np.stack(centers), np.stack(scales)))
        if progress is not None:
            progress(stop)
    keypoints = np.concatenate(out, 0)
    keypoints[:, FACE] = face
    return keypoints


def sapiens2_pose(images, detector, sapiens2, vitpose, bboxes=None, config=None, body_stick_width=-1,
                  hand_stick_width=-1, draw_head=True, draw_threshold=0.5, draw_images=True):
    """The Sapiens2 Pose node: what pipelines/pose.pose_detection returns, (pose_images, pose_data,
    bboxes, key_frame_body_points), with the hybrid keypoints; `draw_images` as there. `detector` is
    not called when `bboxes` is given and may then be None."""
    config = config or PoseConfig()
    pose_data, boxes = detect_with(detector, [sapiens2, vitpose], lambda frames, boxes, progress: hybrid_keypoints(
        sapiens2, vitpose, frames, boxes, progress), images, bboxes=bboxes, config=config)
    return pose_outputs(pose_data, boxes, config, body_stick_width, hand_stick_width, draw_head, draw_threshold,
                        draw_images)
