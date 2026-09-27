"""The pose crop the estimator is fed: its normalisation and its geometry."""
import numpy as np

# The ImageNet normalisation, on the 0..1 RGB frames ComfyUI hands us. RGB is the order ViTPose
# was trained on, and Kijai's node feeds it RGB too; Wan's preprocess feeds it BGR (pose2d.py flips
# its RGB frames once on the way to the pose crop, twice on the way to the detector). The
# pose-parity measurement ran BGR as its own arm and found no net gain; the owner ruled that RGB
# stays.
IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406])
IMG_NORM_STD = np.array([0.229, 0.224, 0.225])
# The crop is cut around the box padded by this factor, in the model input's aspect (height:width
# 4:3 for ViTPose's 256x192 `input_shape`, the only thing taken from the model), and is black
# where it runs off the frame. 1.25 and 256x192 are ViTPose's top-down values, and Wan's and
# Kijai's (pose2d.py ViTPose.preprocess, nodes.py).
POSE_CROP_RESCALE = 1.25


def pose_crop(frame, box, resolution):
    """The estimator input cut around `box` from `frame` and sampled at `resolution`
    (height, width): (img_norm [3, h, w] float32, center, scale)."""
    from ...libs.pose_utils.pose2d_utils import bbox_from_detector, crop

    center, scale = bbox_from_detector(box, resolution, rescale=POSE_CROP_RESCALE)
    img = crop(frame, center, scale, resolution)[0]
    img_norm = ((img - IMG_NORM_MEAN) / IMG_NORM_STD).transpose(2, 0, 1).astype(np.float32)
    return img_norm, center, scale
