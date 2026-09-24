"""The pose crop the estimator is fed: its normalisation and its geometry."""
import numpy as np

# The ImageNet normalisation, on the 0..1 RGB frames ComfyUI hands us.
IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406])
IMG_NORM_STD = np.array([0.229, 0.224, 0.225])
# The crop is cut with a 4:3 geometry around the box padded by this factor; only the
# resolution it is sampled at comes from the model (its `input_shape`, 256x192 for ViTPose).
POSE_CROP_RESCALE = 1.25


def pose_crop(frame, box, resolution):
    """The estimator input cut around `box` from `frame` and sampled at `resolution`
    (height, width): (img_norm [3, h, w] float32, center, scale)."""
    from ...libs.pose_utils.pose2d_utils import bbox_from_detector, crop

    center, scale = bbox_from_detector(box, resolution, rescale=POSE_CROP_RESCALE)
    img = crop(frame, center, scale, resolution)[0]
    img_norm = ((img - IMG_NORM_MEAN) / IMG_NORM_STD).transpose(2, 0, 1).astype(np.float32)
    return img_norm, center, scale
