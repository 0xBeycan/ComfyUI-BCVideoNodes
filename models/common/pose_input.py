"""The pose crop both estimators are fed: its normalisation and its geometry, the same for
ViTPose and RTMW."""
import numpy as np

# Verified in the pipeline.json OpenMMLab ships beside the RTMW ONNX, and in the mmpose
# config behind it: RTMW normalises with mean=[123.675, 116.28, 103.53],
# std=[58.395, 57.12, 57.375] and to_rgb, over a crop taken with padding 1.25. On the 0..1
# RGB frames ComfyUI hands us that is exactly the ImageNet numbers below, and exactly the
# POSE_CROP_RESCALE below, so neither differs between ViTPose and RTMW.
IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406])
IMG_NORM_STD = np.array([0.229, 0.224, 0.225])
# The crop is cut with the same 4:3 geometry for both models - ViTPose samples it at 256x192,
# RTMW at 384x288 - so `bbox_from_detector` cuts the very same region out of the frame and
# only the resolution it is sampled at comes from the model (its `input_shape`).
POSE_CROP_RESCALE = 1.25


def pose_crop(frame, box, resolution):
    """The estimator input cut around `box` from `frame` and sampled at `resolution`
    (height, width): (img_norm [3, h, w] float32, center, scale)."""
    from ...libs.pose_utils.pose2d_utils import bbox_from_detector, crop

    center, scale = bbox_from_detector(box, resolution, rescale=POSE_CROP_RESCALE)
    img = crop(frame, center, scale, resolution)[0]
    img_norm = ((img - IMG_NORM_MEAN) / IMG_NORM_STD).transpose(2, 0, 1).astype(np.float32)
    return img_norm, center, scale
