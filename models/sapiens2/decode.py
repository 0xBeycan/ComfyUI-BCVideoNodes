"""Sapiens2's input crop and heatmap decode, its own UDP pipeline (what the model was trained with),
ported from facebookresearch/sapiens2 (commit 7e5bae88456ac418ff0e58e74106c9fe192055d4): the config
sapiens2_*_keypoints308_shutterstock_goliath_3po-1024x768.py uses codec UDPHeatmap (sigma 6) and
test_pipeline PoseGetBBoxCenterScale -> PoseTopdownAffine(use_udp=True).
- crop: udp_crop_params / udp_warp_matrix / udp_crop, then normalise (ImageNet mean/std on 0..1 RGB,
  models/common/pose_input.py: Sapiens2's data_preprocessor mean [123.675, 116.28, 103.53] / std
  [58.395, 57.12, 57.375] on RGB are the same values on 0..255);
- decode: udp_decode (argmax, DARK-UDP refinement with the official zero-padded blur, UDP scaling back to
  the frame).
The pack's keypoints_from_heatmaps(use_udp=True) (libs/pose_utils/pose2d_utils.py) has the same
coordinate transform, but its post_dark_udp blurs in place with cv2's reflected border instead of the
official zero padding, so keypoints near the heatmap edge would differ; the official decode is ported
instead.

Numpy at module level; cv2 inside the functions that use it.
"""
import numpy as np

from ..common.pose_input import IMG_NORM_MEAN, IMG_NORM_STD

INPUT_SIZE = (768, 1024)     # (width, height), as the official codec's input_size
HEATMAP_SIZE = (192, 256)    # (width, height)
PADDING = 1.25               # PoseGetBBoxCenterScale default (pose_transforms.py L195)
BLUR_KERNEL = 11             # UDPHeatmap blur_kernel_size default (udp_heatmap.py L31); the config keeps it


# the official UDP crop

def udp_crop_params(box, input_size=INPUT_SIZE, padding=PADDING):
    """(center [2], scale [2]) of the crop around box (x1, y1, x2, y2): bbox_xyxy2cs(padding)
    (sapiens/pose/src/datasets/transforms/bbox_transforms.py L30-46) then
    PoseTopdownAffine._fix_aspect_ratio to input w/h (pose_transforms.py L112-129)."""
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    center = np.array([x1 + x2, y1 + y2]) * 0.5
    w, h = (x2 - x1) * padding, (y2 - y1) * padding
    aspect = input_size[0] / input_size[1]
    scale = np.array([w, w / aspect]) if w > h * aspect else np.array([h * aspect, h])
    return center, scale


def udp_warp_matrix(center, scale, output_size=INPUT_SIZE):
    """get_udp_warp_matrix at rotation 0 (bbox_transforms.py L108-137): frame -> crop pixels."""
    input_size = np.asarray(center) * 2
    scale_x = (output_size[0] - 1) / scale[0]
    scale_y = (output_size[1] - 1) / scale[1]
    warp = np.zeros((2, 3), dtype=np.float32)
    warp[0, 0] = scale_x
    warp[0, 2] = scale_x * (-0.5 * input_size[0] + 0.5 * scale[0])
    warp[1, 1] = scale_y
    warp[1, 2] = scale_y * (-0.5 * input_size[1] + 0.5 * scale[1])
    return warp


def udp_crop(frame, center, scale, output_size=INPUT_SIZE):
    """The crop as PoseTopdownAffine.transform makes it (pose_transforms.py L143-158): cv2.warpAffine
    of the uint8 frame, INTER_AREA when shrinking, INTER_CUBIC when enlarging."""
    import cv2
    warp = udp_warp_matrix(center, scale, output_size)
    factor = min(np.linalg.norm(warp[0, :2]), np.linalg.norm(warp[1, :2]))
    interp = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_CUBIC
    return cv2.warpAffine(frame, warp, (int(output_size[0]), int(output_size[1])), flags=interp)


# the official UDP decode

def _gaussian_blur(heatmaps, kernel):
    """post_processing.gaussian_blur (codecs/utils/post_processing.py L142-171): zero padded,
    rescaled to the original maximum. In place on [K, H, W]."""
    import cv2
    border = (kernel - 1) // 2
    K, H, W = heatmaps.shape
    for k in range(K):
        origin_max = np.max(heatmaps[k])
        dr = np.zeros((H + 2 * border, W + 2 * border), dtype=np.float32)
        dr[border:-border, border:-border] = heatmaps[k].copy()
        dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
        heatmaps[k] = dr[border:-border, border:-border].copy()
        heatmaps[k] *= origin_max / np.max(heatmaps[k])
    return heatmaps


def _refine_dark_udp(keypoints, heatmaps, kernel):
    """refinement.refine_keypoints_dark_udp (codecs/utils/refinement.py L113-172) for one
    instance: keypoints [K, 2] heatmap cells, heatmaps [K, H, W] (modified)."""
    K = keypoints.shape[0]
    H, W = heatmaps.shape[1:]
    heatmaps = _gaussian_blur(heatmaps, kernel)
    np.clip(heatmaps, 1e-3, 50.0, heatmaps)
    np.log(heatmaps, heatmaps)
    pad = np.pad(heatmaps, ((0, 0), (1, 1), (1, 1)), mode="edge").flatten()
    index = keypoints[:, 0] + 1 + (keypoints[:, 1] + 1) * (W + 2)
    index += (W + 2) * (H + 2) * np.arange(0, K)
    index = index.astype(int).reshape(-1, 1)
    i_ = pad[index]
    ix1, iy1, ix1y1 = pad[index + 1], pad[index + W + 2], pad[index + W + 3]
    ix1_y1_, ix1_, iy1_ = pad[index - W - 3], pad[index - 1], pad[index - 2 - W]
    derivative = np.concatenate([0.5 * (ix1 - ix1_), 0.5 * (iy1 - iy1_)], axis=1).reshape(K, 2, 1)
    dxx = ix1 - 2 * i_ + ix1_
    dyy = iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1).reshape(K, 2, 2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    return keypoints - np.einsum("imn,ink->imk", hessian, derivative).reshape(K, 2)


def udp_decode(heatmaps, center, scale, input_size=INPUT_SIZE, kernel=BLUR_KERNEL):
    """[K, H, W] heatmaps of one crop -> [K, 3] (x, y in frame pixels, the raw heatmap maximum).
    UDPHeatmap.decode, gaussian type (codecs/udp_heatmap.py L86-125: get_heatmap_maximum,
    refine_keypoints_dark_udp, keypoints / (heatmap size - 1) * input_size), then the demo's crop ->
    frame step (tools/vis/vis_pose.py L92-99: / input_size * bbox_scale + center - 0.5 * scale)."""
    heatmaps = np.array(heatmaps, dtype=np.float32, copy=True)
    K, H, W = heatmaps.shape
    # post_processing.get_heatmap_maximum (L100-139)
    flat = heatmaps.reshape(K, -1)
    y, x = np.unravel_index(np.argmax(flat, axis=1), shape=(H, W))
    locs = np.stack((x, y), axis=-1).astype(np.float32)
    vals = np.amax(flat, axis=1)
    locs[vals <= 0.0] = -1
    locs = _refine_dark_udp(locs, heatmaps, kernel)
    crop_xy = locs / [W - 1, H - 1] * np.asarray(input_size, dtype=np.float64)
    frame_xy = crop_xy / np.asarray(input_size) * scale + center - 0.5 * np.asarray(scale)
    return np.concatenate([frame_xy, vals[:, None]], axis=1).astype(np.float32)


def normalise(crop):
    """uint8 RGB crop [H, W, 3] -> the model input [3, H, W] float32."""
    return ((crop.astype(np.float32) / 255.0 - IMG_NORM_MEAN) / IMG_NORM_STD).transpose(2, 0, 1).astype(np.float32)


def crop_input(frame, box):
    """(model input [3, 1024, 768] float32, center, scale) of `frame` (uint8 RGB) around `box`."""
    center, scale = udp_crop_params(box)
    return normalise(udp_crop(frame, center, scale)), center, scale
