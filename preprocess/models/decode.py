# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Raw pose-model output to keypoints in frame coordinates, [N, K, 3] rows of x, y and a
0..1-ish confidence. Kept apart from the wrappers so the offline conversion can decode
exactly the way the nodes do without importing ComfyUI."""
import numpy as np

from ..pose_utils.pose2d_utils import keypoints_from_heatmaps, transform_preds

# RTMW's head is SimCC, not a heatmap: it classifies each keypoint's column and its row
# separately over the model input's pixels sampled SIMCC_SPLIT_RATIO times each, so the
# model returns simcc_x [N, 133, 288 * 2] and simcc_y [N, 133, 384 * 2].
SIMCC_SPLIT_RATIO = 2.0
# What mmpose calls the SimCC score is min(max simcc_x, max simcc_y), and those are logits,
# not probabilities: measured over 25 frames of a dancer they run 1.8 to 8.0, so the raw
# number means nothing to the 0.3 / 0.5 thresholds the guard, the mask seeding and the
# drawing all apply to ViTPose's heatmap maxima. The divisor is set on the body keypoints:
# by the ratio of the two models' body medians (4.14-5.25 per clip) and by the divisor that
# puts the same fraction of body keypoints below the 0.5 draw threshold as ViTPose does
# (4.57-5.58), it is ~4.6. The earlier 6.0 was calibrated on all 133 keypoints pooled,
# where 68 are face keypoints whose raw score is ~6.5, so it was set for the face and the
# body inherited it.
#
# What it cannot fix: SimCC is less decisive than a heatmap about a keypoint that is not
# there, because it still has to pick some column and some row inside the crop. On keypoints
# the crop cuts off, more stay above 0.3 than ViTPose's do, so the guard sees a few more
# confident keypoints than it would with ViTPose on frames that cut the person.
SIMCC_CONF_SCALE = 4.6


def decode_heatmaps(heatmaps, center, scale, conf_scale=1.0):
    """ViTPose: DARK-refined heatmap maxima mapped back through the crop. The maxima are
    the confidences as they are; a conf_scale other than 1 divides them."""
    points, prob = keypoints_from_heatmaps(heatmaps=heatmaps,
                                        center=center,
                                        scale=scale*200,
                                        unbiased=True,
                                        use_udp=False)
    if conf_scale != 1.0:
        prob = prob / conf_scale
    return np.concatenate([points, prob], axis=2)


def decode_simcc(simcc_x, simcc_y, center, scale, input_size, conf_scale=SIMCC_CONF_SCALE):
    """RTMW: argmax of each SimCC axis, decoded the way mmpose's get_simcc_maximum /
    SimCCLabel.decode do; the score is min(max x, max y) / conf_scale clipped to 0..1.
    `input_size` is the model input's (height, width)."""
    points = np.stack([simcc_x.argmax(axis=2), simcc_y.argmax(axis=2)], axis=-1)
    points = points.astype(np.float32) / SIMCC_SPLIT_RATIO
    vals = np.minimum(simcc_x.max(axis=2), simcc_y.max(axis=2))
    # the points are in the crop `crop` cut, so they go back to the frame through the
    # same transform the heatmap decode uses, over the input grid instead of a heatmap
    height, width = input_size
    for i in range(len(points)):
        points[i] = transform_preds(points[i], center[i], scale[i] * 200, [width, height])
    prob = np.clip(vals / conf_scale, 0.0, 1.0)
    return np.concatenate([points, prob[..., None]], axis=2)
