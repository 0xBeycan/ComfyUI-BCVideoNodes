# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""ViTPose-H as the pose pipeline calls it: the heatmap keypoints of a crop, and the test-time
flip as its capability (flip_keypoints)."""
from ..common.wrapper import NativeModel
from .decode import decode_heatmaps
from .flip import flip_back


class ViTPose(NativeModel):
    architecture = "vitpose"
    # ViTPose's confidences are its heatmap maxima as they are: nothing divides them, and
    # PoseConfig.confidence_scale (RTMW only) is ignored for this model.
    conf_scale = None

    def forward(self, img, center, scale, **kwargs):
        return decode_heatmaps(self.run(img), center, scale)

    def flip_keypoints(self, x, center, scale):
        """ViTPose's keypoints from the crop `x` [1, 3, h, w] and its mirror: the mirror's heatmaps
        flipped back with left and right swapped (`flip.flip_back`), the two averaged and decoded
        as the model decodes one pass."""
        heatmaps = (self.run(x) + flip_back(self.run(x[..., ::-1]))) * 0.5
        return decode_heatmaps(heatmaps, center, scale)
