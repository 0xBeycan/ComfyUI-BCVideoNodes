# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""RTMW-l as the pose pipeline calls it: the SimCC keypoints of a crop."""
from ..common.wrapper import NativeModel
from .decode import SIMCC_CONF_SCALE, SIMCC_SPLIT_RATIO, decode_simcc


class RTMW(NativeModel):
    """RTMW wholebody: the same 133 COCO-WholeBody keypoints as ViTPose, from a SimCC head."""

    architecture = "rtmw"
    # the divisor of the SimCC score; the preprocess may set another (PoseConfig.confidence_scale)
    conf_scale = SIMCC_CONF_SCALE

    def __init__(self, path):
        super().__init__(path)
        height, width = self.input_size
        bins = (self.config["head"]["cls_x"][1], self.config["head"]["cls_y"][1])
        if bins != (width * SIMCC_SPLIT_RATIO, height * SIMCC_SPLIT_RATIO):
            raise ValueError(f"{path}: expected SimCC axes of {width} x {SIMCC_SPLIT_RATIO:g} and "
                             f"{height} x {SIMCC_SPLIT_RATIO:g} bins, found {bins[0]} and {bins[1]}")

    def forward(self, img, center, scale, **kwargs):
        simcc_x, simcc_y = self.run(img)
        return decode_simcc(simcc_x, simcc_y, center, scale, self.input_size, self.conf_scale)
