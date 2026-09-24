"""The timeline image: the metrics over the frames with the flagged frames marked, drawn with
OpenCV."""
from typing import Union

import numpy as np
import torch

from .common import WARNINGS, MaskRow, PoseRow, PreprocessRow, Scail2Row


PANEL_W, PANEL_H, MARGIN_L, MARGIN_R, MARGIN_T, GAP = 1200, 190, 210, 20, 34, 34
PLOT_W = PANEL_W - MARGIN_L - MARGIN_R  # the width of the plot area, the panels' and the flags'
COLORS = {"blue": (31, 119, 180), "orange": (255, 127, 14), "green": (44, 160, 44), "red": (214, 39, 40), "grey": (150, 150, 150)}
# The timeline panels, as (title, [(legend, row key, colour)]). A series keeps its colour in
# every timeline it appears in.
POSE_PANELS = [
    ("pose", [("mean keypoint confidence", "pose_conf", "orange"), ("limbs vs neighbours", "pose_completeness", "green")]),
    ("motion", [("box IoU vs previous", "box_iou_prev", "orange"), ("torso jump / box diagonal", "torso_jump", "green")]),
]
MASK_PANELS = [
    ("mask", [("mask / box area", "mask_to_box", "blue"), ("mask outside box", "mask_outside_box", "orange")]),
    ("mask vs pose", [("keypoints inside mask", "keypoint_recall", "blue"), ("body not drawn", "body_not_drawn", "red")]),
    ("motion", [("mask IoU vs previous", "mask_iou_prev", "blue"), ("box IoU vs previous", "box_iou_prev", "orange")]),
]
PREPROCESS_PANELS = [
    ("mask", [("mask / box area", "mask_to_box", "blue"), ("mask outside box", "mask_outside_box", "orange")]),
    ("pose", [("keypoints inside mask", "keypoint_recall", "blue"), ("mean keypoint confidence", "pose_conf", "orange"),
              ("limbs vs neighbours", "pose_completeness", "green"), ("body not drawn", "body_not_drawn", "red")]),
    ("motion", [("mask IoU vs previous", "mask_iou_prev", "blue"), ("box IoU vs previous", "box_iou_prev", "orange"),
                ("torso jump / box diagonal", "torso_jump", "green")]),
]

SCAIL2_PANELS = [
    ("driving mask", [("mask area / frame", "mask_area", "blue"), ("mask kept by the latent cut", "latent_kept", "green")]),
    ("motion", [("mask IoU vs previous", "mask_iou_prev", "blue")]),
]


def _polyline(img, values, x0, y0, w, h, color):
    import cv2

    pts = [(int(x0 + i / max(len(values) - 1, 1) * w), int(y0 + h - min(max(v, 0.0), 1.0) * h))
           for i, v in enumerate(values) if v is not None]
    for a, b in zip(pts, pts[1:]):
        cv2.line(img, a, b, color, 1, cv2.LINE_AA)


def _panel(img, y0, title, series):
    """One panel: a 0..1 axis with gridlines and the named series drawn over it."""
    import cv2

    x0, w, h = MARGIN_L, PLOT_W, PANEL_H - GAP
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), COLORS["grey"], 1)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = int(y0 + h - frac * h)
        if 0 < frac < 1:
            cv2.line(img, (x0, y), (x0 + w, y), (225, 225, 225), 1)
        cv2.putText(img, f"{frac:.2f}", (x0 - 38, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["grey"], 1, cv2.LINE_AA)
    # title and legend on the line above the panel
    cv2.putText(img, title, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    legend_x = x0 + 90
    for name, values, color in series:
        _polyline(img, values, x0, y0, w, h, color)
        cv2.line(img, (legend_x, y0 - 14), (legend_x + 18, y0 - 14), color, 2)
        cv2.putText(img, name, (legend_x + 24, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 40, 40), 1, cv2.LINE_AA)
        legend_x += 8 * len(name) + 56


def timeline_image(rows: Union[list[PoseRow], list[MaskRow], list[PreprocessRow], list[Scail2Row]], flags, panels):
    """Metrics over frames with the flagged frames marked, as an IMAGE tensor (no plotting
    library needed, drawn with OpenCV). `panels` is one of the *_PANELS lists."""
    import cv2

    n = len(rows)
    names = list(flags) or ["(no flags)"]
    flag_h = 24 * len(names) + 40
    height = MARGIN_T + len(panels) * PANEL_H + flag_h + 30
    img = np.full((height, PANEL_W, 3), 255, np.uint8)
    y = MARGIN_T
    for title, series in panels:
        _panel(img, y, title, [(name, [m[key] for m in rows], COLORS[color]) for name, key, color in series])
        y += PANEL_H
    x0, w = MARGIN_L, PLOT_W
    cv2.putText(img, "flags", (x0, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.rectangle(img, (x0, y), (x0 + w, y + flag_h - 30), COLORS["grey"], 1)
    for row, name in enumerate(names):
        cy = y + 16 + 24 * row
        cv2.putText(img, name, (12, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 40, 40), 1, cv2.LINE_AA)
        color = COLORS["orange"] if name in WARNINGS else COLORS["red"]
        for f in flags.get(name, []):
            cx = int(x0 + f / max(n - 1, 1) * w)
            cv2.rectangle(img, (cx - 2, cy - 4), (cx + 2, cy + 4), color, -1)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        cx = int(x0 + frac * w)
        cv2.putText(img, str(int(frac * (n - 1))), (cx - 8, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLORS["grey"], 1, cv2.LINE_AA)
    cv2.putText(img, "frame", (PANEL_W // 2 - 20, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 40), 1, cv2.LINE_AA)
    return torch.from_numpy(img).float().unsqueeze(0) / 255.0
