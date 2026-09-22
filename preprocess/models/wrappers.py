# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""The detector and pose models as the preprocess calls them: each wraps a native torch
module loaded from its safetensors file, managed by ComfyUI like any other model, and
decodes the raw output into boxes or keypoints in frame coordinates."""
import cv2
import numpy as np
import torch
from comfy import model_management as mm
from comfy.model_patcher import ModelPatcher

from ..pose_utils.pose2d_utils import box_convert_simple
from . import checkpoint
from .decode import SIMCC_CONF_SCALE, SIMCC_SPLIT_RATIO, decode_heatmaps, decode_simcc


def load_models(*models):
    """Bring the models to the compute device together, freeing VRAM held by other models
    if needed; ComfyUI moves them back out when another model needs the room."""
    mm.load_models_gpu([m.patcher for m in models], force_full_load=True)


class NativeModel:
    """A native module loaded from a model file and managed by ComfyUI like any other model."""

    # which module the file has to hold, see checkpoint.ARCHITECTURES
    architecture = None

    def __init__(self, path):
        self.path = path
        self.net = checkpoint.load(path, self.architecture)
        self.config = self.net.config
        # (height, width) of the model input, and the [N, C, H, W] shape the preprocess
        # reads the crop resolution from
        self.input_size = tuple(self.config["input_size"])
        self.input_shape = [1, 3, *self.input_size]
        # the module runs in its weights' precision; float frames are cast to it
        self.input_dtype = next(self.net.parameters()).dtype
        self.patcher = ModelPatcher(self.net, load_device=mm.get_torch_device(), offload_device=mm.unet_offload_device())

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def run(self, x):
        x = torch.from_numpy(np.ascontiguousarray(x)).to(self.patcher.load_device, self.input_dtype)
        with torch.inference_mode():
            out = self.net(x)
        if isinstance(out, tuple):
            return tuple(o.float().cpu().numpy() for o in out)
        return out.float().cpu().numpy()


# The detector's score threshold: boxes below it are dropped before NMS and person selection.
DEFAULT_DETECTION_THRESHOLD = 0.05


class Yolo(NativeModel):
    architecture = "yolov10"

    def __init__(self, checkpoint, threshold_conf=DEFAULT_DETECTION_THRESHOLD, threshold_multi_persons=0.1, input_resolution=(640, 640), threshold_iou=0.5, threshold_bbox_shape_ratio=0.4, cat_id=[1], select_type='max', strict=True, sorted_func=None):
        super().__init__(checkpoint)

        self.input_width = 640
        self.input_height = 640
        if self.input_size != (self.input_height, self.input_width):
            raise ValueError(f"{checkpoint}: expected a detector taking {self.input_height}x{self.input_width} "
                             f"input, found {self.input_size[0]}x{self.input_size[1]}")

        self.threshold_multi_persons = threshold_multi_persons
        self.threshold_conf = threshold_conf
        self.threshold_iou = threshold_iou
        self.threshold_bbox_shape_ratio = threshold_bbox_shape_ratio
        self.input_resolution = input_resolution
        self.cat_id = cat_id
        self.select_type = select_type
        self.strict = strict
        self.sorted_func = sorted_func

    def postprocess(self, output, shape_raw, cat_id=[1]):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """
        # Transpose and squeeze the output to match the expected shape

        outputs = np.squeeze(output)
        if len(outputs.shape) == 1:
            outputs = outputs[None]
        if output.shape[-1] != 6 and output.shape[1] == 84:
            outputs = np.transpose(outputs)

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = shape_raw[1] / self.input_width
        y_factor = shape_raw[0] / self.input_height

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        if outputs.shape[-1] == 6:
            max_scores = outputs[:, 4]
            classid = outputs[:, -1]

            threshold_conf_masks = max_scores >= self.threshold_conf
            classid_masks = classid[threshold_conf_masks] != 3.14159

            max_scores = max_scores[threshold_conf_masks][classid_masks]
            classid = classid[threshold_conf_masks][classid_masks]

            boxes = outputs[:, :4][threshold_conf_masks][classid_masks]
            boxes[:, [0, 2]] *= x_factor
            boxes[:, [1, 3]] *= y_factor
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
            boxes = boxes.astype(np.int32)

        else:
            classes_scores = outputs[:, 4:]
            max_scores = np.amax(classes_scores, -1)
            threshold_conf_masks = max_scores >= self.threshold_conf

            classid = np.argmax(classes_scores[threshold_conf_masks], -1)

            classid_masks = classid!=3.14159

            classes_scores = classes_scores[threshold_conf_masks][classid_masks]
            max_scores = max_scores[threshold_conf_masks][classid_masks]
            classid = classid[classid_masks]

            xywh = outputs[:, :4][threshold_conf_masks][classid_masks]

            x = xywh[:, 0:1]
            y = xywh[:, 1:2]
            w = xywh[:, 2:3]
            h = xywh[:, 3:4]

            left = ((x - w / 2) * x_factor)
            top = ((y - h / 2) * y_factor)
            width = (w * x_factor)
            height = (h * y_factor)
            boxes = np.concatenate([left, top, width, height], axis=-1).astype(np.int32)

        boxes = boxes.tolist()
        scores = max_scores.tolist()
        class_ids = classid.tolist()

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.threshold_conf, self.threshold_iou)
        # Iterate over the selected indices after non-maximum suppression

        results = []
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = box_convert_simple(boxes[i], 'xywh2xyxy')
            score = scores[i]
            class_id = class_ids[i]
            results.append(box + [score] + [class_id])
            # # Draw the detection on the input image

        # Return the modified input image
        return np.array(results)


    def process_results(self, results, shape_raw, cat_id=[1], single_person=True):
        if isinstance(results, tuple):
            det_results = results[0]
        else:
            det_results = results

        person_results = []
        person_count = 0
        if len(results):
            max_idx = -1
            max_bbox_size = shape_raw[0] * shape_raw[1] * -10
            max_bbox_shape = -1

            bboxes = []
            idx_list = []
            for i in range(results.shape[0]):
                bbox = results[i]
                if (bbox[-1] + 1 in cat_id) and (bbox[-2] > self.threshold_conf):
                    idx_list.append(i)
                    bbox_shape = max((bbox[2] - bbox[0]), ((bbox[3] - bbox[1])))
                    if bbox_shape > max_bbox_shape:
                        max_bbox_shape = bbox_shape

            results = results[idx_list]

            for i in range(results.shape[0]):
                bbox = results[i]
                bboxes.append(bbox)
                if self.select_type == 'max':
                    bbox_size = (bbox[2] - bbox[0]) * ((bbox[3] - bbox[1]))
                elif self.select_type == 'center':
                    bbox_size = (abs((bbox[2] + bbox[0]) / 2 - shape_raw[1]/2)) * -1
                bbox_shape = max((bbox[2] - bbox[0]), ((bbox[3] - bbox[1])))
                if bbox_size > max_bbox_size:
                    if (self.strict or max_idx != -1) and bbox_shape < max_bbox_shape * self.threshold_bbox_shape_ratio:
                        continue
                    max_bbox_size = bbox_size
                    max_bbox_shape = bbox_shape
                    max_idx = i

            if self.sorted_func is not None and len(bboxes) > 0:
                max_idx = self.sorted_func(bboxes, shape_raw)
                bbox = bboxes[max_idx]
                if self.select_type == 'max':
                    max_bbox_size = (bbox[2] - bbox[0]) * ((bbox[3] - bbox[1]))
                elif self.select_type == 'center':
                    max_bbox_size = (abs((bbox[2] + bbox[0]) / 2 - shape_raw[1]/2)) * -1

            if max_idx != -1:
                person_count = 1

            if max_idx != -1:
                person = {}
                person['bbox'] = results[max_idx, :5]
                person['track_id'] = int(0)
                person_results.append(person)

            for i in range(results.shape[0]):
                bbox = results[i]
                if (bbox[-1] + 1 in cat_id) and (bbox[-2] > self.threshold_conf):
                    if self.select_type == 'max':
                        bbox_size = (bbox[2] - bbox[0]) * ((bbox[3] - bbox[1]))
                    elif self.select_type == 'center':
                        bbox_size = (abs((bbox[2] + bbox[0]) / 2 - shape_raw[1]/2)) * -1
                    if i != max_idx and bbox_size > max_bbox_size * self.threshold_multi_persons and bbox_size < max_bbox_size:
                        person_count += 1
                        if not single_person:
                            person = {}
                            person['bbox'] = results[i, :5]
                            person['track_id'] = int(person_count - 1)
                            person_results.append(person)
            # people the detector is at least fairly sure of; the 0.05 threshold above also
            # counts shadows and reflections
            strong = int(sum(1 for bbox in results if bbox[-2] >= 0.3))
            for person in person_results:
                person['person_count'] = strong
            return person_results
        else:
            return None


    def postprocess_threading(self, outputs, shape_raw, person_results, i, single_person=True, **kwargs):
        result = self.postprocess(outputs[i], shape_raw[i], cat_id=self.cat_id)
        result = self.process_results(result, shape_raw[i], cat_id=self.cat_id, single_person=single_person)
        if result is not None and len(result) != 0:
            person_results[i] = result


    def forward(self, img, shape_raw, **kwargs):
        """
        Runs the detector on `img` [N, 3, 640, 640] (RGB, 0..1) and returns, per image, the
        selected person(s) as [{'bbox': [x1, y1, x2, y2, score], 'track_id', 'person_count'}]
        in the frame coordinates `shape_raw` [N, (H, W)] gives; the whole frame with score
        -1 where nobody was found. Boxes scoring below `threshold_conf` are dropped.
        """
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
            shape_raw = shape_raw.cpu().numpy()

        outputs = self.run(img)
        person_results = [[{'bbox': np.array([0., 0., 1.*shape_raw[i][1], 1.*shape_raw[i][0], -1]), 'track_id': -1}] for i in range(len(outputs))]

        for i in range(len(outputs)):
            self.postprocess_threading(outputs, shape_raw, person_results, i, **kwargs)
        return person_results


class ViTPose(NativeModel):
    architecture = "vitpose"
    # ViTPose's confidences are its heatmap maxima as they are: nothing divides them, and
    # PoseConfig.confidence_scale (RTMW only) is ignored for this model.
    conf_scale = None

    def forward(self, img, center, scale, **kwargs):
        return decode_heatmaps(self.run(img), center, scale)


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
