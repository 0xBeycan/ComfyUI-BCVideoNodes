"""Face Crop."""

from .common import PREPROCESS
from .unused_outputs import LINK_INPUTS, drop_unwanted, heavy_wanted, wants


class BCVFaceCrop:
    @classmethod
    def INPUT_TYPES(cls):
        from ..pipelines import face

        return {
            "required": {
                "images": ("IMAGE",),
                "pose_data": ("POSEDATA",),
                "face_padding": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1, "tooltip": "Pixels added on every side of the face box before it is cut and resized to 512x512. Ignored when face_bboxes is connected"}),
            },
            "optional": {
                "face_bboxes": ("BBOX", {"tooltip": "Face boxes (x1, y1, x2, y2), one per frame or one for all, cut as they are: pose_data's face keypoints, face_padding and face_box_smoothing are then not used."}),
                "face_box_smoothing": (list(face.FACE_BOX_SMOOTHING), {"default": face.DEFAULT_FACE_BOX_SMOOTHING, "tooltip": f"Temporal smoothing of the face boxes computed from the face keypoints, against the crop's frame-to-frame jitter. off: each frame's own box. median: per-coordinate median over {face.MEDIAN_WINDOW} frames. size: the box centre kept, its width and height averaged over neighbouring frames (Gaussian, sigma {face.SIZE_SIGMA:g} frames), which stops the size pulsing without letting a fast-moving face leave its box (default). Ignored when face_bboxes is connected"}),
            },
            "hidden": dict(LINK_INPUTS),
        }

    RETURN_TYPES = ("IMAGE", "BBOX")
    RETURN_NAMES = ("face_images", "face_bboxes")
    # not cut when nothing links them (nodes/unused_outputs.py)
    HEAVY_OUTPUTS = ("face_images",)
    FUNCTION = "crop"
    CATEGORY = PREPROCESS
    DESCRIPTION = "Cuts the face out of every frame, from the face keypoints in pose_data or from face_bboxes, as 512x512 images (Wan Animate's face_video)."

    def crop(self, images, pose_data, face_padding, face_bboxes=None, face_box_smoothing="size", prompt_graph=None,
             unique_id=None, *, wanted=None):
        """`wanted`: the heavy outputs a wrapper needs, in place of the link stamp's."""
        from ..pipelines import face

        wanted = heavy_wanted(type(self), prompt_graph, unique_id, wanted)
        return drop_unwanted(type(self), face.crop_faces(
            images, pose_data, face_padding=face_padding, face_bboxes=face_bboxes, smoothing=face_box_smoothing,
            crop=wants(wanted, "face_images")), wanted)
