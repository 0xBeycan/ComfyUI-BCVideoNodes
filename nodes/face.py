"""Face Crop."""

from .common import PREPROCESS


class BCVFaceCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "pose_data": ("POSEDATA",),
                "face_padding": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1, "tooltip": "Pixels added on every side of the face box before it is cut and resized to 512x512. Ignored when face_bboxes is connected"}),
            },
            "optional": {
                "face_bboxes": ("BBOX", {"tooltip": "Face boxes (x1, y1, x2, y2), one per frame or one for all, cut as they are: pose_data's face keypoints and face_padding are then not used."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "BBOX")
    RETURN_NAMES = ("face_images", "face_bboxes")
    FUNCTION = "crop"
    CATEGORY = PREPROCESS
    DESCRIPTION = "Cuts the face out of every frame, from the face keypoints in pose_data or from face_bboxes, as 512x512 images (Wan Animate's face_video)."

    def crop(self, images, pose_data, face_padding, face_bboxes=None):
        from ..pipelines import face

        return tuple(face.crop_faces(images, pose_data, face_padding=face_padding, face_bboxes=face_bboxes))
