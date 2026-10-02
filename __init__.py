"""Registration only: the pack's nodes, as ComfyUI lists them."""
from .nodes.sampler import BCVWanAnimate2LongVideoSampler, BCVWanAnimateLongVideoSampler, BCVSCAIL2LongVideoSampler
from .nodes.pose import BCVPoseConfig, BCVPoseDetection
from .nodes.sam3_1_multiplex import BCVSAM3Config, BCVSAM3VideoTrack
from .nodes.face import BCVFaceCrop
from .nodes.guard import BCVMaskGuard, BCVPoseGuard
from .nodes.preprocess import BCVWanAnimatePreprocess, BCVWanAnimatePreprocessGuard
from .nodes.scail2 import BCVSCAIL2ColoredMask, BCVSCAIL2Preprocess, BCVSCAIL2PreprocessGuard
from .nodes.video_input import BCVConformVideo, BCVGetVideoInfo, BCVLoadReferenceImage, BCVLoadVideo, register_plan_route
from .nodes.video_output import BCVSaveVideo, BCVVideoComparer
from .nodes.unused_outputs import register_link_stamp
from .nodes.full_clear import register_full_clear_hook

NODE_CLASS_MAPPINGS = {
    "BCVWanAnimateLongVideoSampler": BCVWanAnimateLongVideoSampler,
    "BCVWanAnimate2LongVideoSampler": BCVWanAnimate2LongVideoSampler,
    "BCVPoseDetection": BCVPoseDetection,
    "BCVPoseConfig": BCVPoseConfig,
    "BCVSAM3VideoTrack": BCVSAM3VideoTrack,
    "BCVSAM3Config": BCVSAM3Config,
    "BCVFaceCrop": BCVFaceCrop,
    "BCVPoseGuard": BCVPoseGuard,
    "BCVMaskGuard": BCVMaskGuard,
    "BCVWanAnimatePreprocess": BCVWanAnimatePreprocess,
    "BCVWanAnimatePreprocessGuard": BCVWanAnimatePreprocessGuard,
    "BCVSCAIL2LongVideoSampler": BCVSCAIL2LongVideoSampler,
    "BCVSCAIL2ColoredMask": BCVSCAIL2ColoredMask,
    "BCVSCAIL2Preprocess": BCVSCAIL2Preprocess,
    "BCVSCAIL2PreprocessGuard": BCVSCAIL2PreprocessGuard,
    "BCVLoadVideo": BCVLoadVideo,
    "BCVGetVideoInfo": BCVGetVideoInfo,
    "BCVLoadReferenceImage": BCVLoadReferenceImage,
    "BCVConformVideo": BCVConformVideo,
    "BCVSaveVideo": BCVSaveVideo,
    "BCVVideoComparer": BCVVideoComparer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BCVWanAnimateLongVideoSampler": "Wan Animate Long Video Sampler",
    "BCVWanAnimate2LongVideoSampler": "Wan Animate 2 Long Video Sampler",
    "BCVPoseDetection": "Pose Detection",
    "BCVPoseConfig": "Pose Config",
    "BCVSAM3VideoTrack": "SAM 3.1 Multiplex Video Track",
    "BCVSAM3Config": "SAM 3.1 Multiplex Config",
    "BCVFaceCrop": "Face Crop",
    "BCVPoseGuard": "Pose Guard",
    "BCVMaskGuard": "Mask Guard",
    "BCVWanAnimatePreprocess": "WanAnimate Preprocess",
    "BCVWanAnimatePreprocessGuard": "WanAnimate Preprocess Guard",
    "BCVSCAIL2LongVideoSampler": "SCAIL-2 Long Video Sampler",
    "BCVSCAIL2ColoredMask": "SCAIL-2 Colored Mask",
    "BCVSCAIL2Preprocess": "SCAIL-2 Preprocess",
    "BCVSCAIL2PreprocessGuard": "SCAIL-2 Preprocess Guard",
    "BCVLoadVideo": "Load Video",
    "BCVGetVideoInfo": "Get Video Info",
    "BCVLoadReferenceImage": "Load Reference Image",
    "BCVConformVideo": "Conform Video",
    "BCVSaveVideo": "Save Video",
    "BCVVideoComparer": "Video Comparer",
}

# writes the link state of the heavy outputs into each prompt, after the other packs' on_prompt
# handlers (nodes/unused_outputs.py)
register_link_stamp(NODE_CLASS_MAPPINGS)
# lets ComfyUI-BCNodes' full clear drop the models the loaders keep (nodes/full_clear.py)
register_full_clear_hook()
# what Load Video will load, for its preview (nodes/video_input.py)
register_plan_route()

# the frontend: the video player of Save Video, Video Comparer and Load Video
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
