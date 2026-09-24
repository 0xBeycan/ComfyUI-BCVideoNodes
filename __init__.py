"""Registration only: the pack's nodes, as ComfyUI lists them."""
from .nodes.sampler import BCVWanAnimate2LongVideoSampler, BCVWanAnimateLongVideoSampler
from .nodes.pose import BCVPoseConfig, BCVPoseDetection
from .nodes.sam3_1_multiplex import BCVSAM3Config, BCVSAM3VideoTrack
from .nodes.face import BCVFaceCrop
from .nodes.guard import BCVMaskGuard, BCVPoseGuard
from .nodes.preprocess import BCVWanAnimatePreprocess, BCVWanAnimatePreprocessGuard

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
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
