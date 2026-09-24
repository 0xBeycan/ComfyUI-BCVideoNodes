"""AnimateAdapter: what one core conditioning node needs from the long-video chunk loop
(pipelines/long_video.py). The class attributes and hooks default to the Wan Animate contract
(WanAnimateToVideo, WanAnimate2ToVideo); a node with another contract overrides them. A subclass
per core node lives in its model package and registers itself in the "animate" family under
ANIMATE_NODE.
"""


def check_pose_percents(start, end):
    """Raises when the pose conditioning would start after it ends."""
    if start > end:
        raise ValueError("pose_start_percent ({}) must not be greater than pose_end_percent ({}).".format(start, end))


class AnimateAdapter:
    """The hooks of one core conditioning node, for one run: the loop creates an instance per
    generate call, so per-run state set in ``prepare`` lives on it."""

    ANIMATE_NODE = ""  # the core node id it adapts, and its registry name
    # the fewest outputs the core node must return, and what to tell the user when it returns fewer
    OUTPUTS = 6  # positive, negative, latent, trim_latent, trim_image, video_frame_offset
    UPDATE_HINT = "Update ComfyUI: this node needs the {} that returns trim_latent / trim_image / video_frame_offset."
    # the videos the core node seeks by video_frame_offset: held on their last frame up to the
    # last frame the plan samples ("pose_video" is the loop's own input, the rest pass through)
    HELD_VIDEOS = ("pose_video", "face_video", "background_video")

    def __init__(self, node_name, last_chunk):
        self.node_name = node_name  # the node's class name, for the log lines
        self.last_chunk = last_chunk  # the last_chunk widget (libs/chunking.LAST_CHUNK), for the log lines

    def prepare(self, animate_cls, animate_inputs, reference_image, width, height, frames_per_chunk):
        """Validate / normalize the pass-through inputs before the loop and
        return the frames the core node trims back off every chained chunk.
        Inputs that are the node's own (not the core node's) are popped here."""
        raise NotImplementedError

    def check_videos(self, pose_video, animate_inputs):
        """Checks between the pass-through videos, before any is held. Raises on a mismatch."""
        return None

    def patch_model(self, patched, animate_inputs):
        """Model-level patches applied once per run, after ModelSamplingSD3."""
        return patched

    def continuation(self, anchor, offset):
        """The core-node inputs that chain a chunk to the one before: the previous decoded
        frames (None on the first chunk) and the offset the previous call returned."""
        return {"continue_motion": anchor, "video_frame_offset": offset}

    def chunk_inputs(self, index, offset, anchor, pose_video, animate_inputs):
        """Core-node inputs that change per chunk beyond the continuation and
        length; merged over the pass-through inputs."""
        return {}

    def unpack(self, outputs, anchor):
        """The core node's outputs as (positive, negative, latent, trim_latent, trim_image,
        video_frame_offset): the latent frames to drop after sampling, the decoded frames to
        drop after decoding, and the offset of the next chunk."""
        return tuple(outputs[:self.OUTPUTS])

    def after_animate(self, positive, negative, trim_image, length, offset, animate_inputs):
        """Repairs on the core node's conditioning before it is sampled.
        ``offset`` is the video_frame_offset the core node was called with."""
        return None
