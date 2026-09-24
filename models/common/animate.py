"""AnimateAdapter: what one Wan Animate core conditioning node needs from the long-video chunk
loop (pipelines/long_video.py), as four hooks with today's defaults. A subclass per core node
lives in its model package and registers itself in the "animate" family under ANIMATE_NODE.
"""


class AnimateAdapter:
    """The hooks of one core conditioning node, for one run: the loop creates an instance per
    generate call, so per-run state set in ``prepare`` lives on it."""

    ANIMATE_NODE = ""  # the core node id it adapts, and its registry name

    def __init__(self, node_name):
        self.node_name = node_name  # the node's class name, for the log lines

    def prepare(self, animate_cls, animate_inputs):
        """Validate / normalize the pass-through inputs before the loop and
        return the frames the core node trims back off every chained chunk.
        Inputs that are the node's own (not the core node's) are popped here."""
        raise NotImplementedError

    def patch_model(self, patched, animate_inputs):
        """Model-level patches applied once per run, after ModelSamplingSD3."""
        return patched

    def chunk_inputs(self, index, offset, anchor, pose_video, animate_inputs):
        """Core-node inputs that change per chunk beyond continue_motion /
        video_frame_offset / length; merged over the pass-through inputs."""
        return {}

    def after_animate(self, positive, negative, trim_image, length, offset, animate_inputs):
        """Repairs on the core node's conditioning before it is sampled.
        ``offset`` is the video_frame_offset the core node was called with."""
        return None
