"""The `sam3` Names the SAM 3.1 Multiplex test bodies read pack names through (tests/names.py),
and what the split SAM test files share: the default config and a 20-point body layout.
"""
import numpy as np

from names import Names, Ref, Seam, refs, seams

sam3 = Names("sam3", {
    **refs("pipelines.sam3_1_multiplex.config", "DETECTION", "META", "OURS", "PROPAGATED",
           "UNIT_RANGE", "SIGNED_RANGE", "TOKEN_0", "BEST_IOU", "CLEANED", "RAW"),
    **refs("pipelines.sam3_1_multiplex.prompt",
           "PROMPT", "anchor_detections", "non_overlapping", "suppress_recently_occluded", "suppress_shrunk",
           "keep_memory", "memory_score", "memory_view", "selected_frames"),
    **refs("pipelines.sam3_1_multiplex.pose",
           "_clear_of_hand_points", "annexed_points", "background_points", "body_points", "box_bounds", "prompt_for",
           "remember_annexed", "spread_points"),
    **refs("pipelines.sam3_1_multiplex.track",
           "MODES", "MODE_BOX_KEYPOINT", "MODE_PROMPT", "parse_bboxes", "parse_coords", "pose_inputs"),
    **refs("models.sam3_1_multiplex.adapter", "MASK_LOGIT_SCALE", "backbone_frame", "track_frame"),
    "_propagation_backbone": Ref("models.sam3_1_multiplex.adapter", "propagation_backbone"),
    **refs("models.sam3_1_multiplex.postprocess", "low_res_logits", "clean_channel_logits"),
    # core's tracker names: the functions import them when called, so they are read and patched on core's module
    **refs("comfy.ldm.sam3.tracker", "fill_holes_in_mask_scores", "MultiplexMaskDecoder", "SAM31Tracker",
           "_upscale_masks"),
    **refs("comfy.ldm.sam3.sam", "MLP", "PositionEmbeddingRandom"),
    "ops": Ref("comfy.ops"),
    **seams("comfy.ldm.sam3.tracker", "MultiplexState", "_prep_frame"),
    "SAM3Config": Ref("pipelines.sam3_1_multiplex.config", "SAM3_1MultiplexConfig"),
    # the names each caller looks up in its own module when called
    "_multiplex_parts": Seam(Ref("pipelines.sam3_1_multiplex.prompt", "multiplex_parts"),
                             Ref("pipelines.sam3_1_multiplex.pose", "multiplex_parts")),
    **seams("pipelines.sam3_1_multiplex.prompt", "detect_person", "encode_prompt"),
    **seams("pipelines.sam3_1_multiplex.pose", "decode", "is_anchor", "keypoint_recall", "propagate"),
    **seams("pipelines.sam3_1_multiplex.track",
            "LOGITS_SINK", "segment_by_pose", "segment_by_prompt", "segment_by_prompt_multi", "track"),
    "load_sam3": Seam(Ref("pipelines.sam3_1_multiplex.track", "load_sam3_1_multiplex")),
    "SAM3_SIZE": Ref("models.sam3_1_multiplex.adapter", "SAM3_1_MULTIPLEX_SIZE"),
    **refs("libs.keypoints", "L_HIP", "L_SHOULDER", "R_ANKLE", "R_FOOT", "R_HIP", "R_SHOULDER"),
    **refs("libs.mask", "clean_mask", "drop_islands", "fill_holes", "to_frame_size"),
    # core names the functions import when called: ProgressBar is patched where they import it from
    "ProgressBar": Seam(Ref("comfy.utils", "ProgressBar")), "mm": Ref("comfy.model_management")})

C = sam3.SAM3Config()


def kps20(conf=0.0):
    """A 20-point body layout, every keypoint at the centre with confidence `conf`."""
    return np.array([[0.5, 0.5, conf]] * 20, dtype=np.float64)
