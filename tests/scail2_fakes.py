"""The `scail2` Names the SCAIL-2 preprocess, guard and adapter test bodies read pack names
through (tests/names.py)."""
from names import Names, refs

scail2 = Names("scail2", {
    **refs("pipelines.scail2", "PALETTE", "WHITE", "BLACK", "backgrounds", "colored_masks", "check_black_background",
           "driving_on_black", "FACE_HEADROOM", "check_face_crop", "face_box", "face_crop_box", "face_reference",
           "extra_reference_mask", "with_extra_reference"),
    **refs("pipelines.face", "FACE_CROP_SCALE", "get_face_bboxes"),
    **refs("libs.mask", "render_identity"),
    **refs("models.scail2.adapter", "ANIMATION", "REPLACEMENT", "character_on_black", "mask_convention"),
    **refs("models.scail2.rope", "official_pose_rope"),
    **refs("models.common.core_nodes", "clip_vision_encode_official"),
    **refs("nodes.sampler", "BCVSCAIL2LongVideoSampler"),
    **refs("pipelines.guard.scail2", "check_scail2", "driving_person", "latent_reading"),
    **refs("libs.resize", "center_crop", "fit"),
    **refs("pipelines.guard.mask", "dropouts"),
    **refs("pipelines.guard", "SCAIL2GuardConfig", "MaskGuardConfig", "SCAIL2_CHECKS", "SCAIL2_ROW", "WARNINGS",
           "GuardFailed", "combine_guards", "check_mask"),
    **refs("pipelines.guard.common", "SCAIL2_REFERENCE", "SCAIL2_EXTRA_REFERENCE"),
})
