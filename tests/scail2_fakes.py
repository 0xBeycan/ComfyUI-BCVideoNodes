"""The `scail2` Names the SCAIL-2 preprocess, guard and adapter test bodies read pack names
through (tests/names.py)."""
from names import Names, refs

scail2 = Names("scail2", {
    **refs("pipelines.scail2", "PALETTE", "WHITE", "BLACK", "backgrounds", "colored_masks", "check_black_background",
           "driving_on_black"),
    **refs("libs.mask", "render_identity"),
    **refs("models.scail2.adapter", "ANIMATION", "REPLACEMENT", "character_on_black", "mask_convention"),
    **refs("nodes.sampler", "BCVSCAIL2LongVideoSampler"),
    **refs("pipelines.guard.scail2", "check_scail2", "center_crop"),
    **refs("pipelines.guard", "SCAIL2GuardConfig", "SCAIL2_CHECKS", "SCAIL2_ROW", "WARNINGS", "GuardFailed",
           "combine_guards"),
    **refs("pipelines.guard.common", "SCAIL2_REFERENCE"),
})
