"""The `scail2` Names the SCAIL-2 preprocess and adapter test bodies read pack names through
(tests/names.py)."""
from names import Names, refs

scail2 = Names("scail2", {
    **refs("pipelines.scail2", "PALETTE", "WHITE", "BLACK", "backgrounds", "colored_masks"),
    **refs("libs.mask", "render_identity"),
    **refs("models.scail2.adapter", "ANIMATION", "REPLACEMENT", "character_on_black", "mask_convention"),
    **refs("nodes.sampler", "BCVSCAIL2LongVideoSampler"),
})
