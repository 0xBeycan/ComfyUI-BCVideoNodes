"""The `full_clear` Names the full-clear tests read pack names through (tests/names.py): the hook
ComfyUI-BCNodes' Process Monitor calls (nodes/full_clear.py), the loaders' unloads it calls and the
weight count they report (libs/tensor_bytes.py)."""
from names import Names, Ref, refs, seams

full_clear = Names("full_clear", {
    **refs("nodes.full_clear", "FULL_CLEAR_HOOKS", "drop_cached_models", "register_full_clear_hook"),
    **seams("nodes.full_clear", "prompt_server"),
    **refs("libs.tensor_bytes", "module_bytes"),
    "pose_unload": Ref("models.common.loader", "unload"),
    **seams("models.common.loader", "_loaded"),
    "sam3_unload": Ref("models.sam3_1_multiplex.loader", "unload"),
    "sam3_loaded": Ref("models.sam3_1_multiplex.loader", "_loaded"),
})
