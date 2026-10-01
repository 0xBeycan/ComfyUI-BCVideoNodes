# Unused outputs

A whole-clip IMAGE or MASK output that nothing is connected to comes out as an empty (0-frame)
tensor instead of staying in ComfyUI's cache until the prompt ends; where it is a step of its own,
the step does not run at all. That covers Pose Detection `pose_images` (not drawn, with either
`pose_model`), Face Crop `face_images` (not cut), WanAnimate Preprocess `pose_images`, `face_images`,
`mask`, `final_mask` and `bg_images` (no SAM track unless `mask`, `final_mask` or `bg_images` is
connected; no final mask unless `final_mask` or `bg_images` is; no painting unless `bg_images`
is), SCAIL-2 Colored Mask `pose_video_mask` (not
rendered), SCAIL-2 Preprocess `pose_video` and `pose_video_mask` (not computed) and `mask`
(computed, since the colored masks are cut by it, then dropped), and Load Video's and the
samplers' `images` (dropped). Connecting such an output later runs the node again.

When a prompt is queued, the pack writes which of these outputs are connected into the node's
inputs (`bcv_linked_heavy`), which makes the link state part of ComfyUI's cache key.

The limit: another custom node pack can change a queued prompt after this pack has read it (an
`on_prompt` handler that runs after this pack's), and a link it adds could then reach a cached empty
output. Once every custom node has loaded, this pack moves its handler (and ComfyUI-BCNodes') after
every other pack's. A handler added later, while ComfyUI runs, still runs after it: for such a
prompt the saving is off, every output comes out full as without this feature, and the console
says "RAM saving of unused outputs is off for this run: <pack> changes the prompt after it."
ComfyUI-BCNodes does the same for its own nodes and is not counted.
