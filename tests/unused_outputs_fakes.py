"""The `unused` Names the unused-heavy-outputs test bodies read pack names through (tests/names.py),
and the prompts they hand a node: the PROMPT graph the on_prompt handler leaves behind."""
from names import Names, Ref, Seam, refs

unused = Names("unused", {
    **refs("nodes.unused_outputs", "STAMP", "LINK_INPUTS", "LinkStamp", "register_link_stamp", "stamps_last",
           "heavy_wanted", "wants", "drop_unwanted", "drop_unlinked_heavy"),
    "draw": Seam(Ref("pipelines.pose", "draw")),
    "render_identity": Seam(Ref("pipelines.scail2", "render_identity")),
    "driving_on_black": Seam(Ref("pipelines.scail2", "driving_on_black")),
    "load_video": Seam(Ref("pipelines.video_input", "load_video")),
    "final_mask": Seam(Ref("libs.mask", "final_mask")),
    "painted_black": Seam(Ref("libs.mask", "painted_black")),
})

# the sampler's chunk loop, under the node_module fixture's alias
sampler_loop = Names("sampler_loop", {"generate": Seam(Ref("pipelines.long_video", "generate"))}, alias="walong")

NODE = "7"


def graph(cls, linked=(), stamp=True, links=None):
    """The PROMPT graph a node with id NODE of `cls` gets: its stamp lists `linked` (no stamp with
    `stamp` False), and a consumer node links each output name of `links` (default: `linked`)."""
    inputs = {unused.STAMP: ",".join(sorted(linked))} if stamp else {}
    prompt = {NODE: {"class_type": cls.__name__, "inputs": inputs}}
    for i, name in enumerate(linked if links is None else links):
        prompt[f"c{i}"] = {"class_type": "Consumer", "inputs": {"x": [NODE, cls.RETURN_NAMES.index(name)]}}
    return prompt


def hidden(cls, linked=(), **kwargs):
    """The node function's hidden inputs (LINK_INPUTS) for `graph(cls, linked, ...)`."""
    values = {"PROMPT": graph(cls, linked, **kwargs), "UNIQUE_ID": NODE}
    return {name: values[kind] for name, kind in unused.LINK_INPUTS.items()}
