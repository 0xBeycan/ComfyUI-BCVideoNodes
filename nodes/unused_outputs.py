"""Unused heavy outputs return empty: a whole-clip IMAGE or MASK that no node consumes is not kept
in ComfyUI's output cache until the prompt ends.

A node lists such outputs in HEAVY_OUTPUTS (names from its RETURN_NAMES), takes the hidden
LINK_INPUTS, reads `heavy_wanted` and returns through `drop_unwanted`: an unlinked heavy output
leaves as a 0-frame tensor of the same dtype and trailing shape. Where the output is a step of its
own that the node's other outputs do not depend on, the node does not compute it at all (`wants`).

The link state must be part of the node's cache key, or connecting the output later would hit the
cache entry holding the empty tensor. ComfyUI builds that key from the node's own inputs and its
ancestors only, and IS_CHANGED gets an empty PROMPT. So an on_prompt handler (`LinkStamp`, added by
`register_link_stamp` from the root __init__) writes the linked heavy outputs into each heavy
node's inputs as STAMP before the prompt is validated: every input key is part of the cache key,
and an input the node does not declare is never passed to it.

- No stamp (the handler did not run: a direct executor call, a node made by expansion, a node
  called by a wrapper): every output full.
- At run time the node also reads the final PROMPT: a link to a heavy output that the stamp does
  not list returns that output full, with a warning.
- An on_prompt handler of another pack that runs after ours could add a link the stamp never saw,
  and a cached empty output would then reach it. So when one is registered after ours, the stamp
  is left out for that prompt (every output full) and the console and a toast (EVENT,
  web/js/unused_outputs.js) name the pack. Handlers marked `bc_link_stamp` (this pack's and
  ComfyUI-BCNodes') only write their own stamps and are not counted.

A wrapper node passes its own wanted set to the nodes it calls (`wanted=`): its outputs carry the
names of theirs.
"""
from ..libs import log
from .common import prompt_server

STAMP = "bcv_linked_heavy"
EVENT = "bcvideonodes.unused_outputs"
# not `prompt`: the preprocess wrappers have a `prompt` widget, which a hidden input of that name
# would overwrite
LINK_INPUTS = {"prompt_graph": "PROMPT", "unique_id": "UNIQUE_ID"}


def _is_link(value):
    """comfy_execution.graph_utils.is_link, restated so this module imports nothing from ComfyUI."""
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], (int, float))


def _heavy_indices(cls):
    """{output index: name} of the HEAVY_OUTPUTS of `cls`."""
    return {cls.RETURN_NAMES.index(name): name for name in getattr(cls, "HEAVY_OUTPUTS", ())}


def _links(prompt, heavy):
    """{node id: names linked} for `heavy` {node id: {output index: name}}: the heavy outputs some
    node of `prompt` links, whether or not that node runs."""
    linked = {node_id: set() for node_id in heavy}
    for node in prompt.values():
        inputs = node.get("inputs") if isinstance(node, dict) else None
        for value in (inputs.values() if isinstance(inputs, dict) else ()):
            if _is_link(value) and value[0] in heavy:
                name = heavy[value[0]].get(int(value[1]))
                if name is not None:
                    linked[value[0]].add(name)
    return linked


def _pack_of(handler):
    """The custom node folder a handler comes from (its module's top-level file), else its module."""
    import functools
    import os
    import sys

    target = handler
    while isinstance(target, functools.partial):
        target = target.func
    module = getattr(target, "__module__", None) or type(target).__module__
    top = module.split(".")[0]
    file = getattr(sys.modules.get(top), "__file__", None)
    if file:
        path = os.path.dirname(file) if os.path.basename(file) == "__init__.py" else os.path.splitext(file)[0]
        return os.path.basename(path)
    # ComfyUI names a custom node module by its path, dots written as _x_
    return os.path.basename(top).replace("_x_", ".")


class LinkStamp:
    """The on_prompt handler: writes STAMP, the sorted comma-joined linked heavy outputs, into the
    inputs of every node of `classes` that declares HEAVY_OUTPUTS, or removes it from all of them
    when another pack's handler runs after this one. `server` is ComfyUI's PromptServer instance."""

    bc_link_stamp = True

    def __init__(self, classes, server):
        self.classes = classes
        self.server = server

    def __call__(self, json_data):
        prompt = json_data.get("prompt") if isinstance(json_data, dict) else None
        if not isinstance(prompt, dict):
            return json_data
        heavy = {}
        for node_id, node in prompt.items():
            cls = self.classes.get(node.get("class_type")) if isinstance(node, dict) else None
            if cls is not None and getattr(cls, "HEAVY_OUTPUTS", ()) and isinstance(node.get("inputs", {}), dict):
                heavy[node_id] = _heavy_indices(cls)
        if not heavy:
            return json_data
        after = self.packs_after()
        stamps = None if after else {node_id: ",".join(sorted(names)) for node_id, names in _links(prompt, heavy).items()}
        # everything is worked out above; only now is the prompt written
        for node_id in heavy:
            inputs = prompt[node_id].setdefault("inputs", {})
            if stamps is None:
                inputs.pop(STAMP, None)
            else:
                inputs[STAMP] = stamps[node_id]
        if after:
            message = (f"RAM saving of unused outputs is off for this run: {', '.join(after)} "
                       f"{'changes' if len(after) == 1 else 'change'} the prompt after it.")
            log.warning(message)
            self.server.send_sync(EVENT, {"message": message}, json_data.get("client_id"))
        return json_data

    def packs_after(self):
        """The packs whose on_prompt handlers run after this one, the stamping handlers left out."""
        handlers = list(getattr(self.server, "on_prompt_handlers", ()))
        index = next((i for i, handler in enumerate(handlers) if handler is self), None)
        if index is None:
            return []
        return sorted({_pack_of(handler) for handler in handlers[index + 1:] if not getattr(handler, "bc_link_stamp", False)})


def register_link_stamp(classes):
    """Adds a LinkStamp for `classes` to ComfyUI's on_prompt handlers and returns it; None outside
    ComfyUI."""
    server = prompt_server()
    if server is None:
        log.info("no ComfyUI server: unused heavy outputs are returned in full")
        return None
    handler = LinkStamp(classes, server)
    server.add_on_prompt_handler(handler)
    return handler


def heavy_wanted(cls, prompt, unique_id, wanted=None):
    """The HEAVY_OUTPUTS of `cls` that some node links, read from the stamp of node `unique_id` in
    the final `prompt`, plus any link the stamp missed (with a warning). None without a stamp:
    every output is wanted. `wanted`, when given (by a wrapper), is the answer as it is."""
    if wanted is not None:
        return wanted
    node = prompt.get(unique_id) if isinstance(prompt, dict) else None
    inputs = node.get("inputs") if isinstance(node, dict) else None
    stamp = inputs.get(STAMP) if isinstance(inputs, dict) else None
    if not isinstance(stamp, str):
        return None
    wanted = {name for name in stamp.split(",") if name}
    missed = _links(prompt, {unique_id: _heavy_indices(cls)})[unique_id] - wanted
    if missed:
        log.warning(f"{cls.__name__} (node {unique_id}): the prompt links {', '.join(sorted(missed))}, which its link stamp "
                    f"does not list; returned in full")
        wanted |= missed
    unlinked = [name for name in cls.HEAVY_OUTPUTS if name not in wanted]
    if unlinked:
        log.info(f"{cls.__name__} (node {unique_id}): {', '.join(unlinked)} not linked, returned empty")
    return wanted


def wants(wanted, name):
    """Whether heavy output `name` is to be computed (`wanted` from heavy_wanted)."""
    return wanted is None or name in wanted


def drop_unwanted(cls, outputs, wanted):
    """`outputs` as a tuple, each heavy output not in `wanted` a new 0-frame tensor of its dtype,
    device and trailing shape (IMAGE [0, H, W, C], MASK [0, H, W]). A new tensor, never a view, so
    it keeps no storage alive. `wanted` None: unchanged."""
    import torch

    outputs = tuple(outputs)
    if wanted is None:
        return outputs
    outputs = list(outputs)
    for index, name in _heavy_indices(cls).items():
        if name not in wanted and isinstance(outputs[index], torch.Tensor):
            outputs[index] = outputs[index].new_empty((0, *outputs[index].shape[1:]))
    return tuple(outputs)


def drop_unlinked_heavy(cls, outputs, prompt, unique_id):
    """drop_unwanted with what the stamp of node `unique_id` says: for a node that computes every
    output and drops the unlinked ones at return."""
    return drop_unwanted(cls, outputs, heavy_wanted(cls, prompt, unique_id))
