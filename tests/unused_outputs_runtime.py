"""The ComfyUI half of tests/nodes/test_unused_outputs_runtime.py, run in a process of its own:
`python -m pytest` puts the repo root on sys.path, where the pack's nodes/ hides ComfyUI's
nodes.py, which ComfyUI's executor imports.

ComfyUI's PromptServer is created first, then the pack is bound as `bcvideonodes` from its
__init__, as ComfyUI loads a custom node: its root __init__ registers the link stamp on that
server. Each scenario then runs prompts the way POST /prompt does (the on_prompt handlers,
validate_prompt, PromptExecutor) and records what happened; the facts are printed as one JSON
line, which the test reads.

The node is SCAIL-2 Colored Mask (pose_video_mask heavy, reference_image_mask one frame), fed by a
synthetic mask source; an output node records the shape it got.

    PYTHONPATH=/path/to/ComfyUI python tests/unused_outputs_runtime.py
"""
import asyncio
import copy
import importlib.util
import json
import logging
import os
import sys
import uuid

TESTS = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(TESTS)
COLORED = "BCVSCAIL2ColoredMask"
FRAMES, H, W = 4, 16, 8


class MaskSource:
    """A driving MASK [FRAMES, H, W] and a reference MASK [1, H, W]."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"seed": ("INT", {"default": 0})}}

    RETURN_TYPES = ("MASK", "MASK")
    FUNCTION = "run"
    CATEGORY = "test"

    def run(self, seed):
        import torch

        generator = torch.Generator().manual_seed(seed)
        return ((torch.rand(FRAMES, H, W, generator=generator) > 0.5).float(),
                (torch.rand(1, H, W, generator=generator) > 0.5).float())


class Shape:
    """An output node: the shape of the IMAGE it got."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "test"

    def run(self, image):
        return {"ui": {"shape": [list(image.shape)]}}


class ExecutorServer:
    """The executor's server: records what it is sent."""
    client_id = None
    last_node_id = None
    sockets_metadata = {}

    def __init__(self):
        self.events = []

    def send_sync(self, event, data, sid=None):
        self.events.append((event, data))

    def queue_updated(self):
        pass


class Lines(logging.Handler):
    """The pack's console lines."""

    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def N(class_type, **inputs):
    return {"class_type": class_type, "inputs": inputs}


# reference_image_mask linked; pose_video_mask (output 0) linked only in A_AND_B
A_ONLY = {"1": N("MaskSource", seed=0),
          "2": N(COLORED, driving_mask=["1", 0], replacement_mode=False, reference_mask=["1", 1]),
          "3": N("Shape", image=["2", 1])}
A_AND_B = {**A_ONLY, "4": N("Shape", image=["2", 0])}


def another_pack(json_data):
    """An on_prompt handler of another pack."""
    return json_data


class BCNodesStamp:
    """ComfyUI-BCNodes' stamping handler, marked as it is there."""
    bc_link_stamp = True

    def __call__(self, json_data):
        return json_data


class Harness:
    def __init__(self, server, execution, unused):
        self.server, self.execution, self.unused = server, execution, unused
        self.toasts = []
        server.send_sync = lambda event, data, sid=None: self.toasts.append([event, data, sid])
        self.lines = Lines()
        logging.getLogger("BCVideoNodes").addHandler(self.lines)

    def executor(self, kind="CLASSIC"):
        args = {"lru": 10 if kind == "LRU" else 0, "ram": 0, "ram_inactive": 0}
        if kind == "RAM_PRESSURE":
            import comfy.model_management as mm

            args["ram_inactive"] = min(128.0, mm.total_ram / 1024.0)
        return self.execution.PromptExecutor(ExecutorServer(), cache_type=getattr(self.execution.CacheType, kind),
                                             cache_args=args)

    async def submit(self, ex, prompt, via_server=True, edit_after=None):
        """POST /prompt's path: the on_prompt handlers (unless `via_server` is False), then
        `edit_after` (a change to the prompt after them), validate, execute. -> the facts."""
        json_data = {"prompt": copy.deepcopy(prompt), "client_id": "test"}
        if via_server:
            json_data = self.server.trigger_on_prompt(json_data)
        if edit_after:
            edit_after(json_data["prompt"])
        run = json_data["prompt"]
        prompt_id = str(uuid.uuid4())
        valid = await self.execution.validate_prompt(prompt_id, run, None)
        if not valid[0]:
            raise RuntimeError(f"invalid prompt: {valid[1]}")
        ex.server.events.clear()
        await ex.execute_async(run, prompt_id, {"client_id": "test"}, valid[2])
        if not ex.success:
            raise RuntimeError(f"execution failed: {[m for m in ex.status_messages if m[0] == 'execution_error']}")
        entry = ex.caches.outputs.get_local("2")
        seen = ex.history_result["outputs"].get("4", {}).get("shape", [None])[0]
        return {"stamp": run["2"]["inputs"].get(self.unused.STAMP),
                "runs": sum(1 for event, data in ex.server.events if event == "executing" and data.get("node") == "2"),
                "seen": seen,
                "cached": "evicted" if entry is None else list(entry.outputs[0][0].shape)}

    async def scenario(self, prompts, kind="CLASSIC", after=None, **options):
        """`prompts` submitted in turn to one executor, with `after` registered after the pack's
        handler meanwhile. -> {"runs": [facts...], "lines": warnings, "toasts": [...]}"""
        del self.lines.lines[:], self.toasts[:]
        if after is not None:
            self.server.add_on_prompt_handler(after)
        try:
            ex = self.executor(kind)
            runs = [await self.submit(ex, prompt, **options) for prompt in prompts]
        finally:
            if after is not None:
                self.server.on_prompt_handlers.remove(after)
        return {"runs": runs, "lines": list(self.lines.lines), "toasts": list(self.toasts)}


async def main():
    try:
        import av  # noqa: F401  (first, as ComfyUI imports it; see tests/conftest.py)
    except ImportError:
        pass
    import execution
    import nodes
    import server
    from app.assets.manager import default_asset_manager

    instance = server.PromptServer(asyncio.get_event_loop(), default_asset_manager())
    # the pack loaded as ComfyUI loads a custom node: its root __init__ registers the stamp
    spec = importlib.util.spec_from_file_location("bcvideonodes", os.path.join(PACK, "__init__.py"),
                                                  submodule_search_locations=[PACK])
    package = importlib.util.module_from_spec(spec)
    sys.modules["bcvideonodes"] = package
    spec.loader.exec_module(package)
    sys.path.insert(0, TESTS)
    from unused_outputs_fakes import unused

    nodes.NODE_CLASS_MAPPINGS.update({"MaskSource": MaskSource, "Shape": Shape, COLORED: package.NODE_CLASS_MAPPINGS[COLORED]})
    h = Harness(instance, execution, unused)
    facts = {"handlers": [type(handler).__name__ for handler in instance.on_prompt_handlers]}
    for kind in ("CLASSIC", "LRU", "RAM_PRESSURE"):
        # P1 unlinked, P2 linked later, P3 the same again, P4 unlinked again, P5 the same again
        facts[f"p1_p5_{kind}"] = await h.scenario([A_ONLY, A_AND_B, A_AND_B, A_ONLY, A_ONLY], kind)
    facts["no_stamp"] = await h.scenario([A_ONLY], via_server=False)
    stale = copy.deepcopy(A_ONLY)
    stale["2"]["inputs"][unused.STAMP] = "pose_video_mask"
    facts["stale_stamp"] = await h.scenario([stale])
    facts["missed_link"] = await h.scenario([A_ONLY], edit_after=lambda prompt: prompt.update({"4": N("Shape", image=["2", 0])}))
    facts["another_pack_after"] = await h.scenario([A_ONLY], after=another_pack)
    facts["bcnodes_after"] = await h.scenario([A_ONLY], after=BCNodesStamp())
    return facts


if __name__ == "__main__":
    print(json.dumps(asyncio.run(main())))
