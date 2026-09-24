"""Wan 2.2 Animate: the animate adapter of the core node WanAnimateToVideo."""
from ..common import registry
from .adapter import WanAnimateAdapter

registry.register("animate", WanAnimateAdapter.ANIMATE_NODE, WanAnimateAdapter)
