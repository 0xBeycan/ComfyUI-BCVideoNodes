"""The animate family of the model registry: the two Wan Animate adapters, registered by their
packages in models/__init__.py's order, each under the core node id it adapts (its ANIMATE_NODE).
Skipped when torch is not installed (importing models/ imports the pose nets).
"""

import pytest

pytest.importorskip("torch")

from bcvideonodes.models.common import registry  # noqa: E402
from bcvideonodes.models.wan_animate.adapter import WanAnimateAdapter  # noqa: E402
from bcvideonodes.models.wan_animate2.adapter import WanAnimate2Adapter  # noqa: E402


def test_animate_family_in_registration_order():
    assert registry.names("animate") == ["WanAnimateToVideo", "WanAnimate2ToVideo"]


def test_animate_entries_are_the_adapters_under_their_core_node_ids():
    assert registry.get("animate", "WanAnimateToVideo") == registry.Entry(WanAnimateAdapter, None)
    assert registry.get("animate", "WanAnimate2ToVideo") == registry.Entry(WanAnimate2Adapter, None)
    for name in registry.names("animate"):
        assert registry.get("animate", name).implementation.ANIMATE_NODE == name
