"""libs/tensor_bytes.py: the bytes a model's weights hold, each storage once."""
import pytest

from full_clear_fakes import full_clear

torch = pytest.importorskip("torch")


def test_module_bytes_counts_each_storage_once():
    a = torch.nn.Linear(10, 10)  # 100 weights + 10 biases, float32
    b = torch.nn.Linear(10, 10, bias=False)
    b.weight = a.weight  # tied: one storage in both
    meta = torch.nn.Linear(1000, 1000, device="meta")  # no memory
    assert full_clear.module_bytes(a) == 110 * 4
    assert full_clear.module_bytes(a, b, None, meta) == 110 * 4
    a.register_buffer("stats", torch.zeros(5, dtype=torch.float16))
    assert full_clear.module_bytes(a) == 110 * 4 + 10
