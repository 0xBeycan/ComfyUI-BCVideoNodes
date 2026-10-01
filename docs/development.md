# Development

## Tests

```
python -m pytest tests
```

`tests/libs/test_chunking.py` covers the length math and
`tests/test_package.py` the node contract of all twenty-one nodes, both without
torch or ComfyUI:
`python -m pytest tests/test_package.py tests/libs/test_chunking.py`.
`tests/pipelines/test_long_video*.py` run the samplers' chunk loop against
stubbed core nodes with real CPU tensors; it is skipped when torch is not
installed. The preprocess tests (`tests/nodes/test_nodes_*.py`,
`tests/pipelines/test_pose.py`, `tests/pipelines/test_sam3_1_multiplex_*.py`,
`tests/pipelines/test_guard*.py`, `tests/models/test_models_*.py`, ...) need
torch and, for most, ComfyUI on the path:
`PYTHONPATH=/path/to/ComfyUI python -m pytest tests`. The video node tests
(`tests/libs/test_video_*.py`, `tests/libs/test_resize.py`,
`tests/pipelines/test_video_input.py`, `tests/nodes/test_video_*.py`) write
small synthetic clips with PyAV; none reads a real clip.
