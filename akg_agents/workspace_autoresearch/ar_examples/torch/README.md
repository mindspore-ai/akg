# torch

Format: a single Python file with `class ModelNew(torch.nn.Module)`.
This is useful as a baseline DSL or for CPU/GPU/NPU framework-only tests.

Run from `akg_agents/workspace_autoresearch` after setting
`defaults.dsl: torch`, `defaults.framework: torch`, and the desired backend:

```text
/autoresearch --ref ar_examples/torch/reference.py --kernel ar_examples/torch/vector_add_kernel.py --op-name vector_add --devices 0
```
