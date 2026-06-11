# triton_ascend

Format: a single Python file with `class ModelNew(torch.nn.Module)`.
The `forward` method launches one or more Triton kernels on Ascend NPU.

Run from `akg_agents/workspace_autoresearch` after setting
`defaults.dsl: triton_ascend`, `defaults.backend: ascend`,
`defaults.framework: torch`:

```text
/autoresearch --ref ar_examples/triton_ascend/reference.py --kernel ar_examples/triton_ascend/vector_add_kernel.py --op-name vector_add --devices 0
```
