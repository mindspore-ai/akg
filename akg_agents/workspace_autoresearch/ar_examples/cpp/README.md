# cpp

Format: a single Python file with `class ModelNew(torch.nn.Module)`.
The model may compile a CPU C++ extension through `torch.utils.cpp_extension`.

Run from `akg_agents/workspace_autoresearch` after setting
`defaults.dsl: cpp`, `defaults.backend: cpu`, `defaults.framework: torch`:

```text
/autoresearch --ref ar_examples/cpp/reference.py --kernel ar_examples/cpp/vector_add_kernel.py --op-name vector_add --devices 0
```
