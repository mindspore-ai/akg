# pypto

Format: a single Python file with `class ModelNew(torch.nn.Module)`.
`ModelNew.forward` can build and call a `@pypto.frontend.jit` kernel.

Run from `akg_agents/workspace_autoresearch` after setting
`defaults.dsl: pypto`, `defaults.backend: ascend`,
`defaults.framework: torch`:

```text
/autoresearch --ref ar_examples/pypto/reference.py --kernel ar_examples/pypto/vector_add_kernel.py --op-name vector_add --devices 0
```
