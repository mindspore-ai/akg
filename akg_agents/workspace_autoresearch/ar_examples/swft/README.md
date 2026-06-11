# swft

Format: a single Python file with `class ModelNew`.
The current adapter uses SWFT binary I/O:

- verifier writes `input*.bin` before calling `ModelNew.forward`;
- `forward` builds and executes the SWFT kernel;
- verifier reads `output*.bin` after `forward` returns.

Run from `akg_agents/workspace_autoresearch` after setting
`defaults.dsl: swft`, `defaults.backend: ascend`,
`defaults.framework: torch`:

```text
/autoresearch --ref ar_examples/swft/reference.py --kernel ar_examples/swft/vector_add_kernel.py --op-name vector_add --devices 0
```
