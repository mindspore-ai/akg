# ascendc

Format: a direct-invoke AscendC project. `--kernel` points at the
`ascendc_op/` directory; its parent directory must contain `kernel.py`.

Required layout:

```text
add_custom/
  reference.py
  kernel.py
  ascendc_op/
    CMakeLists.txt
    op_kernel/
    op_extension/
```

`add_custom` is a CANNBot direct-invoke example targeting
Ascend910B/dav-2201. It builds a PyTorch extension shared library and loads it
from the sibling `kernel.py` wrapper.

Run from `akg_agents/workspace_autoresearch` after setting
`defaults.dsl: ascendc`, `defaults.backend: ascend`,
`defaults.framework: torch`:

```text
/autoresearch --ref ar_examples/ascendc/add_custom/reference.py --kernel ar_examples/ascendc/add_custom/ascendc_op --op-name add_custom --devices 0
```

The AscendC adapter copies the whole `ascendc_op/` project, rebuilds it, and
imports sibling `kernel.py` as the `ModelNew` entry.
