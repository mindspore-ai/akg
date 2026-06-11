# add_custom

This is a CANNBot-style direct-invoke AscendC example for Ascend910B/dav-2201.
It is intended to be used directly as an AutoResearch seed.

Layout:

```text
add_custom/
  reference.py
  kernel.py
  ascendc_op/
    CMakeLists.txt
    op_kernel/
    op_extension/
```

Run from `akg_agents/workspace_autoresearch`:

```text
/autoresearch --ref ar_examples/ascendc/add_custom/reference.py --kernel ar_examples/ascendc/add_custom/ascendc_op --op-name add_custom --devices 0
```

`--kernel` points at the `ascendc_op/` project directory. The AscendC adapter
rebuilds that project, then imports the sibling `kernel.py` wrapper as
`ModelNew`.
