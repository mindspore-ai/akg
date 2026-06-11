---
name: ascendc-registry-to-direct-invoke
description: "把注册式 AscendC 算子迁移为 direct-invoke 工程的保真原则：kernel 算法和 tiling 公式不乱改，只替换注册框架胶水、入口 ABI、host launch 与 PyTorch extension。"
---

# Registry Invoke To Direct Invoke

Use this when an existing AscendC custom-operator or registry-invoke project
must be converted into the WA `kernel.py + ascendc_op/` direct-invoke layout.

## One Principle

Only replace the registry framework glue. Preserve the kernel algorithm and
tiling math unless the user explicitly asks for an algorithmic rewrite.

## Keep Kernel Logic Intact

Do not "simplify" these while porting:

- kernel class layout, member variables, and member functions
- `CopyIn`, `Compute`, `CopyOut`, `Process` ordering
- helper functions used by compute or data movement
- loop structure, branch structure, and arithmetic order
- dtype-specialized paths and tiling-key-specialized paths

Allowed changes are mechanical integration changes:

- replace cross-directory includes with local includes or a small local deps header
- wrap names in a namespace only when needed to avoid collisions
- replace framework macro-generated tiling types with equivalent POD structs
- change the kernel entry ABI to match direct launch

## Keep Tiling Math Intact

Preserve all formulas and branches:

- core count, tile length, base shape, alignment, and tail calculations
- aligned vs non-aligned branches
- dtype, quant, grad, full-load, split-D, and other variant branches
- numerator/denominator order and ceil/floor behavior

Never merge "similar" branches just to reduce code. Tiling drift can compile
cleanly and silently corrupt results.

## Replace The Glue

The required direct-invoke glue is:

- kernel entry: direct `__global__` launcher with explicit arguments
- tiling data: ordinary POD struct shared by host and kernel
- host launch: compute tiling, allocate output, pass stream, launch kernel
- PyTorch extension: register `torch.ops.npu.<op>` and implement Meta function
- `kernel.py`: lazy-load `.so` and call the registered op from `ModelNew.forward()`

Registry macros such as these do not belong in direct-invoke WA seeds:

- `IMPL_OP_OPTILING`
- `REGISTER_TILING_DATA_CLASS`
- runtime `GET_TILING_DATA` dispatch paths that depend on registry glue
- framework-specific `OP_LOG*` macros that are not available in the direct project

Replace logging with `throw std::runtime_error(...)`, `printf`, or Python-side
exceptions as appropriate.

## Dependency Closure

For current-operator source files, copy whole files. For external shared
helpers, copy the minimal symbol closure:

- SDK includes stay as SDK includes
- tiny constants and traits may be localized
- helper functions must be copied with their direct dependencies
- do not drag a large unrelated common header tree into `ascendc_op/`

After migration, `rg '#include "\\.\\.' ascendc_op` should be empty unless the
relative include stays inside the copied project tree by design.

## Host/Device Pass Guard

AscendC `.asc` and `-xasc` style compilation may compile the same source in a
host pass and a device pass. If a kernel implementation header uses device-only
MicroAPI or RegBase types, guard the implementation body:

```cpp
#if !defined(__NPU_HOST__)
// device-only kernel implementation
#endif
```

Do not guard pure POD tiling headers that are required by both host and device.

## Verification Checklist

- The launch ABI matches on host declaration, generated launcher, and kernel entry.
- All tensor pointers are passed consistently as `GM_ADDR` / `uint8_t*` according to the launcher.
- Tiling struct field names, types, order, and size match host and kernel expectations.
- The PyTorch Meta implementation returns correct shape and dtype.
- `kernel.py` does not compile, mutate source, or choose a device at import time.
- WA `--kernel` points to `ascendc_op/`, with sibling `kernel.py`.
