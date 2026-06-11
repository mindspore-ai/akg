---
name: ascendc-crash-debug
description: "AscendC direct-invoke 崩溃/挂起修复索引：Kernel timeout、hang、Segmentation Fault、aic error、buffer 死锁、plog/memcheck 调试。"
---

# AscendC Crash Debug

Use this when WA reports timeout, crash, process abort, aic error, or no result
from a direct-invoke AscendC kernel.

## Triage

| Symptom | Likely Cause | First Check |
|---|---|---|
| CMake configure/build fails | environment, include/lib path, npu arch | `ASCEND_HOME_PATH`, torch_npu, `--npu-arch` |
| `.so` missing after build | target name/output path mismatch | CMake shared target and build logs |
| import/load fails | wrong library selected or missing registration | `torch.ops.load_library`, `TORCH_LIBRARY` namespace |
| timeout/hang | queue deadlock, bad sync, infinite loop | Alloc/Free and EnQue/DeQue pairing |
| aic error | GM/UB out-of-bounds, misalignment | DataCopy length, block/tail math |
| Segmentation Fault | host pointer misuse, null ptr, bad launch ABI | host bridge and kernel declaration |
| output never returns | cross-core flag mismatch | SetFlag/WaitFlag pairs and core participation |

## Build And Load

Before editing kernel math, prove the project is being rebuilt and loaded:

- remove `ascendc_op/build`
- check that CMake builds a `.so` under `build/`
- make `kernel.py` load the intended library by stable file name when possible
- ensure registration namespace matches the Python call

Example mismatch:

```python
torch.ops.npu.my_op(...)
```

requires a C++ registration such as:

```cpp
TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("my_op(Tensor x) -> Tensor");
}
```

## Launch ABI

Host declaration, generated launcher, and kernel entry must match.

Check:

- argument count and order
- pointer type conventions
- tiling pointer or tiling tensor type
- `blockDim`
- stream argument passed by the launcher wrapper

Avoid host-side `void*` to `GM_ADDR` reinterpret tricks that AscendC compilation
or runtime may reject. Keep the launcher ABI simple and consistent.

## Deadlock Checklist

- Every `AllocTensor` has a matching `FreeTensor`.
- Every `EnQue` has a matching `DeQue`.
- Producer and consumer queues use compatible `TPosition` values.
- Cross-core `WaitFlag` has a guaranteed `SetFlag` for every participating core.
- No branch lets some cores skip a synchronization that other cores wait for.
- Tail branches do not exit before freeing queue buffers.

## Out-Of-Bounds Checklist

- `SetGlobalBuffer` length matches each core's actual accessible span.
- `GetBlockIdx()` offset cannot exceed total elements.
- Last core uses tail length, not normal tile length.
- `DataCopy`/`DataCopyPad` length is in the expected unit.
- Output shape from Meta and Python wrapper matches actual writes.

## Logs And Tools

For local/manual debugging outside the WA loop:

- inspect plog for aic/aiv error locations
- use core dump or gdb for host-side segfaults
- use Ascend memory checking tools when crashes are nondeterministic
- add temporary `PRINTF` / `DumpTensor` only around suspected stages

Remove heavy instrumentation before returning to optimization mode.

## WA-Specific Recovery

- If `eval_timeout` kills the process, first reduce to one failing shape and
  run formal verify only.
- If crash happens before `ModelNew.forward()`, focus on import/build/load.
- If crash happens after launch, focus on tiling, launch ABI, and kernel memory.
- Do not mask crashes by catching all exceptions in `kernel.py`; WA needs the
  original error text.
