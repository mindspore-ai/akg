---
name: ascendc-precision-debug
description: "AscendC direct-invoke 精度失败修复索引：输出全 0/随机值、DataCopy 对齐、EnQue/DeQue 同步、FP16/FP32 精度、Cast RoundMode、DumpTensor 分段定位。"
---

# AscendC Precision Debug

Use this when WA reports correctness failure for `dsl=ascendc`.

## First Moves

1. Fix one failing shape first. Do not tune all shapes at once.
2. Prefer a small 32-byte-aligned case, then the failing tail/non-aligned case.
3. Clear stale build output if behavior does not change after editing:

```bash
rm -rf ascendc_op/build
```

In WA, the adapter also cleans per-iteration build directories, but stale local
manual runs can still confuse inspection.

## Symptom Table

| Symptom | Likely Cause | Check |
|---|---|---|
| output all zeros | output never written, wrong queue, stale binary | `CopyOut`, VECOUT queue, rebuild |
| random output | missing DMA sync or uninitialized UB | `EnQue/DeQue`, `PipeBarrier`, buffer init |
| small shapes fail | non-32B aligned copy | use `DataCopyPad` or explicit tail handling |
| tail case fails | tile/tail math mismatch | `tailNum`, last-core length, mask |
| FP32 passes, FP16/BF16 fails | half precision intermediate | cast to FP32 for sensitive math |
| Cast output wrong | wrong round mode | half->float `CAST_NONE`; float->half `CAST_ROUND` |
| BF16 passes but FP16 fails | range/precision or API fallback difference | inspect dtype branches and thresholds |
| code change has no effect | `.so` not rebuilt or wrong `.so` loaded | remove `build/`, make `_load()` choose correct library |

## Pipeline Checks

DMA is asynchronous. Allocating a `LocalTensor` does not mean the data copy has
finished.

For queue-based MemBase code, the common shape is:

```cpp
AscendC::DataCopy(local, gm, len);
queue.EnQue(local);

auto ready = queue.DeQue<T>();
// compute on ready
queue.FreeTensor(ready);
```

If code uses `DataCopy` without `EnQue/DeQue`, add the proper queue handoff or
a targeted `PipeBarrier<PIPE_*>` when queue structure is not appropriate.

## DataCopy And Tail

- Use `DataCopyPad` for non-aligned GM/UB movement.
- Keep tile length in elements, byte length in bytes; do not mix the two.
- Last core length may be zero when block count is overestimated; guard it.
- Match `blockDim` to the tiling struct and kernel's `GetBlockIdx()` logic.

## Numeric Strategy

- For reductions and exp/log/div chains, accumulate or compute sensitive
  intermediates in FP32.
- For softmax-like operators, subtract max before exp.
- Use epsilon for divide/sqrt paths that can hit zero.
- Do not loosen WA tolerances from inside the kernel. Fix the implementation or
  the reference task config.

## DumpTensor / Printf Path

When ordinary inspection fails, insert temporary instrumentation at the stage
boundaries:

- after `CopyIn`
- after `Compute`
- before/after `CopyOut`

Use unique tags or desc IDs per stage, then compare against a CPU/PyTorch golden
for the same failing shape. Remove instrumentation before final submission.

## WA-Specific Reminders

- `framework_output` is produced by reference.py; match shape and dtype exactly.
- `ModelNew.forward()` should return a tensor or tuple/list of tensors, not a
  debug object.
- If the failure report mentions missing `.so`, fix build/load first; do not
  change math.
