[中文版](./CN/VerifierDataCache.md)

# Verifier Data Cache

## Overview

Verifier-side data cache avoids paying the same verification setup cost repeatedly for the same KernelBench task:

- reference data generation: `init_inputs / inputs / outputs`
- baseline profiling: framework latency measurement

The cache is persistent, local-only, and disabled by default.

## What is cached

### Reference data

The implementation reuses the existing `generate_reference_data(save_inputs=True)` output format:

- `outputs`
- `inputs`
- `init_inputs`

On a cache hit, verification uses `use_reference_data + use_reference_inputs`, so the verifier can skip rerunning the framework baseline path.

Cached `.pt` files are validated before reuse. If the payload is unreadable or misses reusable `inputs/outputs`, the verifier removes the stale entry and regenerates reference data.

### Baseline result

The implementation persists the normalized content of `base_profile_result.json`, mainly `avg_time_us`.

On a cache hit, `run_profile()` injects:

- `override_base_time_us`
- `skip_base_profile=True`

## Cache layout

```text
~/.akg/verifier_data_cache/
├── reference/
└── baseline/
```

## Scope

Current scope is `KernelBench` only. `SOL-ExecBench` already ships stable reference files, so the immediate pain point is the KernelBench verifier path.

Reference data cache currently covers static-shape verification. Dynamic-shape tasks (`get_inputs_dyn_list`) skip reference data cache and continue using live input generation. Baseline cache still keys by framework code, backend, arch, DSL, and profile parameters.

## Demo

See:

`examples/kernel_related/run_torch_npu_triton_single_with_cache.py`
