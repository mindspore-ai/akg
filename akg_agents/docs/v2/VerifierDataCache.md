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

The baseline pre-profile path used by `evolve` and `adaptive_search` also reads and writes the same cache. Concurrent warmups for the same baseline are serialized with a cache lock, and waiters re-check the cache before measuring.

## Cache layout

```text
~/.akg/verifier_data_cache/
├── reference/
└── baseline/
```

## Scope

Verifier Data Cache currently covers the `KernelBench` verifier path only. This does not affect the main `SOL-ExecBench` verification/profile path: SOL inputs are normalized separately by the verifier SOL adapter into `definition.json`, `workload.jsonl`, and `reference.py`. SOL data is not cached by Verifier Data Cache because SOL already ships stable reference files, so the immediate pain point is the KernelBench verifier path.

Reference data cache currently covers static-shape verification. Dynamic-shape tasks (`get_inputs_dyn_list`) skip reference data cache and continue using live input generation. Baseline cache still keys by framework code, backend, arch, DSL, and profile parameters.

Cache keys include `task_id` by default to avoid accidental reuse across independent tasks. Set `data_cache.cache_key_id` when a workflow needs multiple verifier task ids to share the same cached reference data and baseline. `adaptive_search`, `evolve`, and `AutoResearch` set a stable `cache_key_id` automatically for one operator workflow. Reusing data for a demo or repeated validation requires the same `task_id` or the same `cache_key_id`.

Configuration:

```yaml
data_cache:
  enabled: true
  cache_reference_data: true
  cache_baseline_result: true
  # optional stable identity for sharing cache across verifier task ids
  cache_key_id: "relu:torch:triton_ascend:ascend:ascend910b4:kernelbench"
```

The default cache directory is expanded by code to `~/.akg/verifier_data_cache`. Set `data_cache.cache_dir` or `AKG_AGENTS_VERIFY_DATA_CACHE_DIR` to override it.

## Demo

See:

`examples/kernel_related/run_torch_npu_triton_single_with_cache.py`

The demo uses a dedicated cache directory (`~/.akg/verifier_data_cache_demo`) and keeps existing cache entries by default, so it can show reuse across separate invocations. Use `--clear-cache` when you want a deterministic cold-cache first run:

```bash
python akg_agents/examples/kernel_related/run_torch_npu_triton_single_with_cache.py --clear-cache
```

With an empty cache:

1. run 1: miss reference/baseline cache, generate reference data, run base + generation profile, then populate cache
2. run 2: hit reference/baseline cache, reuse cached inputs/outputs, and skip base profile

Expected log markers:

- `Verifier Data Cache 未命中：reference data`
- `reference data 已写入 Verifier Data Cache`
- `Verifier Data Cache 命中：reference data`
- `Verifier Data Cache 命中：baseline=... us`
- `跳过 base profile`

## Acceptance

Run the focused verifier cache tests:

```bash
pytest -q akg_agents/tests/op/ut/test_verifier_data_cache.py
```

Run the Triton Ascend end-to-end cache demo on an Ascend environment with `torch_npu` and Triton Ascend installed:

```bash
python akg_agents/examples/kernel_related/run_torch_npu_triton_single_with_cache.py
```

For a deterministic cold-cache demonstration:

```bash
python akg_agents/examples/kernel_related/run_torch_npu_triton_single_with_cache.py --clear-cache
```

Run a whitespace sanity check before submitting:

```bash
git diff --check
```

The demo is intentionally small (`relu`) so review can focus on the Verifier Adapter data path rather than LLM generation quality. It still exercises the actual `KernelVerifier.run()` and `KernelVerifier.run_profile()` paths for `dsl=triton_ascend`, `backend=ascend`.
