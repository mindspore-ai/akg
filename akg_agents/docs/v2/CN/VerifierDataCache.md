[English Version](../VerifierDataCache.md)

# Verifier Data Cache 设计

## 1. 背景

在实际算子验证中，`Verifier` 的两段开销会被反复支付：

1. `task_desc -> init_inputs / inputs / outputs` 的数据生成
2. baseline profile（框架实现性能测量）

对复杂算子来说，这两段通常比单次 impl 校验更慢；在 `adaptive_search`、`evolve`、重复回归测试这类场景中，同一个 task 会被多次验证，重复生成数据和重复测 baseline 的收益很低。

## 2. 目标

- 在 `Verifier` 侧提供一个**默认关闭、显式开启**的本地持久缓存
- 命中时直接复用：
  - reference data（`inputs` / `init_inputs` / `outputs`）
  - baseline profile result（`avg_time_us`）
- 尽量复用现有能力，不引入新的验证数据格式

## 3. 非目标

- 不改 `SOL-ExecBench` 主流程
- 不改 Worker 协议
- 不做跨任务共享的复杂索引服务，只做本地文件缓存

## 4. 设计选择

### 4.1 reference data 复用现有 `.pt` 格式

仓库里已经有两条现成能力：

- `KernelVerifier.generate_reference_data(save_inputs=True)`
- `gen_verify_project()` 中的 `use_reference_data / use_reference_inputs`

因此本设计不新增数据格式，直接复用现有 `.pt`：

- `outputs`: 作为 ground truth
- `inputs`: 命中后直接复用，避免再次跑 `get_inputs()`
- `init_inputs`: 命中后直接复用，避免再次跑 `get_init_inputs()`

命中 reference data 后，验证脚本走 `reference_data + reference_inputs` 分支，只执行 impl，不再重复执行 framework baseline。

### 4.2 baseline 复用 `avg_time_us`

baseline cache 直接保存 `base_profile_result.json` 的核心信息：

- `avg_time_us`
- `execution_time_us`
- `warmup_times`
- `run_times`

命中后由 `KernelVerifier.run_profile()` 注入：

- `override_base_time_us`
- `skip_base_profile=True`

这样现有 `LocalWorker.profile()` 和 `profiler_utils.run_profile_scripts_and_collect_results()` 无需改协议。

### 4.3 缓存目录

默认目录：

```text
~/.akg/verifier_data_cache/
├── reference/
│   ├── {op_name}_{hash}.pt
│   └── {op_name}_{hash}.json
└── baseline/
    └── {op_name}_{hash}.json
```

其中：

- `reference/*.pt` 保存 reference data 二进制
- `reference/*.json` 保存元信息
- `baseline/*.json` 保存 baseline 结果

### 4.4 cache key

reference data key：

- `op_name`
- `framework`
- `backend`
- `arch`
- `bench_type`
- `task_id`
- `framework_code` 的 AST 归一化结果

baseline key：

- 上述字段
- `dsl`
- `warmup_times`
- `run_times`

这样可以避免：

- 不同 task_desc 相互污染
- 不同 task_id 相互污染
- 不同后端/架构相互污染
- 不同 profile 参数误复用 baseline

## 5. 命中/未命中流程

### 5.1 verify

1. 读取 `data_cache.enabled`
2. 先查 reference cache
3. 命中：将缓存内容注入 `use_reference_data/use_reference_inputs/reference_data`
4. 命中后会先校验 `.pt` 内容，损坏或缺少 `inputs/outputs` 时删除旧缓存并重新生成
5. 未命中：调用 `generate_reference_data(save_inputs=True)` 生成并落盘
6. 继续走现有 `gen_verify_project -> run_verify`

动态 shape（`get_inputs_dyn_list`）当前不走 reference data cache，避免错误复用单组静态输入；baseline cache 仍按 profile 参数和代码 key 生效。

### 5.2 profile

1. 读取 `data_cache.enabled`
2. 先查 baseline cache
3. 命中：自动设置 `override_base_time_us + skip_base_profile`
4. 未命中：正常执行 base + generation profile
5. 成功后把 `base_profile_result.json` 写入 cache

### 5.3 baseline_profiler

`evolve` / `adaptive_search` 在进入主循环前会预先 profile baseline。现在这里也先查同一份 baseline cache，命中后直接返回，不再重复测量。

## 6. 配置方式

```yaml
data_cache:
  enabled: true
  cache_reference_data: true
  cache_baseline_result: true
```

默认缓存目录由代码统一展开为 `~/.akg/verifier_data_cache`。如需改到其他目录，可显式配置：

```yaml
data_cache:
  enabled: true
  cache_dir: "/path/to/verifier_data_cache"
```

也支持环境变量：

- `AKG_AGENTS_VERIFY_DATA_CACHE=1`
- `AKG_AGENTS_VERIFY_DATA_CACHE_DIR=/path/to/cache`

## 7. 当前范围

当前实现只覆盖 `KernelBench` 验证链路。原因：

- `SOL` 已经自带 `reference.py` 和 `workload.jsonl`
- 本次痛点主要是 `task_desc -> inputs/outputs` 和 framework baseline profile

后续如果需要扩到 `SOL`，可以只复用 baseline cache，不必引入 reference data cache。

当前 reference data cache 只覆盖静态 shape 验证。动态 shape 场景会自动跳过 reference data cache，继续走实时输入生成与验证流程。

## 8. Triton Ascend 示例

仓库已提供一个不依赖 LLM 的 demo：

`examples/kernel_related/run_torch_npu_triton_single_with_cache.py`

它会：

1. 使用 `relu_torch.py + relu_triton_ascend_torch.py`
2. 第一次运行：生成并写入 reference/baseline cache
3. 第二次运行：直接命中本地 cache

适合做任务验收和流程演示。
