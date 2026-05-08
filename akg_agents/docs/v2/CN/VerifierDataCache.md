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

cache 命中和写入日志会带上本次使用的 `cache_file` 和 `cache_key`，方便直接定位、排查或清理对应缓存文件。

### 4.4 cache key

reference data key：

- `op_name`
- `framework`
- `backend`
- `arch`
- `bench_type`
- `cache_key_id`（未配置时回退到 `task_id`）
- `framework_code` 的 AST 归一化结果

baseline key：

- 上述字段
- `dsl`
- `warmup_times`
- `run_times`

这样可以避免：

- 不同 task_desc 相互污染
- 未显式设置 `cache_key_id` 时，不同 task_id 相互污染
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

多个进程或任务同时预热同一个 baseline 时，会通过 cache lock 串行化；等待者会在拿到锁后再次读取 cache，避免重复测同一份 baseline。

### 5.4 RemoteWorker

RemoteWorker 场景与 LocalWorker 使用同一套 Verifier Data Cache 语义：

1. cache 文件仍保存在发起验证的本机 `cache_dir` 中
2. 未命中时，Verifier 通过 RemoteWorker 在远端生成 reference data，并把返回的 `.pt` bytes 写入本机 cache
3. 命中时，Verifier 将本机 cache 中的 `.pt` 写入验证工程并打包发送给远端 Worker
4. baseline 命中时，Verifier 通过 `override_base_time_us + skip_base_profile` 通知远端跳过 base profile

因此远端 Worker 不需要额外实现一套持久 cache 服务；它只需要支持现有的 `generate_reference` / `verify` / `profile` / `profile_single_task` 接口。远端机器上的临时任务目录仍由 Worker 按原逻辑清理。

## 6. 配置方式

```yaml
data_cache:
  enabled: true
  cache_reference_data: true
  cache_baseline_result: true
  # 可选：用于让同一工作流内多个 task_id 复用同一份数据
  cache_key_id: "relu:torch:triton_ascend:ascend:ascend910b4:kernelbench"
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

当前 Verifier Data Cache 覆盖 `KernelBench` 的 reference data 和 baseline profile 路径。对于 `SOL-ExecBench`，本次只接入 baseline profile cache，不缓存 reference data，原因：

- `SOL` 已经自带稳定的 `definition.json` / `workload.jsonl` / `reference.py`
- 重复开销主要集中在 reference baseline profile
- 复用 SOL reference data 容易绕开 benchmark 自身的输入/容差描述，不作为当前适配范围

`SOL-ExecBench` baseline cache 的 key 会包含 bench type、framework/backend/arch/DSL、profile 参数，以及归一化后的 `definition.json` / `workload.jsonl` / `reference.py` 内容。命中后会给 profile 流程注入 `override_base_time_us` 并设置 `skip_base_profile=True`，从而跳过 SOL base profile；cache miss 时正常测量并写入 baseline cache。

当前 reference data cache 只覆盖 KernelBench 静态 shape 验证。动态 shape 场景会自动跳过 reference data cache，继续走实时输入生成与验证流程。

默认 cache key 包含 `task_id`，用于避免不同独立任务之间误复用。`adaptive_search`、`evolve` 和 `AutoResearch` 会自动设置稳定的 `data_cache.cache_key_id`，让同一算子的多个候选实现复用同一份 reference data / baseline cache。做 demo 或重复回归时，也可以保持 `task_id` 一致，或显式设置相同的 `cache_key_id`。

## 8. Triton Ascend 示例

仓库已提供一个不依赖 LLM 的 demo：

`examples/kernel_related/run_torch_npu_triton_single_with_cache.py`

它会：

1. 使用 `relu_torch.py + relu_triton_ascend_torch.py`
2. 第一次运行：生成并写入 reference/baseline cache
3. 第二次运行：直接命中本地 cache

示例使用独立缓存目录 `~/.akg/verifier_data_cache_demo`，默认保留已有 cache，便于演示跨进程复用。需要稳定看到“第一次填充、第二次复用”时，使用 `--clear-cache`：

```bash
python akg_agents/examples/kernel_related/run_torch_npu_triton_single_with_cache.py --clear-cache
```

预期日志关键字：

- `Verifier Data Cache 未命中：reference data`
- `reference data 已写入 Verifier Data Cache`
- `Verifier Data Cache 命中：reference data`
- `Verifier Data Cache 命中：baseline=... us`
- `跳过 base profile`

适合做任务验收和流程演示。

## 9. 验收建议

运行 Verifier Data Cache 定向单测：

```bash
pytest -q akg_agents/tests/op/ut/test_verifier_data_cache.py
```

在安装了 `torch_npu` 与 Triton Ascend 的 Ascend 环境运行端到端示例：

```bash
python akg_agents/examples/kernel_related/run_torch_npu_triton_single_with_cache.py
```

如果希望从空 cache 开始演示完整 miss/hit 流程：

```bash
python akg_agents/examples/kernel_related/run_torch_npu_triton_single_with_cache.py --clear-cache
```

提交前运行空白字符检查：

```bash
git diff --check
```

该示例刻意使用较小的 `relu` case，目的是验证 Verifier Adapter 的数据路径，而不是验证 LLM 生成质量。它会实际经过 `dsl=triton_ascend`、`backend=ascend` 下的 `KernelVerifier.run()` 与 `KernelVerifier.run_profile()` 流程。

## 10. 新增 bench format 接入指引

如果后续新增 `SOL` 之外的 bench format，建议按下面的顺序接入 Data Cache，避免每种格式各自实现一套不兼容逻辑。

### 10.1 先判断可缓存对象

对新 bench format 先拆分两类开销：

- reference data：是否存在“从任务描述生成 inputs / init_inputs / outputs”的步骤
- baseline profile：是否存在“对参考实现重复计时”的步骤

如果 bench format 已经自带固定 workload 和 reference 文件，通常不需要 reference data cache，只需要 baseline cache。

### 10.2 复用统一 cache helper

新增格式应优先复用 `op/verifier/data_cache.py`：

- `load_verifier_data_cache_config()` 读取开关和目录
- `build_reference_cache_key()` / `build_baseline_cache_key()` 生成 key
- `read_*_from_cache()` / `write_*_to_cache()` 读写文件
- `verifier_data_cache_lock()` 串行化同一 key 的写入

不要在 bench format 目录下重新定义独立 cache layout，避免清理、迁移和排查时出现多套规则。

### 10.3 cache key 字段要求

新增格式的 key 至少应覆盖：

- `op_name`
- `framework`
- `backend`
- `arch`
- `bench_type`
- `cache_key_id` 或 `task_id`
- 会影响输入/输出或 baseline 性能的任务描述内容
- baseline cache 还必须包含 `dsl`、`warmup_times`、`run_times`

如果 format 有额外影响数据生成的文件，例如 `definition.json`、`workload.jsonl`、shape 配置或 dtype 配置，应将其内容 hash 纳入 key。

### 10.4 Verifier 接入点

推荐接入点：

- reference data：在 `KernelVerifier.run()` 生成验证工程前准备并注入 `reference_data`
- baseline：在 `KernelVerifier.run_profile()` 生成 profile 工程前注入 `override_base_time_us`
- 预热 baseline：在 `baseline_profiler.profile_baseline_once()` 中先查 cache，成功后直接返回

新增 format 的生成逻辑可以放在独立模块中，但 cache 命中、失效、写入和加锁逻辑应保持统一。

### 10.5 测试要求

至少补充以下测试：

- 本地 Worker 首次 miss 后写 cache，第二次 hit 不再生成 reference data
- RemoteWorker 模拟场景下，cache hit 时仍会把 `.pt` 打包发送给远端验证
- baseline cache hit 时跳过 base profile
- cache 文件损坏或缺少必要字段时删除并重新生成
- key 中任一关键字段变化时不会误命中

如果 bench format 不支持 reference data cache，应在文档中明确说明“仅支持 baseline cache”。
