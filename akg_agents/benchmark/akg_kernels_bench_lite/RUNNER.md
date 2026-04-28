# Benchmark Lite Runner

本文档说明 `benchmark_lite` runner 的当前实现结构、重构背景、设计约束、模块职责和对外行为。

本文档与同目录下的 [README.md](./README.md) 分工如下：

| 文档 | 内容 |
|---|---|
| `README.md` | `benchmark_lite` 数据集内容 |
| `RUNNER.md` | runner 的实现、接口、输出和测试 |

## 1. 背景

重构前，`benchmark_lite` 相关逻辑分散在两个入口脚本中：

| 文件 | 重构前状态 |
|---|---|
| `akg_agents/examples/kernel_related/run_torch_bench_lite.py` | 多 backend 入口，同时包含 discovery、聚合、summary、JSON 输出 |
| `akg_agents/examples/kernel_related/gpu/run_torch_cuda_triton_bench_lite.py` | GPU 专用入口，同时复制了一套 discovery、聚合、summary、JSON 输出 |

原有实现的主要问题如下：

| 类别 | 问题 |
|---|---|
| 语义 | 文案包含 benchmark / performance 含义，实际输出主要是 `Pass@N` 成功率 |
| 结构 | 两个入口脚本都维护了一套 discovery / aggregation / summary 逻辑 |
| CLI | 文档示例包含 `--backend all`，解析器不支持 |
| 环境 | GPU 路径默认写死 `rtx3090`，默认设备假设为多卡环境 |
| 测试 | 缺少针对 CLI 契约、artifact 结构和聚合逻辑的专门测试 |

## 2. 重构目标

本次重构的目标限定如下：

| 编号 | 目标 |
|---|---|
| 1 | 将 runner 收敛为多模式评测 runner (correctness / performance / full) |
| 2 | 保持数据集 README 不变，不将 runner 说明混入数据集文档 |
| 3 | 将公共逻辑从入口脚本中抽离 |
| 4 | 统一单 backend 与多 backend 的结果结构 |
| 5 | 明确 `all` 模式和退出码语义 |

本次重构不包含以下内容：

| 编号 | 不包含内容 |
|---|---|
| 1 | baseline 性能回归门禁 |
| 2 | `benchmark_lite` case 数据集结构调整 |

## 3. 重构总结

### 3.1 代码组织变更

| 改动类型 | 说明 |
|---|---|
| 新增文件 | `akg_agents/examples/kernel_related/bench_lite_common.py` - 共享逻辑模块（~1500 行） |
| 新增文件 | `akg_agents/benchmark/akg_kernels_bench_lite/RUNNER.md` - 本文档 |
| 修改文件 | `akg_agents/examples/kernel_related/run_torch_bench_lite.py` - 主入口重构（515→610 行） |
| 修改文件 | `akg_agents/examples/kernel_related/gpu/run_torch_cuda_triton_bench_lite.py` - GPU wrapper 简化（540→55 行） |
| 修改文件 | `.gitignore` - 添加 `.claude/` 目录 |

### 3.2 `run_torch_bench_lite.py` 重构详情

**重构前（~515 行）：**
- 内联所有 backend 配置（`BACKEND_CONFIGS` 字典）
- 包含完整的 discovery 逻辑（`discover_bench_lite_cases`）
- 包含 case 读取和 CUDA 转换逻辑
- 包含结果聚合和 summary 计算逻辑
- 包含 JSON artifact 构建和输出逻辑
- 直接调用 `asyncio.run(run_bench_lite_tests(args))` 没有通过 `sys.exit`

**重构后（~610 行）：**
- 添加 shebang `#!/usr/bin/env python3`
- 更新 docstring：支持三种模式（correctness / performance / full）
- 移除所有内联逻辑，改为从 `bench_lite_common.py` 导入
- 新增 `main(argv)` 函数封装，通过 `sys.exit(asyncio.run(...))` 退出
- CLI 解析验证：`--pass-n >= 1`、设备 ID 非负且唯一、性能参数校验
- 退出码同时反映 correctness 和 performance 失败
- `--backend all` 聚合 correctness results 和 performance results 到顶层

### 3.3 `run_torch_cuda_triton_bench_lite.py` 重构详情

**重构前（~540 行）：**
- 完整复制主入口的所有 discovery/aggregation/summary 逻辑
- 包含独立的 `BACKEND_CONFIGS` 配置（仅 GPU 相关）
- 仅用于 GPU，但维护了完整的 runner 实现

**重构后（~55 行）：**
- 添加 shebang `#!/usr/bin/env python3`
- 更新 docstring：反映多模式支持
- 更新为简洁的 wrapper 脚本
- 通过 `sys.path.insert(0, ...)` 导入父目录的主入口
- `main(argv)` 函数转发所有参数到 `benchmark_main`
- 自动过滤并替换 `--backend` 参数为 `--backend gpu`（兼容 `--backend gpu` 和 `--backend=gpu` 两种形式）

### 3.4 `bench_lite_common.py` 新模块详情（~1500 行）

**核心类型定义：**
- `BackendConfig` (TypedDict) - backend 配置类型
- `BenchLiteCase` (dataclass) - 单个 case 元数据
- `DiscoveryResult` (dataclass) - discovery 结果数据

**常量：**
- `RUNNER_VERSION = "benchmark-lite-v2"`
- `DEFAULT_TIERS = ("t1", "t2", "t3")` - 仅作 fallback，正常情况动态发现
- `DEFAULT_TEAM_NAME = "aikg_agent"` - 默认 team 名称
- `BACKEND_CONFIGS` - 统一的 backend 配置字典（`arch`/`dsl`/`backend` 均可通过 CLI 覆盖）

**主要功能函数：**
- `validate_team_name()` - 校验 team_name 防止路径穿越（仅允许 `[A-Za-z0-9_-]`）
- `resolve_backend_config()` - 解析 backend 并允许 arch/dsl/backend 覆盖
- `resolve_devices()` - 解析设备列表（支持环境变量、torch 设备检测）
- `get_bench_lite_dir()` - 从 runner 位置反推数据集目录
- `discover_bench_lite_cases()` - 动态扫描所有 `t\d+` 目录、过滤、跳过 case
- `convert_case_source_for_backend()` - GPU/NPU backend 的 case 源码转换（`cpu()` → `cuda()`/`npu()`），跳过注释行
- `read_case_source()` - 读取 case 文件并做 backend 适配
- `build_task_specs()` - 为每个 Pass@N attempt 构造任务规范
- `aggregate_correctness_results()` - 按 case 聚合 raw task results
- `compute_summary()` - 生成单 backend summary（包含 tier 级统计）
- `build_environment_info()` - 构造 environment metadata
- `build_full_output_payload()` - 构造 JSON artifact payload（所有模式统一使用，唯一的 payload 构建函数）
- `combine_backend_payloads()` - 合并多 backend 的 summary 和 correctness results（`--backend all` 模式）
- `extract_submissions_from_results()` - 从 Agent 结果提取 submission（写入前清理旧目录，目录名使用 `team_name`）
- `write_leaderboard()` - 生成排行榜 JSON（按 tier+case 排序，包含失败原因，`team_name` 可参数化）
- `print_backend_summary()` - 打印 human-readable correctness summary
- `print_full_summary()` - 打印 correctness + performance 联合报告（标题随 mode 变化，使用 `.get()` 防御缺键）
- `_eval_single_case()` - 子进程隔离的单 case 性能评测（spawn 模式，超时后 terminate/kill）
- `_validate_devices()` - GPU/NPU 设备校验（检测可用设备数，过滤无效 ID）

## 4. 当前文件组织

| 文件 | 位置 | 角色 |
|---|---|---|
| 多 backend 主入口 | `akg_agents/examples/kernel_related/run_torch_bench_lite.py` | 主执行入口 |
| GPU 兼容入口 | `akg_agents/examples/kernel_related/gpu/run_torch_cuda_triton_bench_lite.py` | GPU 快捷 wrapper |
| 公共逻辑模块 | `akg_agents/examples/kernel_related/bench_lite_common.py` | discovery、聚合、artifact 等共享逻辑 |
| 数据集说明 | `akg_agents/benchmark/akg_kernels_bench_lite/README.md` | 数据集说明 |
| runner 说明 | `akg_agents/benchmark/akg_kernels_bench_lite/RUNNER.md` | runner 设计与实现说明 |

当前目录组织依据如下：

| 原则 | 说明 |
|---|---|
| 数据集归属 | `benchmark_lite` 数据集继续属于 `benchmark/` |
| 入口归属 | runner 属于执行入口，继续放在 `examples/kernel_related/` |
| 公共模块归属 | 共享逻辑当前仅服务于 runner，先放在 runner 同层目录 |

## 5. 运行模式

Runner 支持三种运行模式，通过 `--mode` 参数选择：

| 模式 | 说明 |
|---|---|
| `correctness` | 默认模式。仅评测 Pass@N 正确性 |
| `performance` | 在 correctness 基础上追加性能评测（speedup + 评分） |
| `full` | 完整评测流水线：correctness → 提交提取 → 性能评测 → 评分 → 排行榜 |

### 5.1 `full` 模式流程

```
Phase 1: Correctness (Pass@N)
  └─ Agent (LangGraphTask) 生成 kernel → 验证正确性

Phase 2: Submission Extraction
  └─ 从成功 attempt 的 final_state['coder_code'] 提取 ModelNew 代码
  └─ 写入 submission 目录: {submission_dir}/agent/{tier}/{case}.py

Phase 3: Performance Evaluation
  └─ 加载 reference Model + generated ModelNew
  └─ 二次正确性验证 (atol/rtol)
  └─ 性能测量: warmup → 多轮迭代 → 取中位数
  └─ 计算 speedup 和加权分数

Phase 4: Report
  └─ 控制台输出完整报告
  └─ JSON artifact (含 correctness + performance)
  └─ leaderboard.json (可选)
```

### 5.2 评分规则

| 项目 | 规则 |
|---|---|
| 正确性前提 | 不正确的 case 得 0 分 |
| speedup < 1.0 | 比 baseline 慢，线性折扣 [0, 60) 分 |
| speedup == 1.0 | 基础分 60 分 |
| speedup > 1.0 | 从 60 向 100 递增 |
| speedup >= 5.0 | 封顶 100 分 |
| Tier 权重 | t1=1.0x, t2=1.5x, t3=2.0x, t4=2.5x, t5=3.0x |
| 加权分 | raw_score × tier_weight |

### 5.3 runner 类型

| 项目 | 当前实现 |
|---|---|
| runner 类型 | `multi-mode runner (correctness / performance / full)` |
| 关注对象 | case 正确性、attempt 成功率、性能 speedup、加权评分 |
| 默认行为 | `--mode correctness`（向后兼容） |

### 5.4 `Pass@N`

| 项目 | 定义 |
|---|---|
| `N` | 每个 case 生成并运行的次数 |
| case `PASS` | `success_count > 0` |
| case `FAIL` | `success_count == 0` |

### 5.4 skip

| 场景 | 当前行为 |
|---|---|
| case 包含 `torch_npu` 且 backend 配置 `skip_npu=True` | 标记为 skipped，`skip_reason="requires torch_npu"` |
| 文件读取失败 | 标记为 skipped，`skip_reason` 包含读取错误信息 |
| skipped case 是否保留 | 是，进入 summary 和 JSON artifact |

### 5.5 environment error

| 来源 | 当前行为 |
|---|---|
| worker 注册失败 | 写入 `summary.environment_error` |
| 配置加载失败 | 写入 `summary.environment_error` |
| 环境检查失败 | 写入 `summary.environment_error` |
| 当前筛选条件下无 runnable cases | 写入 `summary.environment_error` |

## 6. CLI 接口

### 6.1 参数表

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--mode` | `str` | `correctness` | 运行模式：`correctness / performance / full` |
| `--backend` | `str` | `gpu` | `cpu / gpu / npu / all` |
| `--arch` | `str` | `None` | 覆盖 backend 默认架构（如 `rtx3090`, `ascend910b4`） |
| `--dsl` | `str` | `None` | 覆盖 backend 默认 DSL（如 `cpp`, `triton_cuda`, `triton_ascend`, `pytorch`） |
| `--backend-name` | `str` | `None` | 覆盖传递给 worker/config 的 backend 映射名（如 `cpu`, `cuda`, `ascend`） |
| `--pass-n` | `int` | `3` | 每个 case 的生成 / 运行次数 |
| `--devices` | `List[int]` | `None` | 显式指定设备 ID 列表 |
| `--max-concurrent` | `int` | `4` | 最大并发任务数 |
| `--tiers` | `List[str]` | `None` | tier 过滤 |
| `--cases` | `List[str]` | `None` | case 过滤 |
| `--filter` | `str` | `None` | 关键字过滤 |
| `--output` | `str` | `None` | JSON artifact 输出路径 |
| `--submission-dir` | `str` | `None` | Agent 提交文件保存目录（仅 performance/full 模式） |
| `--warmup` | `int` | `10` | 性能测量预热轮数（仅 performance/full 模式） |
| `--iterations` | `int` | `100` | 性能测量每轮迭代次数（仅 performance/full 模式） |
| `--num-trials` | `int` | `3` | 性能测量试验轮数（仅 performance/full 模式） |
| `--rtol` | `float` | `1e-2` | 二次正确性验证相对误差容差 |
| `--atol` | `float` | `1e-2` | 二次正确性验证绝对误差容差 |
| `--timeout` | `int` | `300` | 单 case 性能评测超时秒数（仅 performance/full 模式） |
| `--team-name` | `str` | `aikg_agent` | 提交者/团队标识，用于 submission 目录名和 leaderboard |
| `--workflow` | `str` | `coder_only_workflow` | LangGraph workflow 名称 |
| `--backends` | `List[str]` | `None` | `--backend all` 的执行顺序和范围（如 `--backends gpu npu`） |

**参数校验约束：**

| 参数 | 约束 |
|---|---|
| `--pass-n` | >= 1 |
| `--devices` | 非负、唯一 |
| `--backends` | 仅允许 `cpu`/`gpu`/`npu`；重复值自动去重；只能与 `--backend all` 一起使用 |
| `--team-name` | 仅允许 `[A-Za-z0-9_-]`，不能为空（防止路径穿越） |
| `--arch` / `--dsl` / `--backend-name` | 不能与 `--backend all` 同时使用（会被无差别广播到所有 backend） |
| `--warmup` | >= 0 |
| `--iterations` | >= 1 |
| `--num-trials` | >= 1 |
| `--rtol` | >= 0 |
| `--atol` | >= 0 |
| `--timeout` | >= 1 |

### 6.2 backend 语义

| backend | 行为 |
|---|---|
| `cpu` | 执行 CPU correctness |
| `gpu` | 执行 CUDA correctness |
| `npu` | 执行 Ascend correctness |
| `all` | 串行执行多个 backend 并汇总。默认顺序 `cpu → gpu → npu`，可通过 `--backends` 指定执行范围和顺序 |

### 6.3 backend 默认配置（来自 `BACKEND_CONFIGS`）

以下默认值均可通过对应 CLI 参数覆盖（仅限单 backend 模式；`--backend all` 下禁止使用这些覆盖参数）：

| backend | 默认 arch (`--arch`) | 默认 dsl (`--dsl`) | 默认 backend 映射名 (`--backend-name`) |
|---|---|---|---|
| `cpu` | `x86_64` | `cpp` | `cpu` |
| `gpu` | `rtx3090` | `triton_cuda` | `cuda` |
| `npu` | `ascend910b4` | `triton_ascend` | `ascend` |

### 6.4 devices 解析规则

| backend | 未显式传 `--devices` 时的行为 |
|---|---|
| `cpu` | 使用 `[0]` |
| `gpu` | 从环境变量或 `torch.cuda.device_count()` 解析，失败回退 `[0]` |
| `npu` | 从环境变量或 `torch.npu.device_count()` 解析，失败回退 `[0]` |

## 7. 输出结构

### 7.1 控制台输出

| 输出项 | 当前是否保留 |
|---|---|
| total cases | 是 |
| passed / failed cases | 是 |
| total / successful attempts | 是 |
| per-tier` statistics | 是 |
| per-case `Pass@N` 与 `PASS/FAIL` | 是 |
| `Best Time` | 否 |
| `best_attempt` | 否 |
| `elapsed_time=None` | 否 |

### 7.2 JSON artifact 顶层字段

| 字段 | 模式 | 说明 |
|---|---|---|
| `timestamp` | 全部 | 结果生成时间 |
| `runner_version` | 全部 | runner 版本标识 |
| `mode` | 全部 | `"correctness"` / `"performance"` / `"full"` |
| `config` | 全部 | 执行配置（见 7.2.1） |
| `environment` | 全部 | 运行环境信息（见 7.2.2） |
| `summary` | 全部 | correctness 维度的 summary |
| `results` | 全部 | case 级正确性结果（`--backend all` 时包含各 backend 聚合结果，每条带 `backend` 字段） |
| `performance_config` | performance/full | 性能测量参数 (warmup/iterations/trials/rtol/atol)。早期失败路径也会输出空值版本以保证 schema 一致 |
| `performance_results` | performance/full | case 级性能评测结果。早期失败或 0 提取时为 `[]` |
| `performance_summary` | performance/full | 性能评测汇总 (total_weighted_score/avg_speedup) |
| `submission_dir` | performance/full | Agent 提交文件保存路径 |
| `backend_results` | `--backend all` | 每个 backend 的完整结果 |

#### 7.2.1 `config` 字段

所有模式下 `config` 包含以下运行时行为参数：

| 字段 | 说明 |
|---|---|
| `mode` | 运行模式 |
| `backend` | 当前 backend（`"all"` 表示多 backend） |
| `arch` | 架构（单 backend: 字符串；`all`: 按 backend 的字典） |
| `dsl` | DSL 名称（单 backend 有值；`all` 模式不包含因为每个 backend 独立） |
| `backend_name` | backend 映射名覆盖（`null` 表示使用默认值） |
| `devices` | 设备列表 |
| `pass_n` | Pass@N |
| `max_concurrent` | 最大并发 |
| `tiers` / `cases` / `filter` | 过滤条件 |
| `team_name` | 团队标识 |
| `workflow` | LangGraph workflow 名称 |
| `backends` | 仅 `--backend all`：实际执行的 backend 顺序列表 |
| `warmup` / `iterations` / `num_trials` / `rtol` / `atol` / `timeout` | 性能参数（仅 performance/full 模式） |

`--backend all` 模式下 `config.arch` 为按 backend 聚合的字典（注意：`--arch`/`--dsl`/`--backend-name` 不能与 `--backend all` 一起使用）：

```json
{
  "arch": {
    "cpu": "<BACKEND_CONFIGS['cpu']['arch']>",
    "gpu": "<BACKEND_CONFIGS['gpu']['arch']>",
    "npu": "<BACKEND_CONFIGS['npu']['arch']>"
  },
  "backends": ["cpu", "gpu", "npu"]
}
```

#### 7.2.2 `environment` 字段在 `--backend all` 下的差异

单 backend 模式：

```json
{
  "framework": "torch",
  "dsl": "<来自 BACKEND_CONFIGS>",
  "backend": "<当前 backend>",
  "visible_devices": [0, 1]
}
```

`--backend all` 模式不再有顶层 `visible_devices`，改为 `per_backend` 聚合各 backend 的实际环境：

```json
{
  "framework": "torch",
  "dsl": "multiple",
  "backend": "all",
  "per_backend": {
    "cpu": { "visible_devices": [...], "dsl": "<来自 BACKEND_CONFIGS>" },
    "gpu": { "visible_devices": [...], "dsl": "<来自 BACKEND_CONFIGS>" },
    "npu": { "visible_devices": [...], "dsl": "<来自 BACKEND_CONFIGS>" }
  }
}
```

### 7.3 `summary` 核心字段

| 字段 | 说明 |
|---|---|
| `total_cases` | 总 case 数 |
| `passed_cases` | 至少成功一次的 case 数 |
| `failed_cases` | 全部失败的 case 数 |
| `case_pass_rate` | case 级通过率 |
| `total_attempts` | 总 attempts 数 |
| `successful_attempts` | 成功 attempts 数 |
| `attempt_pass_rate` | attempt 级通过率 |
| `total_wall_time` | 总耗时 |
| `tier_stats` | tier 级统计 |
| `skipped_cases` | 被跳过的 case 及原因 |
| `environment_error` | backend 级初始化或环境错误 |

### 7.4 退出码

| 条件 | 返回码 |
|---|---|
| 全部 case 成功（且 performance 无失败） | `0` |
| 任一 backend 初始化失败 | `1` |
| 任一 correctness case 失败 | `1` |
| performance/full 模式下有 perf 失败 | `1` |
| performance/full 模式下有通过的 case 但 0 个 submission 被抽取 | `1` |

## 8. 当前限制

| 编号 | 限制 |
|---|---|
| 1 | 性能测量需要真实硬件（GPU/NPU），且应在无其他负载的环境下运行 |
| 2 | GPU 兼容入口的 `--help` 文案来自统一入口，因此展示的是通用 runner 参数 |
| 3 | 共享模块仍位于 `examples/kernel_related/`，因为当前只被 runner 使用 |
| 4 | `--mode full` 下的性能评测是串行的，不支持并发性能测量 |
| 5 | 性能评测的 `--timeout` 基于子进程隔离（`multiprocessing` spawn 模式），超时后通过 `terminate()` / `kill()` 终止工作进程并释放 GPU/NPU 资源。超时错误信息包含 `last phase` 用于诊断（可能的阶段：`startup` / `loading_modules` / `model_init` / `correctness_check` / `measuring_baseline` / `measuring_solution`） |
| 6 | 二次正确性验证使用严格 AND 语义（`max_abs_diff <= atol` 且 `max_rel_diff <= rtol` 必须同时满足），比 `torch.allclose` 更严格 |

## 9. 推荐使用方式

### 9.1 正确性评测（默认）

```bash
python akg_agents/examples/kernel_related/run_torch_bench_lite.py --backend gpu
python akg_agents/examples/kernel_related/run_torch_bench_lite.py --backend cpu
python akg_agents/examples/kernel_related/run_torch_bench_lite.py --backend npu
```

### 9.2 完整评测（正确性 + 性能 + 评分）

```bash
python akg_agents/examples/kernel_related/run_torch_bench_lite.py --mode full --backend gpu --output results.json
```

### 9.3 自定义性能参数

```bash
python akg_agents/examples/kernel_related/run_torch_bench_lite.py \
  --mode full --backend gpu \
  --warmup 20 --iterations 200 --num-trials 5 \
  --output results.json
```

### 9.4 多 backend 汇总

```bash
python akg_agents/examples/kernel_related/run_torch_bench_lite.py --backend all --output results.json
```

### 9.5 GPU 快捷入口

```bash
python akg_agents/examples/kernel_related/gpu/run_torch_cuda_triton_bench_lite.py --mode full --pass-n 3
```

### 9.6 组合示例（含默认值说明）

#### 示例 1：最简运行（全部使用默认值）

```bash
python akg_agents/examples/kernel_related/run_torch_bench_lite.py
```

等价于：

```bash
python akg_agents/examples/kernel_related/run_torch_bench_lite.py \
  --mode correctness \
  --backend gpu \
  --arch rtx3090 \
  --dsl triton_cuda \
  --backend-name cuda \
  --pass-n 3 \
  --max-concurrent 4 \
  --team-name aikg_agent \
  --workflow coder_only_workflow
```

> 不指定时，`arch`/`dsl`/`backend-name` 从 `BACKEND_CONFIGS` 中取对应 backend 的默认值。

#### 示例 2：GPU full 评测 + 自定义性能参数 + 输出

```bash
python akg_agents/examples/kernel_related/run_torch_bench_lite.py \
  --mode full \
  --backend gpu \
  --pass-n 5 \
  --warmup 20 --iterations 200 --num-trials 5 \
  --rtol 1e-3 --atol 1e-3 \
  --timeout 600 \
  --output results.json
```

> 缺省 `--warmup 10`、`--iterations 100`、`--num-trials 3`、`--rtol 1e-2`、`--atol 1e-2`、`--timeout 300`。

#### 示例 3：NPU correctness + 覆盖 DSL 和架构

```bash
python akg_agents/examples/kernel_related/run_torch_bench_lite.py \
  --mode correctness \
  --backend npu \
  --arch ascend910b3 \
  --dsl pytorch \
  --devices 0 1
```

> 缺省 `--arch ascend910b4`、`--dsl triton_ascend`（均来自 `BACKEND_CONFIGS['npu']`）。

#### 示例 4：多 backend 汇总 + 自定义执行顺序

```bash
python akg_agents/examples/kernel_related/run_torch_bench_lite.py \
  --mode full \
  --backend all \
  --backends gpu npu \
  --team-name my_team \
  --output results.json
```

> `--backends gpu npu` 只运行 GPU 和 NPU（跳过 CPU），按指定顺序执行。缺省时 `--backends` 为 `cpu gpu npu`。
> `--team-name my_team` 使 submission 目录为 `{submission_dir}/my_team/`，leaderboard 中 team 字段也使用此值。缺省为 `aikg_agent`。

#### 示例 5：CPU correctness + 指定 tier 和 case

```bash
python akg_agents/examples/kernel_related/run_torch_bench_lite.py \
  --backend cpu \
  --tiers t1 t2 \
  --cases softmax relu_forward
```

> 缺省时 `--tiers` 从数据集目录中动态扫描所有 `t\d+` 目录（fallback: `t1 t2 t3`）。`--cases` 缺省时运行所有发现的 case。

#### 示例 6：使用自定义 workflow 进行 full 评测

```bash
python akg_agents/examples/kernel_related/run_torch_bench_lite.py \
  --mode full \
  --backend gpu \
  --workflow my_custom_workflow \
  --output results.json
```

> `--workflow` 决定 LangGraph 使用的工作流名称，缺省为 `coder_only_workflow`。
