# wip/ — Skill 系统对比复现脚本（实验中）

本目录包含 6 个独立复现脚本，用于对比**固定文档导入**和 **Skill 系统导入**两种方式在不同 benchmark 上的算子生成效果。

> **状态**: WIP（实验中）。验证通过后将移至 `reproduce/` 根目录。

---

## 两种导入方式说明

### 固定文档导入 (coder_only_workflow)

使用 `load_config(dsl=, backend=)` 加载默认配置，由 `docs_dir.coder` 指定的固定文档目录（如 `op/resources/docs/triton_ascend_docs`）将**全部文档内容**拼接注入 prompt。不经过任何动态选择。

- 优点：确定性强，每次注入内容完全一致
- 缺点：token 消耗固定且较高，无法根据算子类型和生成阶段调整内容

### Skill 系统导入 (kernelgen_only_workflow)

使用 `triton_ascend_kernelgen_config.yaml` 配置，由 `KernelGen` 内部按生成阶段动态选择 skill：

| 阶段 | 注入内容 |
|------|---------|
| **initial** | fundamental + reference + guide + example（无 case） |
| **debug** | fundamental + reference + guide + example + case（含 evolved-fix） |
| **optimize** | fundamental + reference + guide + example + case（含 evolved-improvement） |

其中 guide / case 由 LLM 根据算子描述从 `SKILL.md` 文档库中动态选择。

- 优点：按需注入，token 消耗更低，不同阶段看到不同层级内容
- 缺点：依赖 LLM 选择准确性，有一定随机性

---

## 脚本列表

| # | 脚本 | Benchmark | 策略 | 说明 |
|---|------|-----------|------|------|
| 1 | `reproduce_mhc_coder_only.py` | EvoKernel MHC | 固定文档 | MHC 算子，按序号指定 |
| 2 | `reproduce_mhc_kernelgen_skill.py` | EvoKernel MHC | Skill 系统 | MHC 算子，按序号指定 |
| 3 | `reproduce_kernelbench_coder_only.py` | KernelBench Level1 | 固定文档 | 默认排除 conv(54-87) |
| 4 | `reproduce_kernelbench_kernelgen_skill.py` | KernelBench Level1 | Skill 系统 | 默认排除 conv(54-87) |
| 5 | `reproduce_akgbench_coder_only.py` | AKGBench Lite | 固定文档 | 按 tier(t1/t2/t3) 选择 |
| 6 | `reproduce_akgbench_kernelgen_skill.py` | AKGBench Lite | Skill 系统 | 按 tier(t1/t2/t3) 选择 |
| 7 | `reproduce_adaptive_search.py` | 所有 benchmark | Adaptive Search | 直接调用 `adaptive_search()` API，UCB 策略多方案探索 |
| 8 | `reproduce_evolve.py` | 所有 benchmark | Evolve | 直接调用 `evolve()` API，多岛屿进化策略 |

公共模块 `_common.py` 提供环境规范采集、通用运行器、报告生成等共享功能。

> **脚本 1-6** 为基础 workflow（单算子直接生成），通过 `LangGraphTask` 调用。
> **脚本 7-8** 为搜索/进化策略（每个算子内部多次尝试），直接调用 `adaptive_search()` / `evolve()` API。
> 它们通过 `--benchmark` 参数支持 kernelbench / akgbench / mhc 三种 benchmark。

---

## 前置条件

```bash
# 1. 激活环境
source env.sh

# 2. 配置 API Key
export AKG_AGENTS_API_KEY=your_key_here

# 3. 下载第三方 benchmark（KernelBench / EvoKernel）
bash akg_agents/download.sh --with_kernelbench --with_evokernel

# 4. 确认 NPU 可用
export DEVICE_ID=0   # 可选，默认 0
```

---

## 运行方式

每个脚本都支持 `--help`：

```bash
python reproduce/wip/reproduce_mhc_coder_only.py --help
```

### 通用参数

基础脚本（1-6）共享以下参数（由 `_common.add_common_args` 提供）：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--device` | NPU 设备 ID（可指定多个以池化，如 `--device 4 5 6 7`） | `$DEVICE_ID` 或 `0` |
| `--concurrency` | 设备任务并行度上限 | `4` |
| `--llm-concurrency` | LLM 请求并发数 | 与 `--concurrency` 相同 |
| `--arch` | 硬件架构 | `ascend910b4` |
| `--pass-n` | Pass@N：每个算子独立运行 N 次 | `1` |
| `--output` | JSON 报告输出路径 | `~/.akg/reproduce_log/<script>_<timestamp>.json` |
| `--profile` | 开启性能测试（默认关闭，开启后验证通过的算子自动跑 speedup） | `false` |

### 搜索/进化策略（独立脚本 7-8）

`adaptive_search` 和 `evolve` 因参数空间完全不同，使用**独立脚本**，通过 `--config` 加载 YAML 配置文件：

- `reproduce_adaptive_search.py` — 直接调用 `adaptive_search()` API
  - 配置文件：`adaptive_search_config.yaml`（`--config` 指定，默认使用项目内置）
  - 内部创建多个 `LangGraphTask(workflow=kernelgen_only_workflow)` 子任务
  - UCB 策略选择父代，灵感采样驱动进化
  - CLI 覆盖：`--max-total-tasks`, `--max-concurrent`, `--exploration-coef` 等
- `reproduce_evolve.py` — 直接调用 `evolve()` API
  - 配置文件：`evolve_config.yaml`（`--config` 指定，默认使用项目内置）
  - 多岛屿模型 + 精英保留 + 迁移策略
  - CLI 覆盖：`--max-rounds`, `--parallel-num`, `--num-islands` 等

配置优先级：**CLI 参数 > config.yaml > 内置默认值**。
JSON 输出中 `adaptive_search_config` / `evolve_config` 字段记录最终使用的全部参数。

### 各脚本特有参数

**MHC 脚本** (`reproduce_mhc_*.py`)：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--op` | MHC 算子序号（空格分隔多个）；不指定则全跑 | 全部（1-15） |

MHC 算子序号对照：

| 序号 | 算子 | 序号 | 算子 | 序号 | 算子 |
|------|------|------|------|------|------|
| 01 | SinkhornKnopp | 06 | MhcUpdate | 11 | FusedMHCKernels |
| 02 | MhcProjector | 07 | MHCModule | 12 | OptimizedMHCLayerWithFusion |
| 03 | StreamWeightedSum | 08 | MHCBlock2d | 13 | StaticMHCHyperConnections |
| 04 | StreamMix | 09 | MHCBlockBottleneck2d | 14 | MhcPreBlock |
| 05 | StreamWrite | 10 | OrthostochasticProject | 15 | MhcPostBlock |

**KernelBench 脚本** (`reproduce_kernelbench_*.py`)：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--tasks` | Level1 任务序号列表（空格分隔） | 全部 1-100（排除 54-87 conv） |
| `--include-conv` | 包含 54-87 号 conv 算子 | `false` |

**AKGBench Lite 脚本** (`reproduce_akgbench_*.py`)：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--tiers` | Tier 列表（空格分隔）；不指定则全跑 | `t1 t2 t3`（全部） |
| `--cases` | 指定算子名称（不含 .py，空格分隔） | 该 tier 全部算子 |

### 运行示例

```bash
# MHC 全部算子（默认）
python reproduce/wip/reproduce_mhc_coder_only.py
python reproduce/wip/reproduce_mhc_kernelgen_skill.py

# MHC 指定序号
python reproduce/wip/reproduce_mhc_coder_only.py --op 5
python reproduce/wip/reproduce_mhc_kernelgen_skill.py --op 1 3 5 7

# KernelBench 默认（排除 conv）
python reproduce/wip/reproduce_kernelbench_coder_only.py
python reproduce/wip/reproduce_kernelbench_kernelgen_skill.py

# KernelBench 指定序号
python reproduce/wip/reproduce_kernelbench_coder_only.py --tasks 1 5 19 42

# KernelBench 包含 conv
python reproduce/wip/reproduce_kernelbench_coder_only.py --include-conv

# AKGBench Lite 全部 tier（默认）
python reproduce/wip/reproduce_akgbench_coder_only.py
python reproduce/wip/reproduce_akgbench_kernelgen_skill.py

# AKGBench Lite 只跑 t1
python reproduce/wip/reproduce_akgbench_coder_only.py --tiers t1

# AKGBench Lite 指定算子
python reproduce/wip/reproduce_akgbench_kernelgen_skill.py --cases gelu softmax

# 多卡池化 + LLM 并发控制
python reproduce/wip/reproduce_kernelbench_coder_only.py --device 4 5 6 7 --concurrency 4 --llm-concurrency 8

# Pass@3：每个算子独立运行 3 次
python reproduce/wip/reproduce_kernelbench_kernelgen_skill.py --pass-n 3

# Adaptive Search — KernelBench 全部算子
python reproduce/wip/reproduce_adaptive_search.py --benchmark kernelbench --max-total-tasks 50

# Adaptive Search — 指定序号 + 调参
python reproduce/wip/reproduce_adaptive_search.py --benchmark kernelbench --tasks 1 5 19 \
    --max-concurrent 4 --exploration-coef 2.0

# Evolve — KernelBench + 多岛屿
python reproduce/wip/reproduce_evolve.py --benchmark kernelbench --max-rounds 5 --num-islands 3

# Evolve — AKGBench
python reproduce/wip/reproduce_evolve.py --benchmark akgbench --tiers 1 2

# 指定架构
python reproduce/wip/reproduce_kernelbench_coder_only.py --arch ascend910b3
```

---

## 结果存储格式

每个脚本运行后生成一个 JSON 报告文件，默认保存到 `~/.akg/reproduce_log/` 目录。

### JSON 结构

```json
{
  "benchmark": "KernelBench_Level1_no_Conv",
  "script": "kernelbench_coder_only",
  "workflow": "coder_only_workflow",
  "pass_n": 1,
  "ops_count": 66,
  "elapsed_s": 456.7,
  "device_ids": [4, 5, 6, 7],
  "max_concurrency": 4,
  "llm_concurrency": 4,
  "env_spec": {
    "arch": "ascend910b4",
    "python": "3.10.0",
    "timestamp": "2026-03-30T14:30:00",
    "torch_npu": "2.7.1",
    "triton_ascend": "3.2.0",
    "commit": "7916e2a8",
    "llm_model": "deepseek-v3.2"
  },
  "task_log_dir": "/home/user/.akg/reproduce_log",
  "stats": {
    "total_ops": 66,
    "passed_ops": 50,
    "failed_ops": 16,
    "pass_rate": 0.758,
    "op_results": {
      "akg_agents_kernelbench_19_ReLU": {
        "passed": 1,
        "total": 1,
        "profile": {
          "gen_time": 12.3,
          "base_time": 15.1,
          "speedup": 1.23
        }
      },
      "akg_agents_kernelbench_42_Max_Pooling_2D": {
        "passed": 0,
        "total": 1
      }
    }
  }
}
```

> `profile` 仅在 profiling 成功时出现（含 `gen_time`=生成耗时、`base_time`=基线耗时、`speedup`=加速比）。
> `adaptive_search` 和 `evolve` 脚本产出的 JSON 额外包含 `adaptive_search_config` / `evolve_config` 字段
> 以及每个算子的 `search_stats` / `evolve_stats` 字段。

### 环境规范字段说明

以下字段均由脚本**运行时自动检测**并写入 JSON 报告，无需手动填写：

| 字段 | 来源 | 说明 |
|------|------|------|
| `arch` | `--arch` 参数 | 硬件架构（如 `ascend910b4`） |
| `python` | `platform.python_version()` | Python 版本 |
| `timestamp` | 运行时刻 | ISO 格式时间戳 |
| `torch_npu` | `torch_npu.__version__` | torch_npu 版本 |
| `triton_ascend` | `triton.__version__` | Triton Ascend 版本 |
| `commit` | `git rev-parse --short HEAD` | akg 项目 git commit |
| `llm_model` | `create_llm_client("standard")` 解析结果 | 实际使用的 LLM 模型名称（env > settings.json） |
| `device_ids` | `--device` 参数 | 使用的设备 ID 列表 |
| `max_concurrency` | `--concurrency` 参数 | 任务并行度上限 |
| `task_log_dir` | config `log_dir` | 任务具体日志存储目录 |

### 对比方法

运行同一 benchmark 的两个脚本（`*_coder_only.py` 和 `*_kernelgen_skill.py`），
对比 JSON 报告中的 `stats` 字段即可得到固定文档 vs Skill 系统的效果差异。

可参照 `reproduce/RESULT_TEMPLATE.md` 模板记录人工对比结论。

---

## 文件说明

```
wip/
├── README.md                                  # 本文件
├── _common.py                                 # 公共模块（环境采集、运行器、报告）
├── reproduce_mhc_coder_only.py                # MHC — 固定文档
├── reproduce_mhc_kernelgen_skill.py           # MHC — Skill 系统
├── reproduce_kernelbench_coder_only.py        # KernelBench — 固定文档
├── reproduce_kernelbench_kernelgen_skill.py   # KernelBench — Skill 系统
├── reproduce_akgbench_coder_only.py           # AKGBench Lite — 固定文档
├── reproduce_akgbench_kernelgen_skill.py      # AKGBench Lite — Skill 系统
├── reproduce_adaptive_search.py               # 所有 benchmark — Adaptive Search (direct API)
└── reproduce_evolve.py                        # 所有 benchmark — Evolve (direct API)
```
