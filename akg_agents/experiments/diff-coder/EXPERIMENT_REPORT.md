# KernelBench × diff 修复（Diff-Coder）对照实验报告

> **定位说明**：本报告对应仓库内 **`akg_agents/experiments/diff-coder`** 矩阵实验（KernelBench + **Triton CUDA** + 首轮错误注入 + **diff 修复 A/B**），与「Triton-Ascend + 166 算子 + 智能选模」类实习报告 **场景不同**：后者偏 **能力边界扫全库 + 路由策略**；本报告偏 **同一 Agent 链路下「diff 修复开/关」对耗时与 LLM 调用的影响**，为后续是否合入主流程提供数据参考。

---

## 一、实验目的与问题陈述

### 1.1 背景

端到端 Coder 在复杂算子上失败后，往往触发 **整段重写**，带来额外 **LLM 调用与验证轮次**。工程上引入 **diff 修复** 思路：在失败日志约束下做 **局部补丁式 / 增量式修改**，期望在部分场景降低总耗时或总 token。（实现侧仍以环境变量 **`AIKG_TARGETED_REPAIR`** 表征该路径开关；profile 中 `repair_variants` 的 `targeted_on` / `targeted_off` 即 **diff 开 / diff 关**。）

### 1.2 本实验要回答的问题

1. 在 **受控的首轮失败**（`syntax` / `import` / `name` / `runtime` / `type`）下，`AIKG_TARGETED_REPAIR=1` 与 `=0` 相比，**墙钟时间、LLM 总耗时、首轮 vs 修复段耗时** 如何变化？  
2. 不同 **KernelBench 用例**（本 profile：`mm` / `conv2d` / `batchnorm`）与不同 **失败类型** 上，收益是否一致？  
3. 在 **当前实现与模型组合** 下，是否存在 **「简单 case 开 diff 修复反而更亏」** 的现象？

### 1.3 非目标（明确边界）

- 本实验 **不** 覆盖 Triton-Ascend / CANN / NPU 全量 249 或 166 扫库。  
- 本实验 **不** 评测 `model_selector` / 路由策略；仅提供 **可复现矩阵 + 计时字段**，供后续接入。  
- 本报告中的 **通过率** 在下列本地聚合样本上为 **pytest 最终轮 `pass_pct`（本数据均为 100%）**；不代表「无注入、无修复」下的真实生成成功率。

---

## 二、实验框架与流水线

### 2.1 执行拓扑

| 层级 | 内容 |
|------|------|
| 驱动 | `run_matrix.py` + profile `profiles/kernelbench_triton_cuda_repair_ab.json` |
| 被测树 | 单树 `akg_agents`（`workspace_root` 指向含 `akg_agents/` 的仓库根） |
| 容器 | profile 默认 Docker（可 `--no-docker` 烟测） |
| Benchmark 脚本 | `scripts/benchmark_local_kernelbench_repeat.sh` |
| Pytest 入口 | 见 profile `benchmarks[].test_target`（`mm`/`batchnorm` 与 `conv2d` 用例不同） |
| 工作流 | `coder_only_workflow`（由测试侧 `LangGraphTask` 指定） |

### 2.2 验证与计时

| 观测项 | 来源 |
|--------|------|
| 每轮 pytest 是否通过 | `summary.tsv` 中 `pytest_rc`、`pass_pct` |
| 墙钟 / 本地耗时 / LLM 耗时 | `wall_seconds`、`local_sum_s`、`llm_sum_s`、`llm_first_sum_s`、`llm_repair_sum_s`、`task_total_sum_s` |
| 细粒度 LLM 调用 | `AIKG_BENCHMARK_LOCAL_TIMING_JSONL` 指向的 `timing_*_attempt*.jsonl`（若需更细可二次分析） |

### 2.3 首轮错误注入（与主链路对齐）

| 环境变量 | 含义 |
|----------|------|
| `AIKG_FORCE_FIRST_FAIL` | profile `force_first_fail: true` 时置 `1`：首轮在 Coder 成功后 **注入一条可控错误**，稳定走「失败 → 修复」 |
| `AIKG_FORCE_FAIL_MODE` | 取 `syntax` / `import` / `name` / `runtime` / `type` 之一，与 `nodes.py` 中 `_inject_forced_first_fail` 一致 |

### 2.4 diff 修复 A/B

| 维度 | `targeted_on`（**diff 开**） | `targeted_off`（**diff 关**） |
|------|-----------------|----------------|
| 环境 | `AIKG_TARGETED_REPAIR=1` | `AIKG_TARGETED_REPAIR=0` |
| 含义 | 走 **diff / 补丁式** 修复路径（详见 `core/agent/coder.py` 等） | 更偏 **全量重写式** 修复路径 |
| 其它 | 与 `repair_variants` 中 `extra_env` 一致；可与 `AIKG_TARGETED_REPAIR_ATTEMPTS` 等组合 | 同左 |

**结论前置（逻辑）**：矩阵对 **每一种 `fail_mode` × 每一种 benchmark × diff 开/关** 各跑 **`repeats` 轮**（本 profile 为 3），因此 **不是「所有错误都只走 diff 修复」**；而是 **同一错误类型下对比 diff 开与关**。

---

## 三、硬件与软件环境（本实验实际使用）

> 以下以 **本仓库 profile 默认值 + 本地一次完整矩阵** 为准；若你更换镜像 / GPU / 模型，请在复现时更新本节。

| 组件 | 规格 / 版本 |
|------|-------------|
| GPU | NVIDIA（pytest 标记 `cuda` / `a100`，物理机可为 A800 等 Ampere 系） |
| Python | `python3`（脚本与容器内一致） |
| 框架 | PyTorch + Triton CUDA（KernelBench level1 / level2 用例） |
| LLM | profile 中 `AKG_AGENTS_MODEL_NAME` / `AIKG_MODEL_NAME`（例如 `deepseek-v4-flash`；以实际 `api_env_exports` 为准） |
| API | DeepSeek 兼容接口；密钥来自仓库根 `.deepseek_api_env`（勿提交） |

---

## 四、数据集与用例配置

### 4.1 Profile 中的 benchmark id

| `benchmarks[].id` | 说明 | `test_target`（摘要） |
|-------------------|------|------------------------|
| `mm` | 矩阵/算子烟测用例 | `test_bench_triton_cuda.py::test_bench_triton_cuda`（level1，**题号以测试代码内 `get_kernelbench_op_name([...])` 为准**） |
| `conv2d` | 更重用例 | `::test_bench_triton_cuda_level_2` |
| `batchnorm` | 命名 id；**须与测试内 KernelBench 序号一致**（若测试仍为 `[19]` 则实为 ReLU 等，**非** `33_BatchNorm`） |

**已知限制**：若需「名实一致」的 BatchNorm，应同步调整 `tests/op/bench/test_bench_triton_cuda.py` 中序号或增加 per-benchmark `test_target_by_tree`；否则报告中的 `batchnorm` 行应理解为 **「profile 标签」** 而非一定对应 `33_BatchNorm.py`。

### 4.2 失败类型（`fail_modes`）

`syntax` / `import` / `name` / `runtime` / `type` —— 与 `nodes.py` 注入 map 一一对应。

### 4.3 矩阵规模（本报告数据来源）

- **diff 修复维度**：2（`targeted_on` / `targeted_off`，即 diff 开 / diff 关）  
- **Benchmark**：3  
- **Fail mode**：5  
- **Repeats**：3（profile `repeats`）  

合计 **2×3×5 = 30** 个输出目录；每个目录 `summary.tsv` 取 **`attempt=final` 且 `counted_for_avg=1`** 共 **90** 条样本行，用于下文聚合。

---

## 五、复现步骤（给评审 / 后续自己）

```bash
# 0) API（勿提交密钥）
# 在 workspace_root 根目录配置 .deepseek_api_env，并在 profile 中正确填写 env_file 绝对路径

# 1) 进入实验目录
cd <repo>/akg_agents/experiments/diff-coder

# 2) 干跑（检查 docker / export 块）
python3 run_matrix.py run --profile profiles/kernelbench_triton_cuda_repair_ab.json --dry-run

# 3) 全矩阵（示例：无 Docker 烟测可改 --no-docker；生产多为 Docker）
python3 run_matrix.py run --profile profiles/kernelbench_triton_cuda_repair_ab.json --continue-on-error

# 4) 汇总（glob 按本机 out_dir_prefix 修改）
python3 run_matrix.py summarize --glob '<repo>/akg/out/diffcoder/kernelbench_triton_cuda_repair_ab_*'
```

**说明**：`out/` 默认在 `<workspace_root>/akg/out/diffcoder`；若 Docker 以 root 写产物，宿主机后续 `git rebase` 可能遇权限问题；无 sudo 时可 **新目录 clone** 做历史改写（见仓库内讨论记录）。

---

## 六、实验结果（基于本机 `summary.tsv` 聚合）

### 6.1 总体：diff 修复开 vs 关（90 条 final 样本）

| 分组 | 样本数 | 平均墙钟 `wall_s` | 平均 `llm_sum_s` | 平均 `llm_first_sum_s` | 平均 `llm_repair_sum_s` | 平均 `pass_pct` |
|------|--------|-------------------|------------------|--------------------------|-------------------------|-----------------|
| `targeted_on`（diff 开） | 45 | **39.23** | **18.67** | 8.13 | 10.54 | 100% |
| `targeted_off`（diff 关） | 45 | **72.02** | **39.90** | 8.00 | 31.90 | 100% |

**解读（本数据集）**：在 **首轮强制失败 + 本 profile 三用例** 组合下，`targeted_on`（diff 开）相对 `targeted_off`（diff 关）**平均墙钟约 −45%**、**平均 LLM 总耗时约 −53%**。主要贡献来自 **`conv2d` 子矩阵**（见 6.3）：diff 关时 `llm_repair` 极高，拉高整体均值。

> 注意：此处为 **「注入失败后的修复实验」** 下的耗时对比，**不能直接外推**为「生产无注入时 diff 修复一定更快」。

### 6.2 按 benchmark（各 30 条 final）

| Benchmark | 平均 `wall_s` | 平均 `llm_sum_s` | 平均 `llm_first_s` | 平均 `llm_repair_s` |
|-----------|---------------|------------------|--------------------|------------------------|
| `mm` | 31.19 | 12.33 | 5.59 | 6.74 |
| `batchnorm` | 31.99 | 13.49 | 6.15 | 7.34 |
| `conv2d` | **103.70** | **62.03** | 12.45 | **49.58** |

### 6.3 按 `fail_mode`（各 18 条 final）

| fail_mode | 平均 `wall_s` | 平均 `llm_sum_s` |
|-----------|---------------|------------------|
| `import` | 52.02 | 25.07 |
| `runtime` | 52.41 | 29.36 |
| `syntax` | 55.78 | 28.87 |
| `type` | 55.76 | 29.38 |
| `name` | **62.15** | **33.72** |

### 6.4 细粒度：同一 `(benchmark, fail_mode)` 下 diff 开 vs diff 关（各 3 轮均值）

下表为 **diff 开（`targeted_on`）相对 diff 关（`targeted_off`）的墙钟变化率**（负表示 diff 开更快）与 **LLM 总耗时变化率**。

| bench | fail_mode | Δ wall % | Δ llm % |
|-------|-----------|----------|---------|
| batchnorm | import | −1.9% | −4.1% |
| batchnorm | name | **+15.0%** | **+35.2%** |
| batchnorm | runtime | **−37.8%** | **−31.9%** |
| batchnorm | syntax | −1.7% | −3.1% |
| batchnorm | type | −8.2% | −19.3% |
| conv2d | import | **−32.5%** | **−52.4%** |
| conv2d | name | **−60.1%** | **−61.4%** |
| conv2d | runtime | **−74.9%** | **−75.1%** |
| conv2d | syntax | **−74.0%** | **−78.5%** |
| conv2d | type | **−65.6%** | **−67.3%** |
| mm | import | +4.5% | +11.5% |
| mm | name | +7.8% | +19.4% |
| mm | runtime | −28.2% | −27.5% |
| mm | syntax | +11.0% | +34.2% |
| mm | type | +9.2% | +24.4% |

**观察**：

1. **`conv2d` + 多数 fail_mode**：diff 开（`targeted_on`）**显著优于** diff 关（墙钟与 LLM 双降），与「复杂用例上 diff / 局部修改更划算」的直觉一致。  
2. **`mm` + 部分 fail_mode**（如 `syntax` / `type` / `name`）：diff 开 **更慢**，符合「简单或单次可收敛场景，diff 修复有额外提示与多轮结构开销」的判断。  
3. **`batchnorm` + name`**：diff 开明显慢于 diff 关，需结合 **该 id 实际对应 KernelBench 题面** 再解读（见 4.1）。

---

## 七、结论与对主流程合入的建议

### 7.1 结论（本实验范围内）

1. **diff 修复不是免费午餐**：在 **mm 等相对轻** 的子矩阵上，部分 `fail_mode` 下 **diff 开比 diff 关更慢**，说明 **策略需按用例/失败类型分流**，不宜默认全局开启 diff。  
2. **在 conv2d（level2）子矩阵上收益很大**：与本 profile 下 **更高验证/修复成本** 一致，diff 开（`targeted_on`）在墙钟与 LLM 上均 **大幅下降**。  
3. **聚合层面**：本批 90 个 final 样本上 **diff 开整体优于 diff 关**，但主要由 **conv2d 子集拉高 diff 关时成本** 驱动；写论文/写 MR 时应 **分层汇报**，避免只报总平均掩盖「简单 case 开 diff 反而吃亏」。

### 7.2 与「166 算子 Ascend 能力边界」类报告的关系

| 维度 | 166 / 249 Ascend 报告（示例） | 本报告 |
|------|------------------------------|--------|
| 目标 | 模型能力边界、选模 ROI | **修复策略 A/B、计时与可观测性** |
| 后端 | Triton-Ascend + NPU | **Triton CUDA + GPU pytest** |
| 数据规模 | 全库 / 多模型 | **小矩阵、强控制变量** |
| 产出 | 路由表 / 复杂度阈值 | **`summary.tsv` / timing jsonl + 复现命令** |

二者 **互补**：本报告支持「**何时开 diff 修复**」的工程决策；Ascend 全库报告支持「**选哪个模型**」。

### 7.3 主流程合入建议（供后续 owner）

1. **先合入工具链**（矩阵、计时、profile、文档）与 **可选特性开关**，默认行为与线上对齐。  
2. **路由/选模** 建议等具备 **Ascend 或生产等价负载** 的分层数据后再合。  
3. **profile 与测试题号** 做一次对齐审计（`batchnorm` id）。  
4. **Docker 产物权限** 写入运维说明，避免本地 `rebase` 被 root 文件阻塞。

---

## 八、附录：聚合脚本（可自行重跑）

将 `<OUT_GLOB>` 换成你的 `out_dir_prefix`：

```python
import csv, glob, re
from pathlib import Path
from statistics import mean

pat = re.compile(
    r"kernelbench_triton_cuda_repair_ab_akg_agents_(targeted_on|targeted_off)_(mm|conv2d|batchnorm)_(syntax|import|name|runtime|type)$"
)
rows = []
for p in sorted(glob.glob(str(Path("<OUT_GLOB>") / "*/summary.tsv"))):
    m = pat.match(Path(p).parent.name)
    if not m:
        continue
    repair, bench, mode = m.groups()
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get("attempt") == "final" and r.get("counted_for_avg") == "1":
                rows.append((repair, bench, mode, float(r["wall_seconds"]), float(r["llm_sum_s"])))
print("n=", len(rows))
```

---

## 九、修订记录

| 日期 | 说明 |
|------|------|
| 2026-05-14 | 首版：结构对齐实习报告体例；数值来自 `<workspace>/akg/out/diffcoder` 一次矩阵跑出的 `summary.tsv` 聚合 |
| 2026-05-14 | 术语统一：报告内以「diff 修复」表述开关维度；技术标识 `targeted_on`/`targeted_off`、`AIKG_TARGETED_REPAIR` 仍与代码一致 |

---

## 许可

与仓库一致：以各文件 SPDX / 仓库根 LICENSE 为准。
