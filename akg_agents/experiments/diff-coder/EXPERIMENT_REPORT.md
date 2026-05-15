# KernelBench × diff 修复（Diff-Coder）对照实验报告

> **场景**：在 **KernelBench + Triton CUDA** 上，对首轮 **受控失败**（`syntax` / `import` / `name` / `runtime` / `type`）对比 **`AIKG_TARGETED_REPAIR` 开（diff 修复开）与关** 的墙钟与 LLM 耗时。与「Ascend 全库能力边界 / 选模」类报告互补：本报告回答 **修复策略 A/B 的工程取舍**，不评测路由或 NPU 全量。

---

## 实验说明（摘要）

- **对比维度**：同一 profile 下 `targeted_on`（diff 开，`AIKG_TARGETED_REPAIR=1`）与 `targeted_off`（diff 关，`=0`）。  
- **矩阵**：3 个 benchmark 标签（`mm` / `conv2d` / `batchnorm`）× 5 种 `fail_mode` × 2 种修复开关 × 每格 **3** 次重复；聚合时取各目录 `summary.tsv` 中 **`attempt=final` 且 `counted_for_avg=1`** 的行，共 **90** 条。  
- **通过率**：本批样本上 pytest **均为 100%**；含义是「在注入失败的前提下仍能跑通当前测试」，**不能**外推为无注入时的真实生成成功率。  
- **边界**：不覆盖 Triton-Ascend / 166 或 249 全库；不评测 `model_selector`。  
- **已知标签问题**：profile 中 `batchnorm` 与测试内 KernelBench **题号**可能不一致；表中 `batchnorm` 行应理解为 **配置标签**，名实对齐需改测试或 profile（见代码侧说明）。

**实现、profile、矩阵驱动与复现命令**不在本报告展开，见下方 GitCode 分支中的 `akg_agents/experiments/diff-coder/`。

---

## 代码与复现（GitCode）

请在浏览器中打开（若分支名有调整，将 URL 中的分支名替换为你的实际分支即可）：

- **仓库**：<https://gitcode.com/zhangyize-2026/akg_3797>  
- **建议入口（含本实验目录）**：<https://gitcode.com/zhangyize-2026/akg_3797/-/tree/br_agents/akg_agents/experiments/diff-coder>  

该目录下包含 `run_matrix.py`、`profiles/kernelbench_triton_cuda_repair_ab.json` 等；原始 `summary.tsv` 聚合路径与本地 `out/` 约定亦以该分支为准。

---

## 观察（基于本机矩阵 `summary.tsv` 聚合）

### 总体（90 条 final 样本）

| 分组 | 样本数 | 平均墙钟 `wall_s` | 平均 `llm_sum_s` | 平均 `llm_first_s` | 平均 `llm_repair_s` | `pass_pct` |
|------|--------|-------------------|------------------|--------------------|---------------------|------------|
| diff 开（`targeted_on`） | 45 | **39.23** | **18.67** | 8.13 | 10.54 | 100% |
| diff 关（`targeted_off`） | 45 | **72.02** | **39.90** | 8.00 | 31.90 | 100% |

在本批数据上，diff **开**相对 **关**：平均墙钟约 **−45%**，平均 LLM 总耗时约 **−53%**；主要因 **`conv2d`** 子矩阵在 diff 关时 **`llm_repair` 很高**，拉高整体均值。

> 上述结论限定在 **「首轮强制失败 + 本 profile」** 的实验设定下，**不可**直接写成「生产环境无注入时 diff 一定更快」。

### 按 benchmark（各 30 条 final）

| Benchmark | 平均 `wall_s` | 平均 `llm_sum_s` | 平均 `llm_repair_s` |
|-----------|---------------|------------------|---------------------|
| `mm` | 31.19 | 12.33 | 6.74 |
| `batchnorm` | 31.99 | 13.49 | 7.34 |
| `conv2d` | **103.70** | **62.03** | **49.58** |

### 同一 `(benchmark, fail_mode)` 下：diff 开相对 diff 关的变化率

（**Δ wall % / Δ llm %**：负表示 diff **开**更快、LLM 更少。）

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

**要点**：`conv2d` 上 diff 开在多数 `fail_mode` 下 **明显占优**；`mm` 上部分模式 diff 开 **更慢**；`batchnorm` + `name` 为 **反例**，需结合该标签对应的真实题号解读。

---

## 结论

1. **diff 修复不是全局免费**：轻量子矩阵（如 `mm`）上，部分失败类型下 **开 diff 比关更慢**，存在提示与轮次结构开销。  
2. **重子矩阵（本数据中的 `conv2d`）收益大**：diff 关在修复段 LLM 成本显著更高，与「复杂用例上局部补丁更划算」的直觉一致。  
3. **总平均「开优于关」主要由 `conv2d` 驱动**；对外汇报应 **分层**（按 benchmark / fail_mode），避免只报总平均掩盖简单 case 的劣势。

---

## 推荐修改方案（对主流程 / 后续 owner）

以下按 **合入拆分 → 默认与开关 → 分流与实现线索 → 配置与可观测性 → 回归与质量 → 工程运维 → 对外沟通 → 后续扩展** 组织，便于拆任务与评审对照。

**路径约定（下文「涉及文件」均以仓库根下含 `akg_agents/` 的 clone 为准）**

| 前缀 | 含义 |
|------|------|
| `experiments/diff-coder/...` | 即 `akg_agents/experiments/diff-coder/...` |
| `python/akg_agents/...` | 即 `akg_agents/python/akg_agents/...`（Python 包根） |
| `tests/op/...` | 即 `akg_agents/tests/op/...`（pytest 常从 `akg_agents/` 目录跑） |

若你当前分支尚未包含矩阵目录或某文件，以 **GitCode 上含本实验的分支**（如 `br_agents`）为准；本地可用 `git grep -n <关键词>` 核对实际路径。

### 1. 合入拆分（建议两阶段 MR）

| 阶段 | 建议内容 | 目的 | 主要涉及文件 / 目录（典型） |
|------|----------|------|------------------------------|
| **A（低风险、可先行）** | 矩阵、profile、`summary.tsv`/timing、文档 | 可复现、**不改**线上默认修复路径 | `experiments/diff-coder/run_matrix.py`；`experiments/diff-coder/profiles/*.json`（如 `kernelbench_triton_cuda_repair_ab.json`）；若有 `profiles/profile.schema.json` 一并维护；`experiments/diff-coder/README.md`、`DEEPSEEK_API_SETUP.md`、本 `EXPERIMENT_REPORT.md`；bench 入口常为 `akg_agents/scripts/benchmark_local_kernelbench_repeat.sh`（由 profile `benchmarks[].script` 引用） |
| **B（需评审 + 灰度）** | 默认是否开 diff、路由/启发式、与选模联动 | 行为变更 | `python/akg_agents/core/agent/coder.py`（diff/全量修复路径）；`python/akg_agents/op/langgraph_op/nodes.py`、`routers.py`、`task.py`（节点与路由）；`python/akg_agents/op/workflows/coder_only_workflow.py`；默认环境/模型可读 `python/akg_agents/core_v2/config/settings.py`、`python/akg_agents/core_v2/llm/factory.py`；选模相关以仓库内 `git grep model_selector` 命中文件为准 |

**原则**：阶段 A 合入后，任何人应能用同一 profile 在本地/CI 跑出 **与本文同口径** 的聚合；阶段 B 再讨论「默认开/关」与分流规则。

### 2. 默认行为与兼容性

- **`AIKG_TARGETED_REPAIR`**：在未完成阶段 B 前，**保持与当前线上一致**（通常为关或现有默认）；不在 profile 里「悄悄改成 1」作为默认。  
- **显式开关**：若未来增加「按 benchmark 自动切换」，建议仍保留 **环境变量 / profile 覆盖** 作为逃生口，便于对照与排障。  
- **与选模解耦**：`model_selector`、Ascend 166/249 扫库类变更 **不依赖** 本实验结论即可独立合入；本实验文档中写明 **不评测选模**，避免评审时范围蔓延。

**涉及文件**：默认 env 常在 **`experiments/diff-coder/profiles/*.json`** 的 `repair_variants[].extra_env` / `extra_env`；若代码内有硬编码默认，搜 **`AIKG_TARGETED_REPAIR`**（`git grep -n AIKG_TARGETED_REPAIR python/akg_agents`）并改对应 `.py`；选模独立 MR 则只改其自身模块（`git grep model_selector` 列出文件）。

### 3. 分流策略（结合本报告数据的落地建议）

在尚无线上全量分布前，可用 **静态规则 + 可观测信号** 组合，后续再数据驱动迭代。

**3.1 静态启发式（与上表一致）**

- **优先倾向「diff 开」**：`test_target` 指向 **level2 / 高成本** 路径时（本数据中 **`conv2d` 全部 `fail_mode`** diff 开均占优），可作为默认候选。  
- **优先倾向「diff 关」或缩短 repair**：**`mm` + `syntax` / `type` / `name`**（本数据中 diff 开墙钟与 LLM 均变差）；可先关 diff 或 **降低 `AIKG_TARGETED_REPAIR_ATTEMPTS`**，观察是否仍满足通过率。  
- **`batchnorm` + `name`**：本数据为反例，但 **标签可能与真实题面不一致**；在 **§6 审计完成前**，不建议单独为「batchnorm」写死全局规则，避免误伤真实 BatchNorm 题。

**涉及文件**：分流规则若写在配置层 → **`experiments/diff-coder/profiles/*.json`**（按 `benchmarks[].id` / `repair_variants` 拆 profile 或多 profile）；若写在代码层 → **`python/akg_agents/op/langgraph_op/routers.py`** 或 **`nodes.py`** 中在进入 Coder/Fix 前设置 `os.environ` / `task` 字段；repair 轮次 → profile `extra_env` 中的 **`AIKG_TARGETED_REPAIR_ATTEMPTS`**（或代码中读取该变量的位置，以 `git grep` 为准）。

**3.2 动态信号（实现侧可预留钩子）**

- 首轮失败后若 **`llm_repair_sum_s` 相对 `llm_first_sum_s` 占比** 持续高于某阈值（需在你们环境标定），可 **下一轮切到 diff 开**；反之轻量 case 可维持关。  
- 若后续接入 **token 计数**，用 **repair 段 token / 总 token** 替代「仅 wall 时间」做分流，更接近成本。

**涉及文件**：读 **`summary.tsv` / timing** 的逻辑若在矩阵侧 → `run_matrix.py` 或 `summarize` 子命令所在模块；若在 Agent 运行时动态切 → **`python/akg_agents/op/langgraph_op/task.py`**（`LangGraphTask` 一轮结束后的决策）、**`nodes.py`**（Fix/Coder 节点之间）；token 累计 → 通常在 **`python/akg_agents/core_v2/agents/base.py`** 的 `run_llm` / 包装层，以 `git grep run_llm` 调用栈为准。

**3.3 与产品沟通的最小表述**

- 「**复杂 / level2 类** 更可能从 diff 开受益；**简单 smoke + 部分语法/类型类错误** 可能 diff 开更慢，需要分流或关。」

**涉及文件**：可仅更新 **`EXPERIMENT_REPORT.md` / `README.md`** 的对外摘要，无需改代码。

### 4. 配置与可观测性（避免「不可比」）

- **Profile 即契约**：在 `profiles/kernelbench_triton_cuda_repair_ab.json`（或等价文件）中 **写全** 本实验依赖项：`repair_variants`、`fail_modes`、`repeats`、`benchmarks[].test_target`、`env_file`、Docker 镜像名、`AIKG_FORCE_*`、`AIKG_TARGETED_REPAIR*`、bench 超时、`AIKG_BENCHMARK_LOCAL_TIMING_JSONL` 等；MR 描述中附 **「与某次报告对齐的 profile 提交 hash」**。  
- **文档交叉引用**：在 `README.md` 或 `DEEPSEEK_API_SETUP.md` 中说明 **`AKG_AGENTS_MODEL_NAME` 与 `AIKG_MODEL_NAME` 的覆盖关系**（避免计费/日志模型名与 profile 不一致）。  
- **聚合口径写死**：对外说明 **只统计 `attempt=final` 且 `counted_for_avg=1`**；若 `summary.tsv` 列名变更，需同步更新本报告与聚合脚本。

**涉及文件**：**`experiments/diff-coder/profiles/*.json`**（主契约）；**`README.md`、`DEEPSEEK_API_SETUP.md`**；写出 `summary.tsv` / `timing_*.jsonl` 列名的代码 → 一般在 **bench 脚本** `akg_agents/scripts/benchmark_local_kernelbench_repeat.sh` 及其调用的 **pytest / task** 路径（`git grep summary.tsv` 或 `counted_for_avg` 定位）；计时上下文 README 常指向 **`python/akg_agents/utils/llm_timing_context.py`**（若路径变更以 `git grep llm_timing` 为准）+ **`python/akg_agents/core_v2/agents/base.py`**（`run_llm`）+ **`python/akg_agents/op/langgraph_op/task.py`**（汇总写 jsonl）；**`run_matrix.py` 内 `summarize`** 若含聚合脚本需同步列名。

### 5. 回归与质量门禁

- **最小子集**：至少保留 **`conv2d × 1～2 个 fail_mode` + `mm × syntax`** 作为每次发布前或 nightly 的 **smoke 矩阵**（覆盖「大赚」与「小亏」两类）。  
- **阈值建议**：对 `wall_seconds`、`llm_sum_s`、`llm_repair_sum_s` 设 **相对基线 ±X%** 或绝对上限（X 由你们历史方差定）；超阈 **阻断发布或仅告警**，二选一写入发布 checklist。  
- **通过率**：本实验 `pass_pct` 在注入条件下为 100%；若作为门禁，应 **单独定义「成功」**（例如 final pass + 无超时），并在注释中说明 **与无注入生产成功率不同**）。

**涉及文件**：CI 流水线 → 仓库 **`.github/workflows/*.yml`**、**`.gitcode/pipelines/*.yml`** 或你们实际使用的 Jenkinsfile（`git grep run_matrix` 定位）；smoke 可 **新增/裁剪 profile** → `experiments/diff-coder/profiles/*.json`；阈值脚本可放在 **`experiments/diff-coder/`** 下小工具或 `akg_agents/scripts/`，与 MR 一并说明入口命令。

### 6. 标签与题面审计（可开独立 issue）

1. 打开 `profiles/...` 中每个 `benchmarks[].id` 与对应 **`test_target`**。  
2. 在 `tests/op/bench/test_bench_triton_cuda.py`（或实际路径）中核对 **`get_kernelbench_op_name([...])` 或 level2 列表** 与 **算子名**。  
3. 对 **`batchnorm`**：要么改题号与 `33_BatchNorm` 一致，要么将 profile id 改为 **`relu_smoke`** 等 **名实一致** 的名称，并在看板中替换旧标签。  
4. 审计结果写入 **一行表格**（id / 测试文件 / KernelBench 题号 / 备注）放在 `README.md` 或 `EXPERIMENT_REPORT` 附录链接，避免口头约定。

**涉及文件**：**`experiments/diff-coder/profiles/*.json`**（`benchmarks[].id` / `test_target`）；**`tests/op/bench/test_bench_triton_cuda.py`**（KernelBench 题号与 marker）；审计表落 **`README.md`** 或本报告新增小节。

### 7. `out/`、Docker 与本地开发体验

- **默认输出路径**：矩阵 slug 形如 `out/diffcoder/...`；在文档中说明 **与 `workspace_root` 的关系**，避免贡献者找不到 `summary.tsv`。  
- **root 写卷**：若容器以 root 写 bind mount，宿主机 `git clean` / `rebase` 可能 **Permission denied**；推荐在文档中写明 **任选其一**：(a) 容器内指定非 root user；(b) 跑完后对输出目录 `sudo chown`；(c) 将 `out` 指到 **仓库外** 专用目录并在 `.gitignore` 已忽略。  
- **CI**：若 CI 内跑矩阵，**产物目录** 应落在 workspace 子路径且 **job 结束清理**，避免磁盘涨满。

**涉及文件**：profile 里 **`out_dir_prefix` / `workspace_root`** → **`experiments/diff-coder/profiles/*.json`**；Docker 镜像名 → 同上 profile；**仓库根 `.gitignore`**（确保 `out/` 或自定义产物目录被忽略）；运维说明 → **`README.md`** 或 **`docs/`** 下贡献指南（若项目有 `CONTRIBUTING.md` 可链过去）；CI job 的 `rm -rf` / volume → **`.github/workflows/*`、`.gitcode/*` 或 Jenkinsfile**。

### 8. 对外汇报与 MR 描述模板（可直接复用）

- **必须呈现**：按 **benchmark** 分组的表；若篇幅允许，附 **本报告 §「同一 (benchmark, fail_mode) Δ%」** 表或链接。  
- **禁止单独使用**：仅一句「平均墙钟降 45%」而无 **分层** 与 **实验条件**（注入失败、CUDA KernelBench、模型名）。  
- **MR 检查项**：是否说明 **profile 版本 / 模型 / repeats**；是否声明 **不覆盖 Ascend 全库**。

**涉及文件**：仅 **MR 描述 + 本 `EXPERIMENT_REPORT.md`**（可选在 `README.md` 加一句「对外汇报见报告 §8」）；无强制代码路径。

### 9. 后续扩展（非阻塞本 MR）

- **Triton-Ascend / NPU**：在等价注入与计时字段就绪后，**平行跑一小矩阵**，验证 CUDA 上结论是否迁移；再决定是否与 `model_selector` 联合优化。  
- **线上分流**：积累 **真实 fail_mode 分布** 与 **repair 占比** 后，用数据替换 §3.1 的静态规则，并保留 profile 级 A/B 开关做金丝雀。

**涉及文件**：Ascend 侧通常 **新增 profile**（`experiments/diff-coder/profiles/` 或 `akg_agents/.../op/config/*.yaml`）+ **对应 pytest / bench 脚本**；与 `model_selector` 联动 → 以 **`git grep model_selector`** 命中模块为准，再复制一版矩阵 `repair_variants` 做金丝雀。

---

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-05-14 | 首版：完整体例与复现、附录脚本 |
| 2026-05-14 | 精简版：实验说明摘要 + GitCode 链接 + 观察/结论/建议；数值与首版同一批 `summary.tsv` 聚合 |
| 2026-05-14 | 扩充「推荐修改方案」：合入顺序、分流启发式、回归基线、`out/` 权限与汇报口径 |
| 2026-05-14 | 「推荐修改方案」细化：分阶段 MR、分流信号、配置契约、回归门禁、审计 checklist、运维与汇报模板、后续扩展 |
| 2026-05-14 | 「推荐修改方案」各条补充 **涉及文件/目录** 与路径约定；对分支差异处注明 `git grep` 核对 |

---

## 许可

与仓库一致：以各文件 SPDX / 仓库根 LICENSE 为准。
