[English Version](../SkillEvolution.md)

# Skill 自进化系统

## 1. 概述

Skill 自进化系统是一个通用的经验提取框架：从 Agent 的运行日志和交互记录中自动提炼可复用的优化知识，生成结构化的 `SKILL.md` 文档，使 Agent 具备"实践 → 总结 → 复用"的闭环学习能力。

当前实现聚焦于算子层，以 SubAgent 形式注册为 `call_skill_evolution` 工具，由 `KernelAgent` 在算子生成/优化流程中调用。

**四种模式**：
- **search_log**：从搜索日志中提取进化链 diff —— 自动化优化模式
- **expert_tuning**：从对话历史中提取人工调优经验 —— "用户建议 → 代码变更 → 性能变化"因果链
- **error_fix**：从错误修复记录中提取调试经验 —— "错误类型 → 修复策略"
- **merge_skills**：将同 DSL 下的 evolved skills 按主题合并去重

**目标**：将自动搜索日志、人工调优经验和错误修复经验统一转化为结构化的优化知识，供后续算子生成时参考。

**架构**：`SkillEvolutionBase`（`core_v2/agents/`）提供工作区管理和日志工具等通用能力，`SkillEvolutionAgent`（`op/agents/`）继承基类并实现算子特有的四种模式逻辑。

## 2. search_log 模式

### 2.1 数据来源

系统从 `logs/` 目录中读取 3 个文件：

| 文件 | 内容 | 关键字段 |
|------|------|---------|
| `verification_results.jsonl` | 验证记录 | `task_id`, `passed`, `verify_dir`, `dsl`, `backend`, `arch` |
| `{op}/profiling/speed_up_record.txt` | 性能记录 | `task_id`, `generation_time`, `speedup` |
| `{op}_lineage_graph.md` | 进化树表格 | `task_id`, `parent_id`, `generation` |

对每个通过验证的任务，从 `verify_dir/*_impl.py` 读取实现代码。

### 2.2 流程

```
1. collect   — 解析 3 个文件 + 读取 impl 代码 → List[TaskRecord]
2. compress  — 建进化树 → 每条路径单调栈 → 注释剥离 → diff
3. LLM      — 最佳代码 + 进化链 diff → 生成 SKILL.md 正文
4. Writer   — YAML frontmatter + 正文 → 写入 SKILL.md
```

### 2.3 数据收集

`collect(log_dir, op_name) -> (records, metadata)`

- 解析 `verification_results.jsonl` 获取 passed 任务及其 `verify_dir`
- 解析 `speed_up_record.txt` 获取 `gen_time` 和 `speedup`
- 解析 `lineage_graph.md` 表格获取 `parent_id` 和 `generation`
- 从每个任务的 `verify_dir` 读取 `*_impl.py` 作为代码
- 返回 `TaskRecord` 列表和环境元数据（`dsl`, `backend`, `arch`）

### 2.4 数据压缩

`compress(records, metadata) -> CompressedData`

**最佳方案**：`gen_time` 最小（执行最快）的记录，完整代码注入 prompt。

**单调栈进化链**：

1. 从 `parent_id` 关系重建进化树
2. DFS 收集所有根→叶路径
3. 对每条路径维护单调栈：只保留 `gen_time` 严格递减（性能严格递增）的节点
4. 对单调栈中相邻节点，注释剥离后生成 unified diff
5. 过滤掉 `MIN_GEN_TIME_IMPROVE_PCT < 0.01` 的微小改进

注释剥离（`strip_comments`）在 diff 前移除 docstring、纯注释行和行内注释，消除注释改写带来的噪声。

### 2.5 LLM 分析

Jinja2 模板（`analyze_search_log.j2`）向 LLM 注入：
- 算子信息（名称、DSL、后端、架构）
- 最佳实现方案（完整代码 + gen_time + speedup）
- 进化链 diff（注释剥离、单调栈过滤后的）
- 性能数据摘要

LLM 直接生成 SKILL.md 正文（Markdown 格式），前两行为 `skill_name` 和 `description`。

**生成目标**：提炼可迁移的通用优化方法论，而非描述当前算子的特性。文档结构为：任务特征（界定问题类别）→ 优化方法（每个方法独立成节，说明条件、做法、原理）→ 适用边界。

### 2.6 写入

组装 YAML frontmatter（name, description, category, backend, dsl, source）+ LLM 正文 → 写入 `op/resources/skills/{dsl}/cases/{skill_name}/SKILL.md`。

**命名规则**：`skill_name` 遵循 `{dsl}-case-{算子类别}-{优化特征}` 格式，例如 `triton-ascend-case-reduction-amin-large`、`triton-ascend-case-elemwise-broadcast-3d`。`category` 统一为 `example`，`source` 为 `search_log`。

### 2.7 核心算法：单调栈

```
原始路径: A(17us) → B(8us) → C(9us) → D(8us)
单调栈:   A(17us) → B(8us)
diff 对:  (A→B) — 唯一性能严格提升的 pair
```

- 性能比较优先用 `gen_time`（越小越好）
- `seen_pairs` 集合防止多条路径共享前缀时重复生成 diff
- `MIN_GEN_TIME_IMPROVE_PCT = 0.01` 过滤掉微小改进

## 3. expert_tuning 模式

从对话历史中提取人工调优经验，生成 SKILL.md。

**适用场景**：用户在对话中手动指导优化（如 "调大 BLOCK_SIZE"、"增加 num_warps"），希望将这些经验沉淀为可复用的 skill。

### 3.1 数据来源

收集器优先读取 `{conversation_dir}/trace.json` 获取对话树结构，通过 DFS 找到所有 root→leaf 路径，每条路径作为一个独立分支。每个分支按路径顺序读取对应节点的 `actions/action_history_fact.json`。

若 `trace.json` 不存在，回退到按节点编号排序（node_001, node_002, ...）。

**多分支处理**：对话树可能包含多个分支（例如 root→node_001...019 和 root→node_020...029），每个分支作为独立的时间线段输出，用分支标题分隔。分支内包含完整的共享前缀节点，确保每个分支的因果链完整。单分支场景（线性链）不添加额外标签，行为与之前一致。

```
root → node_001 → ... → node_019   (分支 1)
     → node_020 → ... → node_029   (分支 2)

输出:
  ## 分支 1 (共 19 个节点)
  ### node_001 Turn 1 — ask_user ...
  ...
  ## 分支 2 (共 10 个节点)
  ### node_020 Turn 1 — ask_user ...
  ...
```

### 3.2 增量式 LLM 压缩

收集器将每个 action 格式化为一个 section，然后增量构建时间线：

```
accumulated = ""
for section in sections:
    if len(accumulated + section) > 阈值(60000字符):
        accumulated = LLM压缩(accumulated)  // 压缩已积累的历史
    accumulated += section                   // 追加新 section
```

每次添加新 section 前检查总长度，超出阈值时先用 LLM 压缩已积累的部分，再追加新内容。这样无论对话多长都能处理。

**LLM 压缩保留原则**（通过 prompt 约束）：

| 类别 | 处理方式 |
|------|---------|
| 用户回复（优化建议） | **完整保留** |
| 代码生成工具产出的完整代码 | **完整保留** |
| 性能数据（gen_time、speedup 等） | **完整保留** |
| 各工具执行状态 | **完整保留** |
| Agent 消息（解释、确认提问） | 可压缩或删除 |
| 工具冗余参数 | 可压缩或删除 |
| 重复的错误信息 | 可压缩或删除 |

### 3.3 流程

```
1. collect          — 从 trace.json DFS 获取分支路径 → 按路径顺序读取 action → 格式化为 section 列表
2. build_timeline   — 增量拼接 section，超阈值时 LLM 压缩已积累部分
3. LLM 分析        — 时间线 → 自行分析因果链 → 生成 SKILL.md 正文
4. Writer           — YAML frontmatter + 正文 → 写入 SKILL.md
```

### 3.4 收集与时间线构建

`collect(conversation_dir, op_name) -> (sections, metadata)`

**职责**：读取对话树结构并格式化。优先从 `trace.json` 获取树结构，DFS 找到所有 root→leaf 路径；若无 `trace.json` 则回退到按节点编号排序。返回 section 列表（多分支时包含分支标题），不做分析或压缩。

`build_timeline(sections, llm_fn, max_chars=60000, work_dir="") -> str`

**职责**：增量拼接 section 并按需压缩，返回最终时间线文本。`work_dir` 可选，指定时会输出压缩前后的中间文件用于调试。

### 3.5 LLM Prompt

模板 `analyze_expert_tuning.j2` 将时间线注入，指示 LLM：

1. 自行识别哪些轮次包含实质性优化建议（忽略"确认"等消息）
2. 自行匹配代码版本和性能数据（profile_kernel 对应之前最近的代码生成）
3. 提炼"用户建议 → 代码变更 → 性能变化"的因果链
4. 生成 SKILL.md 正文

**命名规则**：`skill_name` 遵循 `{dsl}-exp-{算子类别}-{调优特征}` 格式，`source` 为 `expert_tuning`。

## 4. error_fix 模式

从搜索日志中提取"失败→成功"的错误修复记录，并将其持续沉淀到一个调试经验 `SKILL.md` 中。

**适用场景**：代码生成过程中多次失败后被成功修复，希望将这些调试经验沉淀为可复用的 skill，帮助后续生成阶段避免同类错误。

### 4.1 数据来源

与 `search_log` 模式共用同一个 `logs/` 目录，但关注不同的信息：

| 文件 | 内容 | 关键字段 |
|------|------|---------|
| `verification_results.jsonl` | 验证记录（含失败和成功） | `task_id`, `passed`, `verify_dir`, `step` |
| `verify_dir/*_impl.py` | 失败/成功代码 | 代码内容 |

**数据提取逻辑**：对每个 Task，按 step 排序其验证记录，找到第一个 `passed=true` 的条目。取其之前最后一个 `passed=false` 的条目作为"失败版本"。提取失败代码、成功代码和错误日志。

```
Task 验证序列: step2(fail) → step5(fail) → step8(fail) → step11(pass)
提取: failed_code=step8, success_code=step11, error_log=step8的错误
```

**Diff 完整性**：error_fix 模式生成的 diff 不截断（与 search_log 的 200 行上限不同），确保 LLM 看到完整的代码变更。

**多 workflow 兼容**：error_fix 模式仅依赖 `verification_results.jsonl` 和 `verify_dir`，`task_id` 只作为分组 key 使用，因此同时兼容 adaptive_search（如 `_Gen1_Task3`）、evolve（如 `1_Island1_Task0`）和 kernelgen（如 `0`）三种 workflow 的日志格式。

### 4.2 流程

```
1. collect         — 解析 verification_results.jsonl → 对每个 Task 找 fail→success 对
                     → 读取失败/成功代码 + 错误日志 → 完整 diff（不截断）
2. LLM 分析       — 修复案例（error_log + diff）→ 生成错误修复经验
3. LLM 去重       — 若已有 SKILL.md，注入已有内容和新生成内容，LLM 只输出不重复的新增条目
4. Writer          — 首次运行创建 `error-fix/SKILL.md`，后续运行追加去重后的增量内容
```

### 4.3 数据收集

`collect(log_dir, op_name) -> (records, metadata)`

- 解析 `verification_results.jsonl`，按 `task_id` 分组
- 对每个 Task 按 step 排序，找到第一个成功前的最后一个失败
- 从对应 `verify_dir` 读取失败和成功版本的 `*_impl.py`
- 读取失败步骤的 `error_log`（截取末尾 1000 字符）
- 生成失败→成功的 unified diff（完整，不截断）
- 返回 `SuccessfulFixRecord` 列表和环境元数据

**数据结构**：

```python
@dataclass
class SuccessfulFixRecord:
    task_id: str
    op_name: str
    error_log: str       # 截取的错误日志
    error_step: int      # 失败步骤编号
    failed_code: str     # 失败版本代码
    success_code: str    # 成功版本代码
    diff: str            # unified diff（完整，不截断）
    dsl: str
    backend: str
    arch: str
```

### 4.4 LLM Prompt

模板 `analyze_error_fix.j2` 向 LLM 注入所有修复案例（每个案例包含错误日志和完整代码 diff），不注入 conductor 建议。

LLM 任务：
1. 归类常见错误（简短标题）
2. 每种错误只写**报错特征**和**修复方法**，附简短代码对比
3. 合并同类项，聚焦可迁移的通用修复方法

### 4.5 去重与写入

**去重**（`dedup_error_fix.j2`）：当已有 `error-fix/SKILL.md` 时，将已有正文和新生成内容一起注入 LLM，让 LLM 判断哪些条目是新的、哪些已存在。LLM 只输出不重复的增量内容。如果全部重复，输出"无新增内容"，跳过写入。

**写入**（`SkillWriter.write_error_fix`）：

- 固定 skill 目录名：`error-fix`
- 默认输出路径：`op/resources/skills/{dsl}/evolved/error-fix/SKILL.md`
- 如果传入 `--output-dir DIR`：输出到 `DIR/error-fix/SKILL.md`
- 文件不存在时新建（含固定 frontmatter：`name: error-fix`、`category: example`、`metadata.source: error_fix`）
- 文件已存在时追加去重后的增量内容（保留原有 frontmatter 和正文）

```
第 1 次运行: LLM 生成 → 新建 SKILL.md
第 2 次运行: LLM 生成 → 对比已有 → 只追加新增条目
第 N 次运行: 同上，持续累积不重复的调试经验
```

## 5. merge_skills 模式

将同一 DSL 下的多个 evolved skills 按优化主题合并去重，减少重复文档。

**适用场景**：经过多次 `search_log` 和 `expert_tuning` 生成后，evolved 目录下积累了大量 skill，不同算子的 skill 包含相似的优化手段。合并后可以减少重复注入，使描述更泛化。

### 5.1 设计约束

直接将所有 skill 全文注入单次 LLM 调用会导致上下文过长。因此采用**两阶段策略**：
1. **摘要聚类**：LLM 仅看 `name + description`（每个 ~100 字），不传全文
2. **逐簇合并**：按簇独立调用 LLM，每次只传同主题的少量 skill 全文

### 5.2 流程

```
1. scan            — 扫描 evolved/ 下所有 SKILL.md（排除 error-fix/ 和 .archive/）
2. classify        — 提取 name + description → LLM 按优化主题聚类（仅摘要，不传全文）
3. merge per-cluster — 对每个 >=2 个 skill 的簇，传入全文让 LLM 合并去重
                       大簇（>5 个）自动拆分为子批次，滚动合并
4. archive + write — 原始 skill 归档到 .archive/{timestamp}/，写入合并后的 SKILL.md
```

### 5.3 聚类阶段

`classify_skills.j2` 模板仅注入 skill 的 `name` 和 `description`，要求 LLM 按优化主题分组并给出分类原因。输出 JSON 格式：

```json
{
  "clusters": [
    {"reason": "这些 skill 都涉及内存访问模式优化和带宽利用率提升", "skills": ["skill-a", "skill-c"]},
    {"reason": "这些 skill 聚焦于计算块大小调优和寄存器压力控制", "skills": ["skill-b"]}
  ]
}
```

### 5.4 合并阶段

对每个包含 >= 2 个 skill 的簇：

- `merge_cluster.j2` 注入同簇 skill 的全文和 DSL 前缀，要求 LLM 去重合并、泛化（去除特定算子名称）、统一结构，并产出 `skill_name`（格式 `{dsl}-merged-{调优特征}`）和 `description`
- 大簇保护：超过 5 个 skill 时自动拆分为子批次，先合并前 5 个，再将结果作为"已合并文档"与下一批合并
- 只有 1 个 skill 的簇保持原样

### 5.5 合并后的 SKILL.md

合并后的 skill 名称和描述由合并阶段的 LLM 产出，frontmatter 示例：

```yaml
---
name: triton-cuda-merged-memory-access-optimization
description: "合并后的优化方法论描述..."
category: example
metadata:
  source: merged
  backend: cuda
  dsl: triton-cuda
---
```

- `name` 和 `description` 由合并阶段 LLM 生成
- `source` 设为 `merged`

### 5.6 增量合并

后续新生成的 skill 照常写入 `evolved/`。下次运行 `merge_skills` 时：
- 已合并的 merged skill 和新增的单个 skill 都参与聚类
- 如果新 skill 被聚入已有 merged skill 的簇 → 以 merged 为基底增量合并
- 如果新 skill 自成一簇 → 保留为独立 skill

## 6. 文件结构

```
core_v2/agents/
└── skill_evolution_base.py     — SkillEvolutionBase 基类（工作区管理、日志工具）

op/tools/skill_evolution/
├── common.py                   — 公共类型、工具函数、LLM 输出解析、SKILL.md 写入
├── search_log_utils.py         — search_log 模式：collect + compress + to_prompt_vars
├── expert_tuning_utils.py      — expert_tuning 模式：collect + build_timeline + to_prompt_vars
├── error_fix_utils.py          — error_fix 模式：collect + to_prompt_vars
├── merge_utils.py              — merge_skills 模式：扫描、聚类解析、归档、合并写入
└── __init__.py

op/agents/skill_evolution_agent.py — SkillEvolutionAgent（继承基类，四模式分发）

op/resources/prompts/skill_evolution/
├── analyze_search_log.j2       — search_log: 结构化进化链 diff → LLM
├── analyze_expert_tuning.j2    — expert_tuning: action 时间线 → LLM
├── analyze_error_fix.j2        — error_fix: 修复案例 → LLM
├── dedup_error_fix.j2          — error_fix: 已有内容 + 新内容 → LLM 去重，输出增量
├── classify_skills.j2          — merge_skills: name + description → LLM 聚类
└── merge_cluster.j2            — merge_skills: 同簇 skill 全文 → LLM 合并去重

examples/kernel_related/skill_evolution/
├── run_skill_evolution.py      — 独立 CLI 脚本（不依赖 Agent 框架）
├── run_ab_test.py              — A/B 测试批量运行器
├── ab_test_utils.py            — A/B 测试工具函数
└── tracking.md                 — 实验结果跟踪文档
```

## 7. 独立 CLI 脚本

`examples/kernel_related/skill_evolution/run_skill_evolution.py` 提供不依赖 Agent 框架的独立入口。

```bash
# search_log 模式
python examples/kernel_related/skill_evolution/run_skill_evolution.py search_log /path/to/logs relu

# expert_tuning 模式
python examples/kernel_related/skill_evolution/run_skill_evolution.py expert_tuning ~/.akg/conversations/cli_xxx relu

# error_fix 模式
python examples/kernel_related/skill_evolution/run_skill_evolution.py error_fix /path/to/logs matmul

# merge_skills 模式
python examples/kernel_related/skill_evolution/run_skill_evolution.py merge_skills triton_cuda
python examples/kernel_related/skill_evolution/run_skill_evolution.py merge_skills triton_cuda --skills-dir /path/to/evolved -o ./merged

# 指定输出目录和模型级别
python examples/kernel_related/skill_evolution/run_skill_evolution.py error_fix /path/to/logs matmul -o ./output -m complex
```

| 参数 | 说明 |
|------|------|
| `mode` | `search_log`、`expert_tuning`、`error_fix` 或 `merge_skills` |
| `log_dir` / `conversation_dir` | 日志目录（search_log / error_fix）或对话目录（expert_tuning） |
| `op_name` | 算子名称（如 relu、l1norm、matmul） |
| `-o / --output-dir` | SKILL.md 输出目录|
| `-m / --model-level` | LLM 模型级别（默认 standard） |

## 8. 工作区

Agent 模式下保存在 `{cur_path}/logs/skill_evolution/`；CLI 模式下默认保存在 `~/.akg/skill_evolution/{mode}_{op_name}/`（可通过 `-o` 覆盖）：

**search_log 模式：**

| 文件 | 内容 |
|------|------|
| `collected_data.json` | 任务记录摘要（task_id, parent_id, gen_time, speedup, has_code） |
| `compressed_data.json` | 最佳方案 + 进化链 |
| `llm_prompt.txt` | 渲染后的 LLM prompt |
| `llm_response.txt` | LLM 原始输出 |
| `session.log` | 执行日志 |
| `result.json` | 最终结果摘要 |

**expert_tuning 模式：**

| 文件 | 内容 |
|------|------|
| `action_timeline.md` | 格式化的 action 时间线（可能包含压缩标记） |
| `llm_prompt.txt` | 渲染后的 LLM prompt（含时间线） |
| `llm_response.txt` | LLM 原始输出 |
| `session.log` | 执行日志 |
| `result.json` | 最终结果摘要 |

**error_fix 模式：**

| 文件 | 内容 |
|------|------|
| `collected_fix_records.json` | 修复记录摘要（task_id, error_step, has_conductor, diff_lines） |
| `llm_prompt.txt` | 渲染后的 LLM prompt（含修复案例） |
| `llm_response.txt` | LLM 原始输出 |
| `session.log` | 执行日志 |
| `result.json` | 最终结果摘要 |

**merge_skills 模式：**

| 文件 | 内容 |
|------|------|
| `skill_summaries.json` | 所有 skill 的 name + description 摘要 |
| `classify_prompt.txt` | 聚类 LLM prompt |
| `classify_response.txt` | 聚类 LLM 输出 |
| `clusters.json` | 解析后的聚类结果 |
| `merge_{theme}_prompt.txt` | 每个簇的合并 LLM prompt |
| `merge_{theme}_response.txt` | 每个簇的合并 LLM 输出 |
| `result.json` | 最终结果摘要 |

## 9. Workflow 兼容性

不同 workflow 的日志命名差异：

| Workflow | 文件命名示例 | 特征 |
|----------|-------------|------|
| adaptive_search | `Iteration_Gen1_Task3_Step02_{op}_coder_result.txt` | 含 `Gen` + `Task` 层级 |
| evolve | `Iteration1_Island0_Task0_Step05_{op}_coder_prompt.txt` | 含 `Island` + `Task` 层级 |
| kernelgen | `Iteration0_Step01_{op}_kernel_gen_prompt.txt` | 无 Task/Island 层级 |

各模式兼容情况：

| 模式 | adaptive_search | evolve | kernelgen | 说明 |
|------|:-:|:-:|:-:|------|
| **error_fix** | Y | Y | Y | 仅依赖 `verification_results.jsonl` + `verify_dir`，与文件命名无关 |
| **search_log** | Y | - | - | 依赖 `lineage_graph.md` + `speed_up_record.txt`，目前仅 adaptive_search 产生这些文件 |
| **merge_skills** | - | - | - | 不依赖任何日志目录，处理已生成的 evolved skill 文件 |
| **expert_tuning** | - | - | - | 依赖对话目录 `trace.json` / `action_history_fact.json`|

> **注**：`search_log` 模式如需扩展到 evolve/kernelgen，需补充对应 workflow 的 lineage 和性能文件解析逻辑。`error_fix` 模式已天然兼容所有产生 `verification_results.jsonl` 的 workflow。