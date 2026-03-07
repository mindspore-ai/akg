[English Version](../SkillEvolution.md)

# Skill 自进化系统

## 1. 概述

Skill 自进化系统自动提取优化经验，生成可复用的 `SKILL.md` 文档。系统以 SubAgent 形式注册为 `call_skill_evolution` 工具，由 `KernelAgent` 调用。

**两种模式**：
- **search_log**：从搜索日志中提取进化链 diff —— 自动化优化模式
- **expert_tuning**：从对话历史中提取人工调优经验 —— "用户建议 → 代码变更 → 性能变化"因果链

**目标**：形成"优化 → 总结 → 复用"的闭环 —— 将自动搜索日志和人工调优经验统一转化为结构化的优化知识，供后续算子生成时参考。

**架构**：`SkillEvolutionBase`（`core_v2/agents/`）提供工作区管理和日志工具等通用能力，`SkillEvolutionAgent`（`op/agents/`）继承基类并实现算子特有的两种模式逻辑。

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

## 4. 文件结构

```
core_v2/agents/
└── skill_evolution_base.py     — SkillEvolutionBase 基类（工作区管理、日志工具）

op/tools/skill_evolution/
├── common.py                   — 公共类型、工具函数、LLM 输出解析、SKILL.md 写入
├── search_log_utils.py         — search_log 模式：collect + compress + to_prompt_vars
├── expert_tuning_utils.py      — expert_tuning 模式：collect + build_timeline + to_prompt_vars
└── __init__.py

op/agents/skill_evolution_agent.py — SkillEvolutionAgent（继承基类，search_log / expert_tuning 分发）

op/resources/prompts/skill_evolution/
├── analyze_search_log.j2       — search_log: 结构化进化链 diff → LLM
└── analyze_expert_tuning.j2    — expert_tuning: action 时间线 → LLM

tests/op/st/test_skill_evolution.py — 独立 CLI 脚本（不依赖 Agent 框架）
```

## 5. 独立 CLI 脚本

`tests/op/st/test_skill_evolution.py` 提供不依赖 Agent 框架的独立入口。

```bash
# search_log 模式
python akg_agents/tests/op/st/test_skill_evolution.py search_log /path/to/logs relu

# expert_tuning 模式
python akg_agents/tests/op/st/test_skill_evolution.py expert_tuning ~/.akg/conversations/cli_xxx relu

# 指定输出目录和模型级别
python akg_agents/tests/op/st/test_skill_evolution.py expert_tuning /path/to/conv relu -o ./output -m complex
```

| 参数 | 说明 |
|------|------|
| `mode` | `search_log` 或 `expert_tuning` |
| `log_dir` / `conversation_dir` | 日志目录（search_log）或对话目录（expert_tuning） |
| `op_name` | 算子名称（如 relu、l1norm） |
| `-o / --output-dir` | SKILL.md 输出目录（默认写入项目 skills 目录） |
| `-m / --model-level` | LLM 模型级别（默认 standard） |

## 6. 工作区

中间文件保存在 `{cur_path}/logs/skill_evolution/`：

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
