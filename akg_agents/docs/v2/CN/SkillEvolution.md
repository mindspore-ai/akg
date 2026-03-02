[English Version](../SkillEvolution.md)

# Skill 自进化系统

## 1. 概述

Skill 自进化系统从 `adaptive_search` 的搜索日志中自动提取优化经验，生成可复用的 `SKILL.md` 文档。系统以 SubAgent 形式注册为 `call_skill_evolution` 工具，由 `KernelAgent` 调用。

**目标**：形成"搜索 → 总结 → 复用"的闭环 —— 将搜索日志转化为结构化的优化知识，供后续算子生成时参考。

## 2. 数据来源

系统从 `logs/` 目录中读取 3 个文件：

| 文件 | 内容 | 关键字段 |
|------|------|---------|
| `verification_results.jsonl` | 验证记录 | `task_id`, `passed`, `verify_dir`, `dsl`, `backend`, `arch` |
| `{op}/profiling/speed_up_record.txt` | 性能记录 | `task_id`, `generation_time`, `speedup` |
| `{op}_lineage_graph.md` | 进化树表格 | `task_id`, `parent_id`, `generation` |

对每个通过验证的任务，从 `verify_dir/*_impl.py` 读取实现代码。

## 3. 流程

```
1. Collector  — 解析 3 个文件 + 读取 impl 代码 → List[TaskRecord]
2. Compressor — 建进化树 → 每条路径单调栈 → 注释剥离 → diff
3. LLM        — 最佳代码 + 进化链 diff → 生成 SKILL.md 正文
4. Writer     — YAML frontmatter + 正文 → 写入 SKILL.md
```

### 3.1 数据收集（Collector）

`collect(log_dir, op_name) -> (records, metadata)`

- 解析 `verification_results.jsonl` 获取 passed 任务及其 `verify_dir`
- 解析 `speed_up_record.txt` 获取 `gen_time` 和 `speedup`
- 解析 `lineage_graph.md` 表格获取 `parent_id` 和 `generation`
- 从每个任务的 `verify_dir` 读取 `*_impl.py` 作为代码
- 返回 `TaskRecord` 列表和环境元数据（`dsl`, `backend`, `arch`）

### 3.2 数据压缩（Compressor）

`compress(records, metadata) -> CompressedData`

**最佳方案**：`gen_time` 最小（执行最快）的记录，完整代码注入 prompt。

**单调栈进化链**：

1. 从 `parent_id` 关系重建进化树
2. DFS 收集所有根→叶路径
3. 对每条路径维护单调栈：只保留 `gen_time` 严格递减（性能严格递增）的节点
4. 对单调栈中相邻节点，注释剥离后生成 unified diff
5. 过滤掉 `MIN_GEN_TIME_IMPROVE_PCT < 0.01` 的微小改进

注释剥离（`_strip_comments`）在 diff 前移除 docstring、纯注释行和行内注释，消除注释改写带来的噪声。

### 3.3 LLM 分析

Jinja2 模板（`analyze.j2`）向 LLM 注入：
- 算子信息（名称、DSL、后端、架构）
- 最佳实现方案（完整代码 + gen_time + speedup）
- 进化链 diff（注释剥离、单调栈过滤后的）
- 性能数据摘要

LLM 直接生成 SKILL.md 正文（Markdown 格式），前两行为 `skill_name` 和 `description`。

**生成目标**：提炼可迁移的通用优化方法论，而非描述当前算子的特性。文档结构为：任务特征（界定问题类别）→ 优化方法（每个方法独立成节，说明条件、做法、原理）→ 适用边界。

### 3.4 写入（Writer）

组装 YAML frontmatter（name, description, category, backend, dsl）+ LLM 正文 → 写入 `op/resources/skills/{dsl}/cases/{skill_name}/SKILL.md`。

**命名规则**：`skill_name` 遵循 `{dsl}-case-{算子类别}-{优化特征}` 格式，例如 `triton-ascend-case-reduction-amin-large`、`triton-ascend-case-elemwise-broadcast-3d`，与手写 cases 目录保持一致。`category` 统一为 `example`。

## 4. 核心算法：单调栈

```
原始路径: A(17us) → B(8us) → C(9us) → D(8us)
单调栈:   A(17us) → B(8us)
diff 对:  (A→B) — 唯一性能严格提升的 pair
```

- 性能比较优先用 `gen_time`（越小越好）
- `seen_pairs` 集合防止多条路径共享前缀时重复生成 diff
- `MIN_GEN_TIME_IMPROVE_PCT = 0.01` 过滤掉微小改进

## 5. 文件结构

```
op/tools/skill_evolution/
├── models.py      — TaskRecord, EvolutionStep, CompressedData
├── collector.py   — collect(log_dir, op_name) → (records, metadata)
├── compressor.py  — compress(records, metadata) → CompressedData
├── analyzer.py    — prompt 变量转换 + LLM 输出解析
├── writer.py      — YAML frontmatter + 正文 → SKILL.md
└── __init__.py

op/agents/skill_evolution_agent.py — Agent 编排

op/resources/prompts/skill_evolution/analyze.j2 — LLM prompt 模板

tests/op/st/test_skill_evolution.py — 独立 CLI 脚本（不依赖 Agent 框架）
```

## 6. 独立 CLI 脚本

`tests/op/st/test_skill_evolution.py` 提供不依赖 Agent 框架的独立入口，直接复用 collect → compress → LLM → write 流水线。

```bash
# 基本用法
python akg_agents/tests/op/st/test_skill_evolution.py <log_dir> <op_name>

# 指定输出目录和模型级别
python akg_agents/tests/op/st/test_skill_evolution.py /path/to/logs relu -o ./output -m complex
```

| 参数 | 说明 |
|------|------|
| `log_dir` | adaptive_search 的日志目录（节点 logs 路径） |
| `op_name` | 算子名称（如 relu、l1norm） |
| `-o / --output-dir` | SKILL.md 输出目录（默认写入项目 skills 目录） |
| `-m / --model-level` | LLM 模型级别（默认 standard） |

## 7. 工作区

中间文件保存在 `{cur_path}/logs/skill_evolution/`：

| 文件 | 内容 |
|------|------|
| `collected_data.json` | 任务记录摘要（task_id, parent_id, gen_time, speedup, has_code） |
| `compressed_data.json` | 最佳方案 + 进化链 |
| `llm_prompt.txt` | 渲染后的 LLM prompt |
| `llm_response.txt` | LLM 原始输出 |
| `session.log` | 执行日志 |
| `result.json` | 最终结果摘要 |
