# AutoResearch

AutoResearch 是内核优化 agent：给定参考实现和 seed 内核，它在 eval 预算内
迭代编辑内核并逐轮评估，由结构化 plan 和 skill 目录引导方向。所有状态
（plan、skill、counters、op_summary.md、plan_analysis.md）均会持久化，
因此任一 run 都可恢复。

本文档是 `python/akg_agents/op/autoresearch/` 模块的设计契约，描述运行时
架构、组件间的数据流，以及保证主循环终止、上下文有界的不变量。代码库
自己能 grep 到的实现细节（函数体、config 默认值）不在范围内。

---

## 1. 契约

| | |
|---|---|
| 工具名 | `call_autoresearch_workflow` |
| 场景 | 针对已有内核、已知参考实现的迭代优化 |
| 输入 | `op_name`、`task_desc`（`Model/get_inputs/get_init_inputs` 形式的参考实现）、`dsl`、`framework`、`backend`、`arch`、可选 `previous_code`、可选 `max_rounds`（默认 20） |
| 输出 | 优化后的内核 + profile 结果（延迟、相对参考的 speedup） |
| 终止 | eval 预算耗尽、终态 LLM 错误、压缩恢复失败，或 agent 显式调用 `finish` |

task 参数允许的取值（backend / dsl / arch 枚举）定义在仓库根的
`AGENTS.md`。Preflight fail-closed，任何校验失败即中止；见 §3。

### 1.1 何时选用

AutoResearch 是"深度迭代优化"这条路。以下场景应选其它 workflow：

| 能力 | KernelGenOnly | Evolve / AdaptiveSearch | **AutoResearch** |
|---|---|---|---|
| 策略 | 单次生成 | 基于种群 | agent 驱动的 ReAct |
| 迭代次数 | 1 | 多（并行） | 多（顺序） |
| 自主性 | 无 | 模板驱动 | 完全自主：自行规划、读文档、自诊断 |
| 适用 | 初稿 | 广度探索 | 深度调优 |
| 失败恢复 | 无 | 种群吸收 | diagnose subagent + agent 自承认 |

经验法则：初稿用 KernelGen；想并行铺量用 Evolve / AdaptiveSearch；已有
正确内核再想深挖时才上 AutoResearch。

---

## 2. 架构

```
                 call_autoresearch_workflow
                             │
                             ▼
     ┌──────────────────── AgentLoop ────────────────────┐
     │                                                    │
     │   [Preflight] ─► [Baseline eval] ─► [Turn loop]   │
     │                                                    │
     │       ┌───────────── Turn loop ──────────────┐    │
     │       │                                       │    │
     │   LLM call ─► TurnExecutor ─► post-eval hook │    │
     │                    │                          │    │
     │                    ├── tools ─► edit / plan / │    │
     │                    │           skill search / │    │
     │                    │           compact / fin  │    │
     │                    │                          │    │
     │                    └── eval (quick_check →    │    │
     │                           run eval script →   │    │
     │                           git commit if KEEP) │    │
     │       │                                       │    │
     │       └── on failure / no_edit / threshold ─► │    │
     │                   Diagnose subagent            │    │
     └────────────────────────────────────────────────────┘

状态所有者（每种产物单一写方）：
  • plan.md             ← FeedbackBuilder
  • session.json        ← SessionStore（counters + plan + skill 的快照）
  • messages_*.jsonl    ← ConversationBuffer
  • op_summary.md       ← compress.auto_compact (LLM #1；force_rebuild 写 fallback)
  • plan_analysis.md    ← compress.auto_compact (LLM #2；force_rebuild 写 fallback)
  • ranking.md / log.jsonl / perf_log.md / report.png ← RoundLogger / Runner
```

关键结构规则：**每种状态单一所有者**。LLM 从不直接写 plan.md，它只提交
`update_plan(...)`，由 `FeedbackBuilder` 充当唯一的校验者和写方。LLM 也
不能指派 `backing_skill`，它只提交 `keywords`，由 `SkillPool` 和
`TurnExecutor` 做匹配。这样信任边界才清晰。

### 2.1 组件地图

| 模块 | 职责 |
|---|---|
| `agent/loop.py` | AgentLoop —— preflight、baseline、turn loop、post-eval 钩子、每 turn 保存 session |
| `agent/turn.py` | TurnExecutor —— 单 turn：LLM → tool 分派 → eval → feedback |
| `agent/feedback.py` | Plan 状态机 + plan.md 写方 + feedback 拼装 |
| `agent/skill_pool.py` | 关键词排序的候选 skill 列表；refill / match / reference-match |
| `agent/skill_builder.py` | 单 skill registry（selected / active / applied / previously unbound；无终态） |
| `agent/skill_adapter.py` | catalog 过滤、关键词生成、基于关键词的排序 |
| `agent/skill_rendering.py` | skill → markdown（index 模式 vs. full 模式） |
| `agent/subagents.py` | post-eval diagnose subagent |
| `agent/conversation.py` | ConversationBuffer：消息列表、skill 注入、compact 钩子 |
| `agent/compress.py` | microcompact / auto_compact / force_rebuild、STATE_ATTACHMENT 重建 |
| `agent/session.py` | session.json、heartbeat、带 HEAD / dirty guards 的 resume |
| `agent/counters.py` | RunCounters：eval 预算、连续失败记账 |
| `agent/prompt_builder.py` | 系统提示 + 初始 user message 的拼装 |
| `framework/runner.py` | ExperimentRunner：quick_check、eval 子进程、回滚、git commit |
| `framework/logger.py` | RoundLogger：log.jsonl、perf_log.md、ranking.md |
| `framework/git_repo.py` | GitRepo：每轮 snapshot / commit / rollback |

---

## 3. Preflight

Fail-closed 校验在主循环前运行，确保坏掉的环境不会耗用 eval 预算：

1. **Worker 获取**（强制）。无可用 worker → 中止。
2. **参考实现校验** —— 静态 AST 检查（`Model`、`get_inputs`、
   `get_init_inputs` 存在且签名正确）+ 在 worker 上实际运行参考实现。
   任一失败 → 中止。参考实现是 ground truth，它若坏掉，之后的正确性
   检查全部失效。
3. **Seed 解析** —— `previous_code` 经 `KernelVerifier.run()` 做运行时
   校验。失败则退入 KernelGen 路径（最多 `gen_retries` 次重试，每次
   都过 CodeChecker 静态检查 + KernelVerifier 运行时校验），错误反馈
   给下一次 KernelGen。
4. **Worker 释放** 放在 `finally`，任何中止路径都会把 worker 归还池。

Preflight 产出被校验过的 baseline code 与 baseline 指标，主循环从这里
开始。

CodeChecker 被所有生成 kernel 代码的 workflow 调用，不仅用于
AutoResearch。流水线与 YAML 策略
（`op/config/code_checker.yaml`）在
[CodeChecker.md](./CodeChecker.md) 中记录。

---

## 4. 运行时

### 4.1 启动

`AgentLoop` 构造时装配稳定依赖（LLM 客户端、runner、git repo、session
store）与 per-run 状态（feedback、skill builder、skill pool、会话
buffer、counters）。baseline eval 完成并 commit 后，**启动 refill**
与 baseline 测量并发执行，为 skill pool 注入初始候选：

```
SkillPool.refill(mode="replace", include_categories=["guide"])
```

启动期只装 guide（唯一的"可绑定"类）进池。fundamental 不进池 ——
它直接进入系统提示的 `## DSL Fundamentals` 段，由
`build_system_prompt` 从 `task_dir/skills/<name>/SKILL.md` 扫
front-matter 为 `category: fundamental` 的条目拼入。example 和 case
故意延后 —— 要么 agent 通过 `search_skills(hint=...)` 按需拉取，
要么带 `keywords` 的条目找不到可绑候选时系统自动扩池（§6.3）。

如果是 resume 场景，`SkillBuilder.skill_state_from_dict` 恢复 registry
（v4 格式；旧格式整体拒绝），随后 `_rehydrate_pool_from_plan` 会把
被恢复 plan 条目引用的 `backing_skill`、但 refill 后仍不在池里的
skill 按名称补回。

### 4.2 Turn 循环

一个 **turn** 是一次 LLM 调用及其 tool 分派产生的全部后果：

1. **上下文护栏** —— LLM 调用前，`ConversationBuffer` 若估算 token 数
   越过 `compression_threshold × context_limit`，自动触发 auto-compact。
   若仍抛 prompt-too-long，则 `compact_failures` 记一次，路径依次升级：
   auto_compact → force_rebuild → 到 `compact_max_failures` 就中止。
2. **Skill 注入** —— 若当前 active 条目带 `backing_skill`，
   `ConversationBuffer.inject_backing_skill` 把匹配的 SKILL.md 作为
   带 marker 前缀的普通 user message 追加到 buffer
   （`[skill auto-injected for <item> v<N> (<name>)]`），按
   `(plan_version, item_id, skill_name)` dedup。settle 时与 agent
   主动 `read_file('skills/...')` 共享同一条 elision 路径（§8.4）。
3. **LLM 调用** → 返回 tool-call 列表。
4. **Tool 分派**（§4.3）。若发生了任何编辑，turn 进入 **eval 路径**：
   quick_check（import + smoke）→ 完整 eval → KEEP/FAIL/DISCARD 判定
   → KEEP 则 git commit，DISCARD/FAIL 则回滚 →
   `FeedbackBuilder.settle_active` → 拼出下一轮的 feedback。
5. **Post-eval 钩子** —— 只有 `consecutive_failures` 越过
   `diagnose_suggest_threshold` 整数倍时触发 diagnose subagent
   （继而经 `require_replan` 强制 replan）。
   `consecutive_no_edit_turns` 不走 diagnose，是由
   `TurnExecutor._nudge_if_no_edits` 在本轮 buffer 里追加一条
   警告 user message，仅此而已。
6. **Session 保存** —— `SessionStore.save` 每 turn 写一次 counters +
   plan_state + skill_state + last_diagnosis。heartbeat 同步刷新。

### 4.3 Tools

LLM 看到一个稳定的 schema；信任边界在 `TurnExecutor`。

| Tool | 参数 | 效果 |
|---|---|---|
| `update_plan` | `items: [{text, rationale, keywords?}]` | 提交新 plan。rationale 会过长度 + 泛化词校验。keywords 由池匹配指派 `backing_skill`。agent 自报的 `backing_skill` 被剥离。 |
| `read_file` | `path` | 沙箱读。返回截断后的内容 + 一个 stale-file hash，后续 `patch_file` 若文件变了就 fail fast。 |
| `patch_file` | `path, old_str, new_str` | 唯一匹配替换的原地编辑。禁用模式会被拦截。**对绑定条目，在该条目完成 `acknowledge_skill` 之前被拒绝。** |
| `write_file` | `path, content` | 整文件重写。同样受 acknowledge 闸门约束。 |
| `acknowledge_skill` | `plan_item_id, valuable_aspects, kernel_application, applicability` | 在绑定条目首次编辑前必须调用一次。schema 严：`valuable_aspects`（100–500 字符，skill 层面的通用价值）+ `kernel_application`（100–500 字符，对当前 kernel 的具体改动——优先**结构性改动**而非调参）；applicability ∈ {apply, unbind}。`unbind` 释放当前条目的绑定（条目继续作为自由探索），并把 skill 追加到 `unbound_at_versions`，**但 skill 本身不会被永久排除**——之后同一 skill 被 KEEP 会把它提回优先级最高的 tier。 |
| `search_skills` | `hint: str` | `SkillPool.refill(mode="append")`：关键词流水线带 hint 重跑，新候选追加进池。不变更 plan。 |
| `compact` | —— | Agent 主动触发 auto_compact。 |
| `finish` | `summary?` | 显式终止。 |

非法 tool 组合（例如有待 eval 编辑的情况下 `finish`）返回 reject 消息
且不变更状态。reject 消息是自描述的，agent 可以据此自行修正调用。

### 4.4 终止

以下任一触发即终止主循环：

- eval 预算耗尽（`eval_calls_made >= max_rounds`）。
- API 调用预算耗尽（`total_api_calls >= max_rounds × max_turns_multiplier`）。
- 压缩失败次数达到 `compact_max_failures`。
- `llm_max_retries` 重试用完仍 LLM 连接错误。
- agent 显式调用 `finish`。

`best_result`（按 `lower_is_better` / `higher_is_better` 选出 primary
metric 最优的一轮）作为最终产物写回 task 目录；对应的 git commit 就是
可复现的 checkpoint。

---

## 5. Plan 状态

### 5.1 形状

一个 `plan_item` 是含以下字段的 dict：

```
id               "p1"、"p2"……
text             简短的行动描述
rationale        一句话、经校验的理由（必填）
status           pending | active | done_ok | done_fail
keywords         list[str]（系统清洗过）
backing_skill    绑定的 skill 名称，或 None（在 `unbind` ack 后会被清空）
skill_ack        agent 读完注入 SKILL.md 之后提交的结构化承认
                 {valuable_aspects, kernel_application, applicability,
                 skill}；unbound 条目或尚未承认的条目不携带此字段
sketch           可选的代码草稿
```

plan 提交时，第一个 pending 条目升级为 `active`；active 条目就是 agent
要编辑的对象。

### 5.2 生命周期

```
update_plan(items) ──► 所有条目 pending；首个 → active
                         │
                         ▼ （若 item.backing_skill != None）
           loop 自动注入 SKILL.md via inject_backing_skill
                         │
                         ▼
           agent 必须调 acknowledge_skill(item_id,
                        valuable_aspects, kernel_application,
                        applicability)
             │
             ├─ applicability = apply
             │     → skill_ack 存档；编辑闸门打开
             │
             └─ applicability = unbind
                   → SkillBuilder.mark_unbound(backing_skill);
                     item.backing_skill = None；条目降级为 unbound
                     自由探索。skill **不是**终态——仍保留在 registry
                     中、仍可被下一条 item 绑定（仅 tier-2 优先级）
                         │
                         ▼ （patch_file / write_file + eval）
                ┌─ KEEP    → done_ok；SkillBuilder.record_applied(bs, v);
                │            _advance() 激活下一条 pending
                ├─ FAIL    → done_fail；_advance() 激活下一条 pending
                │            （consecutive_failures++ 过阈值会触发
                │            diagnose，但该 item 本身已经 settle、
                │            不再 active）
                └─ DISCARD → done_fail；_advance() 激活下一条 pending

diagnose → require_replan ──► 失败条目打上 "abandoned (diagnose)"
                              （pre-eval 失败时新建 history 条目；
                              post-eval 失败时追溯改写最后一条
                              history，并把 _advance 上来但还没跑
                              的 pending 倒回 pending）；
                              must_replan = true；下次 update_plan
                              替换被放弃的条目。**不**改 skill registry
                              ——diagnose 只回卷 plan，不动技能态。
```

每次 `update_plan` 被接受，`plan_version` 自增。SkillBuilder registry
（§6.4）存 `registered_at_version`、`applied_versions[]`、
`unbound_at_versions[]`，既用于 plan.md 渲染，也是 binding 优先级
tier 计算的依据。

### 5.3 plan.md 布局

`FeedbackBuilder._persist_plan` 每次状态变化都会写 plan.md，分三段：

```
# Plan vN
- [status] p1: text  (keywords: ...) (skill: backing_skill)
  ……

## Optimization History
### Plan v(N-k)
- [O]/[X] …… 带 reason 和 metric 的 settlement 条目
……

## Skill State
### Active (adopting now)
  - [>>>] name (category)  reason / backing items / excerpt
### Applied (pattern adopted successfully)
### Selected (candidates, not yet bound)
### Previously unbound (last-resort tier; still selectable, ranked last)
```

每次渲染时每条 skill 只落入一个桶：当前被 active item 绑定 → Active；
否则 `applied_versions` 非空 → Applied；否则 `unbound_at_versions`
非空 → Previously unbound；否则 Selected。四种状态**全都不是终态**，
Previously unbound 只是降级信号，不是拦截。下一次 KEEP 会把它提回
Applied（tier 0），unbound 历史保留作为徽章。

Optimization History 会随 plan 版本无界增长。Skill State 是恢复点：
每条 skill 的完整历史（`applied_versions`、`unbound_at_versions`）
都保留，所以压缩后的提示仍能看到"X 在 v12 被 unbind、v17 被 apply"，
不必回放链路。

---

## 6. Skills

### 6.1 三层模型

| 层 | 位置 | 内容 |
|---|---|---|
| 0 —— 系统提示 | 静态前导 | `## DSL Fundamentals` 段：`task_dir/skills/` 下每个 `category: fundamental` 的 SKILL.md 全文（受 `system_fundamentals_max_chars` 预算限制，默认 20000），外加 task metadata、context_files（hardware_info 等）、约束、tool protocol。 |
| 1 —— `task_dir/skills/<name>/SKILL.md` | 按需 | 每份 guide / example / case SKILL.md 在脚手架时镜像进 task 目录。agent 通过 `read_file('skills/<name>/SKILL.md')` 拉取；内容只在对应 item 活跃期间停留在 buffer，settle 时由 `unload_item_reads` 清空。 |
| 2 —— skill pool 索引 | 初始 user message | 紧凑索引（`name [category] @ skills/<name>/SKILL.md: desc`）列出可绑定候选（guide）。不含内容。agent 依此挑 `keywords`，框架依此做绑定候选列表。 |

### 6.2 绑定模型

plan 条目上只有一个 skill 槽：

- **`backing_skill`** —— 匹配到的 skill 名（未绑定则为 None）。框架在
  `update_plan` 时根据 `keywords` 设置；agent 自报的值会在信任边界被
  剥离。agent 调 `acknowledge_skill(applicability="unbind")`
  时清空，条目降级为 unbound 自由探索。skill 本身**不**被排除；
  只是在 registry 中的优先级 tier 被降低。

`update_plan` 时的绑定流程：

```
agent 提交带 keywords 的条目
  ↓
pool.match_by_keywords(keywords, exclude_names=已绑)
  → top-1 结果（先按 tier 排序，tier 内按 -score） → item.backing_skill
```

`match_by_keywords` 走 `residual_bindable`（池中可绑 category 的所有
skill——没有终态排除）。命中集合再按 `SkillBuilder.tier()` 排序，
tier 内按 keyword-score 降序：

- **Tier 0（applied）** ——`applied_versions` 非空。已证明在本 run 有
  效，优先再试。
- **Tier 1（fresh）** —— registered、从未 applied、从未 unbound。
- **Tier 2（previously unbound）** ——`unbound_at_versions` 非空且
  `applied_versions` 为空。末位但**仍可选**——之后一次 KEEP 会把它
  提回 tier 0。

### 6.3 池枯竭自动扩池

启动池故意窄（只有 guide），让初始提示保持紧凑。长期运行下，
agent 用 keyword 要的模式可能只出现在 `example` / `case` 里。
`_match_keywords_to_skills` 走两遍：任何带
keyword 的条目若在当前池里没命中，就触发一次
`SkillPool.refill(mode="append", include_categories=["example", "case", "method", "implementation"])`
再对 miss 重试。触发条件是 **"某个 keyword 请求没被满足"**，不是
**"池里零可绑"** —— 因为 guide 仍在、但语义不匹配时池非空却每次都
空匹。

为维持"被 reject 的 `update_plan` 无副作用"：**rationale 校验在扩池
调用之前**，非法 plan 直接返回 reject，不动池。

### 6.4 SkillBuilder registry

SkillBuilder 是一个扁平 registry，**没有终态**。每一条 registered
的 skill 都保持 bindable；两个单调徽章列表决定 binding 优先级 tier：

```
    registered  ◄──► record_applied  → applied_versions = [v, ...]
        │                                        │
        │                                        └─ tier 0（最高优先）
        │
        └──────► mark_unbound   →  unbound_at_versions = [v, ...]
                                              │
                                              └─ tier 2（末位、仍可再选；
                                                  除非同时也 applied）
```

- **register** —— 建条目或更新条目。幂等；重复 register 只会刷新
  `registered_reason` 和 `registered_at_version`。
- **mark_unbound** —— 由 agent 的
  `acknowledge_skill(applicability="unbind")` 承认触发（经
  `FeedbackBuilder.record_skill_acknowledgement`），把 `plan_version`
  追加到 skill 的 `unbound_at_versions`，清空当前条目的
  `backing_skill`。diagnose / `require_replan` **不**动 registry——
  只回卷 plan。**非终态**：skill 仍可 bindable；`tier()` 仅在未被
  applied 时返回 2。
- **record_applied** —— backing skill 对应的条目 settle KEEP 时调用。
  把 `plan_version` 追加到 `applied_versions`。`tier()` 只要
  `applied_versions` 非空就返回 0 —— 哪怕该 skill 之前被 unbound 过，
  一次 KEEP 就能让它回到最高优先 tier。

"当前谁是 active skill"由 `plan_items` 实时派生（状态 `active` 且
`backing_skill != None` 的条目）。registry 不存这个。

Session 持久化格式：v5（使用 `unbound_at_versions` 列表）。v4 payload
（旧的终态 `abandoned=True` + 标量 `abandoned_at_version`）会被
隐式迁移为 `unbound_at_versions=[abandoned_at_version]`——这些 skill
以 tier 2 身份加载，而不是永久排除。

### 6.5 关键词流水线

`skill_adapter.generate_query_keywords` 从 `op_name + task_desc +
stage + dsl/backend/arch/framework` 出发，调一次 LLM 生成
`QueryKeywords` 三元组（带确定性 fallback）。
`rank_skills_by_keywords` 融合三类信号：关键词命中分、当前 stage 下
的 category prior、以及 catalog 里的算子元数据加成。最终输出一条
排序后的列表。

canonical category 会把 `method → guide`、`implementation → example`
折叠，防止磁盘上两套分类法把 prior 拆成两份。

---

## 7. Skill 承认与 diagnose

### 7.1 Agent 自承认（pre-edit 闸门）

没有外部 supervisor。"agent 确实读过也理解了注入的 SKILL.md"的执行
力靠 agent 自己通过结构化工具调用来强制。

当 plan 条目有 `backing_skill != None` 时：

1. turn prologue 调
   `ConversationBuffer.inject_backing_skill(item_id, skill_name,
   content, plan_version)` —— SKILL.md 以带 marker 前缀的 user
   message 进入 buffer，按 `(plan_version, item_id, skill_name)`
   dedup。
2. `TurnExecutor._dispatch_tools` 对该条目的 `patch_file` /
   `write_file` **拦截**，直到 agent 调
   `acknowledge_skill(plan_item_id, valuable_aspects,
   kernel_application, applicability)`。
3. handler 校验 schema（两个字段都是 100–500 字符、
   applicability ∈ {apply, unbind}），把 ack 交给
   `FeedbackBuilder.record_skill_acknowledgement`。tool description
   明确引导 `kernel_application` 优先写**结构性改动**（算法 / 访存
   pattern / 内存层次 / kernel 拆分-融合），只有和结构性假设挂钩的
   参数调整才合格。
4. `unbind` 路径：
   - 调 `SkillBuilder.mark_unbound(backing_skill, reason)`
     （**非终态**——skill 仍留在 registry 中、tier 2）；
   - 清空 `item.backing_skill`（条目降级为 unbound）；
   - 打开编辑闸门 —— 继续以自由探索身份编辑。
5. `apply` 路径：ack 写到条目的 `skill_ack` 字段（plan.md 渲染、
   session.json 持久化）；闸门打开。

未绑定条目完全跳过这条流程 —— 自由探索无需承认。结构化 payload
同时是审计线索：`valuable_aspects` 与 `kernel_application` 会进入
plan 历史，事后能同时看到 agent 对 skill 价值的判断和它准备做的
具体改动。

### 7.2 Post-eval diagnose

在 `consecutive_failures` 越过 `diagnose_suggest_threshold` 整数倍时
触发（`consecutive_no_edit_turns` **不**走这条——那是
`TurnExecutor` 里的一条警告消息，不会起 subagent）。
diagnose subagent 拿到当前 plan、近期历史和可编辑代码，产出诊断
报告；handler 再以这个报告为参数调 `require_replan`，
报告作为 `last_diagnosis` 出现在下一轮提示，agent 被强制进入
replan 模式。

`require_replan` 把"失败的那一项"打上 `abandoned (diagnose)`
（post-eval 路径：追溯修改 settled_history 最后一条 + 把
_advance 升上来的未执行 pending 倒回 pending；pre-eval 路径
`edit_fail` / `quick_check_fail`：当前仍然 active 的条目 close
为 done_fail 并新增一条 history），把 `must_replan` 置 true，
保存 diagnosis。**不**触碰 `SkillBuilder` registry。下一次
`update_plan` 进入 **替换模式**：agent 提交一个或多个替换条目
插入到被废弃条目原位置；旧 plan 的其它 pending 条目原样保留。

`last_diagnosis` 会持久化进 session.json，这样在 replan-required 状态
恢复后，原始推理仍然可见。

---

## 8. 上下文管理

### 8.1 ConversationBuffer

它是 `_msgs` 和 skill-read 跟踪集合的唯一所有者，也是唯一会变更
消息列表的组件。公开操作：

- `append` / `extend` —— 普通追加。
- `inject_backing_skill(item_id, skill_name, content, plan_version,
  max_chars)` —— 把注入的 SKILL.md 作为带 marker 前缀的普通 user
  message 追加（`[skill auto-injected for <item> v<N> (<name>)]`），
  按 `(plan_version, item_id, skill_name)` 幂等；marker 登记到
  `_item_inject_markers[item_id]`，settle 时 `unload_item_reads`
  据此定位。**注意不是裸的 tool_result block —— 那样会破坏
  Anthropic 的 tool_use / tool_result 配对规则。**
- `track_item_skill_read(tool_use_id, item_id)` —— TurnExecutor 在
  agent `read_file('skills/...')`且 `item_id` 为 active 时调用，
  把 tool_use_id 记入 `_item_read_ids[item_id]`。
- `unload_item_reads(item_id)` —— settle 时调。走两条轨：synthetic
  marker（匹配顶层 user message 的 content 前缀）+ 真 tool_use_id
  （下钻 user message 的 content 数组、翻转匹配到的 tool_result
  block）。正文替换为 `[skill read elided — plan item settled]`；
  保留 `tool_use_id`，不破坏 API 配对。
- `on_buffer_rebuilt` —— auto_compact / force_rebuild 后的钩子。
  新 buffer 会把近期轮次原样搬过来，所以**不能**盲清 tracking，
  否则幸存 marker 会重复注入、幸存 tool_result 在 settle 时清不掉：
    * auto-inject 轨：重扫幸存 user message，按
      `[skill auto-injected for pN vM (name)]` 正则抽出
      `(plan_version, item_id, skill_name)`，重建
      `_skill_inject_keys` + `_item_inject_markers`。marker 还在
      recent 就继续拦截 dedup；被压缩掉就释放 dedup、下轮重注入。
    * voluntary-read 轨：收集新 buffer 中仍然作为 `tool_result`
      出现的 tool_use_id 集合，对每个 item 的 id 集合求交，空 item
      项删除。幸存的 id 仍然 elide-able。
- `save_full_increment` / `load_latest` / `save_latest` —— JSONL
  持久化（`messages_full.jsonl` 追加写，`messages_latest.jsonl`
  覆盖写）。

### 8.2 压缩分层

| 层 | 触发时机 | 做什么 |
|---|---|---|
| `microcompact` | 每轮 | 削陈旧的 `tool_result` 正文（只保留最近 N 条）。零 LLM 成本。 |
| `auto_compact` | `estimate > threshold × context_limit`，或 agent 调 `compact` | 多步流水线：**并发 2 次独立 LLM 调用**（operator summary + plan.md 结构化分析）。新 buffer 是 5 条 marker 消息（boundary + bootstrap + 3 个 state attachment）加保留的 recent rounds。见 §8.3。 |
| `force_rebuild` | 第二次 PTL，或 auto_compact 无改动 / 抛异常 | 和 `auto_compact` 同样的 buffer 布局，但 **无 LLM 调用**：operator summary 降级为关键词裸 dump；plan analysis 降级为截断的 raw plan.md 并加"analysis unavailable"标记。 |

`compact_failures` 统计相邻两次 LLM 成功间的 PTL 次数；达到
`compact_max_failures` 时直接中止，不在不可用提示上烧预算。

### 8.3 auto_compact 流水线

`auto_compact` 完成后 buffer 结构：

```
[COMPACT_BOUNDARY]                  ← 仅 marker
[BOOTSTRAP]                         ← 当前 plan vN + [OPERATOR_SUMMARY]
[STATE_ATTACHMENT:KERNEL]           ← editable files 全量（80k sanity cap）
[STATE_ATTACHMENT:PLAN]             ← 5 段结构化 plan 分析
[STATE_ATTACHMENT:RANKING]          ← ranking.md 全量
<从输入保留下来的 recent rounds>
```

**LLM call #1 — operator summary**（`_summarize_operator_from_keywords`）：

- 输入：`config.name`、`reference.py` 头部、从 `_plan_items +
  _settled_history` 聚合的历史关键词频次表（全 run 累积）。
- 输出：≤500 token markdown，三段 `## Operator Shape`、
  `## Computation Components`、`## Exploration Signals`。
- 落盘：`agent_session/op_summary.md`（每次 compact 覆盖写）。
- Fallback：`Keywords seen so far: a×3, b, c (counts: …)` 裸 dump。

**LLM call #2 — plan analysis**（`_analyze_plan_md`）：

- 输入：`task_dir/plan.md` 全文 —— 不截断。
- 输出：≤1500 token markdown，严格五段：
  `## Current Status`、`## What's Working`、`## High-ROI Operations`、
  `## Repeated Failures`、`## Dead Directions`。
- 落盘：`agent_session/plan_analysis.md`（每次 compact 覆盖写）。
- Fallback：raw `plan.md` 截至 `compact_plan_raw_fallback_chars`，附
  "analysis unavailable" 提示。

两次调用走 `asyncio.gather(..., return_exceptions=True)`；其中一个
失败不会阻塞另一个。`kernel.py` 和 `ranking.md` 全量放进 attachment
（`compact_kernel_sanity_cap` 是防御性的 80k 硬顶，**不是** 默认截断）。

### 8.4 Skill 内容生命周期

skill 内容在 **turn prologue**（不是 submit 时）由
`inject_backing_skill` 进入 buffer，按 `(plan_version, item_id,
skill_name)` 注入一次。它是带 marker 前缀的普通 user message，不是
裸 tool_result —— 因为一个没有前置 `tool_use` 的合成 `tool_use_id`
会被 Anthropic API 拒掉。

settle 时由 `unload_item_reads(item_id)` 清两条轨：

- **synthetic 注入** —— 按预存 marker 前缀匹配，顶层 user message 的
  content 整段替换成 elision marker。
- **主动 read_file** —— agent 在该 item 活跃时读了
  `skills/<name>/SKILL.md`；对应 tool_result block 的 `content` 被
  替换，`tool_use_id` 保留，assistant / tool_result 配对不变。

rebuild（auto_compact / force_rebuild）清两条跟踪 + 注入 dedup set；
active 条目的 skill 下轮重新注入进新 buffer。

---

## 9. 持久化与 resume

### 9.1 session.json

```json
{
  "version": 3,
  "task_name": "...",
  "model": "...",
  "counters": { ... RunCounters.to_dict() ... },
  "baseline_commit": "<sha>",
  "head_commit":     "<保存时的 sha>",
  "saved_at":        "YYYY-MM-DD HH:MM:SS",
  "plan_state":      { ... FeedbackBuilder.plan_state_to_dict() ... },
  "skill_state":     { ... SkillBuilder.skill_state_to_dict() ... },
  "last_diagnosis":  "..."
}
```

由 `SessionStore.save()` 每 turn 原子写（临时文件 + os.replace）。
legacy v2 session（counters 在顶层）由 `RunCounters.from_dict` 自动
迁移。

### 9.2 Resume 护栏

`SessionStore.load()` 在以下情况拒绝恢复：

- `task_name` 不匹配（session 属于别的 task）。
- `head_commit` 和 repo 当前 HEAD 不一致（外部推进了代码树，session
  的 eval 历史已经对不上）。
- 语义文件（`editable_files`、`eval_script` 等）在 git 里是 dirty
  状态。

拒绝时日志发一条 warning，run 走全新启动。这是刻意设计：resume 是
优化，不是正确性要求。

### 9.3 Resume 时的池 rehydrate

`SkillBuilder.skill_state_from_dict` 会恢复每一条 SkillRecord（含
previously-unbound 的 —— 它们仍 bindable、仅在 tier 2）。池本身
不持久化，启动 refill 负责重建。由于 refill 有
category 过滤，上一个 session 通过 `search_skills` 加进来的
`example` / `case` 不会自动回到池里。`_rehydrate_pool_from_plan` 会
遍历恢复后的 plan 条目，收集所有 `backing_skill` 名称，按名字调
`pool.append_new(...)` 从 catalog 补回，这样 resume 后 skill 注入的
内容查找仍然可用。

---

## 10. 预算与 counters

`RunCounters`（`agent/counters.py`）持有循环会读的每个语义阈值：

| 字段 | 用途 |
|---|---|
| `eval_calls_made` | 主预算；gate 在 `max_rounds` 上。 |
| `total_api_calls` | 次预算；gate 在 `max_rounds × max_turns_multiplier` 上。 |
| `consecutive_failures` | FAIL **以及 pre-eval 失败**（`edit_fail` / `quick_check_fail`）自增；KEEP **和** DISCARD 清零（correct-but-no-improvement 不算失败）。触发 diagnose 阈值。 |
| `consecutive_no_edit_turns` | 无编辑的 turn 自增；任何 patch 清零。达 `max_no_edit_turns` 时由 turn handler 追加一条 "你必须编辑" 警告 user message（**不**会走 diagnose / replan）。 |
| `consecutive_no_tool_turns` | LLM 回复但完全没有 tool call 时自增。兜底保护。 |
| `compact_failures` | PTL 升级计数。 |

Counters 是单一真相源。任何"要不要 replan / diagnose / abort"的决策
都只读这里 —— 主循环自己不保留重复标志位。

---

## 11. 配置

AutoResearch 读取标准 AKG 配置系统（`load_config` +
`build_langgraph_task_config`）。per-task 覆盖写在 `task.yaml` 的
`agent.config` 块。完整字段清单见 `framework/config.py`；下面是精选。

### 11.1 Task 层字段

| 字段 | 默认值 | 用途 |
|---|---|---|
| `max_step` | 20 | 最大 eval 轮次（`max_rounds` 参数优先级更高）。 |
| `agent_model_config.coder` | `"standard"` | LLM 档位。 |
| `eval_timeout` | 120 | 单轮 eval 超时（秒）。 |
| `workflow_timeout` | 动态 | 整 workflow 超时。自动算成 `max(1800, max_rounds × (eval_timeout + 60) + 300)`。 |
| `profile_settings.run_times` | 50 | profiling 重复次数。 |
| `profile_settings.warmup_times` | 5 | profiling 预热次数。 |

### 11.2 AgentConfig 常用字段

| 字段 | 默认值 | 用途 |
|---|---|---|
| `max_consecutive_failures` | 10 | 连续 N 次失败后中止。计 FAIL + pre-eval 失败（edit / quick_check）；DISCARD 会清零计数器。 |
| `max_turns_multiplier` | 8 | API 调用总数上限 = `max_rounds × this`。 |
| `max_no_edit_turns` | 3 | 连续 N 个 turn 没编辑时 `TurnExecutor._nudge_if_no_edits` 追加警告 user message。**不**强制 replan、**不**触发 diagnose。 |
| `context_limit` | 150000 | 模型上下文窗口（token）。auto_compact 在 `× compression_threshold` 触发。 |
| `compression_threshold` | 0.75 | 触发 auto_compact 的阈值比例。 |
| `compact_max_failures` | —— | PTL 升级次数，达到后中止主循环。 |
| `skill_block_max_chars` | 8000 | 初始提示 skill 索引的总字符预算。 |
| `skill_block_top_k` | 5 | 初始提示 top-K 索引条目数。 |
| `skill_inject_max_chars` | 6000 | 条目激活时 skill 内容注入的上限。 |
| `skill_narrow_timeout` | 30.0 | 关键词生成 LLM 调用的硬超时。 |
| `diagnose_suggest_threshold` | 3 | 每 N 次连续失败触发 diagnose subagent。 |
| `subagent_max_iterations` | 15 | diagnose subagent 迭代上限。 |
| `system_fundamentals_max_chars` | 20000 | 系统提示 `## DSL Fundamentals` 段预算。 |
| `skill_inject_max_chars` | 6000 | 注入的 SKILL.md 主体截断。 |
| `plan_item_rationale_min_chars` | 30 | plan 条目 rationale 最小长度。 |
| `plan_item_rationale_max_chars` | 400 | rationale 截断上限。 |
| `skill_keyword_max_per_item` | 5 | 单条目允许的 `keywords` 上限。 |

### 11.3 字段分组（便于速查）

- **上下文** —— `context_limit`、`compression_threshold`、
  `compact_keep_recent_rounds`、`compact_op_summary_max_tokens`、
  `compact_plan_analysis_max_tokens`、`compact_kernel_sanity_cap`、
  `compact_plan_raw_fallback_chars`、`compact_post_check_ratio`、
  `compact_max_failures`。
- **截断** —— `editable_file_truncate`、
  `system_context_file_truncate`、`system_context_total_truncate`、
  `plan_max_chars`、`skill_block_max_chars`、`skill_inject_max_chars`、
  `eval_feedback_tail`、`raw_output_tail`。
- **循环控制** —— `max_consecutive_failures`、`max_no_edit_turns`、
  `max_turns_multiplier`、`llm_max_tokens`、`thinking_budget`。
- **Skill** —— `skill_*`、`system_fundamentals_max_chars`、
  `plan_item_rationale_*`。
- **Diagnose / feedback** —— `diagnose_suggest_threshold`、
  `compact_diagnosis_truncate`、`ranking_description_truncate`、
  `finish_hint_threshold`。
- **LLM** —— `llm_connection_check_timeout`、`call_timeout`、
  `llm_max_retries`、`chars_per_token`。
- **路径** —— `session_dir`（默认 `"agent_session"`）、`heartbeat_file`。

---

## 12. Task 目录布局

run 过程中产出的所有内容都在 `task_dir` 下：

| 路径 | 写方 | 用途 |
|---|---|---|
| `session.json` | SessionStore | Resume 状态 |
| `plan.md` | FeedbackBuilder | 当前 plan + history + skill state |
| `{session_dir}/messages_full.jsonl` | ConversationBuffer | 完整消息归档（append-only） |
| `{session_dir}/messages_latest.jsonl` | ConversationBuffer | 当前 buffer 快照 |
| `{session_dir}/op_summary.md` | compress.auto_compact (LLM #1) | 每次 compact 重算的 operator 级总结 |
| `{session_dir}/plan_analysis.md` | compress.auto_compact (LLM #2) | 每次 compact 重算的 5 段结构化 plan 分析 |
| `log.jsonl` | RoundLogger | 每轮 eval 一条 JSON |
| `perf_log.md` | RoundLogger | 人类可读的结果表 |
| `ranking.md` | RoundLogger | top-K 正确结果 + failed 尝试 |
| `report.png` | Runner | 优化曲线图 |
| `agent.log` | FileLogger | stdout 带时间戳 tee |
| `RUNNING` | SessionStore | PID + 状态 heartbeat |

`git_repo.py` 的策略是这些 run artifact **绝不进 git** —— 仓库只留
eval checkpoint（每个 KEEP 一个 commit），task 目录是不入 git 的
工作区。

---

## 13. 不变量

系统依赖若干不变量，每次改代码都必须保持：

1. **每种产物单一写方。** plan.md → 只由 FeedbackBuilder 写；
   session.json → 只由 SessionStore 写；messages_*.jsonl → 只由
   ConversationBuffer 写。想改以上任意一种状态，必须走所有者。
2. **被 reject 的 tool 调用无副作用。** `update_plan` 因 rationale
   校验被拒，则 SkillPool、SkillBuilder、plan.md 不变；`patch_file`
   因 stale-hash 被拒，则文件系统不变。
3. **SkillBuilder 没有终态。** 每一条 registered skill 都保持
   bindable；`unbound_at_versions` 只在 binding 优先级排序时把 skill
   降到 tier 2，之后一次 KEEP 就能把它提回 tier 0。
4. **预算记账只靠 counters。** 主循环的 "继续 / 终止 / 升级" 决策
   只读 `RunCounters`，不引入临时标志位。
5. **plan.md 是恢复点。** 任何压缩周期后，plan.md + session.json 足以
   重建 agent 的世界观。section-aware 截断正是为了保护这点。
6. **Tool schema 是信任边界。** 所有超出声明 tool 参数范围的字段
   （包括 agent 自报的 `backing_skill`）由 `TurnExecutor` 剥离，
   然后才有可能影响状态。

---

## 14. 使用方式

生产环境的唯一入口是 CLI 脚本 `scripts/run_autoresearch.py`。四种输入组合：

```bash
# 自然语言（LLM 写参考、KernelGen 出 seed）
python scripts/run_autoresearch.py \
  --desc "fused ReLU + LayerNorm, input (32, 1024), fp16" --backend cuda

# 自然语言 + 初始内核（跳过 KernelGen）
python scripts/run_autoresearch.py \
  --desc "fused ReLU + LayerNorm, input (32, 1024), fp16" \
  --kernel path/to/kernel.py --backend cuda

# 参考文件（KernelGen 出 seed）
python scripts/run_autoresearch.py --ref path/to/reference.py --backend cuda

# 参考文件 + 初始内核（跳过全部生成）
python scripts/run_autoresearch.py \
  --ref path/to/reference.py --kernel path/to/kernel.py --backend cuda
```

主要参数：

| 参数 | 说明 |
|------|------|
| `--desc` / `--ref` / `--kernel` | 输入来源；见上述四种组合。 |
| `--op-name` | 算子名（影响任务目录与日志分类）。 |
| `--backend` / `--arch` | 目标后端与架构；必须与执行 eval 的 worker 注册值完全一致。 |
| `--max-rounds` | eval 预算上限（覆盖 `task.yaml` 的 `max_step`）。 |
| `--device-id` | 本地 worker 的设备号（默认 0）。与 `--worker-url` 互斥。 |
| `--worker-url` | 远端 Worker Service HTTP 地址（`host:port`，多台用逗号分隔）。本地无 NPU / CUDA 时用它把 eval 转发给远端。Worker 启动（`akg_cli worker --start`）见 [AKG_CLI.md](./AKG_CLI.md) §4；Server / Worker / WorkerManager 整体架构见 [ServerArchitecture.md](../../v1/CN/ServerArchitecture.md)。 |

> KernelAgent 在用户请求"深度迭代优化"时会在内部通过
> `ToolExecutor → prepare_config() → build_initial_state() → workflow
> execution` 自动触发同一 workflow，终端用户无需直接调用。

> **远端 Worker 快速拉起**：在带 NPU / CUDA 的机器上 `akg_cli worker
> --start --backend <backend> --arch <arch> --devices <ids>
> --port <port>`，本地若无公网直连则通过 SSH 隧道转发该端口，再用
> `--worker-url host:port` 跑。详见 [AKG_CLI.md](./AKG_CLI.md) §4。

### 14.1 调试入口

仅用于调试 / 排查的非生产入口。都绕过了 CLI 脚本做的前处理（参数校验、
Worker 注册、scaffold），调用方需自行保证参数合法性，非法输入将在组件
边界处直接抛错。

**（a）直接驱动 `LangGraphTask`**（跳过 `run_autoresearch.py`、手工
装配 config）：

```python
from akg_agents.op.langgraph_op.task import LangGraphTask
from akg_agents.op.config.config_validator import load_config
from akg_agents.core.worker.manager import register_local_worker
from akg_agents.utils.task_label import resolve_task_label

await register_local_worker([0], backend="cuda", arch="a100")

config = load_config(dsl="triton_cuda", backend="cuda")
config["task_label"] = resolve_task_label(op_name="my_op", parallel_index=1)
config["max_step"] = 20

task = LangGraphTask(
    op_name="my_op",
    task_desc=open("reference.py").read(),
    task_id="my_op_001",
    backend="cuda", arch="a100",
    dsl="triton_cuda", framework="torch",
    config=config,
    workflow="autoresearch",
)
op_name, success, final_state = await task.run()
```

**（b）直接进 `AgentLoop` / `manual_eval`**（绕过 workflow 层；必须用
完整模块路径，相对导入使裸 `python manual_eval.py` 无法运行）：

```bash
# 直接进 AgentLoop（跳过 preflight / seed / workflow wrapper）
python -m akg_agents.op.autoresearch.agent \
  --task <task_dir> --max-rounds 20 --device-id 0

# 手动 eval 辅助（无 LLM；单轮 eval / status / report）
python -m akg_agents.op.autoresearch.manual_eval \
  --task <task_dir> --eval-only
python -m akg_agents.op.autoresearch.manual_eval \
  --task <task_dir> --status
python -m akg_agents.op.autoresearch.manual_eval \
  --task <task_dir> --report
```

---

## 15. 测试

测试自动检测 NPU / GPU，都没有则跳过。smoke test 算子是 `relu(x)`，
shape 为 `(11, 37, 8191)` —— 素数维度强制 mask 处理；
`8191 = 2^13 − 1` 压 UB 尺寸。

| 套件 | 路径 | 覆盖 |
|---|---|---|
| 端到端 | `tests/op/st/test_autoresearch.py` | preflight → seed verify → AgentLoop → eval_fn → KernelVerifier；带性能门（speedup > 1.0×）。 |

单元测试在 `tests/op/ut/`：

- `test_skill_builder.py` —— SkillBuilder 状态迁移（settle / supersede / replan / serialize）。
- `test_skill_pipeline.py` —— 关键词流水线原语 + `SkillPool` 读侧。
- `test_acknowledge_skill.py` —— 工具 schema 校验、ack → feedback 接线、`inject_backing_skill` / `unload_item_reads` 在 buffer 上往返。
- `test_fundamentals_layout.py` —— task_dir/skills 布局、Layer 0 fundamentals 扫描、预算上限。
- `test_runtime_skill_binding.py` —— `mark_selected` 幂等、`SkillPool.refill`（replace + append）、`_handle_update_plan` 增强、`_handle_search_skills` 薄封装。
- `test_session_persistence.py` —— resume 流程 + skill 状态 round-trip。
- `test_skill_prompt_injection.py` —— 初始消息措辞 + 压缩 bootstrap 的 skill 索引渲染。
- `test_conversation_buffer.py` —— buffer 的 skill 注入生命周期。
- `test_scaffold_roundtrip.py` —— `scaffold_task_dir` → `load_yaml_config` 字段保留。

---

## 附录 A：公共常量

```
compress.COMPACT_BOUNDARY    "[COMPACT_BOUNDARY]"
compress.STATE_ATTACHMENT    "[STATE_ATTACHMENT]"
compress.BOOTSTRAP_MARKER    "[BOOTSTRAP]"

skill_adapter.TRACKABLE_PATTERN_CATEGORIES
    frozenset({"guide", "example", "case", "method", "implementation"})

skill_pool.SkillPool._REFERENCE_CATEGORIES
    frozenset({"fundamental", "reference"})
```

## 附录 B：参数枚举

task 参数（`framework`、`backend`、`dsl`、`arch`）的合法取值定义在
仓库根的 `AGENTS.md`；`TaskConfig` 构造时会强制校验。
