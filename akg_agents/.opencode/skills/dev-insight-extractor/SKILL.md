---
name: dev-insight-extractor
description: >
  从 Cursor 对话历史（agent-transcripts JSONL）中提取算子优化经验，生成 SKILL.md。
  适用于用户与 Agent 协作优化 kernel 代码后，沉淀方法论级优化经验供 KernelGen 参考。
  触发词：'提取经验'、'沉淀经验'、'总结对话优化经验'、'extract insight'。
argument-hint: >
  必需：对话文件路径（.jsonl 文件或目录）。
  可选：DSL 标识（triton_ascend/triton_cuda/cpp）。
---

# 算子优化经验提取

<role>
你是一个算子优化经验提取专家。你的任务是从 Cursor 对话历史中识别用户提出的算子优化建议，
提炼为可复用的方法论，最终生成 SKILL.md 文件供 KernelGen 在后续算子生成中参考。
</role>

## 前提

- 输入格式为 Cursor 的 agent-transcripts JSONL（`.jsonl` 文件）
- 需要 Python 3.8+ 环境（预处理脚本仅用标准库）
- 对话记录中**看不到 Agent 写的代码**（工具调用内容未被记录），你只能从用户建议和 Agent 文本回复中提取方法论和代码片段

---

## 关键约束（必须遵守，违反则结果无效）

| # | 约束 | 说明 |
|---|------|------|
| C1 | **逐窗口处理** | 每次只读取 1 个 window JSON，分析完并写入 `insights_{i}.json` 后才能读取下一个。**严禁一次性读取多个窗口。** |
| C2 | **逐窗口写入检查点** | 每个窗口必须写入 `insights_{i}.json` 后才能处理下一个窗口。不得跳过。 |
| C3 | **输出路径固定** | 最终 SKILL.md 写入 `~/.akg/evolved_skills/{dsl_key}/{skill_name}/SKILL.md`。**不得写入 `.opencode/skills/` 或其他位置。** |
| C4 | **按算子特性分组** | 经验按算子特性大类 + 问题子类型二级分组。大类：`elemwise`（逐元素）、`reduction`（归约）、`matmul`（矩阵乘）、`attention`（注意力机制）。子类型描述问题特征（如 broadcast、large-reduce、fused-elemwise）。skill_name 格式为 `op-insight-{category}-{subtype}`，如 `op-insight-elemwise-broadcast`、`op-insight-reduction-large-reduce`。一个 skill 包含该子类型下的**所有**优化手段。 |
| C5 | **泛化要求** | 正文和 description 中禁止出现具体算子名（softmax、relu、layernorm、gelu 等）和具体 shape，必须用问题类别（如"逐元素类算子"、"归约类算子"）替代。可附关键代码片段来说明做法。 |

---

## Step 1: 确认输入

询问用户：

> 请提供要提取优化经验的对话文件路径（`.jsonl` 文件或目录）。
> 另外请确认 DSL 标识（如 triton_ascend、triton_cuda、cpp），用于组织输出路径。

确认文件存在后进入 Step 2。

---

## Step 2: 预处理

运行预处理脚本：

```bash
python @scripts/preprocess.py <文件路径> --output-dir <工作目录>
```

默认工作目录为 `~/.akg/dev_insight_workdir/{timestamp}/`。

脚本完成后：

1. 读取 `manifest.json`，向用户报告压缩情况和窗口数量
2. **记录工作目录绝对路径**，后续所有步骤直接使用此路径读写文件，不要再去推断或搜索

向用户报告时**必须包含工作目录路径**，格式示例：

> 预处理完成。工作目录：`C:\Users\xxx\.akg\dev_insight_workdir\20260321_143025\`
> 原始对话 xxx 字符 → 压缩后 xxx 字符（xxx 倍），共 N 个窗口。

---

## Step 3: 逐窗口提取

**严格遵守：读一个窗口 → 生成该窗口的 insights → 写入文件 → 再读下一个窗口。**

流程伪代码（以 3 个窗口为例）：

```
读取 {工作目录}/window_0.json → 分析 → 写入 {工作目录}/insights_0.json
读取 {工作目录}/window_1.json → 分析 → 写入 {工作目录}/insights_1.json
读取 {工作目录}/window_2.json → 分析 → 写入 {工作目录}/insights_2.json
```

**禁止**：先读完所有 window 再批量生成 insights。每个窗口必须走完"读取→分析→写入"完整周期后才能开始下一个。

---

对 manifest 中的每个窗口 i（从 0 到 N-1），执行以下操作：

### 3.1 读取当前窗口

读取 `{工作目录}/window_{i}.json`（**只读这一个文件**），将 turns 格式化为：

```
[Turn 0] 用户: <user_content>
[Turn 0] Agent: <assistant_content>
...
```

### 3.2 分析并提取

仔细阅读当前窗口的对话内容，识别**用户明确提出的算子优化建议**。关注：

1. **优化策略**：用户建议了什么做法（如调整 block size、改变 tiling 方式、利用 shared memory）
2. **适用条件**：该策略在什么场景下适用（算子类型特征、数据特征、硬件特征）
3. **效果数据**：用户反馈的 speedup、性能对比
4. **失败尝试**：用户建议了但没效果或变差的方法

**提取原则**：
- 只提取用户明确说的，不要补充教科书知识
- 用问题类别（"归约类算子"、"切片重排操作"）替代具体算子名（"softmax"、"window_partition"）——参考 C5 约束
- 与算子优化无关的内容一律跳过
- method 字段用文字描述方法论，关键代码片段放 code 字段
- 不要重复写前面已经提到的内容，宁少勿滥

每条经验格式：

```json
{
  "title": "2-3 个关键词概括",
  "category": "算子特性大类：elemwise / reduction / matmul / attention",
  "subtype": "问题子类型（kebab-case，如 broadcast、large-reduce、fused-qkv），不要太细",
  "scenario": "什么场景下考虑此优化（不含具体算子名和具体 shape）",
  "method": "方法论描述",
  "code": "对话中出现的关键代码片段（如有，无则留空字符串）",
  "effect": "性能数据或失败原因",
  "evidence": "用户原话摘录",
  "is_effective": true
}
```

`category` 判定标准：
- **elemwise**：逐元素运算（激活函数、逐点变换、广播运算、elementwise 融合等）
- **reduction**：归约类运算（求和、求最值、归一化、扫描等，含沿某维度的聚合操作）
- **matmul**：矩阵乘法类（含批量矩阵乘、线性层、卷积等计算密集型操作）
- **attention**：注意力机制类（含 QKV 变换、score 计算、mask、softmax-V 等组合模式）

如果一条经验跨多个类别（如"融合 reduction + elemwise"），归入主要受益的那个类别，不要多处提及，能合并就合并。

### 3.3 写入检查点

将提取结果写入 `{工作目录}/insights_{i}.json`：

```json
{ "window_id": 0, "insights": [ ... ] }
```

如果无算子优化内容，写入 `{ "window_id": 0, "insights": [] }`。

**写入完成后，才能开始处理下一个窗口（回到 3.1 读取 window_{i+1}）。**

### 3.4 报告进度

每个窗口处理完毕后简要报告：`窗口 {i}: 提取了 N 条`（空窗口无需报告）。

---

## Step 4: 归并去重

### 4.1 汇总

读取所有 `{工作目录}/insights_*.json`，合并全部经验。

如果总数为 0，告知用户"该对话中未发现可提取的算子优化经验"，流程结束。

### 4.2 子类型语义合并（关键步骤）

不同窗口提取的 subtype 可能名称不同但本质相同（因为各窗口独立命名）。**必须先合并语义相近的子类型，再生成 skill。**

合并方法：

1. **列出所有 (category, subtype) 组合及其经验条数**
2. **逐对比较**同一 category 下的不同 subtype，分析它们的 method、code、scenario 详细内容：
   - 操作模式相似？（如都涉及切片/重排/索引映射）
   - 优化手法可复用？（如都用了批量加载、并行策略）
   - 适用场景重叠？（如都针对非连续内存访问）
3. **合并判定**：满足以上任意两条即应合并，合并后取一个更泛化的 subtype 名
4. 合并后每个 category 下的 subtype **最多不超过 3 个**，如果超过则继续合并最相似的

示例（用户实际遇到的情况）：

```
合并前（7 个 elemwise subtype）：
  window-slice, index-select, window-partition, index-calculation,
  task-scheduling, unbind, autograd

分析：window-slice / window-partition / index-select / unbind / index-calculation / task-scheduling
      → 都涉及切片、索引、重排，优化手法（批量加载、并行分配）可复用
      → 合并为 slice-rearrange
      autograd → 前后向分离，保留

合并后（3 个 elemwise subtype）：
  op-insight-elemwise-slice-rearrange
  op-insight-elemwise-autograd
```

### 4.3 生成 SKILL.md

对合并后的每个 (category, subtype) 组执行：

1. **经验去重**：语义相同的只保留描述最完整的版本
2. **泛化检查**：确认所有 scenario/method/description 中不含具体算子名和具体 shape（C5）
3. **质量过滤**：去掉无性能数据佐证且用户未确认的推测性经验
4. **失败归类**：`is_effective: false` 的经验放在对应 skill 的"适用边界"中

**只为有经验的组合**生成 SKILL.md——空组不生成文件。

**命名规范**（参考 `triton-ascend-case-{category}-{subtype}` 风格）：
- skill_name = `op-insight-{category}-{subtype}`
- `{category}` 是 elemwise / reduction / matmul / attention
- `{subtype}` 用 kebab-case 描述合并后的**问题子类型**（1-3 个词，泛化的问题特征，不是优化手法）
- 一个 skill 包含该子类型下的**所有**优化手段
- 参考 cases：`elemwise-broadcast`、`elemwise-cast`、`reduction-amax-large`
- 示例：`op-insight-elemwise-slice-rearrange`、`op-insight-reduction-large-axis`

**description 规范**（参考 cases 的 description 风格）：
- 50-100 字，概括该子类型下**包含的优化手段**及其适用场景
- 不含具体算子名和具体 shape
- 示例：`"逐元素切片重排类算子优化：批量加载减少内存访问、固定核心数交错分配实现并行、直接切片替代索引映射降低计算开销，适用于涉及非连续内存读写的切片和重排场景"`

### 4.4 输出路径（C3）

```
~/.akg/evolved_skills/{dsl_key}/{skill_name}/SKILL.md
```

### 4.5 文件格式

frontmatter：

```yaml
---
name: op-insight-{category}-{subtype}
description: "{该子类型下包含的优化手段概括及适用场景，50-100 字，不含具体算子名和 shape}"
category: cases
version: "1.0.0"
metadata:
  source: conversation-extraction
  dsl: {dsl_key}
---
```

正文（方法论描述 + 关键代码片段）：

```markdown
# {中文标题}

## 适用场景

{综合 insights 的 scenario，1-2 句话}

## 优化方法

### 1. {insight.title}

**场景**：{insight.scenario}
**做法**：{insight.method}
**效果**：{insight.effect}

（如果对话中有相关代码片段，附在此处用 code fence 包裹）

### 2. ...

## 适用边界

{boundaries}
```

### 4.6 写入前自检

- [ ] 每个 category 下的 subtype 不超过 3 个（经过 4.2 语义合并）
- [ ] skill_name 格式为 `op-insight-{category}-{subtype}`，category 是 elemwise/reduction/matmul/attention 之一（C4）
- [ ] subtype 描述的是合并后的泛化问题子类型（如 slice-rearrange），不是优化手法（C4）
- [ ] 输出路径在 `~/.akg/evolved_skills/` 下（C3）
- [ ] description 概括该子类型下包含的优化手段及适用场景，50-100 字，不含具体算子名和 shape（C5）
- [ ] description 风格参考 cases（列举优化手段 + 说明适用什么场景）
- [ ] 正文中不含具体算子名和具体 shape（C5）
- [ ] 每条 insight 的 title、method、effect 非空
- [ ] 描述内容和 dsl 匹配（如后端是 NPU 不要描述 GPU 特有知识）

### 4.7 报告

向用户报告：生成了几个 skill、各自路径、共包含多少条经验。
