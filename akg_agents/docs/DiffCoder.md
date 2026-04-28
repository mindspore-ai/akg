# DiffCoder 方案讨论

## 1. 问题背景

在 AI Kernel Generator 的 `CoderOnly` 工作流中，每次验证失败后 Coder 都会重新生成完整的 Triton 算子代码。这带来了三个问题：

1. **效率低**：大部分代码是正确的，完全重新生成浪费计算资源
2. **不稳定**：重新生成可能引入新的 Bug，导致"修了一个 Bug，冒出两个"
3. **上下文丢失**：每次重新生成，之前修复的内容可能被覆盖

一个典型场景：LLM 生成的 Triton kernel 中使用了 `tl.tanh`（Triton 不支持），只需要把这 1-2 行改成近似计算，而不需要重写整个算子。

## 2. Diff 格式选型：为什么选 Search/Replace

在设计增量修复方案前，首先需要确定 LLM 应该用什么格式来描述"改动"。主流方案对比如下：

### 2.1 候选方案

| 方案 | 格式示例 | LLM 友好度 | 可靠性 |
|------|---------|-----------|--------|
| Unified Diff | `@@ -10,3 +10,4 @@` | 低 | 低（LLM 经常算错行号和行数计数） |
| 行号定位 + 替换 | `{start_line: 42, end_line: 43, new_lines: [...]}` | 中 | 中（多个编辑时行号互相影响） |
| Search/Replace | `{old_string: "...", new_string: "..."}` | 高 | 高 |
| LINE#HASH（乐观锁） | `{pos: "42#VK", lines: "..."}` | 低 | 高（适合并发编辑场景） |
| AST 级别修改 | 解析 → 修改 AST 节点 → 重建代码 | 不适用 | 高（但会丢失格式和注释） |

### 2.2 选型结论

**选择 Search/Replace**，原因：

- LLM 擅长文本复述和修改，不擅长计算行号 / 行数 / 哈希值
- Search/Replace 天然支持删一行加两行、改多行等各种情况（old 和 new 行数可以不同）
- 不需要额外的基础设施（如哈希计算、行号追踪）
- 业界验证：Cursor（Edit 工具）、Aider、Claude Code 最终都收敛到了 search/replace 变体

### 2.3 排除 LINE#HASH 的原因

LINE#HASH（如 oh-my-opencode 项目中的 hashline-edit）通过对每行内容计算 2 字符哈希作为"乐观锁"，编辑时校验哈希是否匹配来防止覆盖他人修改。这个机制解决的是**并发冲突**问题。在我们的场景中：

- 没有并发 —— 生成和修复是同一个 pipeline
- 生成者就是修复者 —— LLM 完全知道当前内容
- 改动范围小且确定 —— 定位到错误行，换成正确实现

因此 LINE#HASH 的哈希校验在此场景下是纯开销，不引入。

## 3. 核心设计

### 3.1 Modification 数据结构

```python
@dataclass
class Modification:
    old_string: str       # 要替换的原始代码片段
    new_string: str       # 替换后的新代码片段
    reason: str = ""      # 修改原因
    replace_all: bool = False  # 是否替换所有匹配项
    anchor: str = ""      # 可选：附近的唯一标识（如函数签名），用于多匹配消歧
```

### 3.2 LLM 输出格式

```json
{
    "analysis": "对问题的分析说明",
    "modifications": [
        {
            "old_string": "要替换的原始代码片段（精确匹配原始代码）",
            "new_string": "替换后的新代码片段",
            "reason": "这处修改的原因",
            "replace_all": false,
            "anchor": ""
        }
    ],
    "summary": "修改总结"
}
```

### 3.3 各种改动场景的覆盖

```
场景 1：改两行（行数不变）
old: "    output = tl.tanh(input)\n    return output"
new: "    output = input * tl.sigmoid(1.702 * input)\n    return output"

场景 2：删一行加两行
old: "    output = tl.tanh(input)"
new: "    t = 1.702 * input\n    output = input * tl.sigmoid(t)"

场景 3：纯删除
old: "    # TODO: fix this\n    output = tl.tanh(input)"
new: "    output = input * tl.sigmoid(1.702 * input)"

场景 4：多处相同错误
old: "tl.tanh", new: "triton_gelu_approx", replace_all: true

场景 5：多处相似代码只改其中一处
old: "tl.tanh(x)", new: "x * tl.sigmoid(1.702 * x)", anchor: "def forward("
```

## 4. 多级匹配策略

LLM 输出的 `old_string` 可能与原始代码存在空白差异，需要多级容错匹配。

### 4.1 匹配降级流程

```
收到 old_string
    │
    ▼
L1: 精确匹配（str.find）
    │ 失败
    ▼
L2: 行级 trim 匹配（每行 strip 后滑动窗口比较）
    │ 失败
    ▼
L3: 空白规范化匹配（连续空白合并为单个空格后比较）
    │ 失败
    ▼
L4: 模糊匹配（基于编辑距离，需要置信度检查）
    │ 失败
    ▼
报错，返回失败信息
```

### 4.2 各级策略说明

| 优先级 | 策略 | 说明 | 适用场景 |
|--------|------|------|----------|
| 1 | 精确匹配 | 字符串完全一致 | 理想情况 |
| 2 | 行级 trim 匹配 | 忽略每行首尾空白 | LLM 输出的缩进有偏差 |
| 3 | 空白规范化匹配 | 连续空白统一为单个空格 | 空白字符不一致 |
| 4 | 模糊匹配 | 基于编辑距离的相似度匹配 | 有少量字符差异 |

### 4.3 关键设计细节

**匹配结果返回原始内容**：所有匹配级别找到匹配后，返回的是原始代码中的对应片段（而非 LLM 给的 old_string）。这确保后续 `str.replace()` 一定能替换成功。

```python
# 例：LLM 给的 old_string 缩进是 2 空格，原始代码是 4 空格
# trimmed_line_match 匹配成功后，返回原始代码中的 4 空格版本
# 这样 content.replace(matched, new_string) 才能正确工作
```

### 4.4 模糊匹配的置信度检查

模糊匹配存在**匹配到错误位置**的风险，需要加置信度防护：

```python
# 风险场景：Triton 算子中有两段相似代码
# 第 15 行: output = tl.load(input_ptr + offsets, mask=mask, other=0.0)
# 第 42 行: result = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
# 如果 LLM 的 old_string 有 typo，模糊匹配可能选错位置

# 防护策略：如果最佳匹配和次佳匹配的相似度太接近，拒绝匹配
if best_similarity - second_best_similarity < 0.1:
    return None  # 不够置信，报错让 LLM 重试
```

## 5. 重复匹配处理

当 `old_string` 在代码中出现多次时，需要明确的消歧策略。

### 5.1 决策流程

```
精确查找 old_string 出现次数
    │
    ├── 0 次 → 降级到模糊匹配
    │
    ├── 1 次 → 直接替换
    │
    └── N 次 → 有 anchor？
                ├── 有 → 从 anchor 位置开始找 old_string → 找到 → 替换
                │                                        → 没找到 → 报错
                └── 没有 → replace_all == true？
                            ├── 是 → 全部替换
                            └── 否 → 报错，提示三种消歧方式：
                                    1. 在 old_string 中加更多上下文
                                    2. 设 replace_all: true
                                    3. 用 anchor 指定位置
```

### 5.2 三种消歧手段

| 手段 | 适用场景 | 示例 |
|------|---------|------|
| 加上下文 | old_string 太短导致歧义 | 多包含前后几行，直到唯一 |
| replace_all | 所有匹配都需要改 | 全局替换 `tl.tanh` → 近似实现 |
| anchor | 多处相似代码只改其中一处 | `anchor: "def forward("` 指定在哪个函数里改 |

### 5.3 anchor 机制说明

anchor 不依赖行号，而是用**附近的唯一标识**定位。函数签名、类名、注释标记等天然唯一，LLM 指定这些比指定行号可靠得多。

```python
def apply_edit_with_anchor(code: str, anchor: str, old: str, new: str) -> str:
    anchor_pos = code.find(anchor)
    if anchor_pos == -1:
        raise EditError(f"Anchor not found: {anchor}")
    match_pos = code.find(old, anchor_pos)
    if match_pos == -1:
        raise EditError(f"'{old}' not found after anchor '{anchor}'")
    return code[:match_pos] + new + code[match_pos + len(old):]
```

## 6. 多个 Modification 的顺序问题

当一次修复包含多个 Modification 时，需要注意**前一个替换可能影响后续匹配**。

### 6.1 问题示例

```python
# 原始代码：
# 第 10 行: output = tl.tanh(x)
# 第 20 行: grad = tl.tanh(dx) * (1 - output ** 2)

# Modification 1: 改第 10 行（增加了 1 行 → 后续行号偏移）
# Modification 2: 改第 20 行（old_string 仍基于原始代码，但代码已被 mod1 改动）
```

### 6.2 当前处理方式

逐个顺序应用，每次替换后在**修改后的代码**上执行下一个匹配。这意味着：

- 如果 mod2 的 old_string 没有被 mod1 改动过，匹配正常
- 如果 mod1 恰好修改了 mod2 的 old_string 所在区域，mod2 会匹配失败并报错

### 6.3 可选增强：冲突预检测

在应用前扫描所有 modification 的 old_string，检测是否存在重叠：

```python
def detect_conflicts(code: str, modifications: List[Modification]) -> List[str]:
    warnings = []
    for i, mod_a in enumerate(modifications):
        for j, mod_b in enumerate(modifications):
            if i >= j:
                continue
            # 如果 mod_a 的 old_string 包含 mod_b 的 old_string 的一部分
            if mod_a.old_string in mod_b.old_string or mod_b.old_string in mod_a.old_string:
                warnings.append(f"Modification {i+1} and {j+1} may conflict (overlapping regions)")
    return warnings
```

## 7. 工作流集成

### 7.1 整体流程

```
Coder → Verifier → (通过) → END
                 ↘ (失败)
               Conductor（分析错误，生成专家建议）
                    │
                    ▼
               DiffCoder（增量修复）→ Verifier → (通过) → END
                    ▲                        ↘ (失败)
                    └──────── Conductor ←──────┘
```

### 7.2 与 CoderOnly Workflow 的对比

| 特性 | CoderOnly Workflow | DiffCoder Workflow |
|------|-------------------|-------------------|
| 初始代码生成 | Coder | Coder |
| 失败后修复方式 | Coder 完全重新生成 | DiffCoder 增量修复 |
| Conductor 分析 | 有 | 有 |
| 修改粒度 | 整个文件 | 精确到代码片段 |
| 修复稳定性 | 可能引入新问题 | 保留正确代码 |

### 7.3 DiffCoder 的数据流

```
原始代码 + 错误日志 + Conductor建议
        │
        ▼
   ┌─────────┐
   │ DiffCoder│  ─── 调用 LLM，生成 JSON 格式的修改方案
   └────┬─────┘
        │
        ▼
  ┌────────────┐
  │ 解析修改列表 │  ─── 从 LLM 输出中提取 Modification 列表
  └────┬───────┘
        │
        ▼
  ┌────────────┐
  │ CodeMatcher │  ─── 多级匹配策略定位代码片段
  └────┬───────┘
        │
        ▼
  ┌────────────┐
  │ DiffApplier │  ─── 执行替换，生成 unified diff
  └────┬───────┘
        │
        ▼
    DiffResult（修改后的代码 + diff + 统计信息）
```

### 7.4 LangGraph 集成

DiffCoder 通过 `NodeFactory.create_diff_coder_node()` 创建 LangGraph 节点。

**输入（从 State 中读取）**：
- `coder_code`：当前代码
- `verifier_error`：验证错误日志
- `conductor_suggestion`：Conductor 的专家建议

**输出（写回 State）**：
- `coder_code`：更新为修改后的代码
- `diff_coder_success`：是否修改成功
- `diff_coder_diff`：unified diff 文本
- `diff_coder_message`：结果消息
- `diff_coder_modifications`：成功应用的修改数

**路由**：
- `RouterFactory.create_verifier_router_with_conductor()`：Verifier 失败后路由到 Conductor
- `RouterFactory.create_conductor_router_to_diff_coder()`：Conductor 分析后路由到 DiffCoder

### 7.5 配置文件

配置文件位于 [`config/diff_coder_workflow.yaml`](../python/ai_kernel_generator/config/diff_coder_workflow.yaml)：

```yaml
agent_info:
  coder:
    possible_next_agent: [verifier]
  verifier:
    possible_next_agent: [finish, conductor]
  conductor:
    possible_next_agent: [finish, diff_coder]
  diff_coder:
    possible_next_agent: [verifier]

start_agent: coder
mandatory_llm_analysis: [verifier]

limitation_info:
  required:
    max_step: 20
```

## 8. 匹配失败的处理

当所有匹配级别都失败时，需要一个清晰的错误恢复策略。

### 8.1 当前行为

匹配失败的 Modification 被跳过，记录到 `DiffResult.errors`。只要至少有一个 Modification 成功，整体结果仍为 `success=True`。

### 8.2 可选增强：错误信息回传 LLM

将匹配失败的信息（包含代码中的相似片段）回传给 LLM，让它修正 old_string 后重试：

```python
# 回传信息示例
error_feedback = """
Modification 2 failed: old_string not found in code.

Your old_string:
    output = tl.tanh(input)

Most similar fragment in code (similarity=0.85):
    output = tl.tanh(x)

Please correct old_string and retry.
"""
```

这可以通过在 Conductor → DiffCoder 的循环中传递 `diff_coder_errors` 来实现。

## 9. Prompt 设计要点

### 9.1 核心规则

1. **唯一性**：`old_string` 必须在原始代码中只出现一次。如果太短可能匹配多处，多包含前后几行上下文直到唯一。如果确实要改所有匹配项，设 `replace_all: true`
2. **精确复述**：`old_string` 必须是原始代码中精确存在的片段，包括所有空格、缩进和换行符
3. **保持格式**：`new_string` 必须保持与原代码相同的缩进风格
4. **最小修改**：每次修改应尽量小，不要重写不必要的代码
5. **顺序执行**：如果有多处修改，按照从上到下的顺序排列，避免位置冲突
6. **不要转义**：JSON 中的换行符应该是实际换行，不要使用字面量 `\n` 字符串

### 9.2 Prompt 模板参考

```
你是一个专业的代码修改专家 Agent，擅长对 Kernel 代码进行精确的修改和修复。

## 任务

根据原始代码和错误信息，生成精确的搜索替换修改指令。

## 原始代码
{{ original_code }}

## 错误日志
{{ error_log }}

## 专家建议
{{ conductor_suggestion }}

## 输出格式

输出 JSON，包含 modifications 数组，每项有 old_string / new_string / reason / replace_all / anchor。

## 关键规则

1. old_string 必须在原始代码中只出现一次（加足够上下文确保唯一）
2. 如果所有同名调用都要改，设 replace_all: true
3. 如果多处相似代码只改某一处，用 anchor 指定函数签名
4. old_string 和 new_string 行数可以不同（可以删行、加行）
5. 只改必要的部分，不要重写正确的代码
```

## 10. 核心组件

| 组件 | 文件路径 | 职责 |
|------|----------|------|
| `DiffCoder` | `core/agent/diff_coder.py` | 核心 Agent，调用 LLM 生成修改方案并应用 |
| `CodeMatcher` | `utils/diff_utils.py` | 多级代码匹配器，定位需要修改的代码片段 |
| `DiffApplier` | `utils/diff_utils.py` | 差异应用器，执行修改并生成 unified diff |
| `Modification` | `utils/diff_utils.py` | 数据类，表示单个修改操作 |
| `DiffResult` | `utils/diff_utils.py` | 数据类，表示修改操作的完整结果 |
| `DiffCoderWorkflow` | `workflows/diff_coder_workflow.py` | 包含 DiffCoder 的 LangGraph 工作流 |
| `diff_edit.j2` | `resources/prompts/diff/diff_edit.j2` | Prompt 模板 |

## 11. 测试

- **单元测试**：`aikg/tests/ut/test_diff_coder.py`（覆盖解析、匹配、应用等核心逻辑）
- **集成测试**：`aikg/tests/bench/test_bench_diff_coder_workflow.py`（端到端工作流测试）

## 12. 待讨论 / 后续增强

### 12.1 当前方案的改进项

| 增强项 | 优先级 | 说明 |
|--------|--------|------|
| anchor 锚点消歧 | 高 | Modification 增加 anchor 字段，用函数签名等唯一标识定位，解决多匹配消歧 |
| 模糊匹配置信度检查 | 高 | 最佳/次佳匹配相似度差值不足时拒绝匹配，防止选错位置 |
| 匹配失败回传 LLM | 中 | 将失败信息（含最相似片段）回传给 LLM 重试 |
| Modification 冲突预检测 | 中 | 应用前检测 old_string 之间是否重叠 |
| Prompt 唯一性规则强化 | 中 | 从"2-3 行上下文"改为"确保只出现一次" |
| 性能优化 | 低 | Levenshtein 在大文件场景下可用 difflib.SequenceMatcher 替代 |

### 12.2 业界替代方案调研

以下是业界在 LLM 代码修复领域的其他方案，可作为后续演进方向参考。

#### 方案 A：Semantic Edit（语义编辑 / Fast Apply）

**代表项目**：Morph（morph-v3-fast）、Relace（relace-apply-3）、Cursor（Instant Apply）

**核心思路**：LLM 不输出 old/new 文本对，而是输出带省略标记（`// ... existing code ...`）的代码片段，只写真正改动的部分。然后由一个专门训练的轻量 apply model（通常 7B）将编辑片段与原始文件合并。

```python
# LLM 输出示例（不是完整文件，也不是 search/replace）
@triton.jit
def kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # ... existing code ...
    
    t = 1.702 * input
    output = input * tl.sigmoid(t)  # GELU approximation
    
    # ... existing code ...
```

**关键数据**：
- Morph Fast Apply：10,500 tokens/s，98% 合并准确率，500 行文件约 0.8s
- Cursor：1,000 tokens/s（70B 模型），比原始推理提速 9-13x
- 对比 search/replace：token 消耗减少约 40%

**优势**：LLM 写起来最自然（不需要精确复述 old_string）；准确率远高于 search/replace（98% vs 70-80%）；大文件场景优势显著。

**劣势**：需要额外部署一个 apply model（7B 模型）；需要 GPU 推理资源。

**适用建议**：如果 Triton kernel 较长（200+ 行）且当前 search/replace 匹配失败率较高，可优先调研。Morph 提供 API 可直接调用，无需自建。

**参考资料**：
- Morph：https://morphllm.com/edit-formats
- Cursor Blog：https://cursor.com/blog/instant-apply

#### 方案 B：Whole File Rewrite + Speculative Decoding

**代表项目**：Cursor

**核心思路**：不做 diff，直接让 LLM 输出整个修改后的文件。但利用 speculative decoding 加速：原始文件的每一行作为"draft token"，模型只需要在真正改动的地方否决 draft 并生成新 token，其余直接 accept。

```
原文件行:  output = tl.tanh(input)     ← draft → 模型否决 → 生成新行
原文件行:  return output               ← draft → 模型接受 → 跳过
```

**关键数据**：Cursor 70B 模型下约 1,000 tokens/s，验证命中率约 70%。

**优势**：彻底消除匹配失败问题（不需要匹配）；输出结果就是最终代码，无需合并。

**劣势**：需要自建 speculative decoding 推理基础设施，工程门槛高；短文件无优势。

**适用建议**：当前阶段暂不适合，但如果后续有大量 kernel 批量修复需求且有 GPU 推理基础设施，值得考虑。

#### 方案 C：RL 训练的专用 Triton 修复模型

**代表项目**：Dr. Kernel（2026.02）、TritonRL（ICLR 2026 submitted）

**核心思路**：不使用通用 LLM + diff 的两阶段方式，而是用强化学习专门训练一个理解 Triton 语义的模型，端到端地生成/修复 kernel 代码。

**Dr. Kernel（14B）**：
- 基于 KernelGYM 环境训练，支持多轮交互（生成 → 编译 → profile → 修复）
- 奖励函数基于实际 profiling 结果（Profiling-based Rewards），不只是"编译通过"
- 使用 TRLOO（Turn-level Reinforce-Leave-One-Out）解决多轮 RL 的梯度偏差问题
- KernelBench Level-2：31.6% 的 kernel 达到 1.2x 加速（超过 Claude-4.5-Sonnet 的 26.7%），多次尝试选最优可达 47.8%
- 代码和模型已开源

**TritonRL（Qwen3-8B）**：
- 使用分层奖励分解（hierarchical reward decomposition）和数据混合策略
- KernelBench 上达到 SOTA 的正确率和加速比

**优势**：端到端理解 Triton 语义，不需要设计 diff 格式；修复质量更高（考虑性能而非仅正确性）。

**劣势**：需要自行训练或 fine-tune；当前仍处于研究阶段；模型泛化到新 kernel 模式的能力待验证。

**适用建议**：长期方向。如果目标不仅是修复编译错误，而是生成高性能 kernel，值得持续关注。Dr. Kernel 的 KernelGYM 环境可以直接用于评估。

**参考资料**：
- Dr. Kernel：https://arxiv.org/abs/2602.05885
- TritonRL：https://openreview.net/forum?id=feJ5T9sFSJ

#### 方案 D：增强反馈重试循环（RGym 模式）

**代表项目**：RGym（NeurIPS 2025）

**核心思路**：不改 diff 格式本身，而是优化修复的反馈循环。将编译/运行的错误信息（call stack、blamed commits、定位信息）精准回传给 LLM，进行多轮重试。

```
Round 1: LLM 修复 tl.tanh → tl.sigmoid 近似
         编译错误: shape mismatch at line 45
Round 2: LLM 根据错误信息定位到 line 45，修复 shape
         运行通过，但精度不够
Round 3: LLM 调整近似精度
         验证通过
```

**关键数据**：GPT-5 Thinking 在此模式下达到 43.36% 的 kernel bug 修复率，成本约 $0.20/bug。

**优势**：与当前 search/replace 方案完全兼容，不需要改底层机制；实现成本极低；立即可用。

**劣势**：修复率受限于 LLM 能力和错误信息的质量。

**适用建议**：**最推荐短期落地的增强方向**。当前 Conductor → DiffCoder → Verifier 循环已具备雏形，可以借鉴 RGym 的以下做法：
1. 在 Verifier 的错误反馈中加入更精准的定位信息（call stack、具体报错行）
2. 将 DiffCoder 匹配失败的信息（含代码中最相似片段）也作为下一轮输入
3. 多轮重试的退出策略优化

**参考资料**：
- RGym：https://arxiv.org/abs/2511.15757

#### 方案 E：行范围编辑（SWE-agent 模式）

**代表项目**：SWE-agent

**核心思路**：交互式导航 + 行号范围编辑。Agent 先打开文件、滚动到目标位置，再用行号范围指定编辑区域。

```
open file.py
goto 42
edit 42:44
    t = 1.702 * input
    output = input * tl.sigmoid(t)
end_of_edit
```

使用环境变量（`CURRENT_FILE`、`CURRENT_LINE`、`WINDOW`）追踪当前位置。

**优势**：交互式导航天然解决定位问题；行号范围编辑精确无歧义。

**劣势**：更适合交互式 agent，不太适合批量 pipeline；LLM 的行号感知仍然不够可靠。

**适用建议**：可以作为 search/replace 的 **fallback 策略** —— 当 old_string 匹配不到时，降级为"请指定行号范围"让 LLM 用行号重试。

### 12.3 工程性优化

以下是对当前代码实现的工程性优化点，不涉及架构变动，每项独立可做，风险低。

#### 12.3.1 Levenshtein 性能优化

当前 `CodeMatcher.levenshtein_distance` 每次分配完整的 O(n*m) 二维矩阵。对 200 行 kernel 的模糊匹配，每个滑动窗口都会创建约 200x200 的矩阵。

**方案 a**：改为两行 DP，空间从 O(n*m) 降到 O(min(n,m))：

```python
def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    prev = list(range(len(s2) + 1))
    curr = [0] * (len(s2) + 1)
    for i in range(1, len(s1) + 1):
        curr[0] = i
        for j in range(1, len(s2) + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            curr[j] = min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + cost)
        prev, curr = curr, prev
    return prev[len(s2)]
```

**方案 b**：直接用 `difflib.SequenceMatcher`（已 import `difflib`），更适合代码文本的相似度计算：

```python
from difflib import SequenceMatcher

def similarity_ratio(s1: str, s2: str) -> float:
    return SequenceMatcher(None, s1, s2).ratio()
```

#### 12.3.2 模糊匹配窗口行数容差

当前 `fuzzy_match` 用 `search_line_count` 锁死滑动窗口大小。如果 LLM 的 `old_string` 比实际代码多了或少了一个空行，行数不匹配就完全匹配不到。

**优化**：尝试 `search_line_count - 1` 到 `search_line_count + 1` 的窗口范围：

```python
for delta in [0, -1, 1]:
    window_size = search_line_count + delta
    if window_size <= 0:
        continue
    for i in range(len(content_lines) - window_size + 1):
        window = '\n'.join(content_lines[i:i + window_size])
        # ... 计算相似度
```

#### 12.3.3 空编辑检测

如果 LLM 输出了 `old_string == new_string` 的 modification，当前代码会"成功替换"但实际没有变化，虚增 `success_count`。

```python
# 在 apply_single_replacement 开头加
if old_string == new_string:
    return content, False, "old_string and new_string are identical, skipping", 0
```

#### 12.3.4 错误日志截断策略

当前 `error_log[:5000]` 直接截取前 5000 字符，可能把关键的 traceback 尾部（实际错误信息和最近的 call frame）截掉。

**优化**：保留头部 + 尾部：

```python
def truncate_error_log(error_log: str, max_len: int = 5000) -> str:
    if len(error_log) <= max_len:
        return error_log
    head_len = max_len // 3
    tail_len = max_len - head_len - 50
    return (
        error_log[:head_len]
        + f"\n\n... ({len(error_log) - head_len - tail_len} chars truncated) ...\n\n"
        + error_log[-tail_len:]
    )
```

#### 12.3.5 匹配级别追踪

当前 `CodeMatcher.find_match` 只在 `logger.debug` 记录了用哪一级匹配成功，没有把信息传递到 `DiffResult`。这对分析 LLM 输出质量很有价值（如果模糊匹配占比过高，说明 prompt 需要优化）。

**优化**：让 `find_match` 返回匹配级别：

```python
@classmethod
def find_match(cls, content: str, search: str) -> Tuple[Optional[str], str]:
    """返回 (匹配结果, 匹配级别)"""
    result = cls.exact_match(content, search)
    if result:
        return result, "exact"
    result = cls.trimmed_line_match(content, search)
    if result:
        return result, "trimmed"
    result = cls.whitespace_normalized_match(content, search)
    if result:
        return result, "whitespace_normalized"
    result = cls.fuzzy_match(content, search, threshold=0.8)
    if result:
        return result, "fuzzy"
    return None, "none"
```

然后在 `DiffResult` 中统计各级别的匹配次数，用于后续的可观测性分析。

#### 12.3.6 Prompt 示例换行符写法

当前 prompt 示例中使用 `\n` 表示换行：

```json
"old_string": "def relu(x):\n    return x if x > 0 else 0"
```

LLM 可能会学着用字面量 `\n` 而不是真实换行，导致解析后的 `old_string` 与实际代码不匹配。建议示例改为真实的多行 JSON string，或在注意事项中明确说明 `old_string` 中的换行就是实际换行符，不要手写 `\n`。

#### 12.3.7 DiffResult.errors 默认值

当前使用 `None` + `__post_init__` 处理 mutable default argument。更 Pythonic 的方式：

```python
from dataclasses import dataclass, field

@dataclass
class DiffResult:
    # ...
    errors: List[str] = field(default_factory=list)
```

#### 12.3.8 记录 LLM 原始输出

当前 `_parse_modifications` 解析失败时只 log 了 error，没有保存 LLM 的原始输出。调试"为什么 LLM 没给出正确 modification"时需要回溯原始输出。

**优化**：在 `DiffResult` 中加一个可选的 `raw_llm_output` 字段：

```python
@dataclass
class DiffResult:
    # ... 现有字段
    raw_llm_output: str = ""
```

#### 12.3.9 补充 CodeMatcher 单元测试

当前只有端到端测试（调用 LLM）。`CodeMatcher` 的四级匹配逻辑是纯确定性代码，应该有独立的单元测试覆盖边界情况：

- 精确匹配出现多次时的行为
- trim 匹配时 search 末尾有空行
- 空白规范化匹配时 tab vs space
- 模糊匹配的阈值边界（0.79 vs 0.80）
- 模糊匹配是否返回最佳匹配（而非第一个超过阈值的）
- 空字符串、search 比 content 长等边界输入

这些测试不依赖 LLM，可在 CI 中秒级完成。

#### 12.3.10 工程优化汇总

| 优化项 | 类型 | 影响 | 难度 |
|--------|------|------|------|
| Levenshtein 两行 DP / SequenceMatcher | 性能 | 内存减少 ~200x | 低 |
| 模糊匹配窗口 +/-1 行容差 | 鲁棒性 | 提升匹配成功率 | 低 |
| old==new 空编辑检测 | 正确性 | 防止虚增 success_count | 低 |
| 错误日志保留头尾 | 效果 | LLM 能看到关键错误信息 | 低 |
| 匹配级别追踪 | 可观测性 | 可分析 LLM 输出质量 | 低 |
| Prompt 示例换行符修正 | 鲁棒性 | 减少 LLM 换行符混淆 | 低 |
| DiffResult.errors 用 field(default_factory) | 代码质量 | 避免 mutable default 陷阱 | 低 |
| 记录 LLM 原始输出 | 可调试性 | 修复失败时可回溯 | 低 |
| CodeMatcher 单元测试 | 质量保障 | CI 快速验证匹配逻辑 | 中 |

### 12.4 方案对比总结

| 方案 | 匹配准确率 | 工程复杂度 | 落地时间 | 核心价值 |
|------|-----------|-----------|---------|---------|
| 当前：Search/Replace | 70-80% | 低 | 已实现 | 简单可靠的基线方案 |
| A：Semantic Edit | 98% | 中 | 1-2 周 | 大幅提升匹配准确率 |
| B：Speculative Decoding | ~100% | 高 | 1-2 月 | 消除匹配问题，但基础设施要求高 |
| C：RL 专用模型 | - | 高 | 3+ 月 | 端到端理解 Triton，长期方向 |
| D：增强反馈循环 | 不改变匹配率 | 低 | 1 周内 | 通过更好的错误反馈提升修复成功率 |
| E：行范围编辑 | 中 | 低 | 1 周内 | 作为 search/replace 的 fallback |

**建议推进顺序**：D（反馈增强）→ 12.1 的改进项 → A（Semantic Edit 调研）→ C（长期关注）

## 13. 实现进度表

> **说明**：本项目实际实现命名为 **FixCodeGen**（而非 DiffCoder），集成在现有 CoderOnlyWorkflow 中，由 Conductor 决策路由到 `coder`（完全重写）或 `fix_code_gen`（增量修复）。

### 13.1 核心组件实现状态

| 设计文档章节 | 设计要素 | 实现状态 | 实现文件 | 备注 |
|------------|---------|---------|---------|------|
| 3.1 | `Modification` 数据类 | ✅ 已完成 | `op/utils/diff_utils.py` | MVP 版本含 `old_string`, `new_string`, `reason` |
| 3.1 | `Modification.replace_all` | ❌ 待实现 | — | 留给实习生 |
| 3.1 | `Modification.anchor` | ❌ 待实现 | — | 留给实习生 |
| — | `DiffResult` 数据类 | ✅ 已完成 | `op/utils/diff_utils.py` | 使用 `field(default_factory=list)` |
| 4.1 | L1 精确匹配 | ✅ 已完成 | `op/utils/diff_utils.py` → `CodeMatcher.exact_match` | |
| 4.1 | L2 行级 trim 匹配 | ✅ 已完成 | `op/utils/diff_utils.py` → `CodeMatcher.trimmed_line_match` | |
| 4.1 | L3 空白规范化匹配 | ❌ 待实现 | — | 留给实习生 |
| 4.1 | L4 模糊匹配 | ❌ 待实现 | — | 留给实习生 |
| 4.4 | 模糊匹配置信度检查 | ❌ 待实现 | — | 依赖 L4 |
| — | `DiffApplier` 差异应用器 | ✅ 已完成 | `op/utils/diff_utils.py` → `DiffApplier` | 含 unified diff 生成 |
| — | `parse_modifications` 解析器 | ✅ 已完成 | `op/utils/diff_utils.py` | 支持 JSON / markdown 包裹容错 |
| 9.2 | Prompt 模板 | ✅ 已完成 | `op/resources/prompts/fix_code_gen/edit.j2` | |
| 10 | FixCodeGen 节点 | ✅ 已完成 | `op/langgraph_op/nodes.py` → `NodeFactory.create_fix_code_gen_node` | |
| 7.4 | State 字段扩展 | ✅ 已完成 | `op/langgraph_op/state.py` | `fix_code_gen_success/diff/message` |
| 7.4 | Conductor 路由扩展 | ✅ 已完成 | `op/langgraph_op/routers.py` | `enable_fix_code_gen` 参数 |
| 7.4 | Conductor Prompt 调整 | ✅ 已完成 | `op/resources/prompts/conductor/analyze.j2` | A1/A2 分类引导 |
| 7.1 | Workflow 集成 | ✅ 已完成 | `op/workflows/coder_only_workflow.py` | 默认启用 fix_code_gen |

### 13.2 测试实现状态

| 测试类型 | 实现状态 | 文件 | 覆盖内容 |
|---------|---------|------|---------|
| UT — CodeMatcher | ✅ 已完成 | `tests/ut/test_fix_code_gen.py` | L1/L2 匹配、边界情况、降级逻辑 |
| UT — DiffApplier | ✅ 已完成 | `tests/ut/test_fix_code_gen.py` | 单/多修改、部分失败、空编辑检测 |
| UT — parse_modifications | ✅ 已完成 | `tests/ut/test_fix_code_gen.py` | JSON / markdown 包裹 / 格式异常 |
| UT — FixCodeGen 节点 (Mock) | ✅ 已完成 | `tests/ut/test_fix_code_gen.py` | Mock LLM 验证 State 更新 |
| ST — 真实 LLM 调用 | ✅ 已完成 | `tests/op/st/test_fix_code_gen.py` | 修复缺少 import torch 的代码 |
| 集成测试 — 端到端工作流 | ❌ 待实现 | — | 留给实习生 |

### 13.3 待实现功能（按优先级排序，可分配给实习生）

| 优先级 | 功能 | 对应设计文档章节 | 难度 | 负责人 |
|--------|------|----------------|------|--------|
| 高 | `replace_all` 全局替换 | 5.1 | 低 | |
| 高 | `anchor` 锚点消歧 | 5.3 | 中 | |
| 高 | L3 空白规范化匹配 | 4.2 | 中 | |
| 高 | L4 模糊匹配 + 置信度检查 | 4.2 + 4.4 | 中 | |
| 中 | 匹配失败信息回传 LLM 重试 | 8.2 | 高 | |
| 中 | Modification 冲突预检测 | 6.3 | 中 | |
| 中 | 匹配级别追踪（可观测性） | 12.3.5 | 低 | |
| 中 | 模糊匹配窗口行数 +/-1 容差 | 12.3.2 | 低 | |
| 中 | 错误日志截断策略（保留头尾） | 12.3.4 | 低 | |
| 低 | Levenshtein 性能优化 | 12.3.1 | 中 | |
| 低 | 记录 LLM 原始输出到 DiffResult | 12.3.8 | 低 | |
| 低 | 增强反馈循环（RGym 模式） | 12.2 方案 D | 高 | |
| 低 | 行范围编辑 fallback | 12.2 方案 E | 高 | |
| 低 | 端到端集成测试 | 11 | 高 | |
