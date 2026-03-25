---
name: skills_creator
description: |
  用于从对话历史中提取可复用知识并生成新的 OpenCode skill。当用户要求：
  (1) "凝练当前上下文" 或 "总结上下文为skills";
  (2) "任务成功" 后希望保存经验;
  (3) 查看 "available skills" 并希望创建新技能时使用。将有价值的发现转化为可重复使用的技能。
version: 1.0.0
date: 2026-03-04
allowed-tools:
  - Read
  - Write
  - Edit
  - Grep
  - Glob
  - WebSearch
  - WebFetch
  - Skill
  - Bash
---

# Skills 创建器

## Problem

当用户完成复杂的、有价值的任务后，这些经验通常只存在于对话历史中。下次遇到类似问题时，需要重新探索，浪费时间和精力。需要将对话中的知识提取为可复用的 skill。

## Context / Trigger Conditions

当以下任一条件满足时，调用此技能：

- 用户明确要求"凝练当前上下文"、"总结上下文为skills"或"skills summary"
- 用户说"任务成功"并希望保存经验
- 用户询问 "available skills" 后希望创建新技能
- 完成了复杂的调试任务，发现了非显而易见的解决方案
- 通过试错发现了有效的解决方案

## Solution

### Step 1: 分析对话

从对话历史中提取以下信息：

- **原始目标**：用户最初希望完成什么任务
- **核心挑战**：任务中的关键难点和决策点
- **解决步骤**：抽象出可泛化的行动序列
- **避坑指南**：常见的错误或陷阱
- **所需工具**：完成任务需要的工具或资源

### Step 2: 评估知识价值

在创建 skill 前验证：

- **可复用性**：这个知识对未来任务有帮助吗？（不仅是当前案例）
- **非平凡性**：这个知识需要发现，而不是简单的文档查阅？
- **具体性**：能否描述具体的触发条件和解决方案？
- **已验证**：解决方案是否真正有效？（不仅是理论上的）

### Step 3: 生成技能文档

创建符合 OpenCode 规范的 `SKILL.md` 文件：

```markdown
---
name: <技能名称>
description: |
  <精确描述，包括：(1) 具体用例，(2) 触发条件如具体错误信息或症状，
  (3) 解决的问题。足够具体以便语义匹配能准确检索
  (4) 限制字符长度为1-1024>
version: 1.0.0
date: <YYYY-MM-DD>
---

# <技能名称>

## Problem
<该技能解决的问题清晰描述>

## Context / Trigger Conditions
<何时使用此技能？包含具体的错误信息、症状或场景>

## Solution
<逐步解决方案或要应用的知识>

## Verification
<如何验证解决方案有效>

## Example
<应用此技能的具体示例>

## Notes
<任何注意事项、边缘情况或相关考虑>

## References
<可选：链接到官方文档、文章或参考资料>
```

### Step 4: 保存技能

将新技能保存到适当位置：

- **项目级技能**：`.opencode/skills/[skill-name]/SKILL.md`
- **用户级技能**：`~/.opencode/skills/[skill-name]/SKILL.md`

## Verification

创建 skill 后，确认以下检查项：

- [ ] Description 包含具体的触发条件
- [ ] Solution 已验证有效
- [ ] 内容足够具体以便可执行
- [ ] 内容足够通用以便可复用
- [ ] 不包含敏感信息（凭证、内部 URL 等）
- [ ] 不重复现有文档或技能

## Notes

- 技能名称格式：仅使用小写字母、数字、连字符（kebab-case）
- description 是关键字段，要包含具体症状和上下文标记
- 不是每个任务都需要创建 skill，只提取真正有价值的知识
- 建议使用中文编写内容（除非用户指定其他语言）

## References

- [Claudeception: Autonomous Skill Extraction for LLM Agents](https://github.com/blader/Claudeception )
- [OpenCode Skills 架构](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills )