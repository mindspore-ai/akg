# AIKG 重构文档索引

> 重构相关文档的导航指南
> 更新时间: 2026-01-21

---

## 📚 核心文档（必读）

### 1. [refactor.md](refactor.md) ⭐⭐⭐
**整体架构框架 - 从这里开始**

**内容**：
- 核心设计决策
- 整体架构图（5 层）
- 五大重构任务
- 关键机制（混合调用、Trace System、无限运行、Skills 动态加载）
- 目录结构
- 迁移路线

**阅读优先级**：最高  
**适合**：所有人，必读

---

### 2. [REFACTOR_IMPLEMENTATION_PLAN.md](REFACTOR_IMPLEMENTATION_PLAN.md) ⭐⭐⭐
**实施方案 - 如何落地**

**内容**：
- 代码复用分析（哪些可以不改）
- 6 周开发计划（5 个阶段）
- TDD 开发流程（每个阶段都有详细测试）
- 测试策略（单元/集成/E2E）
- 里程碑与交付
- 风险与应对

**阅读优先级**：最高  
**适合**：开发人员，必读

---

### 3. [calling_patterns_summary.md](calling_patterns_summary.md) ⭐⭐
**KernelAgent 调用模式**

**内容**：
- KernelAgent 的手动主循环设计
- Workflow、SubAgent、Tools 的调用方式
- HITL 场景支持
- 与 ReAct 的融合
- 代码示例

**阅读优先级**：高  
**适合**：需要理解 Agent 调用逻辑的开发人员

**关联**：依赖 `refactor.md` 中的整体架构

---

## 📋 详细设计文档

### 4. [FileSystem_State_Design.md](FileSystem_State_Design.md) ⭐⭐
**FileSystem 状态管理设计**

**内容**：
- 文件系统目录结构
- 核心文件详解（trace.json, state.json, thinking.json, action_history_fact.json 等）
- 增量保存机制（避免重复）
- 状态恢复流程
- 调试流程
- LangGraph Checkpointer 集成

**阅读优先级**：高  
**适合**：需要实现状态管理的开发人员

**关联**：`FileSystem_Trace_Design.md`, `refactor.md`

---

### 5. [FileSystem_Trace_Design.md](FileSystem_Trace_Design.md) ⭐⭐
**Trace 系统多分支设计**

**内容**：
- 单一树结构（只有 node，没有 branch）
- 节点分叉和切换机制
- 对话式命令设计（`/trace`, `/parallel`）
- 并行探索支持
- TraceSystem 核心实现

**关键命令**：
- `/trace show` - 显示树结构
- `/trace switch <node>` - 切换节点
- `/trace compare <n1> <n2>` - 对比节点路径
- `/parallel N <action>` - 并行探索

**阅读优先级**：中  
**适合**：需要实现 Trace 系统的开发人员

**关联**：`FileSystem_State_Design.md`, `refactor.md`

---

### 6. [ask_user_and_plan_analysis.md](ask_user_and_plan_analysis.md) ⭐⭐
**ask_user 与 PlanAgent 的关系分析**

**内容**：
- ask_user 导致 plan 失效的问题分析
- PlanAgent 触发条件（定期 + ask_user + 主动检测）
- plan 过期检测规则（关键词匹配 + 执行偏离）
- 三种解决方案对比
- 推荐方案的详细实现
- 实现建议（MVP → 优化 → 智能评估）

**阅读优先级**：高  
**适合**：需要实现 PlanAgent 和 HITL 的开发人员

**关联**：`refactor.md`, `REFACTOR_IMPLEMENTATION_PLAN.md`

---

### 7. [trace_and_skills_analysis.md](trace_and_skills_analysis.md) ⭐
**Trace System 和 Skills System 详细分析**

**内容**：
- Trace System 与 LangGraph Checkpointer 的配合
- Skills 组织和动态加载机制
- Skills 与 SYSTEM_PROMPT 的组合
- 实现细节

**阅读优先级**：中  
**适合**：需要实现 Trace 或 Skills 的开发人员

**关联**：`refactor.md`, `FileSystem_State_Design.md`

---

### 8. [refactor_code_related.md](refactor_code_related.md) ⭐
**代码分析与融合策略**

**内容**：
- 现有 LangGraph 代码分析
- 新架构与现有代码的关系
- 迁移难度评估
- 融合策略
- 代码复用建议

**阅读优先级**：中  
**适合**：需要理解现有代码的开发人员

**关联**：`refactor.md`, `REFACTOR_IMPLEMENTATION_PLAN.md`

---

### 9. [CLI_DESIGN.md](CLI_DESIGN.md) ⭐⭐
**CLI 交互层设计**

**内容**：
- 斜杠命令系统设计（`/trace`, `/parallel`, `/help` 等）
- 自动补全机制（基于 prompt_toolkit）
- 命令注册系统（装饰器风格）
- 简化面板设计（去掉 Top5 框架）
- 与现有 Runner 的集成方案
- 实施计划（3-5天）

**关键特性**：
- 命令式交互（参考 qwen-agent）
- Tab 自动补全 + 描述提示
- 装饰器注册，易扩展
- op 和 common 统一体验

**阅读优先级**：高  
**适合**：需要实现 CLI 交互的开发人员

**关联**：`refactor.md`（任务7）

---

## 📖 阅读建议

### 新人入门路径

1. **第 1 天**：阅读 `refactor.md`，理解整体架构
2. **第 2 天**：阅读 `REFACTOR_IMPLEMENTATION_PLAN.md`，理解实施计划
3. **第 3 天**：阅读 `calling_patterns_summary.md`，理解 Agent 调用模式
4. **第 4 天**：根据分工，阅读对应的详细设计文档

### 开发人员路径

**如果你要开发**：
- **核心框架层**：先读 `refactor.md` → `refactor_code_related.md`
- **HITL 编排层**：先读 `calling_patterns_summary.md` → `ask_user_and_plan_analysis.md` → `refactor.md`
- **PlanAgent**：先读 `ask_user_and_plan_analysis.md` → `REFACTOR_IMPLEMENTATION_PLAN.md`（阶段2）
- **状态管理**：先读 `FileSystem_State_Design.md` → `FileSystem_Trace_Design.md` → `trace_and_skills_analysis.md`
- **CLI 交互层**：先读 `CLI_DESIGN.md` → `refactor.md`（任务7）
- **算子专用层**：先读 `refactor.md` → `REFACTOR_IMPLEMENTATION_PLAN.md`

---

## 🗂️ 文档分类

### 架构设计
- `refactor.md`
- `calling_patterns_summary.md`
- `refactor_code_related.md`

### HITL & PlanAgent
- `ask_user_and_plan_analysis.md`
- `calling_patterns_summary.md`

### 状态管理
- `FileSystem_State_Design.md`
- `FileSystem_Trace_Design.md`
- `trace_and_skills_analysis.md`

### CLI 交互层
- `CLI_DESIGN.md`

### 实施计划
- `REFACTOR_IMPLEMENTATION_PLAN.md`

---

## 📝 更新记录

### 2026-01-25
- 新增 `CLI_DESIGN.md`（CLI 交互层设计）
- 更新 `refactor.md`：添加任务7（CLI 交互层重构）
- 更新索引：添加 CLI 设计文档和开发路径

### 2026-01-21
- 删除旧的讨论文档（1.md, ARCHITECTURE_CN.md, POC*.md, RefactoringPlan.md, HITL-*.md 等）
- 创建 `FileSystem_State_Design.md`（取代 FileSystem_Content_Comparison.md）
- 去除所有外部框架提及，只聚焦 AIKG 新架构
- 更新索引结构

### 2026-01-20
- 创建初始索引
- 添加核心文档

---

**总结**：
- 从 `refactor.md` 开始，理解整体架构
- 根据职责，选择对应的详细设计文档
- CLI 开发者必读 `CLI_DESIGN.md`
- 开发前必读 `REFACTOR_IMPLEMENTATION_PLAN.md`
