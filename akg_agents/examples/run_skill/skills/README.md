# AKG Agents 测试 Skills

> **用于测试 Skill System 的完整示例**

---

## 📊 Skills概览

本目录包含 **8个测试Skills**，覆盖L1-L4所有层级：

| 层级 | 语义 | 数量 | Skills |
|------|------|------|---------|
| **L1** | 流程/编排层 | 2 | standard-workflow, adaptive-evolve |
| **L2** | 组件/执行层 | 4 | coder-agent, designer-agent, verifier-agent, operator-knowledge-dispatcher |
| **L3** | 方法/策略层 | 3 | cuda-basics, triton-syntax, triton-ascend |
| **L4** | 实现/细节层 | 1 | error-handling |
| **-** | 其他 | 1 | web-scraping |

---

## 🏗️ Skills结构

### L1: Workflow Skills

#### 1. standard-workflow
- **Category**: workflow
- **描述**: 标准的算子生成工作流
- **子 Skills**: coder-agent, verifier-agent（L2）
- **适用场景**: 常规算子开发
- **内容**: 完整的三阶段工作流程说明

#### 2. adaptive-evolve
- **Category**: workflow
- **描述**: 自适应进化工作流
- **子 Skills**: designer-agent, coder-agent, verifier-agent（L2）
- **适用场景**: 复杂算子优化
- **内容**: 进化算法原理、配置参数、成功案例

---

### L2: Agent Skills

#### 3. coder-agent
- **Category**: agent
- **描述**: 代码生成 Agent
- **子 Skills**: cuda-basics, triton-syntax（L3）
- **能力**: 
  - 多 DSL 支持（CUDA, Triton, OpenCL）
  - 多后端支持（NVIDIA, AMD, Intel）
  - 代码优化建议
- **内容**: 代码生成策略、模板、RAG 增强

#### 4. designer-agent
- **Category**: agent
- **描述**: 算法设计 Agent
- **子 Skills**: (无)
- **能力**:
  - 算法分析
  - 设计方案生成
  - 性能估算
- **内容**: 设计模式库、决策树、进化策略

#### 5. verifier-agent
- **Category**: agent
- **描述**: 验证 Agent
- **子 Skills**: (无)
- **能力**:
  - 正确性验证
  - 性能 Profiling
  - 资源分析
- **内容**: 验证流程、测试用例生成、性能基准

#### 6. operator-knowledge-dispatcher
- **Category**: dispatcher
- **描述**: 根据算子类型动态加载知识文档
- **子 Skills**: elementwise-knowledge, reduce-knowledge, matmul-knowledge, fusion-knowledge（未加载）
- **说明**: 声明性子 Skill，按需加载

---

### L3: 方法/策略层 Skills

#### 7. cuda-basics
- **Category**: dsl
- **描述**: CUDA 编程基础知识
- **内容**:
  - 线程层次结构
  - 内存层次
  - 访问优化技巧
  - 同步机制
  - 完整代码示例
- **大小**: ~6,285 字符

#### 8. triton-syntax
- **Category**: dsl
- **描述**: Triton 语言语法
- **内容**:
  - Python-like 语法
  - Block 编程模型
  - 完整示例（向量加法、MatMul、Softmax）
  - 调优技巧
  - Triton vs CUDA 对比
- **大小**: ~10,000+ 字符

#### 9. triton-ascend
- **Category**: dsl
- **描述**: Triton on Ascend 适配指南
- **内容**: 昇腾 NPU 编程适配

---

### L4: 实现/细节层 Skills

#### 10. error-handling
- **Category**: implementation
- **描述**: GPU 代码错误处理
- **内容**:
  - CUDA 错误检查
  - Kernel 内边界检查
  - 数值稳定性检查
  - Python/Triton 错误处理
  - 错误恢复策略
  - 完整模板

### 其他 Skills

#### 11. web-scraping
- **Category**: utility
- **描述**: Web scraping 最佳实践
- **说明**: 示例性 Skill，无层级

---

## 🔗 层级关系图

```
adaptive-evolve (L1)
├── designer-agent (L2)
├── coder-agent (L2)
│   ├── cuda-basics (L3)
│   └── triton-syntax (L3)
└── verifier-agent (L2)

standard-workflow (L1)
├── coder-agent (L2)
│   ├── cuda-basics (L3)
│   └── triton-syntax (L3)
└── verifier-agent (L2)

error-handling (L4) - 独立使用
```

---

## 🚀 如何使用

### 1. 加载Skills

```python
from skill_system import SkillLoader, SkillRegistry
from pathlib import Path

# 加载
loader = SkillLoader()
skills = loader.load_from_directory(Path("./skills"))

# 注册
registry = SkillRegistry()
registry.register_batch(skills)

print(f"加载了 {len(skills)} 个Skills")
```

### 2. 查询Skills

```python
# 按名称查询
cuda_skill = registry.get("cuda-basics")

# 按层级查询
from skill_system import SkillLevel
l2_skills = registry.get_by_level(SkillLevel.L2_AGENT)

# 按模式查询
agent_skills = registry.filter(name_pattern="*-agent")
```

### 3. 访问内容

```python
skill = registry.get("cuda-basics")
if skill:
    print(f"名称: {skill.name}")
    print(f"描述: {skill.description}")
    print(f"层级: {skill.level.value}")
    print(f"内容长度: {len(skill.content)}")
    print(f"\n内容:\n{skill.content}")
```

### 4. 使用层级关系

```python
# 获取子 Skills
children = registry.get_children("standard-workflow")
print(f"子 Skills: {[s.name for s in children]}")

# 查看 category
for child in children:
    print(f"  {child.name} - {child.category}")
```

---

## 🧪 运行测试

### 简单测试

```bash
cd demo/skill_system
python test_skills_simple.py
```

**预期输出**：
```
成功加载 8 个Skill
成功注册 8 个Skill
总计:
  L1: 2个
  L2: 3个
  L3: 2个
  L4: 1个
[SUCCESS] 测试完成！
```

### 完整示例

```bash
cd examples
python examples.py              # 基础示例
python example_llm_driven.py    # LLM 驱动选择示例（需要 LLM 环境）
```

---

## 📝 Skill格式说明

每个Skill都是一个SKILL.md文件，格式如下：

```markdown
---
name: skill-name
description: "Skill 描述"
level: L1/L2/L3/L4/L5
category: workflow/agent/dsl/implementation/utility  # 用户自定义
version: "x.y.z"
license: MIT
structure:  # 可选，用于声明层级关系
  child_skills:
    - child1
    - child2
  default_children:
    - child1
  exclusive_groups:
    - [skill-a, skill-b]
---

# Skill 标题

## 正文内容

详细的 Skill 内容...
```

---

## 📊 内容统计

| Skill | 字符数 | 行数 | 章节数 |
|-------|--------|------|--------|
| adaptive-evolve | ~15,000 | ~350 | 15+ |
| standard-workflow | ~2,500 | ~80 | 10 |
| coder-agent | ~10,000 | ~300 | 12 |
| designer-agent | ~8,000 | ~250 | 10 |
| verifier-agent | ~9,000 | ~280 | 11 |
| cuda-basics | ~6,285 | ~351 | 41 |
| triton-syntax | ~10,000 | ~300 | 20 |
| error-handling | ~8,000 | ~250 | 15 |
| **总计** | **~68,785** | **~2,161** | **134** |

---

## ✨ 特色功能

### 1. 完整性
- ✅ 覆盖所有4个层级
- ✅ 包含真实的技术内容
- ✅ 提供具体的代码示例

### 2. 实用性
- ✅ 基于真实 AKG Agents 项目场景
- ✅ 包含最佳实践和陷阱
- ✅ 提供性能分析和优化建议

### 3. 关联性
- ✅ Skills之间有明确的依赖关系
- ✅ 支持层级化管理
- ✅ 可组合使用

### 4. 可测试性
- ✅ 每个Skill都可以独立加载
- ✅ 支持层级关系验证
- ✅ 提供完整的测试脚本

---

## 🎓 学习路径

### 初学者
1. 先阅读 **L3 Skills** (cuda-basics, triton-syntax)
   - 理解基础知识
2. 然后阅读 **L2 Skills** (coder-agent, verifier-agent)
   - 理解如何使用基础知识
3. 最后阅读 **L1 Skills** (standard-workflow)
   - 理解完整工作流

### 进阶用户
1. 从 **L1 Skills** 开始
2. 按照依赖关系向下阅读
3. 重点关注进化算法和优化策略

---

## 🔄 扩展建议

可以继续添加的Skills：

### L1层级
- multi-stage-workflow (多阶段优化)
- distributed-workflow (分布式执行)

### L2层级
- optimizer-agent (参数优化)
- profiler-agent (性能分析)

### L3层级
- opencl-basics (OpenCL编程)
- optimization-techniques (通用优化技巧)
- memory-patterns (内存访问模式)

### L4层级
- debugging-tools (调试工具)
- testing-framework (测试框架)
- performance-patterns (性能模式)

---

## 📚 参考资料

Skills内容参考了：
- CUDA Programming Guide
- Triton Documentation
- AKG Agents 实际项目经验
- GPU优化最佳实践

---

## 🎯 使用建议

1. **学习**: 按层级从底向上阅读
2. **开发**: 按层级从顶向下使用
3. **测试**: 使用提供的测试脚本验证
4. **扩展**: 参考现有格式添加新Skills

---

**更新时间**: 2026年1月27日  
**Skills 总数**: 11个  
**版本**: v0.2.0  
**状态**: ✅ 完整可用

