---
name: operator-knowledge-dispatcher
level: L2
category: dispatcher
version: "1.0.0"
description: "根据算子类型动态加载对应的实现知识文档"
license: MIT
structure:
  child_skills:
    - elementwise-knowledge
    - reduce-knowledge
    - matmul-knowledge
    - fusion-knowledge
  default_children:
    - elementwise-knowledge
---

# 算子知识调度器

## 概述

根据算子类型，动态加载对应的实现知识库。支持：
- **Elementwise算子**: 逐元素运算（add, mul, relu等）
- **Reduce算子**: 归约运算（sum, softmax, layernorm等）
- **MatMul算子**: 矩阵乘法类运算（matmul, conv2d等）
- **Fusion算子**: 融合算子（多个算子组合）

## 使用方法

### 方法1: 通过SkillStackComposer

```python
from skill_system import SkillStackComposer, SkillMatchContext

# 创建context，指定算子类型
context = SkillMatchContext(
    task_type="code_generation",
    state={
        "operator_type": "relu",  # ← 指定算子类型
        "backend": "cuda",
        "hardware": "A100"
    }
)

# 动态组装Skill栈
composer = SkillStackComposer(registry, hierarchy, engine)
stack = composer.compose_stack("operator-knowledge-dispatcher", context)

# 结果：只激活了elementwise-knowledge
print(stack.active_skills)  # ['elementwise-knowledge']
```

### 方法2: 直接使用OrchestrationEngine

```python
from skill_system import OrchestrationEngine

# 获取dispatcher skill
dispatcher = registry.get("operator-knowledge-dispatcher")

# 创建编排引擎
engine = OrchestrationEngine()

# 评估规则
state = {"operator_type": "matmul"}
actions = engine.evaluate_rules(dispatcher.orchestration.rules, state)

# 执行actions
for action in actions:
    if action.action_type == ActionType.ACTIVATE_SKILL:
        skill = registry.get(action.skill_id)
        print(f"激活: {skill.name}")
```

## 配置说明

### operator_type支持的值

**Elementwise类**:
- `add`, `mul`, `div`, `sub`
- `relu`, `sigmoid`, `tanh`, `gelu`
- `exp`, `log`, `sqrt`, `pow`

**Reduce类**:
- `sum`, `mean`, `max`, `min`
- `softmax`, `logsoftmax`
- `layernorm`, `batchnorm`

**MatMul类**:
- `matmul`, `bmm` (batch matmul)
- `conv2d`, `conv3d`
- `linear`, `dense`

**Fusion类**:
- `fusion` (通用融合算子)
- 自动检测：`operator_complexity=complex`

## 扩展方式

### 添加新算子类型

1. 创建新的L3 Skill（如`conv-knowledge`）
2. 在`orchestration.child_skills`中添加
3. 添加对应的规则

```yaml
orchestration:
  child_skills:
    - elementwise-knowledge
    - reduce-knowledge
    - matmul-knowledge
    - conv-knowledge  # ← 新增
  
  rules:
    - rule_id: dispatch_conv
      condition:
        type: state
        field: operator_type
        operator: "in"
        value: ["conv1d", "conv2d", "conv3d"]
      action:
        type: activate_skill
        skill_id: conv-knowledge
```

## 相关Skills

- **elementwise-knowledge** (L3): Elementwise算子实现知识
- **reduce-knowledge** (L3): Reduce算子实现知识
- **matmul-knowledge** (L3): MatMul算子实现知识
- **fusion-knowledge** (L3): Fusion算子实现知识

## 最佳实践

1. **明确算子类型**: 在调用前确定`operator_type`
2. **使用标准名称**: 遵循PyTorch/TensorFlow算子命名
3. **处理未知类型**: 设置默认fallback规则
4. **组合多个知识**: 复杂算子可以激活多个child skills

## 示例场景

### 场景1: 生成ReLU算子

```python
context = SkillMatchContext(
    state={"operator_type": "relu"}
)
stack = composer.compose_stack("operator-knowledge-dispatcher", context)
# 激活: elementwise-knowledge
```

### 场景2: 生成LayerNorm算子

```python
context = SkillMatchContext(
    state={"operator_type": "layernorm"}
)
stack = composer.compose_stack("operator-knowledge-dispatcher", context)
# 激活: reduce-knowledge
```

### 场景3: 生成融合算子

```python
context = SkillMatchContext(
    state={
        "operator_type": "fusion",
        "sub_operators": ["matmul", "relu", "add"]
    }
)
stack = composer.compose_stack("operator-knowledge-dispatcher", context)
# 激活: fusion-knowledge（可能同时激活matmul-knowledge, elementwise-knowledge）
```
