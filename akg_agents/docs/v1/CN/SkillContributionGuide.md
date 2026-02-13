# Skill 文档贡献指南

欢迎贡献 Skill 文档！本指南将帮助你快速了解 Skill 文档规范。

---

## YAML Frontmatter 规范

### 必填字段

```yaml
---
name: triton-ascend-case-xxx               # 小写字母+连字符
description: "详细说明..."                  # 包含数据规模、优化技术、性能指标、适用场景
level: L5                                  # L3（基础/方法）、L4（算子策略）、L5（具体案例）
category: example                          # fundamental、method、implementation、example
version: "1.0.0"                           # 版本号
metadata:
  backend: ascend                          # 硬件后端
  dsl: triton-ascend                       # DSL 类型
  hardware: ascend910b4                    # 硬件型号（可选）
---
```

### 字段说明

- **name**：使用小写字母和连字符，格式为 `{dsl}-{类型}-{具体名称}`
- **description**：**LLM 筛选的核心依据**，应该包含：
  - 数据规模（千级/万级/百万级元素）
  - 核心优化技术（如计算重组、二次切分、原子操作）
  - 性能指标（最优配置、性能提升倍数）
  - 适用场景（明确数据特征和问题类型）
- **level**：
  - L3：基础概念/通用方法
  - L4：某类算子通用策略
  - L5：具体案例+完整性能数据
- **category**：fundamental/method/implementation/example（可自定义）
- **metadata**：`backend` ，`dsl` 和 `hardware` 等，按需添加

---

## 内容结构

### 1. 任务特征（L5 案例必需）
```markdown
## 任务特征
- **操作类型**：elementwise、reduction、matmul 等
- **数据尺寸**：(M, N, K) 及规模描述
- **数据类型**：输入/输出数据类型
- **任务特点**：关键特征和挑战
```

### 2. 优化要点
```markdown
## 优化：{技术名称}

### 简单方式
\`\`\`python
# 未优化代码
\`\`\`

### 优化方式
\`\`\`python
# 优化后代码
\`\`\`

### 优化内容
- 说明优化原理
- 性能提升数据

### 总结
关键原则和适用场景
```

### 3. 配置参数（如有）
```markdown
## Autotune 配置
\`\`\`python
configs = [
    triton.Config({'BLOCK_SIZE': 1024}),  # 性能：xxx
    triton.Config({'BLOCK_SIZE': 2048}),  # 性能：xxx   最优
]
\`\`\`
```

### 4. 总结
```markdown
### 总结
1. 关键优化原则 1
2. 关键优化原则 2
3. 适用场景和边界条件
```

---

## 层级定义与示例

| 层级 | 类别 | 说明 | 示例 |
|------|------|------|------|
| L3 | fundamental | 核心概念、标准模式 | `triton-ascend-basics` |
| L3 | method | 通用优化方法 | `triton-ascend-optimization` |
| L4 | implementation | 某类算子通用策略 | `triton-ascend-reduce`（sum/mean/max/softmax） |
| L5 | example | 具体案例+完整性能数据 | `triton-ascend-case-matmul-swizzle2d` |

---

## 📚 参考示例

我们推荐参考以下 4 个文档，了解不同类型的 Skill 文档写法：

### 1. 基础文档（L3/fundamental）
**文件**：`python/akg_agents/op/resources/skills/triton-ascend/triton-ascend-basics/SKILL.md`
- 展示基础概念、核心术语、标准模式

### 2. 算子类文档（L4/implementation）
**文件**：`python/akg_agents/op/resources/skills/triton-ascend/triton-ascend-reduce/SKILL.md`
- 展示如何为一类算子（归约）提供通用优化策略
- 涵盖多个算子：sum/mean/max/softmax/layernorm

### 3. 特定优化案例（L5/example）
**文件**：`python/akg_agents/op/resources/skills/triton-ascend/triton-ascend-case-matmul-swizzle2d/SKILL.md`
- 展示如何写一个完整的具体优化案例
- 包含任务特征、优化技术、代码示例、性能数据

### 4. 通用优化文档（L3/method）
**文件**：`python/akg_agents/op/resources/skills/triton-ascend/triton-ascend-optimization/SKILL.md`
- 展示通用优化技术、API 限制、性能调优方法

### 更多示例
浏览 `python/akg_agents/op/resources/skills/triton-ascend/` 目录，可查看更多 Skill 文档示例：
- **特定算子策略**：`triton-ascend-elementwise`、`triton-ascend-matmul`、`triton-ascend-attention`
- **优化案例**：`triton-ascend-case-*` 系列（21 个 static shape 优化案例）
- **工具文档**：`triton-ascend-debugging`、`triton-ascend-memory`、`triton-ascend-grid-config`

---

## 关键原则

1. **Description 是 LLM 筛选的核心依据**，必须信息丰富
2. **代码示例使用对比格式**，突出优化前后差异
3. **性能数据必须真实可靠**，包含具体配置和测试环境
4. **避免过度细化 metadata**，保持通用性和可复用性
5. **结构清晰**，易于理解和快速应用

---

## 提交流程

1. 按照本规范创建 Skill 文档
2. 将文件放入合适目录：`python/akg_agents/op/resources/skills/{dsl}/{skill-name}/SKILL.md`
3. 确保文档格式正确，代码可运行
4. 提交 Pull Request，并附上详细的优化技术说明和性能提升数据

---

## 问题与反馈

如果对 Skill 文档规范有疑问，请：
- 参考现有 Skill 文档示例
- 在 GitHub 提交 Issue
- 联系项目维护者

感谢你的贡献！
