# akg-op 端到端使用指南

## 概述

akg-op 是 AKG_Agents 与 OpenCode 集成后的算子优化 Agent，用于生成高性能算子代码。支持两种模式：

| 模式 | 触发方式 | 典型场景 |
|------|---------|---------|
| **单算子** | 指定具体算子或提供代码 | "帮我生成一个 relu 算子"、"优化这段 layernorm 代码" |
| **融合** | 要求分析模型融合机会 | "分析 Qwen2 的算子融合机会并生成融合算子" |

---

## 端到端流程

```
┌─────────────────────────────────────────────────────────────┐
│  用户在 opencode CLI 中切换到 akg-op Agent                    │
│  （Tab 键或 /agents 命令）                                   │
└────────────────────┬────────────────────────────────────────┘
                     ▼
          ┌─────────────────────┐
          │  Phase 0  环境准备   │  检查 akg_agents 可用性
          │  & 参数确认          │  🛑 确认 framework/backend/arch/dsl
          └─────────┬───────────┘
                    ▼
        ┌───────────────────────┐
        │  是否需要融合分析？     │
        └───┬───────────────┬───┘
            │ 是            │ 否
            ▼               │
   ┌────────────────┐       │
   │  Phase 1       │       │
   │  融合分析       │       │
   │  🛑 用户选择    │       │
   │  融合机会       │       │
   └───────┬────────┘       │
           │                │
           ▼                ▼
  ┌─────────────────────────────┐
  │  Phase 2  构建任务描述        │  从代码提取算子逻辑
  │  自动验证 + 🛑 用户确认       │  → {op_name}.py
  └─────────────┬───────────────┘
                ▼
  ┌─────────────────────────────┐
  │  Phase 3  生成算子            │  选择 workflow 运行
  │  🛑 用户选择 workflow         │  (kernelgen / adaptive_search / evolve)
  └─────────────┬───────────────┘
                │
            ┌───┴───┐
            │ 失败？ │──── 是 ──→ 输出失败报告，任务结束
            └───┬───┘
                │ 否
                ▼
  ┌─────────────────────────────┐
  │  Phase 4  确认结果            │  🛑 展示 generated_code.py
  │                              │  接受 / 重新生成
  │  重新生成 → 回到 Phase 3      │
  └─────────────┬───────────────┘
                │ 接受
                ▼
  ┌─────────────────────────────┐
  │  Phase 5  代码集成            │  复制生成代码到工作目录
  │  （如提供了原始代码）          │  备份 + import 替换
  └─────────────┬───────────────┘
                ▼
  ┌─────────────────────────────┐
  │  Phase 6  输出报告            │  report.md
  │  展示配置、结果、文件变更     │
  └─────────────────────────────┘
```

---

## 使用示例

### 示例 1：单算子 — 仅生成

```
用户> 帮我生成一个 relu 算子，输入 shape (128, 4096)，fp32，在 A100 上用 triton

Phase 0: 检查环境（缓存命中，跳过安装）→ 用户确认参数 framework=torch, backend=cuda, arch=a100, dsl=triton_cuda
Phase 2: 生成 relu.py → 验证 → 用户确认
Phase 3: 用户选择 kernelgen → 运行 → 生成成功
Phase 4: 展示代码 → 用户接受
Phase 5: 复制 relu_generated.py 到工作目录
Phase 6: 输出报告
```

### 示例 2：单算子 — 优化已有代码

```
用户> 帮我优化 /path/to/model.py 中的 layernorm，用 Triton

Phase 0: 检查环境（缓存命中）→ 用户确认参数
Phase 2: 从 model.py 提取 layernorm → 生成 layernorm.py → 验证 → 用户确认
Phase 3: 用户选择 adaptive_search → 后台静默运行（约 15 分钟）→ 生成成功
Phase 4: 展示代码 → 用户接受
Phase 5: 备份 model.py → 添加 from layernorm_generated import ModelNew → 保存集成文件
Phase 6: 输出报告，含文件变更记录
```

### 示例 3：融合模式（首次使用，无环境缓存）

```
用户> 分析 Qwen2 模型的算子融合机会并实现

Phase 0: 检查环境（无缓存 → 检测/安装 akg_agents → 采集硬件/Framework/DSL → 写入缓存）
         → 用户确认参数 framework=torch, backend=ascend, arch=ascend910b4, dsl=triton_ascend
Phase 1: 分析 Qwen2 forward → 发现 3 个融合机会 → 用户选择其中 2 个
  对每个选中的融合机会：
    Phase 2: 构建融合算子任务描述 → 验证 → 用户确认
    Phase 3: 生成融合算子
    Phase 4: 展示代码 → 接受
    Phase 5: 备份 → 集成到模型代码
Phase 6: 输出汇总报告
```

---

## 工作目录

所有产物保存在 `~/akg_agents_logs/op_{op_name}_{时间戳}_{随机ID}/` 下：

| 文件 | 说明 |
|------|------|
| `{op_name}.py` | 任务描述（算子参考实现 + 输入定义） |
| `{op_name}_generated.py` | 用户最终接受的生成算子代码 |
| `{model}_generated.py` | 集成后的原代码副本（含 `from {op_name}_generated import ModelNew`） |
| `output/{workflow}_{n}/` | 每次 workflow 运行的完整输出（代码、摘要、日志） |
| `backup/` | 被替换文件的原始副本（可用于回滚） |
| `report.md` | 最终报告 |

---

## 可选 Workflow

| Workflow | 策略 | 典型耗时 | 适用场景 |
|----------|------|---------|---------|
| `kernelgen` | 迭代生成→验证→修复（SubAgent） | 1-5 分钟 | 需求明确、快速出结果（**默认**） |
| `adaptive_search` | UCB 自适应搜索（静默模式） | 10-30 分钟 | 希望更高质量，愿意等待 |
| `evolve` | 岛屿模型进化算法（静默模式） | 15-60 分钟 | 需要多样性探索，多卡并行 |

生成不满意可选择换 workflow 重新生成，历史结果保留在不同子目录中互不覆盖。

---

## 架构

| 组件 | 类型 | 说明 |
|------|------|------|
| `akg-op` | Agent | 主编排器：模式判定、阶段调度、用户交互 |
| `akg-env-setup` | Skill | 环境检查、硬件/Framework/DSL 采集、依赖安装 |
| `op-task-extractor` | Skill | 从代码或自然语言构建任务文件 |
| `vllm-ascend-operator-fusion` | Skill | 模型融合分析 |
| `kernelgen` | SubAgent | 迭代式代码生成+验证工作流 |
| `kernel-generator` | Skill | DSL 感知的算子代码生成（被 kernelgen 调用） |
| `kernel-verifier` | Skill | 多框架、多后端正确性验证（被 kernelgen 调用） |
| `search-workflow` | Skill | adaptive_search / evolve 后台执行 |

---

## 注意事项

- 每个阶段都有人工确认环节，不会自动执行到底
- 生成失败时直接报告错误，不会自动尝试修复
- 替换代码前总会先备份原文件，可从 `backup/` 恢复
- `adaptive_search` 和 `evolve` 以静默模式后台运行，约 1 分钟轮询一次
