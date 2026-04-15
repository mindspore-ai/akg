# docs/ — 文档规范

## 职责

存放 akg_agents 的设计文档和使用文档。

## 目录结构

```
docs/
├── v1/           # 旧版文档（保留备查，不再更新）
│   ├── CN/       #   中文版
│   └── assets/   #   配图
└── v2/           # 当前主文档
    └── CN/       #   中文版
```

## 开发约定

### 文档索引（docs/v2/）

| 文档 | 说明 |
|------|------|
| `Architecture.md` | 整体架构与分层 |
| `Workflow.md` | LangGraph 工作流设计 |
| `SkillSystem.md` | Skill 注册/加载/选择 |
| `SkillContributionGuide.md` | Skill 编写规范 |
| `SkillEvolution.md` | Skill 自进化机制 |
| `AgentSystem.md` | Agent 继承体系 |
| `Configuration.md` | settings.json 详解 |
| `LLM.md` | LLM Provider/Client |
| `KernelGen.md` / `KernelDesigner.md` / `KernelAgent.md` | 各算子 Agent |
| `Trace.md` | 推理追踪与断点恢复 |
| `Tools.md` | 工具注册与执行 |
| `AKG_CLI.md` | akg_cli 命令参考 |
| `OpOptimizer.md` | 算子优化流程 |
| `LocalModelDeployment.md` | 本地 LLM 部署 |
| `SolarRoofline.md` | Solar roofline 依赖安装与运行时接入 |

### 规范

- 新文档放 `v2/`，同步提供 `v2/CN/` 中文版
- 不在 `v1/` 中新增内容
- 配图使用相对路径引用

## 不做什么

- **不要**在 `v1/` 中更新内容——只在 `v2/` 工作
- **不要**把 API reference 放在这里——API 说明随代码走（docstring）
