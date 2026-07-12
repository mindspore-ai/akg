# Agent 上下文图辅助代码仓理解实验报告

## 一、实验说明

本实验面向 AKG Agents 仓库中的 Code Agent 上下文组织问题：当前仓库主要依赖 `AGENTS.md`、`SPEC.md`、`README` 等自然语言文档声明规则，能够说明开发约束，但对代码入口、调用关系、影响范围和验证路径的表达仍然偏散。实验目标是验证：在传统文档摘要之外，额外提供面向 Agent 的结构化代码上下文图，是否能帮助 Code Agent 更准确地理解仓库、定位改动位置，并减少改错模块的风险。

- **代码实现分支**: [zhangyize-2026/akg_3377 br_agents](https://gitcode.com/zhangyize-2026/akg_3377/tree/br_agents)
- **代码实现提交**: `8068bb08cc97193994eb685079f432d47f92efe8`
- **实验任务对象**: AKG Agents 中 Triton Ascend matmul 生成/验证相关改造的上下文定位问题
- **实验模型**: DeepSeek API
- **实验方式**: 让模型回答同一个仓库理解问题，对比传统摘要上下文与传统摘要 + CODEGRAPH/context_graph 上下文的回答质量
- **注意事项**: 实验不要求模型生成 matmul kernel 代码，也不向模型提供 Triton Ascend matmul 的具体优化规则，重点评估模型能否找到正确代码上下文、影响面和验证方式

代码实现分支中已经补充了两类 Agent artifact：

| 类型 | 路径 | 作用 |
|------|------|------|
| 分析文档 | `akg_agents/docs/v2/CN/AgentContextGraph.md` | 说明为什么需要在 AGENTS/SPEC 之外增加上下文图 |
| 实验文档 | `akg_agents/docs/v2/CN/AgentContextGraphExperiment.md` | 记录对照实验设计、指标和结果 |
| 示例目录说明 | `akg_agents/python/akg_agents/op/verifier/SPEC.md` | 补充 verifier 层职责和边界 |
| 子目录 Agent 化示例 | `akg_agents/python/akg_agents/op/verifier/adapters/dsl/SPEC.md` | 描述 DSL adapter 子目录的开发规则 |
| 人类可读代码图 | `akg_agents/python/akg_agents/op/verifier/adapters/dsl/CODEGRAPH.md` | 给出阅读路径、关键节点、影响矩阵和反模式 |
| 机器可读代码图 | `akg_agents/python/akg_agents/op/verifier/adapters/dsl/context_graph.json` | 给 Code Agent 使用的结构化索引 |

## 二、实验设计

实验让两组 DeepSeek 都回答同一个问题：

> 现在要在 AKG Agents 仓库中做一个和 Triton Ascend matmul 生成/验证相关的改造，目标是让 Code Agent 更容易找到正确上下文并避免改错位置。请不要生成 matmul kernel 代码，也不要凭空写 Triton Ascend matmul 具体优化规则。请基于给定仓库上下文回答：优先阅读哪些文件、哪些文件可能需要修改或不应修改、影响哪些调用方或流程、如何验证、如何把流程沉淀给 Code Agent。

对照组只提供传统 `AGENTS.md` / `SPEC.md` 风格摘要，主要包含模块职责、关键文件列表和约束说明，例如 `op/verifier` 通过 backend、dsl、framework 三类 adapter 验证 kernel，per-DSL 行为应通过 `DSLAdapter` hook 表达，不应在核心流程里硬编码 DSL 分支。

实验组在对照组全部内容基础上，额外提供 CODEGRAPH/context_graph 摘要，包含：

- 根目录: `akg_agents/python/akg_agents/op/verifier/adapters/dsl/`
- 关键节点: `DSLAdapter`、`DSLAdapterTritonAscend`、`get_dsl_adapter`、`KernelVerifier`、`CodeChecker`
- Triton Ascend 阅读路径: adapter 实现、base 抽象、factory 注册、config 配置、verifier 调用方、triton-ascend Skill 和 docs
- 影响矩阵: factory mapping、config validation、generated verify script、profile script、artifact readiness、CodeChecker、LocalWorker profile dispatch、workspace_autoresearch handoff
- 验证建议: `compileall`、`json.tool`、相关 verifier/profile 流程
- 反模式: 不要在 `KernelVerifier` 中增加 `if dsl == "triton_ascend"` 之类分支

评分采用静态命中项，总分 11 分，每项 1 分。指标不是评估模型是否能写对 matmul kernel，而是评估模型是否能在不给具体优化规则的情况下，找全仓库改造所需的上下文。

| 指标 | 含义 |
|------|------|
| `mentions_triton_ascend_adapter` | 是否提到 `triton_ascend.py` / `DSLAdapterTritonAscend` |
| `mentions_dsl_base` | 是否提到 `base.py` / `DSLAdapter` |
| `mentions_factory` | 是否提到 `factory.py` / `get_dsl_adapter` |
| `mentions_config_utils` | 是否提到 `config_utils.py` / DSL 配置 |
| `mentions_kernel_verifier` | 是否提到 `KernelVerifier` |
| `mentions_code_checker` | 是否提到 `CodeChecker` / 静态检查 |
| `mentions_triton_ascend_skills` | 是否提到 `resources/skills/triton-ascend` |
| `mentions_triton_ascend_docs` | 是否提到 `resources/docs/triton_ascend_docs` |
| `avoids_kernel_verifier_branch` | 是否明确避免在 `KernelVerifier` 中加 DSL 特判 |
| `mentions_validation` | 是否提出 compileall / pytest / verifier / profile 等验证 |
| `mentions_agent_artifacts` | 是否提到 `CODEGRAPH.md` / `context_graph.json` / 结构化索引 |

## 三、观察

| 组别 | 上下文输入 | 得分 | 主要命中 |
|------|------------|------|----------|
| 对照组 | 传统 `AGENTS.md` / `SPEC.md` 摘要 | 7/11 | adapter、base、factory、config、KernelVerifier、triton-ascend Skill、验证方案 |
| 实验组 | 对照组全部内容 + CODEGRAPH/context_graph 摘要 | 10/11 | adapter、base、factory、config、KernelVerifier、CodeChecker、triton-ascend Skill、triton_ascend_docs、验证方案、Agent artifact |

实验中可以观察到三个差异：

1. **阅读路径更完整**
   对照组能够找到主路径文件，例如 `triton_ascend.py`、`base.py`、`factory.py` 和 `kernel_verifier.py`，但更容易漏掉二级影响点。实验组由于拿到 CODEGRAPH 中的 reading path 和 impact matrix，更容易同时覆盖 `CodeChecker`、Triton Ascend docs、Skill 资源和结构化索引本身。

2. **影响范围描述更接近真实改造流程**
   对照组回答通常停留在“修改 adapter 并注册 factory”的层面。实验组更容易说明 adapter 改动会影响 factory mapping、config validation、verify script、profile script、artifact readiness、LocalWorker profile dispatch 和 autoresearch handoff 等流程。

3. **Agent 沉淀物更明确**
   对照组即使能描述文档更新，也不一定会主动提出 `CODEGRAPH.md`、`context_graph.json` 这类可复用 Agent artifact。实验组会更自然地把“本次如何找上下文”沉淀成后续 Code Agent 可读取的结构化索引。

实验组唯一未稳定命中的指标是 `avoids_kernel_verifier_branch`。实际回答中已经表达了不应在 `KernelVerifier` 中增加 `if self.dsl == "triton_ascend"`，但评分脚本的正则识别不够宽松，因此该项存在评分脚本误差。这个现象也说明后续实验应同时保留人工复核，避免完全依赖关键词命中。

## 四、结论

本实验说明，在不泄露 matmul kernel 实现、也不给具体 Triton Ascend matmul 优化规则的前提下，CODEGRAPH/context_graph 能提升模型对代码仓的上下文覆盖能力。

对照组得分为 **7/11**，实验组得分为 **10/11**。提升主要来自三个方面：

- 更容易找到完整代码入口，包括 DSL adapter、factory、config、verifier、CodeChecker、Skill 和 docs
- 更容易描述改动影响面，而不是只停留在单文件修改
- 更容易提出可沉淀给 Code Agent 的结构化索引产物

因此，这个实验不证明模型已经具备稳定生成高质量 Triton Ascend matmul kernel 的能力，也不证明运行性能提升；它证明的是：在复杂代码仓中，结构化上下文图能增强 Code Agent 的仓库理解、任务定位和改造规划能力。

## 五、最终推荐修改方案

建议 AKG Agents 在继续保留 `AGENTS.md`、`SPEC.md`、`README` 的基础上，引入“人类可读 + 机器可读”的 Agent 上下文图机制。

推荐方案如下：

1. **在关键代码子目录补充 `CODEGRAPH.md`**
   用于描述该目录的职责、关键类/函数、推荐阅读路径、调用方、影响矩阵、验证命令和禁止改法。它面向人类开发者和 Code Agent 共同阅读。

2. **在同一目录补充 `context_graph.json`**
   用结构化 JSON 表达节点、边、入口文件、影响面、验证项和反模式。它面向 Code Agent 自动检索和拼接上下文，减少模型只读到零散文档的问题。

3. **先选择 `op/verifier/adapters/dsl/` 作为示例目录落地**
   该目录天然适合作为示例：它有明确的 DSL adapter 抽象、Triton Ascend/CUDA 具体实现、factory 注册点、KernelVerifier 调用方和 config 约束，能够清楚展示“不要在核心流程硬编码 DSL 特判，而是把差异放到 adapter hook 中”的设计原则。

4. **把验证命令写入上下文图**
   示例目录当前推荐至少保留：

   ```bash
   python -m compileall akg_agents/python/akg_agents/op/verifier/adapters/dsl
   python -m json.tool akg_agents/python/akg_agents/op/verifier/adapters/dsl/context_graph.json >/dev/null
   ```

   后续若补充单测或 profile 流程，也应同步更新到 `CODEGRAPH.md` 和 `context_graph.json`。

5. **将上下文图纳入后续 Agent 开发流程**
   对新增 DSL、verifier、kernel generation、workspace_autoresearch 等复杂模块，建议在新增或重构代码时同步维护对应上下文图。这样 Code Agent 接任务时可以先读取结构化图，再读取源文件，降低误改核心流程或漏看调用方的概率。

最终推荐采用的改造形态是：

```text
AGENTS.md / SPEC.md / README
        +
CODEGRAPH.md
        +
context_graph.json
```

其中 `AGENTS.md` 和 `SPEC.md` 负责声明规则，`CODEGRAPH.md` 负责解释代码地图，`context_graph.json` 负责给 Agent 稳定读取。三者组合比单独依赖自然语言文档更适合 AKG Agents 这类多模块、多流程、多 Agent 协作的代码仓。
