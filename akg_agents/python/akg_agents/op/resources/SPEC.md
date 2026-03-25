# op/resources/ — Prompt / Skill / Doc 资源

## 职责

存放算子场景的所有非代码资源：Agent prompt 模板、内置 Skill、DSL 文档、代码生成模板。

## 目录结构

```
resources/
├── prompts/           # Agent prompt 模板（.j2），按 agent 分子目录
│   ├── kernel_gen/
│   ├── kernel_designer/
│   ├── conductor/
│   ├── checker/
│   ├── sketch/
│   ├── task_constructor/
│   ├── op_task_builder/
│   ├── skill_evolution/
│   └── utils/
├── skills/            # 内置 Skill（SKILL.md + 可选 scripts/references）
│   ├── triton-ascend/
│   ├── triton-cuda/
│   ├── tilelang/
│   ├── task-constructor/
│   ├── kernel-agent/
│   └── ...
├── docs/              # DSL 文档（供 Agent 参考）
│   ├── triton_ascend_docs/
│   ├── triton_cuda_docs/
│   ├── tilelang_*/
│   ├── cuda_docs/
│   ├── cpu_docs/
│   ├── hardware/
│   └── ...
└── templates/         # 代码生成模板（prof_generation、cmake 等）
```

## 开发约定

### Prompt 模板（prompts/）

- 格式：`.j2`（Jinja2）
- 路径约定：`prompts/<agent_name>/<template_name>.j2`
- `get_prompt_path()`（`utils/common_utils.py`）固定指向此目录
- `AgentBase.load_template()` 基于此路径解析
- 模板变量通过 `render(**kwargs)` 传入

### 内置 Skill（skills/）

- 每个 Skill 一个子目录：`skills/<skill-name>/SKILL.md`
- 附带资源放在 `scripts/`、`references/` 等子目录
- 元数据格式见 [core_v2/skill/SPEC.md](../../core_v2/skill/SPEC.md)

### DSL 文档（docs/）

- 按 DSL 分目录存放
- 这些文档会被 Agent 在生成代码时参考
- 目录路径通过 `config/` 中的 yaml（`docs_dir` 字段）配置

## 不做什么

- **不要**把 Python 逻辑代码放在 prompt 模板里——模板只做文本渲染
- **不要**把测试数据放在这里——归 `tests/op/resources/`
- **不要**修改 `docs/` 中的上游文档——保持与源文档同步
