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
│   │   ├── api/       # Triton Ascend API 拆分文档 + manifest + 离线快照
│   │   └── ...
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

#### Triton Ascend API 文档（`docs/triton_ascend_docs/api/`）

| 文件 | 说明 |
|------|------|
| `api_manifest.json` | API 聚合清单；定义章节标题、条目顺序和条目文件名 |
| `*.md` | 单 API 条目文档；每个文件描述一个 API |
| `api.md` | 聚合后的离线快照；用于兜底加载 |

#### Triton Ascend API 文档加载

由 `op/utils/triton_ascend_api_docs.py` 统一负责加载，优先级如下：

1. server 本地 SDK 环境可见的聚合结果
2. 匹配 `backend/arch` 的 worker 环境返回的文档
3. 仓库内离线快照 `docs/triton_ascend_docs/api/api.md`

#### Triton Ascend API 文档维护

1. 在 `docs/triton_ascend_docs/api/` 下新增或更新单 API 文档
2. 单 API 文档的第一个三级标题必须为 API 签名，例如 `### tl.load(...)`、`### triton.cdiv(a, b)`、`### @triton.jit`
3. 在 `api_manifest.json` 中登记条目文件，并放入对应章节的 `sections[].entries`
4. 在安装 Triton Ascend SDK 的环境中执行 `python -m akg_agents.op.utils.triton_ascend_api_docs`，刷新离线快照 `api.md`
5. 如需校验，执行 `pytest akg_agents/tests/op/ut/test_triton_ascend_api_docs.py -q`

#### Triton Ascend API 文档来源

- 事实来源为 `api_manifest.json` 和拆分后的 `*.md`
- `api.md` 为生成产物，不作为手工维护入口
- 当前环境不存在的 API 会在聚合时写入 `api.md` 末尾的“当前版本不存在的 API”章节

## 不做什么

- **不要**把 Python 逻辑代码放在 prompt 模板里——模板只做文本渲染
- **不要**把测试数据放在这里——归 `tests/op/resources/`
- **不要**继续维护 `skills/triton-ascend/.../triton-ascend-api/SKILL.md` 作为 API 文档入口
- **不要**手工编辑 `docs/triton_ascend_docs/api/api.md`——应修改条目文件和 `api_manifest.json` 后重新生成
- **不要**随意改动 `docs/` 中的上游镜像文档
