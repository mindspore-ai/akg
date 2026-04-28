# examples/ — 示例规范

## 职责

提供 akg_agents 各功能模块的可运行示例代码。

## 目录结构

| 子目录 | 说明 |
|--------|------|
| `build_a_simple_react_agent/` | 最小 ReAct Agent 示例（含 README） |
| `build_a_simple_workflow/` | 最小 LangGraph 工作流示例 |
| `run_skill/` | Skill 加载/选择/过滤系列脚本 + 示例 SKILL.md |
| `kernel_related/` | 算子生成（单卡/批量/CPU/GPU/Ascend/evolve/adaptive_search/AB test） |
| `common/` | 其他公共示例 |
| `settings.example.json` | 配置文件样例 |
| `settings.example.more.json` | 多厂商 extra_body 配置示例 |

## 开发约定

### 新增示例的标准结构

1. 在对应子目录下创建独立文件夹或 `.py` 文件
2. 提供 `README.md` 说明运行方式和前置条件
3. 示例应当自包含、可独立运行
4. 不依赖用户的特定环境配置（API Key 等用占位符）

## 不做什么

- **不要**把生产级代码放在示例中——示例是教学性质的
- **不要**把测试代码放在这里——归 `tests/`

