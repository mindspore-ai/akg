# op/config/ — 算子配置

## 职责

管理算子/内核生成各 DSL 和 workflow 的默认配置（YAML）和配置校验逻辑。

## 文件结构

| 类型 | 文件 | 说明 |
|------|------|------|
| Python | `config_validator.py` | `ConfigValidator`、`load_config()`、`normalize_dsl()` |
| YAML | `default_{dsl}_config.yaml` | 各 DSL 的默认配置（14 个 DSL） |
| YAML | `{dsl}_{workflow}_config.yaml` | 特定 DSL + workflow 组合的配置 |
| YAML | `evolve_config.yaml` | 进化搜索批跑配置（顶层配置，引用基础 yaml） |
| YAML | `adaptive_search_config.yaml` | 自适应搜索配置（同上） |

## YAML 结构

### 算子主配置（`default_*` / `*_workflow_*`）

```yaml
log_dir: ~/akg_agents_logs        # 必需，校验时会展开 ~ 并追加唯一子目录
default_workflow: kernelgen_only   # 默认工作流名
max_step: 10                       # 最大迭代步数
docs_dir:                          # 必需，dict 类型
  designer: path/to/docs
  coder: path/to/docs
profile_settings:
  run_times: 50
  warmup_times: 5
verify_timeout: 120                # 验证超时（秒）
```

### `load_config()` 逻辑

1. `normalize_dsl(dsl, backend)` 规范化 DSL 名称
2. 配置文件查找：`config_path`（直传） > `{dsl}_{workflow}_config.yaml` > `default_{dsl}_config.yaml`
3. `ConfigValidator` 校验 `log_dir`（展开路径）和 `docs_dir`（必须为 dict，路径必须存在）
4. 返回校验后的 config 字典

## 开发约定

### 新增 DSL 配置

1. 创建 `default_{dsl}_config.yaml`，至少包含 `log_dir`、`default_workflow`、`docs_dir`
2. 如需特定 workflow 配置，创建 `{dsl}_{workflow}_config.yaml`
3. `docs_dir` 中的路径指向 `../resources/docs/` 下的对应目录

## 不做什么

- **不要**在 yaml 中硬编码绝对路径——使用 `~` 或相对路径
- **不要**在 yaml 中放业务逻辑——yaml 只描述配置参数
