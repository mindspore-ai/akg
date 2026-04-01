# LLM Cache CLI 使用说明

## 简介

本目录提供缓存读写与查看能力。

## 查看缓存

### 命令入口

```bash
python -m akg_agents.core_v2.llm.cache.view_cache [cache_file] [--stats] [-v]
python /home/liting/akg/akg_agents/python/akg_agents/core_v2/llm/cache/view_cache.py [cache_file] [--stats] [-v]
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `cache_file` | 可选，缓存文件路径，默认 `~/.akg/llm_cache/llm_test_cache.json` |
| `--stats` | 输出缓存统计摘要 |
| `-v`, `--verbose` | 输出缓存条目详情 |

### 常用命令

```bash
# 仅看统计
python -m akg_agents.core_v2.llm.cache.view_cache --stats

# 仅看详情
python -m akg_agents.core_v2.llm.cache.view_cache -v

# 同时看统计和详情
python -m akg_agents.core_v2.llm.cache.view_cache --stats -v

# 指定缓存文件看统计
python -m akg_agents.core_v2.llm.cache.view_cache /tmp/llm_cache.json --stats
```

### 统计字段说明

| 字段 | 说明 |
|------|------|
| 缓存条目总数 | 当前读取到的总条目数 |
| 本地文件缓存条目数 | 缓存文件中的条目数 |
| 内存缓存条目数 | 该 CLI 默认无法直接读取其他进程的实时内存缓存，显示为 `N/A` |
| 过期条目数 | 依据 `create_time + expire_seconds` 计算 |
| 按会话/前缀聚合统计 | 以 key 前缀聚合（包含 `session_hash:agent_hash` 风格） |
| 缓存文件路径与大小 | 文件元信息 |

### 常见问题

1. 报错 `unrecognized arguments: --stats`

请确认代码已更新到包含 `--stats` 参数的版本，并使用如下命令查看帮助：

```bash
python -m akg_agents.core_v2.llm.cache.view_cache -h
```

2. `--stats` 下内存缓存条目数显示 `N/A`

这是预期行为。该 CLI 读取的是缓存文件视角，无法直接访问其他 Python 进程中的内存态缓存。

## Cache 模式显式设置

以 `kernel_related/cpu/run_torch_cpu_cpp_single_record_and_replay.py` 为例，示例脚本支持通过参数显式设置 cache 模式：

```bash
python examples/kernel_related/cpu/run_torch_cpu_cpp_single_record_and_replay.py --cache-mode off
python examples/kernel_related/cpu/run_torch_cpu_cpp_single_record_and_replay.py --cache-mode record
python examples/kernel_related/cpu/run_torch_cpu_cpp_single_record_and_replay.py --cache-mode replay --cache-session-hash <session_hash>
```

约定：

- 默认 `cache_mode=off`
- `cache_mode=replay` 时必须提供 `cache_session_hash`

## 文件结构

```text
cache/
├── __init__.py        # 模块导出入口，统一暴露 LLMCache 与常用缓存接口
├── cache_config.py    # 缓存配置加载与默认值管理
├── cache_decorator.py # client.generate 缓存装饰器，支持 record/replay 逻辑
├── cache_utils.py     # 缓存键生成、文件读写与兼容性工具函数
├── llm_cache.py       # 双层缓存核心实现（内存 + 本地文件）
├── view_cache.py      # 缓存查看 CLI，支持统计与详细条目展示
└── README.md          # 本文档
```
