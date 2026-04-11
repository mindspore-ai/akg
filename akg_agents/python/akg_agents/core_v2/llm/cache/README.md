# LLM Cache 模块说明

## 简介

本目录实现 LLM 缓存核心能力，支持三种模式：

- off：不读不写缓存，直接调用实时 LLM。
- record：可读缓存，未命中时调用实时 LLM，并写入缓存。
- replay：只读缓存，未命中直接抛异常，不回退实时 LLM。

该行为由 cache_mode 单一开关控制，历史参数 cache_enable 仅作兼容并被忽略。

## 核心行为

### 1) replay 严格语义

- replay 未命中时抛出 RuntimeError：
  Replay cache miss for session/content keys; aborting to avoid live LLM call
- 保证回放稳定与可追溯，不发生隐式在线调用。

### 2) replay 命中顺序

当前命中顺序如下：

1. session 精确键：session_hash:agent_hash
2. content 精确键：消息与参数哈希键
3. session 前缀兜底：session_hash:*（仅在 1/2 都 miss 后启用）

该顺序用于兼容 agent hash 漂移，同时避免前缀命中过早覆盖精确命中。

### 3) 双层缓存

- 内存层：LRU
- 本地文件层：JSON 文件
- 支持过期清理（expire_seconds）

## 配置与环境变量

默认缓存配置来自模块内默认值与 YAML 配置，支持环境变量覆盖：

- AKG_AGENTS_CACHE_CONFIG_PATH（兼容 AIKG_CACHE_CONFIG_PATH）
- AKG_AGENTS_CACHE_FILE_PATH（兼容 AIKG_CACHE_FILE_PATH）

默认缓存文件路径：

- ~/.akg/llm_cache/llm_test_cache.json

## 常用命令

### 查看缓存

```bash
python -m akg_agents.core_v2.llm.cache.view_cache [cache_file] [--stats] [-v]
```

示例：

```bash
python -m akg_agents.core_v2.llm.cache.view_cache --stats
python -m akg_agents.core_v2.llm.cache.view_cache -v
python -m akg_agents.core_v2.llm.cache.view_cache --stats -v
python -m akg_agents.core_v2.llm.cache.view_cache /tmp/llm_cache.json --stats
```

### 示例脚本切换 cache_mode

```bash
python examples/kernel_related/cpu/run_torch_cpu_cpp_single_record_and_replay.py --cache-mode off
python examples/kernel_related/cpu/run_torch_cpu_cpp_single_record_and_replay.py --cache-mode record
python examples/kernel_related/cpu/run_torch_cpu_cpp_single_record_and_replay.py --cache-mode replay --cache-session-hash <session_hash>
```

## Replay 样本目录与发现

模块提供固定三组 CPU attention replay 样本发现接口：

- discover_cpu_attention_replay_scenarios
- get_project_cache_dir
- ReplayCacheScenario

默认样本目录：

- akg_agents/.cache/

## 测试结构（已迁移）

### 1) 快速 UT

- tests/ut/cache_tests/

包含：

- 缓存核心读写
- replay miss 严格保护
- session 前缀兜底命中
- replay catalog 与环境变量覆盖

执行：

```bash
PYTHONPATH=akg_agents/python python -m pytest akg_agents/tests/ut/cache_tests -q
```

### 2) ST：replay 编译与精度验证

- tests/op/st/test_cache_replay_kernel_verification.py

执行：

```bash
cd akg_agents/tests/op/st
PYTHONPATH=../../../python python -m pytest test_cache_replay_kernel_verification.py -v
```

### 3) ST：record/replay 真实样本生成与导出

- tests/op/st/test_cache_record_replay_real_samples.py

该用例会在 record 后导出样本，并进行“最终收敛版本归一化”：

- 若存在 session:0@1, session:0@2...，导出时统一提升为最终收敛 step
- 防止 replay 命中早期失败版本

执行（真实调用，依赖模型环境）：

```bash
cd akg_agents/tests/op/st
PYTHONPATH=../../../python python -m pytest test_cache_record_replay_real_samples.py -v
```

## 文件结构

```text
cache/
├── __init__.py
├── cache_config.py
├── cache_decorator.py
├── cache_utils.py
├── llm_cache.py
├── replay_catalog.py
├── view_cache.py
└── README.md
```
