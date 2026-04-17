[English Version](../Configuration.md)

# 配置系统

## 1. 概述

AKG Agents 使用多层级配置系统管理 LLM 服务、Embedding 模型和框架设置。

### 优先级（从高到低）

1. **环境变量**（`AKG_AGENTS_*` 或 `AIKG_*`）
2. **本地配置**：`.akg/settings.local.json`（个人配置，gitignored）
3. **项目配置**：`.akg/settings.json`（项目共享，提交到 git）
4. **用户配置**：`~/.akg/settings.json`（全局，跨项目）
5. **默认值**

## 2. settings.json

### 完整示例

```json
{
  "models": {
    "complex": {
      "base_url": "https://api.deepseek.com/beta/",
      "api_key": "your-deepseek-api-key",
      "model_name": "deepseek-reasoner",
      "temperature": 0.0,
      "max_tokens": 8192,
      "extra_body": {
        "thinking": {"type": "enabled"}
      }
    },
    "standard": {
      "base_url": "https://api.deepseek.com/beta/",
      "api_key": "your-deepseek-api-key",
      "model_name": "deepseek-chat"
    },
    "fast": {
      "base_url": "https://api.deepseek.com/beta/",
      "api_key": "your-deepseek-api-key",
      "model_name": "deepseek-chat"
    }
  },
  "embedding": {
    "base_url": "https://api.siliconflow.cn/v1",
    "api_key": "your-api-key",
    "model_name": "BAAI/bge-large-zh-v1.5"
  },
  "default_model": "standard",
  "context_window": 128000,
  "stream_output": false
}
```

> 各 provider 的 `extra_body` 配置示例（OpenAI、Claude、通义千问、智谱、Kimi、豆包等），请参考 [`settings.example.more.json`](../../examples/settings.example.more.json)。

## 3. ModelConfig

单个 LLM 模型的配置。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `base_url` | string | — | API 端点 URL |
| `api_key` | string | — | API 密钥 |
| `model_name` | string | — | 模型名称 |
| `provider_type` | string | `"openai"` | Provider 类型：`"openai"`（OpenAI 兼容协议）或 `"anthropic"`（Anthropic 协议） |
| `temperature` | float | `0.2` | 采样温度 |
| `max_tokens` | int | `8192` | 最大输出 token 数 |
| `top_p` | float | `0.9` | Top-p 采样 |
| `frequency_penalty` | float | `None` | 频率惩罚（可选） |
| `presence_penalty` | float | `None` | 存在惩罚（可选） |
| `timeout` | int | `300` | 请求超时时间（秒） |
| `extra_body` | object | `{}` | 透传到 API 请求体的额外参数（如 thinking/reasoning 配置） |

### Provider 类型选择

AKG Agents 支持两种 API 协议：

| `provider_type` | 协议 | API 路径 | 适用场景 |
|-----------------|------|---------|---------|
| `"openai"` | OpenAI 兼容协议 | `/chat/completions` | DeepSeek、OpenAI、智谱、通义千问、豆包、SiliconFlow 等大部分服务商 |
| `"anthropic"` | Anthropic 协议 | `/v1/messages` | Kimi Coding Plan、Claude API 等 Anthropic 协议服务商 |

**重要**：使用 Kimi Coding Plan 时，必须设置 `provider_type=anthropic`：

```json
{
  "models": {
    "standard": {
      "base_url": "https://api.kimi.com/coding",
      "api_key": "sk-kimi-xxx",
      "model_name": "kimi-for-coding",
      "provider_type": "anthropic"
    }
  }
}
```

或通过环境变量：

```bash
export AKG_AGENTS_BASE_URL="https://api.kimi.com/coding"
export AKG_AGENTS_API_KEY="sk-kimi-xxx"
export AKG_AGENTS_MODEL_NAME="kimi-for-coding"
export AKG_AGENTS_PROVIDER_TYPE="anthropic"
```

### 模型级别

| 级别 | 典型用途 |
|------|----------|
| `complex` | 复杂推理任务（如算法设计） |
| `standard` | 通用任务（默认） |
| `fast` | 简单任务，要求低延迟 |

也支持自定义级别（如 `"coder"`、`"designer"`）。

## 4. EmbeddingConfig

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `base_url` | string | — | Embedding API 端点 |
| `api_key` | string | — | API 密钥 |
| `model_name` | string | — | Embedding 模型名称 |
| `timeout` | int | `60` | 请求超时时间（秒） |

## 5. AKGSettings

顶层设置数据类。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `models` | Dict[str, ModelConfig] | `{}` | 按级别的模型配置 |
| `embedding` | EmbeddingConfig | — | Embedding 模型配置 |
| `default_model` | string | `"standard"` | 默认模型级别 |
| `context_window` | int | `128000` | 上下文窗口大小（token） |
| `stream_output` | bool | `None` | 启用流式输出 |

## 6. 环境变量

### 单模型配置（应用到所有级别）

```bash
export AKG_AGENTS_BASE_URL="https://api.deepseek.com/beta/"
export AKG_AGENTS_API_KEY="YOUR_API_KEY"
export AKG_AGENTS_MODEL_NAME="deepseek-chat"
export AKG_AGENTS_MODEL_ENABLE_THINK="enabled"  # 或 "disabled"
```

### 按级别配置

```bash
export AKG_AGENTS_COMPLEX_BASE_URL="https://api.openai.com/v1"
export AKG_AGENTS_COMPLEX_API_KEY="YOUR_API_KEY"
export AKG_AGENTS_COMPLEX_MODEL_NAME="gpt-4"

export AKG_AGENTS_STANDARD_BASE_URL="https://api.deepseek.com/beta/"
export AKG_AGENTS_STANDARD_API_KEY="YOUR_API_KEY"
export AKG_AGENTS_STANDARD_MODEL_NAME="deepseek-chat"
```

### Embedding

```bash
export AKG_AGENTS_EMBEDDING_BASE_URL="https://api.siliconflow.cn/v1"
export AKG_AGENTS_EMBEDDING_API_KEY="YOUR_API_KEY"
export AKG_AGENTS_EMBEDDING_MODEL_NAME="BAAI/bge-large-zh-v1.5"
```

### 其他

```bash
export AKG_AGENTS_STREAM_OUTPUT="on"   # 或 "off"
```

> 注意：旧版 `AIKG_*` 前缀仍然兼容支持。

## 7. API 参考

| 函数 | 说明 |
|------|------|
| `get_settings()` | 加载并返回合并后的 `AKGSettings` 实例。 |
| `get_settings_path()` | 返回当前活跃的配置文件路径。 |
| `get_all_settings_paths()` | 返回所有候选配置文件路径。 |
| `save_settings_file(path, settings)` | 将配置保存到 JSON 文件。 |
| `create_default_settings_file(path)` | 创建默认配置文件。 |
| `load_settings_file(path)` | 从指定文件加载配置。 |
| `print_settings_info()` | 在控制台打印当前配置信息。 |
| `check_model_config(level)` | 检查模型级别是否已正确配置。 |
