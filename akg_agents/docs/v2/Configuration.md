[中文版](./CN/Configuration.md)

# Configuration

## 1. Overview

AKG Agents uses a multi-level configuration system for managing LLM services, embedding models, and framework settings.

### Priority (High to Low)

1. **Environment variables** (`AKG_AGENTS_*` or `AIKG_*`)
2. **Local config**: `.akg/settings.local.json` (per-user, gitignored)
3. **Project config**: `.akg/settings.json` (shared in repo)
4. **User config**: `~/.akg/settings.json` (global, cross-project)
5. **Defaults**

## 2. settings.json

### Full Example

```json
{
  "models": {
    "complex": {
      "base_url": "https://api.openai.com/v1",
      "api_key": "your-api-key",
      "model_name": "gpt-4",
      "temperature": 0.2,
      "max_tokens": 8192
    },
    "standard": {
      "base_url": "https://api.deepseek.com/beta/",
      "api_key": "your-deepseek-api-key",
      "model_name": "deepseek-chat",
      "thinking_enabled": true
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

## 3. ModelConfig

Configuration for a single LLM model.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `base_url` | string | — | API endpoint URL |
| `api_key` | string | — | API key |
| `model_name` | string | — | Model name |
| `temperature` | float | `0.2` | Sampling temperature |
| `max_tokens` | int | `8192` | Maximum output tokens |
| `top_p` | float | `0.9` | Top-p sampling |
| `frequency_penalty` | float | `None` | Frequency penalty (optional) |
| `presence_penalty` | float | `None` | Presence penalty (optional) |
| `timeout` | int | `300` | Request timeout in seconds |
| `thinking_enabled` | bool | `None` | Enable thinking mode (for models that support it) |

### Model Levels

| Level | Typical Use |
|-------|-------------|
| `complex` | Complex reasoning tasks (e.g., algorithm design) |
| `standard` | General tasks (default) |
| `fast` | Simple tasks requiring low latency |

Custom levels (e.g., `"coder"`, `"designer"`) are also supported.

## 4. EmbeddingConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `base_url` | string | — | Embedding API endpoint |
| `api_key` | string | — | API key |
| `model_name` | string | — | Embedding model name |
| `timeout` | int | `60` | Request timeout in seconds |

## 5. AKGSettings

The top-level settings dataclass.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `models` | Dict[str, ModelConfig] | `{}` | Model configurations by level |
| `embedding` | EmbeddingConfig | — | Embedding model config |
| `default_model` | string | `"standard"` | Default model level |
| `context_window` | int | `128000` | Context window size in tokens |
| `stream_output` | bool | `None` | Enable streaming output |
| `data_collect` | bool | `None` | Enable data collection |

## 6. Environment Variables

### Single Model (applies to all levels)

```bash
export AKG_AGENTS_BASE_URL="https://api.deepseek.com/beta/"
export AKG_AGENTS_API_KEY="YOUR_API_KEY"
export AKG_AGENTS_MODEL_NAME="deepseek-chat"
export AKG_AGENTS_MODEL_ENABLE_THINK="enabled"  # or "disabled"
```

### Per-Level Configuration

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

### Other

```bash
export AKG_AGENTS_STREAM_OUTPUT="on"   # or "off"
```

> Note: Legacy `AIKG_*` prefix is also supported for backwards compatibility.

## 7. API Reference

| Function | Description |
|----------|-------------|
| `get_settings()` | Load and return the merged `AKGSettings` instance. |
| `get_settings_path()` | Return the path to the active settings file. |
| `get_all_settings_paths()` | Return all candidate settings file paths. |
| `save_settings_file(path, settings)` | Save settings to a JSON file. |
| `create_default_settings_file(path)` | Create a default settings file. |
| `load_settings_file(path)` | Load settings from a specific file. |
| `print_settings_info()` | Print current settings info to console. |
| `check_model_config(level)` | Check if a model level is properly configured. |
