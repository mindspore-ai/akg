# API Configuration Guide

This document provides a detailed explanation of the API configurations used in the AIKG project.

---

## 1. Configuration File Locations and Priority

Configuration is loaded with the following priority (high to low):

| Priority | Scope | Location | Affects | Shared with Team? |
|----------|-------|----------|---------|-------------------|
| 1 | **Environment Variables** | `AKG_AGENTS_*` | Runtime | No |
| 2 | **Local** | `.akg/settings.local.json` | This project, this user only | No (gitignored) |
| 3 | **Project** | `.akg/settings.json` | All collaborators of this project | Yes (committed to git) |
| 4 | **User** | `~/.akg/settings.json` | All projects | No |
| 5 | **Defaults** | Built-in code | - | - |

---

## 2. Configuration File Format (`settings.json`)

```json
{
  "models": {
    "complex": {
      "base_url": "https://api.deepseek.com/beta/",
      "api_key": "sk-xxx",
      "model_name": "deepseek-reasoner",
      "temperature": 0.0,
      "max_tokens": 8192,
      "thinking_enabled": true
    },
    "standard": {
      "base_url": "https://api.deepseek.com/beta/",
      "api_key": "sk-xxx",
      "model_name": "deepseek-chat",
      "temperature": 0.1
    },
    "fast": {
      "base_url": "https://api.openai.com/v1",
      "api_key": "sk-xxx",
      "model_name": "gpt-4o-mini"
    }
  },
  "embedding": {
    "base_url": "https://api.siliconflow.cn/v1",
    "api_key": "sk-xxx",
    "model_name": "BAAI/bge-large-zh-v1.5"
  },
  "default_model": "standard",
  "stream_output": false
}
```

**Model Level Description:**
- `complex`: For complex tasks (e.g., deepseek-reasoner, o1)
- `standard`: For standard tasks (default)
- `fast`: For quick responses

**LLM Model Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | string | `https://api.openai.com/v1` | API base URL |
| `api_key` | string | - | API key |
| `model_name` | string | `gpt-4` | Model name |
| `temperature` | float | `0.2` | Temperature parameter |
| `max_tokens` | int | `8192` | Maximum generated tokens |
| `top_p` | float | `0.9` | Nucleus sampling parameter |
| `frequency_penalty` | float | - | Frequency penalty (optional) |
| `presence_penalty` | float | - | Presence penalty (optional) |
| `timeout` | int | `300` | Timeout in seconds |
| `thinking_enabled` | bool | `false` | Enable thinking mode |

**Embedding Model Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | string | - | Embedding API base URL |
| `api_key` | string | - | API key |
| `model_name` | string | - | Embedding model name |
| `timeout` | int | `60` | Timeout in seconds |

---

## 3. Environment Variable Configuration

### 3.1 Single Model Configuration (Recommended)

Setting environment variables will automatically override all levels (`complex` / `standard` / `fast`) configuration:

```bash
# Basic configuration
export AKG_AGENTS_BASE_URL="https://api.deepseek.com/beta/"
export AKG_AGENTS_API_KEY="sk-xxx"
export AKG_AGENTS_MODEL_NAME="deepseek-chat"

# Optional parameters
export AKG_AGENTS_TEMPERATURE="0.0"
export AKG_AGENTS_MAX_TOKENS="8192"
export AKG_AGENTS_TIMEOUT="300"
export AKG_AGENTS_MODEL_ENABLE_THINK="enabled"  # Enable thinking mode

# Other settings
export AKG_AGENTS_DEFAULT_MODEL="standard"
export AKG_AGENTS_STREAM_OUTPUT="on"
```

**Thinking mode values:**
- DeepSeek style: `enabled` / `disabled`
- GLM style: `true` / `false`
- Other valid values: `1` / `yes` / `on`

### 3.2 Multi-Model Configuration

Set different configurations for each level:

```bash
# Complex level
export AKG_AGENTS_COMPLEX_BASE_URL="https://api.deepseek.com/beta/"
export AKG_AGENTS_COMPLEX_API_KEY="sk-xxx"
export AKG_AGENTS_COMPLEX_MODEL_NAME="deepseek-reasoner"

# Standard level
export AKG_AGENTS_STANDARD_BASE_URL="https://api.deepseek.com/beta/"
export AKG_AGENTS_STANDARD_API_KEY="sk-xxx"
export AKG_AGENTS_STANDARD_MODEL_NAME="deepseek-chat"

# Fast level
export AKG_AGENTS_FAST_BASE_URL="https://api.openai.com/v1"
export AKG_AGENTS_FAST_API_KEY="sk-xxx"
export AKG_AGENTS_FAST_MODEL_NAME="gpt-4o-mini"
```

### 3.3 Embedding Model Configuration

```bash
# Embedding configuration
export AKG_AGENTS_EMBEDDING_BASE_URL="https://api.siliconflow.cn/v1"
export AKG_AGENTS_EMBEDDING_API_KEY="sk-xxx"
export AKG_AGENTS_EMBEDDING_MODEL_NAME="BAAI/bge-large-zh-v1.5"
export AKG_AGENTS_EMBEDDING_TIMEOUT="60"  # Optional
```

---

## 4. Usage Examples

### 4.1 LLM Client

```python
from akg_agents.core_v2 import create_llm_client, get_settings

# Method 1: Use model level from configuration
client = create_llm_client(model_level="standard")
result = await client.generate([{"role": "user", "content": "Hello"}])

# Method 2: Directly specify parameters (override config)
client = create_llm_client(
    model_name="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="sk-xxx",
    temperature=0.7
)

# Method 3: Use default configuration
client = create_llm_client()  # Uses default_model
```

### 4.2 Embedding Model

```python
from akg_agents.core_v2.llm import create_embedding_model

# Method 1: Use configuration (environment variables or settings.json)
embedding = create_embedding_model()

# Generate embeddings for documents
vectors = embedding.embed_documents(["text 1", "text 2", "text 3"])

# Generate embedding for a single query
query_vec = embedding.embed_query("search query")

# Method 2: Directly specify parameters (override config)
embedding = create_embedding_model(
    base_url="https://api.siliconflow.cn/v1",
    api_key="sk-xxx",
    model_name="BAAI/bge-large-zh-v1.5"
)
```

---

## 5. Supported LLM Services

Uses a unified OpenAI-compatible interface, supporting the following services:

| Service | base_url | Description |
|---------|----------|-------------|
| OpenAI | `https://api.openai.com/v1` | GPT-4, GPT-4o, o1, etc. |
| DeepSeek | `https://api.deepseek.com/beta/` | deepseek-chat, deepseek-reasoner |
| GLM (Zhipu) | `https://open.bigmodel.cn/api/paas/v4/` | glm-4, glm-4-plus |
| Moonshot | `https://api.moonshot.cn/v1` | moonshot-v1-8k, etc. |
| SiliconFlow | `https://api.siliconflow.cn/v1` | Various models |
| vLLM | `http://localhost:8000/v1` | Local deployment |
| Ollama | `http://localhost:11434/v1` | Local running |

---

## 6. View Current Configuration

```python
from akg_agents.core_v2.config import print_settings_info

# Print current configuration info
print_settings_info()
```

Example output:
```
============================================================
🌍 Environment Variables (highest priority):
   ✓ AKG_AGENTS_BASE_URL=https://api.deepseek.com/beta/
   ✓ AKG_AGENTS_API_KEY=sk-235e0***dc3f
   ✓ AKG_AGENTS_MODEL_NAME=deepseek-chat
💡 Priority: Environment Variables > Local > Project > User > Defaults
============================================================
```
