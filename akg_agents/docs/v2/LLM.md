[中文版](./CN/LLM.md)

# LLM Integration

## 1. Overview

The LLM module provides a unified interface for large language model access, supporting two API protocols:

- **OpenAI-compatible protocol** (default) — For most LLM providers
- **Anthropic protocol** — Only for Kimi Coding Plan (api.kimi.com/coding)

Key components:

- **LLMProvider** — OpenAI-compatible API provider
- **AnthropicProvider** — Anthropic protocol API provider (Kimi Coding Plan)
- **LLMClient** — High-level client with token counting and streaming
- **Factory functions** — `create_llm_client()` and `create_embedding_model()`
- **OpenAICompatibleEmbeddings** — Embedding model for RAG

## 2. Supported Providers

### OpenAI-Compatible Protocol (`provider_type="openai"`, default)

| Provider | Examples |
|----------|----------|
| OpenAI | GPT-4o, o3, o4-mini |
| DeepSeek | deepseek-chat, deepseek-reasoner |
| Claude | claude-3-5-sonnet (via OpenAI-compatible layer) |
| GLM (Zhipu) | GLM-4 series |
| Moonshot / Kimi | kimi-k2, kimi-k2.5 (standard API) |
| Qwen / DashScope | qwen-plus, qwq |
| Doubao / Volcengine | doubao-seed |
| SiliconFlow | Various models via SiliconFlow |
| vLLM | Local deployment |
| Ollama | Local models |

### Anthropic Protocol (`provider_type="anthropic"`)

| Provider | Example Models | Notes |
|----------|----------------|-------|
| Kimi Coding Plan | kimi-for-coding | `https://api.kimi.com/coding` endpoint uses Anthropic protocol |

> **Note**: Kimi's standard API (e.g., kimi-k2) uses OpenAI-compatible protocol. Only Kimi Coding Plan (api.kimi.com/coding) requires `provider_type="anthropic"`.

> See [`settings.example.more.json`](../../examples/settings.example.more.json) for provider-specific configuration examples including thinking/reasoning parameters.

## 3. Provider Selection

Select API protocol via `provider_type` parameter:

```python
from akg_agents.core_v2.llm import create_llm_client

# OpenAI-compatible protocol (default, for most providers)
client = create_llm_client(
    model_name="deepseek-chat",
    base_url="https://api.deepseek.com/beta/",
    api_key="your-key",
    provider_type="openai"  # optional, default value
)

# Kimi Coding Plan (requires Anthropic protocol)
client = create_llm_client(
    model_name="kimi-for-coding",
    base_url="https://api.kimi.com/coding",
    api_key="your-key",
    provider_type="anthropic"  # must be specified
)
```

Or via config file/environment variables:

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

```bash
export AKG_AGENTS_PROVIDER_TYPE="anthropic"
```

## 4. LLMProvider (OpenAI-Compatible Protocol)

`LLMProvider` is the low-level API client based on `AsyncOpenAI`.

```python
provider = LLMProvider(
    model_name="deepseek-reasoner",
    base_url="https://api.deepseek.com/beta/",
    api_key="your-api-key",
    extra_body={"thinking": {"type": "enabled"}}  # Provider-specific params, passed through to API
)

# Non-streaming
result = await provider.generate(messages, temperature=0.2)

# Streaming
async for chunk in provider.generate_stream(messages, temperature=0.2):
    print(chunk)
```

The `extra_body` parameter is passed directly to the API request body, allowing you to configure provider-specific features like thinking/reasoning. Different providers use different parameter formats — see [`settings.example.more.json`](../../examples/settings.example.more.json) for examples.

## 4. LLMClient

`LLMClient` wraps `LLMProvider` with additional features:

- **Token counting**: Tracks total tokens used
- **Streaming UI**: Sends streaming output to UI via session_id
- **Reasoning content**: Automatically handles `reasoning_content` from thinking models

```python
from akg_agents.core_v2.llm import create_llm_client

client = create_llm_client(model_level="standard", session_id="my_session")

result = await client.generate(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    stream=True,
    agent_name="MyAgent"
)

content = result["content"]
reasoning = result.get("reasoning_content", "")
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | LLMProvider | — | LLM provider instance |
| `session_id` | string | `None` | UI session ID for streaming |
| `temperature` | float | `0.2` | Sampling temperature |
| `max_tokens` | int | `8192` | Maximum output tokens |
| `top_p` | float | `0.9` | Top-p sampling |

## 5. Factory Functions

### create_llm_client

Create an `LLMClient` from configuration.

```python
from akg_agents.core_v2.llm import create_llm_client

# From config level
client = create_llm_client(model_level="complex")
client = create_llm_client(model_level="standard", session_id="xxx")
client = create_llm_client(model_level="fast")

# Custom level (defined in settings.json)
client = create_llm_client(model_level="coder")

# Direct parameters (override config)
client = create_llm_client(
    model_name="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="your-key",
    temperature=0.5
)
```

### create_embedding_model

Create an embedding model for RAG.

```python
from akg_agents.core_v2.llm import create_embedding_model

embedding = create_embedding_model()
# Uses embedding config from settings.json
```

## 6. OpenAICompatibleEmbeddings

A LangChain-compatible embedding model that works with any OpenAI-compatible embedding API.

```python
from akg_agents.core_v2.llm import OpenAICompatibleEmbeddings

embeddings = OpenAICompatibleEmbeddings(
    base_url="https://api.siliconflow.cn/v1",
    api_key="your-key",
    model_name="BAAI/bge-large-zh-v1.5"
)

vectors = embeddings.embed_documents(["Hello", "World"])
query_vector = embeddings.embed_query("Hello")
```
