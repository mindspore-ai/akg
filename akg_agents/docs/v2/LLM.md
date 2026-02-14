[中文版](./CN/LLM.md)

# LLM Integration

## 1. Overview

The LLM module provides a unified interface for accessing Large Language Models through OpenAI-compatible APIs. It supports multiple providers with a single API surface.

Key components:

- **LLMProvider** — OpenAI-compatible API provider
- **LLMClient** — High-level client with token counting and streaming
- **Factory functions** — `create_llm_client()` and `create_embedding_model()`
- **OpenAICompatibleEmbeddings** — Embedding model for RAG

## 2. Supported Providers

All providers are accessed through the same OpenAI-compatible interface:

| Provider | Examples |
|----------|----------|
| OpenAI | GPT-4o, o3, o4-mini |
| DeepSeek | deepseek-chat, deepseek-reasoner |
| Claude | Via Anthropic's OpenAI-compatible layer |
| GLM (Zhipu) | GLM-4 series |
| Moonshot / Kimi | kimi-k2, kimi-k2.5 |
| Qwen / DashScope | qwen-plus, qwq |
| Doubao / Volcengine | doubao-seed |
| SiliconFlow | Various models via SiliconFlow |
| vLLM | Local deployment |
| Ollama | Local models |

> See [`settings.example.more.json`](../../examples/settings.example.more.json) for provider-specific configuration examples including thinking/reasoning parameters.

## 3. LLMProvider

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
