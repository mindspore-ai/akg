[English Version](../LLM.md)

# LLM 接入

## 1. 概述

LLM 模块提供统一的大语言模型访问接口，支持两种 API 协议：

- **OpenAI 兼容协议**（默认）— 适用于大部分 LLM 服务商
- **Anthropic 协议** — 仅用于 Kimi Coding Plan（api.kimi.com/coding）

核心组件：

- **LLMProvider** — OpenAI 兼容 API 提供者
- **AnthropicProvider** — Anthropic 协议 API 提供者（Kimi Coding Plan）
- **LLMClient** — 高层客户端，带 Token 计数和流式输出
- **工厂函数** — `create_llm_client()` 和 `create_embedding_model()`
- **OpenAICompatibleEmbeddings** — 用于 RAG 的 Embedding 模型

## 2. 支持的提供商

### OpenAI 兼容协议（`provider_type="openai"`，默认）

| 提供商 | 示例模型 |
|--------|----------|
| OpenAI | GPT-4o, o3, o4-mini |
| DeepSeek | deepseek-chat, deepseek-reasoner |
| Claude | claude-3-5-sonnet（通过 OpenAI 兼容层） |
| 智谱 GLM | GLM-4 系列 |
| Moonshot / Kimi | kimi-k2, kimi-k2.5（普通 API） |
| 通义千问 / DashScope | qwen-plus, qwq |
| 豆包 / 火山引擎 | doubao-seed |
| 硅基流动 SiliconFlow | 各类模型 |
| vLLM | 本地部署 |
| Ollama | 本地模型 |

### Anthropic 协议（`provider_type="anthropic"`）

| 提供商 | 示例模型 | 说明 |
|--------|----------|------|
| Kimi Coding Plan | kimi-for-coding | `https://api.kimi.com/coding` 端点使用 Anthropic 协议 |

> **注意**：Kimi 的普通 API（如 kimi-k2）使用 OpenAI 兼容协议，只有 Kimi Coding Plan（api.kimi.com/coding）需要设置 `provider_type="anthropic"`。

> 各 provider 的具体配置示例（含 thinking/reasoning 参数），请参考 [`settings.example.more.json`](../../examples/settings.example.more.json)。

## 3. Provider 选择

通过 `provider_type` 参数选择 API 协议：

```python
from akg_agents.core_v2.llm import create_llm_client

# OpenAI 兼容协议（默认，适用于大部分服务商）
client = create_llm_client(
    model_name="deepseek-chat",
    base_url="https://api.deepseek.com/beta/",
    api_key="your-key",
    provider_type="openai"  # 可省略，默认值
)

# Kimi Coding Plan（需要 Anthropic 协议）
client = create_llm_client(
    model_name="kimi-for-coding",
    base_url="https://api.kimi.com/coding",
    api_key="your-key",
    provider_type="anthropic"  # 必须指定
)
```

或通过配置文件/环境变量：

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

## 4. LLMProvider（OpenAI 兼容协议）

`LLMProvider` 是基于 `AsyncOpenAI` 的底层 API 客户端。

```python
provider = LLMProvider(
    model_name="deepseek-reasoner",
    base_url="https://api.deepseek.com/beta/",
    api_key="your-api-key",
    extra_body={"thinking": {"type": "enabled"}}  # 透传到 API 请求的额外参数
)

# 非流式
result = await provider.generate(messages, temperature=0.2)

# 流式
async for chunk in provider.generate_stream(messages, temperature=0.2):
    print(chunk)
```

`extra_body` 参数会直接透传到 API 请求体中，用于配置各 provider 的特殊功能（如 thinking/reasoning）。不同 provider 的参数格式不同，详见 [`settings.example.more.json`](../../examples/settings.example.more.json)。

## 4. LLMClient

`LLMClient` 封装 `LLMProvider`，提供额外功能：

- **Token 计数**：追踪总 Token 使用量
- **流式 UI**：通过 session_id 将流式输出发送到 UI
- **推理内容**：自动处理 thinking 模型的 `reasoning_content`

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

### 构造函数参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `provider` | LLMProvider | — | LLM 提供者实例 |
| `session_id` | string | `None` | UI 会话 ID，用于流式输出 |
| `temperature` | float | `0.2` | 采样温度 |
| `max_tokens` | int | `8192` | 最大输出 token 数 |
| `top_p` | float | `0.9` | Top-p 采样 |

## 5. 工厂函数

### create_llm_client

从配置创建 `LLMClient`。

```python
from akg_agents.core_v2.llm import create_llm_client

# 使用配置中的模型级别
client = create_llm_client(model_level="complex")
client = create_llm_client(model_level="standard", session_id="xxx")
client = create_llm_client(model_level="fast")

# 自定义级别（在 settings.json 中定义）
client = create_llm_client(model_level="coder")

# 直接指定参数（覆盖配置）
client = create_llm_client(
    model_name="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="your-key",
    temperature=0.5
)
```

### create_embedding_model

创建用于 RAG 的 Embedding 模型。

```python
from akg_agents.core_v2.llm import create_embedding_model

embedding = create_embedding_model()
# 使用 settings.json 中的 embedding 配置
```

## 6. OpenAICompatibleEmbeddings

兼容 LangChain 的 Embedding 模型，适用于任何 OpenAI 兼容的 Embedding API。

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
