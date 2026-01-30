# LLM API 配置说明

此文档用于详细说明 AIKG 项目中涉及的 LLM API 配置。

---

## 1. 配置文件位置与优先级

配置按以下优先级加载（从高到低）：

| 优先级 | 作用域 | 位置 | 影响范围 | 与团队共享？ |
|-------|--------|------|----------|-------------|
| 1 | **环境变量** | `AKG_AGENTS_*` | 运行时 | 否 |
| 2 | **Local** | `.akg/settings.local.json` | 仅本人此项目 | 否（gitignored） |
| 3 | **Project** | `.akg/settings.json` | 此项目所有协作者 | 是（提交到 git） |
| 4 | **User** | `~/.akg/settings.json` | 跨所有项目 | 否 |
| 5 | **默认值** | 代码内置 | - | - |

---

## 2. 配置文件格式 (`settings.json`)

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

**模型级别说明：**
- `complex`：复杂任务模型（如 deepseek-reasoner、o1）
- `standard`：标准任务模型（默认）
- `fast`：快速响应模型

**LLM 模型参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `base_url` | string | `https://api.openai.com/v1` | API 基础 URL |
| `api_key` | string | - | API 密钥 |
| `model_name` | string | `gpt-4` | 模型名称 |
| `temperature` | float | `0.2` | 温度参数 |
| `max_tokens` | int | `8192` | 最大生成 token 数 |
| `top_p` | float | `0.9` | 核采样参数 |
| `frequency_penalty` | float | - | 频率惩罚（可选） |
| `presence_penalty` | float | - | 存在惩罚（可选） |
| `timeout` | int | `300` | 超时时间（秒） |
| `thinking_enabled` | bool | `false` | 是否启用思考模式 |

**Embedding 模型参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `base_url` | string | - | Embedding API 基础 URL |
| `api_key` | string | - | API 密钥 |
| `model_name` | string | - | Embedding 模型名称 |
| `timeout` | int | `60` | 超时时间（秒） |

---

## 3. 环境变量配置

### 3.1 单模型配置（推荐）

设置环境变量会自动覆盖 `standard` 级别的配置：

```bash
# 基础配置
export AKG_AGENTS_BASE_URL="https://api.deepseek.com/beta/"
export AKG_AGENTS_API_KEY="sk-xxx"
export AKG_AGENTS_MODEL_NAME="deepseek-chat"

# 可选参数
export AKG_AGENTS_TEMPERATURE="0.0"
export AKG_AGENTS_MAX_TOKENS="8192"
export AKG_AGENTS_TIMEOUT="300"
export AKG_AGENTS_MODEL_ENABLE_THINK="enabled"  # 启用 thinking 模式

# 其他设置
export AKG_AGENTS_DEFAULT_MODEL="standard"
export AKG_AGENTS_STREAM_OUTPUT="on"
```

**thinking 模式值：**
- DeepSeek 风格：`enabled` / `disabled`
- GLM 风格：`true` / `false`
- 其他有效值：`1` / `yes` / `on`

### 3.2 多模型配置

为不同级别分别设置：

```bash
# Complex 级别
export AKG_AGENTS_COMPLEX_BASE_URL="https://api.deepseek.com/beta/"
export AKG_AGENTS_COMPLEX_API_KEY="sk-xxx"
export AKG_AGENTS_COMPLEX_MODEL_NAME="deepseek-reasoner"

# Standard 级别
export AKG_AGENTS_STANDARD_BASE_URL="https://api.deepseek.com/beta/"
export AKG_AGENTS_STANDARD_API_KEY="sk-xxx"
export AKG_AGENTS_STANDARD_MODEL_NAME="deepseek-chat"

# Fast 级别
export AKG_AGENTS_FAST_BASE_URL="https://api.openai.com/v1"
export AKG_AGENTS_FAST_API_KEY="sk-xxx"
export AKG_AGENTS_FAST_MODEL_NAME="gpt-4o-mini"
```

### 3.3 Embedding 模型配置

```bash
# Embedding 配置
export AKG_AGENTS_EMBEDDING_BASE_URL="https://api.siliconflow.cn/v1"
export AKG_AGENTS_EMBEDDING_API_KEY="sk-xxx"
export AKG_AGENTS_EMBEDDING_MODEL_NAME="BAAI/bge-large-zh-v1.5"
export AKG_AGENTS_EMBEDDING_TIMEOUT="60"  # 可选
```

---

## 4. 使用示例

### 4.1 LLM 客户端

```python
from akg_agents.core_v2 import create_llm_client, get_settings

# 方式 1：使用配置中的模型级别
client = create_llm_client(model_level="standard")
result = await client.generate([{"role": "user", "content": "Hello"}])

# 方式 2：直接指定参数（覆盖配置）
client = create_llm_client(
    model_name="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="sk-xxx",
    temperature=0.7
)

# 方式 3：使用默认配置
client = create_llm_client()  # 使用 default_model
```

### 4.2 Embedding 模型

```python
from akg_agents.core_v2.llm import create_embedding_model

# 方式 1：使用配置（环境变量或 settings.json）
embedding = create_embedding_model()

# 为文档生成向量
vectors = embedding.embed_documents(["文本1", "文本2", "文本3"])

# 为单个查询生成向量
query_vec = embedding.embed_query("搜索查询")

# 方式 2：直接指定参数（覆盖配置）
embedding = create_embedding_model(
    base_url="https://api.siliconflow.cn/v1",
    api_key="sk-xxx",
    model_name="BAAI/bge-large-zh-v1.5"
)
```

---

## 5. 支持的 LLM 服务

统一使用 OpenAI 兼容接口，支持以下服务：

| 服务 | base_url | 说明 |
|-----|----------|------|
| OpenAI | `https://api.openai.com/v1` | GPT-4, GPT-4o, o1 等 |
| DeepSeek | `https://api.deepseek.com/beta/` | deepseek-chat, deepseek-reasoner |
| 智谱 GLM | `https://open.bigmodel.cn/api/paas/v4/` | glm-4, glm-4-plus |
| Moonshot | `https://api.moonshot.cn/v1` | moonshot-v1-8k 等 |
| 硅基流动 | `https://api.siliconflow.cn/v1` | 多种模型 |
| vLLM | `http://localhost:8000/v1` | 本地部署 |
| Ollama | `http://localhost:11434/v1` | 本地运行 |

---

## 6. 查看当前配置

```python
from akg_agents.core_v2.config import print_settings_info

# 打印当前配置信息
print_settings_info()
```

输出示例：
```
============================================================
🌍 环境变量 (最高优先级):
   ✓ AKG_AGENTS_BASE_URL=https://api.deepseek.com/beta/
   ✓ AKG_AGENTS_API_KEY=sk-235e0***dc3f
   ✓ AKG_AGENTS_MODEL_NAME=deepseek-chat
💡 优先级: 环境变量 > Local > Project > User > 默认值
============================================================
```
