# RAG (检索增强生成) 模块设计文档

## 概述
AKG Agents 中的 RAG 模块通过 `VectorStore` 抽象基类和 `OpenAICompatibleEmbeddings` 实现向量检索增强生成能力。模块已重构为支持多种嵌入模型后端，包括 OpenAI 兼容的远程 API（如 OpenAI、DeepSeek、硅流平台、vLLM 本地部署）和本地 HuggingFace 模型，并集成统一的配置管理系统。

## 核心特性
- **向量存储**：基于 FAISS 的高效向量索引
- **多嵌入模型支持**：支持 OpenAI 兼容 API（远程）和 HuggingFace（本地）双模式
- **统一配置管理**：通过 `settings.json` 或环境变量统一管理 Embedding 配置
- **自动文档生成**：从算子元数据自动生成检索文档
- **多种检索方式**：相似度搜索（Similarity Search）、最大边际相关性搜索（MMR）
- **索引管理**：支持插入、删除、清空等操作

## 架构概览

```
core_v2/llm/
├── factory.py                          # create_embedding_model() 工厂函数
├── providers/
│   └── embedding_provider.py           # OpenAICompatibleEmbeddings 实现
└── ...

database/
├── vector_store.py                     # VectorStore 抽象基类
└── ...

op/database/
├── coder_vector_store.py               # CoderVectorStore 算子代码专用
└── ...
```

## 核心组件

### OpenAICompatibleEmbeddings
位于 `core_v2/llm/providers/embedding_provider.py`，实现 LangChain `Embeddings` 接口，支持任何 OpenAI 兼容格式的 Embedding API。

**支持的后端：**
- OpenAI Embeddings
- DeepSeek Embeddings
- 硅流平台（SiliconFlow）
- vLLM 本地部署
- 其他 OpenAI 兼容 API

**初始化参数：**
| 参数名 | 类型 | 说明 |
|--------|------|------|
| api_url | str | Embedding API 的完整 URL（如 `http://localhost:8001/v1/embeddings`）|
| model_name | str | 模型名称 |
| api_key | str | API 密钥（可选，远程 API 需要）|
| verify_ssl | bool | 是否验证 SSL 证书（默认 False）|
| timeout | int | 超时时间（秒，默认 60）|

**核心方法：**
- `embed_documents(texts)` → `List[List[float]]`：为文档列表生成嵌入向量
- `embed_query(text)` → `List[float]`：为单条查询生成嵌入向量

### create_embedding_model() 工厂函数
位于 `core_v2/llm/factory.py`，根据配置自动创建 Embedding 模型实例。

**配置优先级（从高到低）：**
1. 函数参数（直接指定）
2. 环境变量 `AKG_AGENTS_EMBEDDING_*`
3. `settings.json` 中的 `embedding` 配置

**使用示例：**
```python
from akg_agents.core_v2.llm import create_embedding_model

# 方式 1：使用配置（环境变量或 settings.json）
embedding = create_embedding_model()

# 方式 2：直接指定参数
embedding = create_embedding_model(
    base_url="https://api.siliconflow.cn/v1",
    api_key="sk-xxx",
    model_name="BAAI/bge-large-zh-v1.5"
)
```

### VectorStore (抽象基类)
位于 `database/vector_store.py`，RAG 系统的基础，提供向量存储和检索能力。

**嵌入模型加载策略（按优先级）：**
1. **远程 API**：优先尝试通过 `create_embedding_model()` 使用 OpenAI 兼容 API（自动检查环境变量和配置）
2. **本地 HuggingFace 模型**：如远程 API 不可用，加载本地 `~/.akg_agents/text2vec-large-chinese` 模型
3. **错误提示**：如所有加载方式都失败，抛出异常并提示配置方法

**关键特性：**
- 所有向量存储实现的抽象基类
- 基于 FAISS 的向量索引
- 从算子元数据自动生成文档（子类实现 `gen_document`）
- 支持递归目录遍历构建索引

### 专用向量存储

#### CoderVectorStore
专门用于代码生成场景的向量存储。

**核心特性：**
- 专注于计算相关特征：`["op_name", "op_type", "input_specs", "output_specs", "computation"]`
- 实现层次搜索能力
- 支持代码相似性匹配

## 嵌入模型配置

### 方式 1：环境变量（推荐快速配置）
```bash
# 新版前缀（推荐）
export AKG_AGENTS_EMBEDDING_BASE_URL="https://api.siliconflow.cn/v1"
export AKG_AGENTS_EMBEDDING_MODEL_NAME="BAAI/bge-large-zh-v1.5"
export AKG_AGENTS_EMBEDDING_API_KEY="sk-xxx"
export AKG_AGENTS_EMBEDDING_TIMEOUT="60"

# 旧版前缀（兼容）
export AIKG_EMBEDDING_BASE_URL="https://api.siliconflow.cn/v1"
```

### 方式 2：settings.json 配置文件
```json
{
  "embedding": {
    "base_url": "https://api.siliconflow.cn/v1",
    "api_key": "sk-xxx",
    "model_name": "BAAI/bge-large-zh-v1.5",
    "timeout": 60
  }
}
```

配置文件位置优先级（从高到低）：
1. `.akg/settings.local.json`（仅本人，gitignored）
2. `.akg/settings.json`（项目级，团队共享）
3. `~/.akg/settings.json`（用户级，跨项目）

### 方式 3：本地 HuggingFace 模型（离线环境）
```bash
# 下载本地模型
bash download.sh --with_local_model
```
模型会下载到 `~/.akg_agents/text2vec-large-chinese`，在远程 API 不可用时自动作为降级方案使用。

## 索引管理

### 自动索引构建
- 索引从 `metadata.json` 文件自动构建
- 支持递归目录遍历
- 优雅处理空数据库
- 使用 FAISS 持久化存储

### 文档存储结构
```
{database_path}/
├── {doc_path_1}/
│   ├── metadata.json        # 元数据文件
│   └── {document_file}      # 文档内容文件
├── {doc_path_2}/
│   └── ...
└── {index_name}/
    └── index.faiss          # FAISS 索引文件
```

## 检索接口

### similarity_search
**功能**：执行语义搜索并返回匹配的文档  
**参数**：
- `query`: 查询字符串
- `k`: 返回的文档数量（默认：5）
- `fetch_k`: 候选文档数量（默认：20）

**返回**：匹配的 Document 对象列表

### max_marginal_relevance_search
**功能**：执行最大边际相关搜索，平衡相似度和多样性  
**参数**：
- `query`: 查询字符串
- `k`: 返回的文档数量（默认：5）

**返回**：经过 MMR 重排序的 Document 对象列表

**特点**：
- `lambda_mult=0.2`：极致多样性设置
- `fetch_k=max(20, 5 * k)`：动态候选数量

### similarity_search_with_score
**功能**：执行语义搜索并返回匹配的文档及其相似度得分  
**参数**：
- `query`: 查询字符串
- `k`: 返回的文档数量（默认：5）
- `fetch_k`: 候选文档数量（默认：20）

**返回**：`(Document, score)` 元组列表

## 使用示例

### 基础向量搜索
```python
from akg_agents.op.database.coder_vector_store import CoderVectorStore

# 初始化向量存储（自动选择可用的 Embedding 模型）
vector_store = CoderVectorStore(
    database_path="/path/to/database",
    config=config
)

# 执行相似度搜索
docs = vector_store.similarity_search(query, k=5)

# 执行 MMR 搜索以获得多样性
docs = vector_store.max_marginal_relevance_search(query, k=5)
```

### 索引管理操作
```python
# 插入新文档
vector_store.insert("path/to/your/document")

# 删除文档
vector_store.delete("path/to/your/document")

# 清空所有文档
vector_store.clear()
```
