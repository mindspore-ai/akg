# RAG (Retrieval-Augmented Generation) Module Design Document

## Overview
The RAG module in AKG Agents implements vector retrieval-augmented generation capabilities through the `VectorStore` abstract base class and `OpenAICompatibleEmbeddings`. The module has been refactored to support multiple embedding model backends, including OpenAI-compatible remote APIs (OpenAI, DeepSeek, SiliconFlow, vLLM local deployment) and local HuggingFace models, integrated with a unified configuration management system.

## Core Features
- **Vector Storage**: Efficient vector indexing based on FAISS
- **Multi-Embedding Model Support**: Dual-mode support for OpenAI-compatible APIs (remote) and HuggingFace (local)
- **Unified Configuration Management**: Unified Embedding configuration via `settings.json` or environment variables
- **Automatic Document Generation**: Automatic generation of retrieval documents from operator metadata
- **Multiple Retrieval Methods**: Similarity Search, Maximum Marginal Relevance (MMR) Search
- **Index Management**: Support for insert, delete, and clear operations

## Architecture Overview

```
core_v2/llm/
├── factory.py                          # create_embedding_model() factory function
├── providers/
│   └── embedding_provider.py           # OpenAICompatibleEmbeddings implementation
└── ...

database/
├── vector_store.py                     # VectorStore abstract base class
└── ...

op/database/
├── coder_vector_store.py               # CoderVectorStore for operator code
└── ...
```

## Core Components

### OpenAICompatibleEmbeddings
Located in `core_v2/llm/providers/embedding_provider.py`, implements the LangChain `Embeddings` interface, supporting any OpenAI-compatible Embedding API.

**Supported Backends:**
- OpenAI Embeddings
- DeepSeek Embeddings
- SiliconFlow Platform
- vLLM Local Deployment
- Other OpenAI-compatible APIs

**Initialization Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| api_url | str | Full URL for Embedding API (e.g., `http://localhost:8001/v1/embeddings`) |
| model_name | str | Model name |
| api_key | str | API key (optional, required for remote APIs) |
| verify_ssl | bool | Whether to verify SSL certificates (default: False) |
| timeout | int | Timeout in seconds (default: 60) |

**Core Methods:**
- `embed_documents(texts)` → `List[List[float]]`: Generate embedding vectors for a list of documents
- `embed_query(text)` → `List[float]`: Generate embedding vector for a single query

### create_embedding_model() Factory Function
Located in `core_v2/llm/factory.py`, automatically creates an Embedding model instance based on configuration.

**Configuration Priority (highest to lowest):**
1. Function parameters (direct specification)
2. Environment variables `AKG_AGENTS_EMBEDDING_*`
3. `embedding` configuration in `settings.json`

**Usage Example:**
```python
from akg_agents.core_v2.llm import create_embedding_model

# Method 1: Use configuration (environment variables or settings.json)
embedding = create_embedding_model()

# Method 2: Directly specify parameters
embedding = create_embedding_model(
    base_url="https://api.siliconflow.cn/v1",
    api_key="sk-xxx",
    model_name="BAAI/bge-large-zh-v1.5"
)
```

### VectorStore (Abstract Base Class)
Located in `database/vector_store.py`, the foundation of the RAG system, providing vector storage and retrieval capabilities.

**Embedding Model Loading Strategy (by priority):**
1. **Remote API**: First attempts to use OpenAI-compatible API via `create_embedding_model()` (auto-checks environment variables and configuration)
2. **Local HuggingFace Model**: If remote API is unavailable, loads local `~/.akg_agents/text2vec-large-chinese` model
3. **Error Prompt**: If all loading methods fail, raises an exception with configuration guidance

**Key Features:**
- Abstract base class for all vector store implementations
- FAISS-based vector indexing
- Automatic document generation from operator metadata (subclasses implement `gen_document`)
- Supports recursive directory traversal for index building

### Specialized Vector Stores

#### CoderVectorStore
Vector storage specialized for code generation scenarios.

**Core Features:**
- Focused on computation-related features: `["op_name", "op_type", "input_specs", "output_specs", "computation"]`
- Hierarchical search capabilities
- Code similarity matching support

## Embedding Model Configuration

### Method 1: Environment Variables (Recommended for Quick Setup)
```bash
# New prefix (recommended)
export AKG_AGENTS_EMBEDDING_BASE_URL="https://api.siliconflow.cn/v1"
export AKG_AGENTS_EMBEDDING_MODEL_NAME="BAAI/bge-large-zh-v1.5"
export AKG_AGENTS_EMBEDDING_API_KEY="sk-xxx"
export AKG_AGENTS_EMBEDDING_TIMEOUT="60"

# Legacy prefix (compatible)
export AIKG_EMBEDDING_BASE_URL="https://api.siliconflow.cn/v1"
```

### Method 2: settings.json Configuration File
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

Configuration file location priority (highest to lowest):
1. `.akg/settings.local.json` (personal, gitignored)
2. `.akg/settings.json` (project-level, team shared)
3. `~/.akg/settings.json` (user-level, cross-project)

### Method 3: Local HuggingFace Model (Offline Environments)
```bash
# Download local model
bash download.sh --with_local_model
```
The model is downloaded to `~/.akg_agents/text2vec-large-chinese` and automatically used as a fallback when remote API is unavailable.

## Index Management

### Automatic Index Building
- Indexes are built automatically from `metadata.json` files
- Supports recursive directory traversal
- Graceful handling of empty databases
- Persistent storage using FAISS

### Document Storage Structure
```
{database_path}/
├── {doc_path_1}/
│   ├── metadata.json        # Metadata file
│   └── {document_file}      # Document content file
├── {doc_path_2}/
│   └── ...
└── {index_name}/
    └── index.faiss          # FAISS index file
```

## Retrieval Interfaces

### similarity_search
**Function**: Execute semantic search and return matching documents
**Parameters**:
- `query`: Query string
- `k`: Number of documents to return (default: 5)
- `fetch_k`: Number of candidate documents (default: 20)

**Returns**: List of matching Document objects

### max_marginal_relevance_search
**Function**: Execute Maximum Marginal Relevance search, balancing similarity and diversity
**Parameters**:
- `query`: Query string
- `k`: Number of documents to return (default: 5)

**Returns**: List of Document objects reordered by MMR

**Features**:
- `lambda_mult=0.2`: Extreme diversity setting
- `fetch_k=max(20, 5 * k)`: Dynamic candidate count

### similarity_search_with_score
**Function**: Execute semantic search and return matching documents with similarity scores
**Parameters**:
- `query`: Query string
- `k`: Number of documents to return (default: 5)
- `fetch_k`: Number of candidate documents (default: 20)

**Returns**: List of `(Document, score)` tuples

## Usage Examples

### Basic Vector Search
```python
from akg_agents.op.database.coder_vector_store import CoderVectorStore

# Initialize vector store (automatically selects available Embedding model)
vector_store = CoderVectorStore(
    database_path="/path/to/database",
    config=config
)

# Execute similarity search
docs = vector_store.similarity_search(query, k=5)

# Execute MMR search for diversity
docs = vector_store.max_marginal_relevance_search(query, k=5)
```

### Index Management Operations
```python
# Insert new document
vector_store.insert("path/to/your/document")

# Delete document
vector_store.delete("path/to/your/document")

# Clear all documents
vector_store.clear()
```

## Relationship with Skill System
In the new architecture, dynamic knowledge injection is primarily handled by the **Skill System** (see [Skill System Documentation](./SkillSystem.md)). The RAG module is mainly used for vector retrieval at the `Database` layer (e.g., sample retrieval in `CoderDatabase`), while the Skill System handles higher-level knowledge selection and injection. The two are complementary:
- **RAG**: Suitable for large-scale vector retrieval scenarios (e.g., retrieving similar implementations from thousands of historical samples)
- **Skill System**: Suitable for dynamic selection and injection of structured knowledge (e.g., DSL knowledge, hardware optimization strategies)
