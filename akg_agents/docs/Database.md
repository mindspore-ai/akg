# Database Module Design Document

## Overview
The Database module is the management framework:

- **Generic Base Class** (`database/`): Provides the `Database` abstract base class and `VectorStore` abstract base class, defining the generic framework for document storage, retrieval, and vector indexing
- **Operator-Specific Implementation** (`op/database/`): `CoderDatabase` and `CoderVectorStore`, inheriting from the generic base classes, implementing operator-specific logic

## Architecture

```
database/                           # Generic base classes
├── database.py                     # Database abstract base class
└── vector_store.py                 # VectorStore abstract base class

op/database/                        # Operator-specific implementations
├── coder_database.py               # CoderDatabase (operator code database)
└── coder_vector_store.py           # CoderVectorStore (operator vector store)
```

## Generic Base Classes

### Database (Abstract Base Class)

Provides a generic framework for document management. Subclasses must implement `_do_insert` and `_do_delete` methods.

**Core Methods:**
- `_insert_with_vectors(doc_id, content, mode)`: Generic insert method (with vector store sync), supports `skip` and `overwrite` modes
- `_delete_with_vectors(doc_id)`: Generic delete method (with vector store sync and empty directory cleanup)
- `clear()`: Clear the database and all vector stores

**Initialization Parameters:**

| Parameter | Type/Required | Description |
|-----------|--------------|-------------|
| database_path | str (Optional) | Database storage root directory |
| vector_stores | List[VectorStore] (Optional) | List of vector stores |
| config | dict (Required) | Configuration dictionary, must contain `agent_model_config` |

### VectorStore (Abstract Base Class)

Provides a generic framework for vector storage and retrieval. Subclasses must implement the `gen_document` method.

**Core Features:**
- FAISS-based vector indexing
- Supports OpenAI-compatible embedding models (via `core_v2`'s `create_embedding_model`)
- Automatic index building from metadata.json files
- Supports similarity search and MMR search

**Embedding Model Loading:**
VectorStore now loads embedding models via `core_v2.llm.create_embedding_model()`, supporting OpenAI-compatible Embedding APIs (see [RAG Documentation](./RAG.md)). Loading priority:
1. OpenAI-compatible Embedding model from configuration
2. Local HuggingFace model specified
3. Graceful degradation to disable vector store functionality

**Retrieval Interfaces:**

| Method | Description |
|--------|-------------|
| `similarity_search(query, k, fetch_k)` | Semantic similarity search |
| `max_marginal_relevance_search(query, k)` | MMR search (balancing similarity and diversity) |
| `similarity_search_with_score(query, k, fetch_k)` | Semantic search with scores |
| `insert(doc_path)` | Add document to vector store |
| `delete(doc_path)` | Remove document from vector store |
| `clear()` | Clear vector store |

## Operator-Specific Implementation

### CoderDatabase

Inherits from `Database`, specialized for operator code storage and retrieval.

**Core Features:**
- Singleton pattern to avoid resource duplication
- Hierarchical search based on computation logic (computation similarity → shape matching)
- Feature extraction support (extracting operator features from code via LLM)
- Auto-update from benchmark directory

**Retrieval Strategies:**

| Strategy | Description | Use Case |
|----------|-------------|----------|
| RANDOMICITY | Random sampling | Testing, baseline comparison |
| HIERARCHY | Hierarchical retrieval (computation → shape) | Code generation scenarios (default) |
| NAIVETY | Direct vector similarity | Simple matching |
| MMR | Maximum Marginal Relevance | Diversity requirements |

**Core Methods:**

```python
# Retrieve similar operator solutions
results = await coder_db.samples(
    output_content=["impl_code"],
    sample_num=3,
    framework_code=framework_code,
    backend="ascend",
    arch="ascend910b4",
    dsl="triton_ascend"
)

# Insert new solution
await coder_db.insert(
    impl_code=impl_code,
    framework_code=framework_code,
    backend="ascend",
    arch="ascend910b4",
    dsl="triton_ascend",
    framework="torch"
)

# Delete solution
coder_db.delete(
    impl_code=impl_code,
    backend="ascend",
    arch="ascend910b4",
    dsl="triton_ascend"
)

# Auto-update from benchmark directory
await coder_db.auto_update(
    dsl="triton_ascend",
    framework="torch",
    backend="ascend",
    arch="ascend910b4",
    ref_type="docs"  # or "impl"
)
```

### CoderVectorStore

Inherits from `VectorStore`, focused on operator computation feature vector indexing.

**Core Features:**
- Focused on computation features: `["op_name", "computation"]`
- Singleton pattern
- Feature invariants filtering support

## Storage Structure

```
database/
├── ascend910b4/
│   ├── triton/
│   │   ├── {md5_hash_1}/
│   │   │   ├── metadata.json     # Operator features
│   │   │   ├── triton.py          # DSL implementation code
│   │   │   ├── torch.py           # Framework adapter code
│   │   │   └── doc.md             # Optimization document (optional)
│   │   └── {md5_hash_2}/
│   │       └── ...
│   └── swft/
│       └── ...
└── a100/
    └── triton/
        └── ...
```

## Configuration

```python
config = {
    "agent_model_config": {
        "feature_extraction": "standard"  # Feature extraction model
    },
    "database_config": {
        "enable_rag": True,          # Whether to enable RAG
        "sample_num": 2,             # Default sampling count
    }
}
```

## Usage Examples

```python
from akg_agents.op.database.coder_database import CoderDatabase

# Initialize (singleton pattern)
coder_db = CoderDatabase(config=config)

# Auto-update database from benchmark
await coder_db.auto_update(
    dsl="triton_ascend",
    framework="torch",
    backend="ascend",
    arch="ascend910b4"
)

# Retrieve similar solutions
results = await coder_db.samples(
    output_content=["impl_code", "framework_code"],
    sample_num=3,
    impl_code=my_impl_code,
    framework_code=my_framework_code,
    backend="ascend",
    arch="ascend910b4",
    dsl="triton_ascend",
    framework="torch"
)
```

## Related Documentation
- [RAG Module Documentation](./RAG.md)
- [RAG Usage Guide](./RAG_Usage.md)
