# Database Module Design Documentation

## Overview
The Database module is an operator optimization solution management framework, responsible for storage, retrieval, verification and management of operator optimization solutions. It enables rapid matching of similar solutions through feature extraction and vector search.

## Core Features
- **Multi-strategy similarity search**: Supports COSINE/EUCLIDEAN_DISTANCE and other similarity calculation strategies
- **Solution lifecycle management**: Provides complete operations for solution insertion, update, deletion and lookup
- **Two-stage verification**: Initial retrieval + secondary verification mechanism ensures result accuracy

## Initialization Parameters
| Parameter | Type/Required | Default | Description |
|----------|--------------|---------|-------------|
| config_path | str (optional) | database_config.yaml | Configuration file path. embedding_model can be model name or local model path; distance_strategy and verify_distance_strategy support EUCLIDEAN_DISTANCE, MAX_INNER_PRODUCT, DOT_PRODUCT, JACCARD and COSINE |
| database_path | str (optional) | ../database | Root directory for operator solution storage |

### database_config.yaml example:
```yaml
# Embedding model configuration
embedding_model: "GanymedeNil/text2vec-large-chinese"  # Model name or local path e.g. "xxx/thirdparty/text2vec-large-chinese"

# Vector database configuration
distance_strategy: "COSINE"
verify_distance_strategy: "EUCLIDEAN_DISTANCE"

# Feature extraction preset
agent_model_config:
  feature_extraction: deepseek_r1_default
```

## Core Methods
### sample
**Function**: Retrieve similar operator optimization solutions  
**Parameters**:
- `impl_code`: Operator implementation code
- `backend`: Compute backend (ascend/cuda/cpu)
- `arch`: Hardware architecture (e.g. ascend910b4)
- `impl_type`: Implementation type (triton/swft)

**Returns**:
- Recall rate, list of solutions with similarity scores

### insert
**Feature generation rules**:
1. Generate feature invariants using `get_md5_hash()`
2. Directory structure: `{database_path}/operators/{arch}/{impl_type}/{md5_hash}/`
3. Metadata saved as metadata.json

### delete
**Deletion logic**:
1. Locate directory using md5_hash generated from code content
2. Cascade delete empty parent directories
3. Synchronously update vector storage index

### verify
**Verification process**:
1. Re-retrieve using alternative distance strategy
2. Calculate recall rate of both retrieval results
3. Return verification pass rate

## Usage Example
```python
# Initialize RAG system
rag = DatabaseRAG(config_path="custom_rag.yaml")

# Insert new solution
rag.insert(
    impl_code=matmul_impl,
    framework_code=mindspore_adapter,
    backend="ascend",
    arch="ascend910b4",
    impl_type="triton",
    framework="mindspore"
)

# Retrieve similar solutions
recall, results = rag.sample(
    impl_code=new_matmul_impl,
    backend="ascend",
    arch="ascend910b4",
    impl_type="triton"
)
```