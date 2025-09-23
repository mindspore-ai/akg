# Database Module Design Documentation

## Overview
The Database module is an operator optimization solution management framework, responsible for storage, retrieval, verification and management of operator optimization solutions. It provides a structured approach to organizing and accessing operator implementations across different hardware architectures and DSLs, enabling rapid matching of operator solutions through feature extraction and vector retrieval.

## Core Features
- **Multi-strategy retrieval**: Supports various retrieval strategies including random sampling, similarity search, and rule-based filtering
- **Solution lifecycle management**: Provides complete operations for solution insertion, update, deletion and lookup
- **Feature extraction**: Automatic extraction of operator characteristics from implementation code
- **Hierarchical organization**: Structured storage by architecture, DSL, and unique identifiers

### Retrieval Strategies
The Database module supports multiple retrieval strategies for finding operator optimization solutions:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **RANDOMICITY** | Random sampling of operators from the database | Testing, baseline comparison, unbiased sampling |
| **NAIVETY** | Direct similarity search based on feature vectors | Simple similarity matching, straightforward retrieval |
| **MMR** | Maximum Marginal Relevance balancing similarity and diversity | Avoiding redundant results, ensuring diverse solutions |
| **OPTIMALITY** | Performance-optimized retrieval strategy | High-performance scenarios requiring fastest retrieval |
| **RULE** | Rule-based search with custom criteria | Custom filtering logic, domain-specific requirements |
| **HIERARCHY** | Hierarchical search across different abstraction levels | Multi-level analysis, progressive refinement |
| **FUSION** | Multi-strategy fusion combining different approaches | Comprehensive search, leveraging multiple methods |

## Initialization Parameters
| Parameter | Type/Required | Default | Description |
|----------|--------------|---------|-------------|
| database_path | str (optional) | ../database | Root directory for operator solution storage |
| vector_stores | List[VectorStore] (optional) | [] | List of VectorStores for similarity search |
| config | dict (required) | None | Configuration dictionary containing agent_model_config |

### Configuration Structure:
```python
config = {
    "agent_model_config": {
        "feature_extraction": "deepseek_r1_default"
    },
    "database_config": {
        "embedding_device": "cpu"  # or "cuda" for GPU
    }
}
```

## Core Methods

### samples
**Function**: Retrieve similar operator optimization solutions using specified strategies  
**Parameters**:
- `output_content`: List of content types to retrieve (e.g., ["impl_code", "framework_code"])
- `strategy_modes`: List of retrieval strategies to use
- `sample_num`: Number of samples to retrieve (default: 5)
- `impl_code`: Operator implementation code
- `framework_code`: Framework adapter code
- `backend`: Compute backend (ascend/cuda/cpu)
- `arch`: Hardware architecture (e.g. ascend910b4)
- `dsl`: Domain-specific language (triton/swft)
- `framework`: Framework name (mindspore/pytorch)

**Returns**: List of retrieved operator solutions

### insert
**Function**: Insert new operator implementation into database  
**Parameters**:
- `impl_code`: Operator implementation code
- `framework_code`: Framework adapter code
- `backend`: Compute backend (ascend/cuda/cpu)
- `arch`: Hardware architecture
- `dsl`: Domain-specific language
- `framework`: Framework name
- `profile`: Performance profile (default: inf)

**Storage Structure**:
1. Generate md5_hash using `get_md5_hash()`
2. Directory structure: `{database_path}/{arch}/{dsl}/{md5_hash}/`
3. Save metadata as metadata.json
4. Save implementation code as {dsl}.py
5. Save framework code as {framework}.py

### delete
**Function**: Delete operator implementation from database  
**Parameters**:
- `impl_code`: Operator implementation code
- `backend`: Compute backend
- `arch`: Hardware architecture
- `dsl`: Domain-specific language

**Deletion Process**:
1. Generate md5_hash to locate directory
2. Remove operator directory and files
3. Cascade delete empty parent directories
4. Update VectorStore indices

### extract_features
**Function**: Extract operator features from implementation code  
**Parameters**:
- `impl_code`: Operator implementation code
- `framework_code`: Framework adapter code
- `backend`: Compute backend
- `arch`: Hardware architecture
- `dsl`: Domain-specific language
- `profile`: Performance profile

**Returns**: Dictionary containing extracted features (op_name, op_type, input_specs, output_specs, computation, schedule, etc.)

## Usage Example
```python
from ai_kernel_generator.database.database import Database, RetrievalStrategy

database = Database(
    database_path="/path/to/database",
    vector_stores=[vector_store],
    config=config
)

# Insert new solution
await database.insert(
    impl_code=matmul_impl,
    framework_code=mindspore_adapter,
    backend="ascend",
    arch="ascend910b4",
    dsl="triton",
    framework="mindspore"
)

# Retrieve similar solutions
results = await database.samples(
    output_content=["impl_code", "framework_code"],
    strategy_modes=[RetrievalStrategy.NAIVETY],
    sample_num=5,
    impl_code=new_matmul_impl,
    framework_code=new_framework_code,
    backend="ascend",
    arch="ascend910b4",
    dsl="triton",
    framework="mindspore"
)

```