# Database Module Design Documentation

## Overview
The Database module is an operator optimization solution management framework, responsible for storage, retrieval, verification and management of operator optimization solutions. It provides a structured approach to organizing and accessing operator implementations across different hardware architectures and DSLs, enabling rapid matching of operator solutions through feature extraction and vector retrieval.

## Core Features
- **Multi-strategy retrieval**: Supports various retrieval strategies including random sampling, similarity search, and rule-based filtering
- **Solution lifecycle management**: Provides complete operations for solution insertion, update, deletion and lookup
- **Feature extraction**: Automatic extraction of operator characteristics from implementation code
- **Hierarchical organization**: Structured storage by architecture, DSL, and unique identifiers

## Core Components

### Database (Base Class)
Core database class for operator solution management, providing fundamental functionality.

**Core Methods:**
- `extract_features()`: Extract operator features from implementation code
- `samples()`: Retrieve similar operator solutions using specified strategies
- `insert()`: Insert new operator implementation into database
- `delete()`: Delete operator implementation from database
- `get_case_content()`: Retrieve content from specific operator cases

### CoderDatabase (Code Generation Specialized)
Inherits from Database, specialized for code generation scenarios.

**Core Features:**
- Single instance pattern for efficient resource management
- Computation-focused vector store for code similarity
- Hierarchical search: computation logic → shape matching

### EvolveDatabase (Evolution Optimization Specialized)
Inherits from Database, specialized for evolutionary optimization scenarios.

**Core Features:**
- Multiple vector stores for different schedule aspects (base, pass, text)
- Fusion search with Reciprocal Rank Fusion (RRF)
- Maximum Marginal Relevance (MMR) reranking
- Optimality search for performance-based selection

## Storage Structure

The database uses a hierarchical file system structure:

```
database/
├── ascend910b4/
│   ├── triton/
│   │   ├── {md5_hash_1}/
│   │   │   ├── metadata.json
│   │   │   ├── triton.py
│   │   │   └── torch.py
│   │   └── {md5_hash_2}/
│   │       ├── metadata.json
│   │       ├── triton.py
│   │       └── mindspore.py
│   └── swft/
│       └── {md5_hash_3}/
│           ├── metadata.json
│           ├── swft.py
│           └── numpy.py
└── cuda/
    └── triton/
        └── {md5_hash_4}/
            ├── metadata.json
            ├── triton.py
            └── torch.py
```

## Retrieval Strategies

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

## Usage Guide

### Initialization Parameters
| Parameter | Type/Required | Default | Description |
|----------|--------------|---------|-------------|
| database_path | str (optional) | ../database | Root directory for operator solution storage |
| vector_stores | List[VectorStore] (optional) | [] | List of VectorStores for similarity search |
| config | dict (required) | None | Configuration dictionary containing agent_model_config |

### Configuration
```python
config = {
    "agent_model_config": {
        "feature_extraction": "deepseek_r1_default"  # Feature extraction model configuration
    },
    "database_config": {
        "enable_rag": True,          # Whether to enable RAG functionality
        "sample_num": 2,             # Default sampling number
        "embedding_device": "cpu"    # Embedding model device: cpu or cuda
    }
}
```

**Configuration Parameters:**
- `feature_extraction`: Specifies the model used for feature extraction
- `enable_rag`: Controls whether to enable vector retrieval functionality
- `sample_num`: Sets the default number of retrieved samples
- `embedding_device`: Specifies the device for running embedding models

### Core Methods

#### samples
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

#### insert
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

#### delete
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

#### extract_features
**Function**: Extract operator features from implementation code  
**Parameters**:
- `impl_code`: Operator implementation code
- `framework_code`: Framework adapter code
- `backend`: Compute backend
- `arch`: Hardware architecture
- `dsl`: Domain-specific language
- `profile`: Performance profile

**Returns**: Dictionary containing extracted features (op_name, op_type, input_specs, output_specs, computation, schedule, etc.)

## Usage Examples

### Basic Database Operations
```python
from ai_kernel_generator.database.database import Database, RetrievalStrategy

# Initialize database
database = Database(
    database_path="/path/to/database",
    vector_stores=[vector_store],
    config=config
)

# Insert new solution
await database.insert(
    impl_code=impl_code,
    framework_code=framework_code,
    backend="ascend",
    arch="ascend910b4",
    dsl="triton",
    framework="mindspore"
)

# Retrieve similar solutions
results = await database.samples(
    output_content=["impl_code"],
    strategy_modes=[RetrievalStrategy.NAIVETY],
    sample_num=5,
    impl_code=impl_code,
    framework_code=framework_code,
    backend="ascend",
    arch="ascend910b4",
    dsl="triton",
    framework="mindspore"
)

# Delete solution
await database.delete(
    impl_code=impl_code,
    backend="ascend",
    arch="ascend910b4",
    dsl="triton"
)
```

### Specialized Database Usage

#### CoderDatabase Example
```python
from ai_kernel_generator.database.coder_database import CoderDatabase

# Initialize code generation database
coder_db = CoderDatabase(config=config)

# Hierarchical search for code generation
results = await coder_db.samples(
    output_content=["impl_code"],
    sample_num=3,
    framework_code=framework_code,
    backend="ascend",
    arch="ascend910b4",
    dsl="triton"
)
```

#### EvolveDatabase Example
```python
from ai_kernel_generator.database.evolve_database import EvolveDatabase

# Initialize evolution optimization database
evolve_db = EvolveDatabase(config=config)

# Fusion search optimization
results = await evolve_db.samples(
    output_content=["impl_code"],
    sample_num=5,
    impl_code=impl_code,
    backend="ascend",
    arch="ascend910b4",
    dsl="triton"
)
```

## Performance Optimization

- **Single Instance Pattern**: Database classes use singleton pattern to avoid resource duplication
- **Lazy Loading**: Vector stores are built only when needed
- **Efficient Indexing**: FAISS-based vector indexing for fast similarity search
- **Memory Management**: Automatic cleanup of empty directories during deletion
- **Concurrent Safety**: Thread-safe singleton implementation with locking mechanisms