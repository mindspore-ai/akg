# RAG (Retrieval-Augmented Generation) Module Design Documentation

## Overview
The RAG module in AIKG is currently implemented through VectorStore classes, providing vector-based document retrieval capabilities. Through vector retrieval, it can quickly find similar relevant document content.

## Core Features
- **Vector Storage**: Efficient vector indexing based on FAISS
- **Embedding Models**: Support for HuggingFace embedding models
- **Automatic Document Generation**: Automatic generation of retrieval documents from operator metadata
- **Multiple Retrieval Methods**: Similarity search, Maximum Marginal Relevance search
- **Index Management**: Support for insert, delete, clear operations

## Core Components

### VectorStore (Abstract Base Class)
The foundation of the RAG system, providing vector storage and retrieval capabilities.

**Key Features:**
- Abstract base class for all vector storage implementations
- Singleton pattern for resource efficiency
- HuggingFace embedding model support (default: GanymedeNil/text2vec-large-chinese)
- FAISS-based vector indexing
- Automatic document generation from operator metadata

### Specialized Vector Stores

#### CoderVectorStore
Vector storage specialized for code generation scenarios.

**Core Features:**
- Focuses on computation-related features: ["op_name", "op_type", "input_specs", "output_specs", "computation"]
- Implements hierarchical search capabilities
- Supports code similarity matching

#### EvolveVectorStore
Vector storage specialized for evolutionary optimization scenarios.

**Core Features:**
- Handles schedule-related features: ["base", "pass", "text"]
- Supports multiple schedule aspects for diverse optimization
- Specialized handling of schedule block fields

## Embedding Model Support

### Model Loading Mechanism
VectorStore supports flexible embedding model loading strategies:

**Loading Priority:**
1. **Specified Model**: Prioritize loading the HuggingFace model specified in configuration
2. **Environment Variable**: If specified model fails, try loading local model from EMBEDDING_MODEL_PATH environment variable
3. **Graceful Degradation**: If all loading methods fail, automatically disable vector store functionality


### Device Configuration
- **CPU Mode**: Default configuration, suitable for development and testing
- **CUDA Mode**: Enable GPU acceleration via `embedding_device: "cuda"`

## Index Management

### Automatic Index Building
- Indexes are built automatically from metadata.json files
- Supports recursive directory traversal
- Graceful handling of empty databases
- Persistent storage using FAISS

## Usage Guide

### Document Storage Structure
Each document and its metadata files are stored in a separate folder:
```
{doc_path}/
├── metadata.json    # Metadata file
└── {document_file}       # Document content file
```

The `doc_path` parameter points to the folder path containing the document, relative to database_path.

### Index Operation Interfaces

#### insert
**Function**: Add new documents to vector storage  
**Parameters**:
- `doc_path`: Path of document to insert

**Operation Process**:
1. Load metadata.json from specified path
2. Generate document object
3. Remove existing identical documents (deduplication)
4. Add to vector storage and save index

#### delete
**Function**: Remove specified documents from vector storage  
**Parameters**:
- `doc_path`: Path of document to delete

**Deletion Process**:
1. Iterate through existing document IDs
2. Match file paths
3. Delete matching documents
4. Save updated index

#### clear
**Function**: Clear all documents from vector storage  
**Operation Process**:
1. Delete all document IDs
2. Save empty index file

### Retrieval Interfaces

#### similarity_search
**Function**: Execute semantic search and return matching documents  
**Parameters**:
- `query`: Query string
- `k`: Number of documents to return (default: 5)
- `fetch_k`: Number of candidate documents (default: 20, for improving recall)

**Returns**: List of matching Document objects

#### max_marginal_relevance_search
**Function**: Execute Maximum Marginal Relevance search, balancing similarity and diversity  
**Parameters**:
- `query`: Query string
- `k`: Number of documents to return (default: 5)

**Returns**: List of Document objects reordered by MMR

**Features**:
- `lambda_mult=0.2`: Extreme diversity setting
- `fetch_k=max(20, 5 * k)`: Dynamic candidate count

#### similarity_search_with_score
**Function**: Execute semantic search and return matching documents with similarity scores  
**Parameters**:
- `query`: Query string
- `k`: Number of documents to return (default: 5)
- `fetch_k`: Number of candidate documents (default: 20)

**Returns**: List of (Document, score) tuples

## Document Generation

### Automatic Document Creation
VectorStore automatically generates retrieval documents from operator metadata. Subclasses need to implement the `gen_document` method to define specific document generation logic.

### Specialized Document Generation

#### CoderVectorStore Document Generation
- Extracts computation-related features from metadata
- Builds documents containing operator type, file path, and other information
- Supports feature invariant filtering

#### EvolveVectorStore Document Generation
- Specialized handling of schedule block fields
- Expands scheduling information into key-value pair format
- Supports feature extraction for multiple schedule aspects

## Usage Examples

### Basic Vector Search
```python
from ai_kernel_generator.database.coder_vector_store import CoderVectorStore

# Initialize vector store
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
vector_store.insert("ascend910b4/triton/md5_hash_123")

# Delete document
vector_store.delete("ascend910b4/triton/md5_hash_123")

# Clear all documents
vector_store.clear()
```

## Performance Optimization

### Singleton Pattern
- Prevents resource duplication
- Thread-safe implementation with locking
- Efficient memory usage

### FAISS Optimization
- Fast similarity search
- Configurable fetch_k for improved recall
- Configurable lambda_mult for MMR

### Error Handling
- Graceful degradation when embedding models fail
- Automatic fallback to local models
- Comprehensive logging

## Future Extensions

### Potential Extensions
- **API Documentation Integration**: Support for AscendC API manual
- **Multi-Source RAG**: Integration with external knowledge sources
- **Custom Document Adapters**: Support for PDF, markdown, and other formats
- **Advanced Fusion Strategies**: More sophisticated result combination methods
- **Query Expansion**: Automatic query enhancement for better retrieval