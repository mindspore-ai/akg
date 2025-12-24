# RAG Usage Guide

RAG (Retrieval-Augmented Generation) enables vector-based retrieval from historical operator codebases to find similar implementations, providing references for operator generation. This document describes how to install and use the RAG feature.

## 1. Install Dependencies

Install RAG-related dependencies by running the following command in the `aikg` directory:

```bash
pip install -r requirements_rag.txt
```

## 2. Download Embedding Model

RAG requires an embedding model to generate vector representations. Download the model by running:

```bash
bash download.sh --with_local_model
```

This command downloads the `text2vec-large-chinese` model to `~/.aikg/text2vec-large-chinese`.

**Notes:**
- Model download requires network connectivity and may take some time on first run
- The script skips download if the model directory already exists
- To re-download, delete the `~/.aikg/text2vec-large-chinese` directory first

## 3. AKG CLI Use RAG Feature

Enable RAG when generating operators by adding the `--rag` flag:

```bash
akg_cli op \
  --framework torch \
  --backend cuda \
  --arch a100 \
  --dsl triton_cuda \
  --worker-url 127.0.0.1:9001 \
  --rag
```

**Notes:**
- `--rag` enables RAG, `--no-rag` disables it (default: disabled)
- RAG automatically retrieves similar historical implementations as references during operator generation
- Other parameters work the same as regular operator generation; see [AKG CLI documentation](./AKG_CLI.md) for details

**Complete example:**

```bash
akg_cli op \
  --framework torch \
  --backend cuda \
  --arch a100 \
  --dsl triton_cuda \
  --worker-url 127.0.0.1:9001 \
  --rag \
  --intent "implement fused softmax kernel for input [batch, head, seq, dim]"
```

## 4. Notes

- RAG requires the vector database to be indexed; indexing may be needed on first use
- Retrieval effectiveness depends on the quality and quantity of historical implementations in the vector database
- If no similar implementations are found, the system loads default examples and proceeds with normal operator generation without impact

