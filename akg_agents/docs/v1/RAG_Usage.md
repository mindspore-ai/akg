# RAG Usage Guide

RAG (Retrieval-Augmented Generation) enables vector-based retrieval from historical operator codebases to find similar implementations, providing references for operator generation. This document describes how to configure and use the RAG feature.

## 1. Configure Embedding Model

The RAG feature requires an Embedding model to generate vector representations. The system supports two modes: **Remote API** and **Local Model**.

### Method A: Use Remote Embedding API (Recommended)

Specify an OpenAI-compatible Embedding API via environment variables or configuration file.

**Environment Variables:**
```bash
# Using SiliconFlow
export AKG_AGENTS_EMBEDDING_BASE_URL="https://api.siliconflow.cn/v1"
export AKG_AGENTS_EMBEDDING_MODEL_NAME="BAAI/bge-large-zh-v1.5"
export AKG_AGENTS_EMBEDDING_API_KEY="sk-xxx"
```

**Configuration File:**

Add to `.akg/settings.json` (project-level) or `~/.akg/settings.json` (user-level):
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

> **Configuration Priority**: Environment variables > `.akg/settings.local.json` > `.akg/settings.json` > `~/.akg/settings.json`

### Method B: Use Local HuggingFace Model (Offline Environments)

Run the following command to download the model:
```bash
bash download.sh --with_local_model
# For slow downloads, try a mirror
HF_ENDPOINT=https://hf-mirror.com bash download.sh --with_local_model
```

This downloads the `text2vec-large-chinese` model to `~/.akg_agents/text2vec-large-chinese`.

**Notes:**
- If a remote API is configured, it takes priority; the local model serves as a fallback
- Model download requires network connectivity; first download may take some time
- The script skips download if the model directory already exists

## 2. Install Dependencies

Install RAG-related dependencies in the `akg_agents` directory:

```bash
pip install -r requirements_rag.txt
```

## 3. AKG CLI RAG Usage

Enable RAG when generating operators by adding the `--rag` flag:

```bash
# Ascend 910B2
akg_cli op --framework torch --backend ascend --arch ascend910b2 --dsl triton_ascend --worker-url 127.0.0.1:9001 --rag

# CUDA A100
akg_cli op --framework torch --backend cuda --arch a100 --dsl triton_cuda --worker-url 127.0.0.1:9001 --rag
```

**Notes:**
- `--rag` enables RAG, `--no-rag` disables it (default: disabled)
- RAG automatically retrieves similar historical implementations as references during operator generation
- Other parameters work the same as regular operator generation; see [AKG CLI documentation](./AKG_CLI.md) for details

**Complete example:**

```bash
# Ascend 910B2 with RAG and intent
akg_cli op --framework torch --backend ascend --arch ascend910b2 --dsl triton_ascend --worker-url 127.0.0.1:9001 --rag --intent "implement fused softmax kernel for input [batch, head, seq, dim]"

# CUDA A100 with RAG and intent
akg_cli op --framework torch --backend cuda --arch a100 --dsl triton_cuda --worker-url 127.0.0.1:9001 --rag --intent "implement fused softmax kernel for input [batch, head, seq, dim]"
```

## 4. Notes

- RAG requires the vector database to be indexed; indexing may be needed on first use
- Retrieval effectiveness depends on the quality and quantity of historical implementations in the vector database
- If no similar implementations are found, the system loads default examples and proceeds with normal operator generation
- Remote API mode does not require local model download but needs network connectivity
- Local model mode does not require network connectivity but needs the model file to be downloaded first
