# Local Model Deployment Guide

This document describes how to integrate locally deployed LLMs with AKG Agents.

## Configuring Local Models

If you already have your own deployed model service, simply configure the local service `base_url` in `~/.akg/settings.json`:

```json
{
  "models": {
    "standard": {
      "base_url": "http://localhost:8000/v1",
      "api_key": "EMPTY",
      "model_name": "your-model-name"
    }
  },
  "default_model": "standard"
}
```

**Configuration Details**:
- `base_url`: API address of your local model service (typically ends with `/v1`, compatible with OpenAI API format)
- `api_key`: If your local service doesn't require authentication, use `"EMPTY"` or any string
- `model_name`: The name of your deployed model

## Local Model Deployment Frameworks

If you haven't deployed a local model service yet, consider these open-source frameworks:

| Framework | Official Link |
|-----------|---------------|
| **vLLM** | [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) |
| **vLLM-Ascend** | [https://github.com/vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend) |
| **vLLM-MindSpore** | [https://atomgit.com/mindspore/vllm-mindspore](https://atomgit.com/mindspore/vllm-mindspore) |
| **Ollama** | [https://ollama.com](https://ollama.com) |

## Model Selection Recommendations

For **Kernel code generation tasks**, which demand strong code understanding and generation capabilities, we recommend the following types of models:

### Recommended Models

Due to the high difficulty of kernel code generation, we recommend using powerful models such as **DeepSeek V3.2**, or kernel-specific models designed for such tasks.

### Configuration Example

```json
{
  "models": {
    "standard": {
      "base_url": "http://localhost:8000/v1",
      "api_key": "EMPTY",
      "model_name": "deepseek-v3"
    }
  },
  "default_model": "standard"
}
```

## Verifying Configuration

After configuration, verify the model connection by launching `akg_cli op`:

```bash
# Ascend NPU
akg_cli op --framework torch --backend ascend --arch ascend910b2 \
  --dsl triton_ascend --devices 0,1,2,3,4,5,6,7

# CUDA GPU
akg_cli op --framework torch --backend cuda --arch a100 \
  --dsl triton_cuda --devices 0,1,2,3,4,5,6,7

# CPU
akg_cli op --framework torch --backend cpu --arch x86_64 \
  --dsl cpp --devices 0
```

If the model is configured correctly, the system will start normally and respond to your requests.

## Related Documentation

- [Configuration Documentation](./Configuration.md) - Detailed configuration options
- [LLM Documentation](./LLM.md) - LLM provider and client architecture
- [Kernel Agent Documentation](./KernelAgent.md) - Kernel code generation scenario details
