# 本地部署大模型接入指南

本文档介绍如何将本地部署的大模型接入 AKG Agents。

## 配置本地模型

如果您已经拥有自己部署的模型服务，只需在 `~/.akg/settings.json` 中配置本地服务的 `base_url` 即可：

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

**配置说明**：
- `base_url`: 本地模型服务的 API 地址（通常以 `/v1` 结尾，兼容 OpenAI API 格式）
- `api_key`: 如果本地服务不需要认证，可以填写 `"EMPTY"` 或任意字符串
- `model_name`: 您部署的模型名称

## 本地模型部署框架

如果您还没有部署本地模型服务，可以参考以下开源框架：

| 框架 | 官方链接 |
|------|---------|
| **vLLM** | [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) |
| **vLLM-Ascend** | [https://github.com/vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend) |
| **vLLM-MindSpore** | [https://atomgit.com/mindspore/vllm-mindspore](https://atomgit.com/mindspore/vllm-mindspore) |
| **Ollama** | [https://ollama.com](https://ollama.com) |

## 模型选择建议

对于 **Kernel 代码生成任务**，由于其对代码理解和生成能力要求较高，我们推荐使用以下类型的模型：

### 推荐模型

由于 Kernel 代码生成难度高，推荐使用 **DeepSeek V3.2** 等能力较强的大模型，或 Kernel 类任务专用模型。

### 配置示例

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

## 验证配置

配置完成后，可以通过启动 `akg_cli op` 命令验证模型连接：

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

如果模型配置正确，系统将正常启动并响应您的请求。

## 相关文档

- [配置系统文档](./Configuration.md) - 详细的配置选项说明
- [LLM 接入文档](./LLM.md) - LLM 提供者和客户端架构
- [Kernel Agent 文档](./KernelAgent.md) - Kernel 代码生成场景说明
