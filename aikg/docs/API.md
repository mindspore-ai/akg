# API Configuration Guide

This document provides a detailed explanation of the API configurations used in the AIKG project.


## 1. Environment Variable Configuration (API Key)

We use environment variables to set the API keys and service endpoints for various Large Language Model (LLM) services. This approach helps maintain the privacy of sensitive information (like API keys) and allows for easy switching between different environments.

Supported services and their corresponding environment variables are as follows:

```bash
# VLLM (https://github.com/vllm-project/vllm)
export AIKG_VLLM_API_BASE=http://localhost:8000/v1

# Ollama (https://ollama.com/)
export AIKG_OLLAMA_API_BASE=http://localhost:11434

# SiliconFlow (https://www.siliconflow.cn/)
export AIKG_SILICONFLOW_API_KEY=sk-xxxxxxxxxxxxxxxxxxx

# DeepSeek (https://www.deepseek.com/)
export AIKG_DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxx

# Volcengine (https://www.volcengine.com/)
export AIKG_HUOSHAN_API_KEY=0cbf8bxxxxxx
```

## 2. LLM Model Configuration (`llm_config.yaml`)

This file defines all the underlying LLM models available for use by AIKG. Each model has a unique name and includes the parameters required to call it.

**File Path**: `aikg/python/ai_kernel_generator/core/llm/llm_config.yaml`

**Functionality**:
-   **Register Models**: Integrate new LLM models into the AIKG framework.
-   **Preset Parameters**: Pre-configure the necessary calling parameters for each model.

**Common Parameters**:
- `api_base`: The base URL for the API.
- `model`: The model name.
- `max_tokens`: The maximum number of tokens to generate.
- `temperature`: A parameter to control randomness.
- `top_p`: A nucleus sampling parameter to control diversity.
- `frequency_penalty`: A penalty to discourage repetition.
- `presence_penalty`: A penalty to discourage topic repetition.

**How to Use**:
After adding a new model configuration to `llm_config.yaml`, you can directly call the model using `create_model("my_model_name")`.

## 3. Task Flow Configuration (`xxx_config.yaml`)

Task flow configuration files are used to orchestrate and organize which model defined in `llm_config.yaml` is used by each sub-task (Agent) in a complete kernel generation task.

**Default Configuration Directory**: `aikg/python/ai_kernel_generator/config/`

**Functionality**:
-   **Task Orchestration**: Assign different LLM models to different Agents like `Designer`, `Coder`, and `Conductor`.
-   **Flexible Combinations**: Create multiple configuration files for different scenarios. For example, one flow might use a model deployed with vLLM, while another uses the official DeepSeek API.
-   **Default Configuration**: `default_config.yaml` is the default flow configuration.

**Example (`vllm_dsr1_with_official_dsv3_config.yaml`)**:
This file demonstrates how to configure different LLM models for various Agents (e.g., Coder, Designer) within a kernel generation flow. For instance, the code fixing (`swft_coder_fix`) and checking (`conductor_check`) tasks use the official `deepseek_v3_default` model, while most other tasks use the `vllm_deepseek_r1_default` model deployed via vLLM.

```yaml
# Model preset configuration
agent_model_config:
  aul_designer: vllm_deepseek_r1_default
  aul_designer_fix: vllm_deepseek_r1_default
  swft_coder: deepseek_v3_default
  swft_coder_api: vllm_deepseek_r1_default
  swft_coder_fix: deepseek_v3_default
  triton_coder: vllm_deepseek_r1_default
  triton_coder_fix: vllm_deepseek_r1_default
  conductor_check: vllm_deepseek_r1_default
  conductor_analyze: deepseek_v3_default

# Log configuration
log_dir: "~/aikg_logs"
```

**How to Use**:
In your code, you can load a specific task flow configuration using `load_config()`.
```python
# Load the default configuration: default_config.yaml
config = load_config()

# Load a specific configuration file
config = load_config("/path/to/your/vllm_dsr1_with_official_dsv3_config.yaml")

# Use the configuration in a task
task = Task(
    # ...
    config=config,
)
```
This approach allows you to flexibly configure and switch the underlying large language models for different task flows. 