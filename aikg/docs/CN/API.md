# API 配置说明

此文档用于详细说明 AIKG 项目中涉及的 API 配置。


## 1. 环境变量配置 (API Key)

我们通过环境变量来设置不同大语言模型（LLM）服务的 API Key 和服务地址（Endpoint）。这样可以保持敏感信息（如 API Key）的私密性，并方便在不同环境中切换。

支持的服务及对应的环境变量如下：

```bash
# VLLM (https://github.com/vllm-project/vllm)
export AIKG_VLLM_API_BASE=http://localhost:8000/v1

# Ollama (https://ollama.com/)
export AIKG_OLLAMA_API_BASE=http://localhost:11434

# 硅基流动 (https://www.siliconflow.cn/)
export AIKG_SILICONFLOW_API_KEY=sk-xxxxxxxxxxxxxxxxxxx

# DeepSeek (https://www.deepseek.com/)
export AIKG_DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxx

# 火山引擎 (https://www.volcengine.com/)
export AIKG_HUOSHAN_API_KEY=0cbf8bxxxxxx
```

## 2. LLM 模型配置 (`llm_config.yaml`)

此文件定义了所有可供 AIKG 使用的底层 LLM 模型。每个模型都有一个唯一的名称，并包含调用该模型所需的参数。

**文件路径**: `aikg/python/ai_kernel_generator/core/llm/llm_config.yaml`

**功能**:
-   **注册模型**: 将新的 LLM 模型接入 AIKG 框架。
-   **参数预设**: 为每个模型预设必要的调用参数。

**通用参数**:
- `api_base`: API 基础 URL
- `model`: 模型名称
- `max_tokens`: 最大生成 token 数
- `temperature`: 温度参数，控制随机性
- `top_p`: 核采样参数，控制多样性
- `frequency_penalty`: 频率惩罚，控制重复
- `presence_penalty`: 存在惩罚，控制主题重复

**如何使用**:
在 `llm_config.yaml` 中添加一个新模型配置后，可以通过 `create_model("my_model_name")` 来直接调用该模型。

## 3. 任务流程配置 (`xxx_config.yaml`)

任务流程配置文件用于编排和组织一个完整的算子生成任务中，每个子任务（Agent）具体使用哪个在 `llm_config.yaml` 中定义的模型。

**默认配置文件目录**: `aikg/python/ai_kernel_generator/config/`

**功能**:
-   **任务编排**: 为 `Designer`, `Coder`, `Conductor` 等不同的 Agent 指派不同的 LLM 模型。
-   **灵活组合**: 可以创建多个配置文件，以应对不同场景，例如，某个流程使用 vLLM 部署的模型，另一个流程使用 DeepSeek 的官方 API。
-   **默认配置**: `default_config.yaml` 是默认的流程配置。

**示例 (`vllm_dsr1_with_official_dsv3_config.yaml`)**:
这个文件展示了如何为一个算子生成流程中的不同 Agent（如 Coder, Designer）配置不同的 LLM 模型。例如，代码修复 (`swft_coder_fix`) 和检查 (`conductor_check`) 任务使用了官方的 `deepseek_v3_default` 模型，而其他大部分任务使用了通过 vLLM 部署的 `vllm_deepseek_r1_default` 模型。

```yaml
# 模型预设配置
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

# 日志配置
log_dir: "~/aikg_logs"
```

**如何使用**:
在代码中，可以通过 `load_config()` 来加载指定的任务流程配置。
```python
# 加载默认配置 default_config.yaml
config = load_config()

# 加载指定的配置文件
config = load_config("/path/to/your/vllm_dsr1_with_official_dsv3_config.yaml")

# 在任务中使用该配置
task = Task(
    # ...
    config=config,
)
```
通过这种方式，您可以灵活地为不同的任务流程配置和切换底层的大语言模型。
