# LLM API 配置说明

此文档用于详细说明 AIKG 项目中涉及的 LLM API 配置。


## 1. 环境变量配置 (API Key)

我们通过环境变量来设置不同大语言模型（LLM）服务的 API Key 和服务地址（Endpoint）。这样可以保持敏感信息（如 API Key）的私密性，并方便在不同环境中切换。

支持的服务及对应的环境变量如下：

```bash
# VLLM (https://github.com/vllm-project/vllm)
export AIKG_VLLM_API_BASE=http://localhost:8000/v1

# Ollama (https://ollama.com/)
export AIKG_OLLAMA_API_BASE=http://localhost:11434

# 硅基流动 (https://www.siliconflow.cn/)
export AIKG_SILICONFLOW_API_KEY=sk-xxx

# DeepSeek (https://www.deepseek.com/)
export AIKG_DEEPSEEK_API_KEY=sk-xxx

# 火山引擎 (https://www.volcengine.com/)
export AIKG_HUOSHAN_API_KEY=0cbf8bxxxxxx

# Moonshot (https://www.moonshot.cn/)
export AIKG_MOONSHOT_API_KEY=sk-xxx

# 智谱大模型 (https://www.bigmodel.cn/)
export AIKG_ZHIPU_API_KEY=sk-xxx
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

**使用方式**:

在 `llm_config.yaml` 中添加一个新模型配置后，可以通过 `create_model("my_model_name")` 来直接调用该模型。

## 3. 任务编排方案配置 (`xxx_custom_plan.yaml`)

任务编排方案配置文件用于编排和组织一个完整的算子生成任务中，每个子任务（Agent）具体使用哪个在 `llm_config.yaml` 中定义的模型。

**默认配置文件目录**: `aikg/python/ai_kernel_generator/config/`

**功能**:
-   **任务编排**: 为 `designer`、`coder`、`conductor` 等 Agent 指派不同的 LLM 模型。
-   **灵活组合**: 可以创建多个配置文件，以应对不同场景（如本地 vLLM 与云端 API 混合）。
-   **默认方案**: 按 DSL 提供默认方案，例如 `default_triton_cuda_config.yaml` 或 `default_triton_ascend_config.yaml`。

**示例（coder-only，本地 vLLM）**：`vllm_triton_coderonly_config.yaml`
为 coder-only 流程配置统一的本地 vLLM 模型。

```yaml
# 模型预设配置
agent_model_config:
  designer: vllm_deepseek_r1_default
  coder: vllm_deepseek_r1_default
  conductor: vllm_deepseek_r1_default
  api_generator: vllm_deepseek_r1_default

# 日志配置
log_dir: "~/aikg_logs"
```

**使用方式**:

在代码中，通过 `load_config()` 加载配置。
```python
# 按 DSL 加载默认方案
config = load_config(dsl="triton_ascend", backend="ascend")  # 或使用 "triton_cuda" 用于 CUDA 后端

# 或加载指定的配置文件
config = load_config(config_path="/path/to/your/vllm_custom_plan.yaml")

# 在任务中使用该配置
task = Task(
    # ...
    config=config,
)
```
通过这种方式，您可以灵活地为不同的任务流程配置和切换底层的大语言模型。

## 4. 全局环境变量设置

您可以通过高优先级环境变量直接指定 LLM API。

```bash
export AIKG_BASE_URL="https://api.example.com/v1"
export AIKG_MODEL_NAME="your-model-name"
export AIKG_API_KEY="your-api-key"
```
