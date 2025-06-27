# AI-driven Kernel Generator(AIKG)

## 1. 项目简介
AIKG是一款基于大模型驱动的AI Kernel生成工具，利用大语言模型（LLM）的代码生成能力，完成多后端多类型的AI算子生成与自动优化。


## 2. 安装流程

```bash
# 使用conda环境（可选）
conda create -n aikg python=3.11
conda activate aikg

# 或者创建虚拟环境（可选）
python -m venv .venv
source .venv/bin/active

# pip安装依赖
pip install -r requirements.txt

# setup & install
bash build.sh
pip install output/ai_kernel_generator-*-py3-none-any.whl
```


## 3. 配置

### 配置API环境变量（可选）
```
export AIKG_VLLM_API_BASE=http://localhost:8000/v1 # 本地或远程VLLM服务器地址
export AIKG_OLLAMA_API_BASE=http://localhost:11434 # 本地或远程Ollama服务器地址
export AIKG_SILICONFLOW_API_KEY=sk-xxxxxxxxxxxxxxxxxxx # 硅流key
export AIKG_DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxx # DeepSeek key
export AIKG_HUOSHAN_API_KEY=0cbf8bxxxxxx # 火山key
```

### Ascend310P SWFT后端依赖
请参考：https://gitee.com/mindspore/akg/swft

### Ascend910 Triton后端依赖
请参考：https://gitee.com/ascend/triton-ascend

### MindSpore前端2.7版本依赖
支持python版本：python3.11、python3.10、python3.9
支持系统版本：aarch64、x86_64
```
# python3.11 + aarch64的安装包示例
pip install https://repo.mindspore.cn/mindspore/mindspore/version/202506/20250619/master_20250619160020_1261ff4ce06d6f2dc4ce446139948a3e4e9c966b_newest/unified/aarch64/mindspore-2.7.0-cp311-cp311-linux_aarch64.whl
```

### NVIDIA GPU Triton后端依赖
请参考：https://github.com/triton-lang/triton


## 4. 运行
请参考`examples`目录中示例代码。


## 5. 适配新模型
在 `ai_kernel_generator/core/llm/llm_config.yaml` 文件中可以配置新的模型。每个模型配置包含以下参数：

### 通用参数
- `api_base`: API 基础 URL
- `model`: 模型名称
- `max_tokens`: 最大生成 token 数
- `temperature`: 温度参数，控制随机性
- `top_p`: 核采样参数，控制多样性
- `frequency_penalty`: 频率惩罚，控制重复
- `presence_penalty`: 存在惩罚，控制主题重复

配置完成后，可以通过 `create_model("my_model_name")` 来使用新配置的模型。

### 全流程LLM配置 & 通用设置
在aikg完整流程里，可以通过设置自定义`config.yaml`的方式来控制每个子任务调用的LLM
```python
config = load_config() # 调用默认配置 default_config.yaml
config = load_config("/your-path-to-config/vllm_deepseek_r1_config.yaml")
task = Task(
    ...
    config=config,
)
```




## 6. 设计文档
### AIKG通用框架
TODO（TaskPool、DevicePool、Trace）

### Designer
请参考 [Designer](./ai_kernel_generator/docs/README_Designer.md)

### Coder
请参考 [Coder](./ai_kernel_generator/docs/README_Coder.md)

### Tester
请参考 [Tester](./ai_kernel_generator/docs/README_Tester.md)

### SWFT Backend
请参考 [SWFT](./ai_kernel_generator/docs/README_SWFT.md)

### Triton Backend
请参考 [TRITON](./ai_kernel_generator/docs/README_TRITON.md)