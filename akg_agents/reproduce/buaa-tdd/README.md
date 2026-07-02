# 总文档

# 基准版本与实验配置

- workflow：
    - mathir_coder_workflow
    - mathir_multi_kernel_gen_workflow
- LLM model：参考模型使用
- gpu：nvidia a800 80G
    - torch：2.9.0+cu128
    - triton：3.5.0
- npu：ascend 910b3
    - torch_npu：2.10.0
    - triton：3.2.0
    - torch：2.10.0+cpu
- 测试集：KernelBench Level 1 及 third_party attn（flash\page\radix）

## 模型使用

- ~/.akg/settings.json

```python
{
  "models": {
    "deepseek_v4_flash_default": {
      "base_url": "https://api.deepseek.com",
      "api_key": "xxx",
      "model_name": "deepseek-v4-flash",
      "temperature": 0.2,
      "max_tokens": 16384,
      "top_p": 0.9,
      "frequency_penalty": 0.0,
      "presence_penalty": 0.0
    },
    "deepseek_v4_flash_thinking_high": {
      "base_url": "https://api.deepseek.com",
      "api_key": "xxx",
      "model_name": "deepseek-v4-flash",
      "temperature": 0.1,
      "max_tokens": 65536,
      "top_p": 0.9,
      "frequency_penalty": 0.5,
      "presence_penalty": 0.5,
      "extra_body": {
        "reasoning_effort": "max"
      }
    },
    "deepseek_v4_flash_thinking_high_max_tokens": {
      "base_url": "https://api.deepseek.com",
      "api_key": "xxx",
      "model_name": "deepseek-v4-flash",
      "temperature": 0.1,
      "max_tokens": 393216,
      "top_p": 0.9,
      "frequency_penalty": 0.5,
      "presence_penalty": 0.5,
      "extra_body": {
        "reasoning_effort": "max"
      }
    },
    "deepseek_v4_pro_thinking_high": {
      "base_url": "https://api.deepseek.com",
      "api_key": "xxx",
      "model_name": "deepseek-v4-pro",
      "temperature": 0.1,
      "max_tokens": 393216,
      "top_p": 0.9,
      "frequency_penalty": 0.0,
      "presence_penalty": 0.0,
      "thinking_mode": "enabled",
      "extra_body": {
        "reasoning_effort": "max"
      }
    },
    "complex": {
      "base_url": "https://api.deepseek.com",
      "api_key": "xxx",
      "model_name": "deepseek-v4-flash",
      "temperature": 0.1,
      "max_tokens": 65536,
      "top_p": 0.9,
      "frequency_penalty": 0.5,
      "presence_penalty": 0.5,
      "extra_body": {
        "reasoning_effort": "max"
      }
    },
    "standard": {
      "base_url": "https://api.deepseek.com",
      "api_key": "xxx",
      "model_name": "deepseek-v4-flash",
      "temperature": 0.1,
      "max_tokens": 65536,
      "top_p": 0.9,
      "frequency_penalty": 0.5,
      "presence_penalty": 0.5,
      "extra_body": {
        "reasoning_effort": "max"
      }
    },
    "fast": {
      "base_url": "https://api.deepseek.com",
      "api_key": "xxx",
      "model_name": "deepseek-v4-flash",
      "temperature": 0.2,
      "max_tokens": 16384,
      "top_p": 0.9,
      "frequency_penalty": 0.0,
      "presence_penalty": 0.0
    }
  },
  "default_model": "standard",
  "context_window": 128000,
  "stream_output": false
}
```

## 环境安装

具体细节可以参考解决方案一节。

### database必须

- qdrant
    - 启动docker
    
    ```python
    docker pull qdrant/qdrant
    docker run -p 6333:6333 -p 6334:6334 --name lt_qdrant \
        -v "/home/lutao/workspaces/aikg/qdrant_storage:/qdrant/storage:z" \
        qdrant/qdrant
    ```
    
    - 依赖
    
    ```python
    python -m pip install "qdrant-client>=1.9.0"
    python -m pip install "sentence-transformers>=2.2.0"
    ```
    
- embedding model：建议下载完成后，将下载路径配置到yaml中进行使用。
    
    ```python
    python - <<'PY'
    from sentence_transformers import SentenceTransformer
    
    SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="/home/lutao/workspaces/aikg/aikg_new/aikg/python/ai_kernel_generator/database/api/embed_model_cache",
    )
    print("embedding model ready")
    PY
    ```
    
    - 具体位置随意，可以在config.yaml中指定要使用的路径。
- smoke测试：

```python
# 根据当前本地triton重建数据库
python -m akg_agents.database.api.api_db_bootstrap \
  --target-backend ascend \
  --force-rebuild
# smoke测试
python akg_agents/reproduce/buaa-tdd/smoke_api_database.py \
  --config akg_agents/python/akg_agents/op/config/triton_ascend_mathir_config.yaml
```

### MathIR可选

- 如果需要借助MLIR信息增强生成MathIR生成，需要安装MLIR、torchMLIR、LLVM-tools包
    - 当前MLIR环境只在cuda上进行成功导出，在Ascend上需要手动编译。可以在cuda上由MLIR路径生成MathIR后将结果迁移到Ascend使用。二者MathIR通用，但是MathIR lowering doc不一致。
    - cuda环境安装
        
        ```python
        pip install --pre "torch-mlir==20260103.681" --no-deps -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
        conda install mlir llvm-tools
        ```
        

# **背景与目标**

## TDD

TDD即测试驱动开发是软件开发过程中的重要手段。已有工作表明TDD过程在AI生成通用算法代码中的有效性。但是通用算法场景测试样例构造简单，通常由人工将测试和需求一起交给模型进行生成；当前kernel形式benchmark缺少测试样例来源，且kernel的测试样例通常为大型tensor，难以填入生成上下文。且经过测试表明，如果交由AI自主从0生成测试，TDD的质量高低受限于模型对于代码的理解，这造成了互为前提的循环依赖问题：如果AI善于理解输入模型的代码，那么自然就能生成正确性高的代码，不需要测试进行补充；如果AI不善于理解输入模型的代码，那么从0生成的测试也是基本无效的。需要通过其他手段实现在kernel生成中的TDD，指导生成过程。

## Database增强

数据库增强是当前 Agent 工程中一种常见且有效的成功率提升手段，其核心思想是在 Agent 生成、执行和修复任务的过程中，引入外部数据库作为知识补充来源。相比仅依赖大语言模型自身的参数记忆，数据库增强可以为 Agent 提供更加精确、可追溯且与当前任务相关的上下文信息，例如历史错误案例、已有代码实现、API 使用方式、编译报错与修复记录、性能优化经验、硬件架构说明以及相似任务的成功轨迹等。当 Agent 在面对复杂任务时，可以先根据当前输入、错误日志或中间代码表示检索数据库中的相似样例，再将检索结果作为额外上下文提供给模型，从而降低无依据生成和盲目搜索的概率。对于 kernel 生成、程序修复、自动调优等任务而言，这种方法尤其重要，因为许多错误并不是简单的语法问题，而是与具体后端、库版本、数据布局、边界条件或硬件约束密切相关。通过数据库增强，Agent 可以复用过去积累的调试经验和优化策略，将原本完全依赖模型推理的开放式生成过程，转化为“检索相关经验—生成候选方案—执行验证—反馈修复”的闭环流程，从而提高复杂工程任务中的正确率、稳定性和迭代效率。

## 目标

本次工作聚焦生成能力的提高，*以项目开启前提供的固定aikg commit作为标准*：

1. *KernelBench level1：当前AIKG框架在KernelBench level1可在Triton-GPU、Triton-NPU、CPU等多个后端代码正确生成的Pass@1提升10+%。*
2. *attn：FlashAttention、PagedAttention、RadixAttention算子可在Triton-GPU、Triton-NPU、CPU等多个后端代码正确生成。（由于未指定FlashAttention、PagedAttention、RadixAttention算子代码内容，所以自行实现）*

拟使用TDD行为和Database增强行为，有效增加生成成功率。具体落实到代码workflow中的不同解决方案。

# 问题描述与解决方案

## TDD in kernelbench：mathIR 与 multi kernel gen

### 背景

- 复杂算子生成是当前 Agent 工程中的重要瓶颈。对于简单逐元素算子、规则矩阵乘法、简单 reduction 或固定模式的 fused operator，Agent 往往可以依赖已有代码模板、编译反馈和少量执行测试完成自动修复。但是当目标算子变成 FlashAttention、paged attention、fused MLP、复杂 reduction、动态 shape 算子或跨层融合算子时，单纯依靠“原始 PyTorch 代码 + Prompt + 编译/执行反馈”的方式通常不够稳定。其核心困难不只是代码生成本身，而是 Agent 需要同时完成以下几件事：
    1. 理解原始 PyTorch 程序语义；
    2. 推断输入输出张量的 shape、stride、layout、broadcast、mask、dtype 和数值精度要求；
    3. 将PyTorch 程序语义映射到目标后端的并行执行模型，例如 Triton block/program、CUDA thread block/warp、Ascend AIC/AIV 等；
    4. 设计分块、访存、同步、reduction、softmax 稳定性和中间结果格式；
    5. 在出错后根据有限的编译或运行反馈进行修复。
    
    这些任务高度耦合，对模型的推理能力和上下文窗口要求强，导致复杂 kernel 的自动生成存在问题。
    
- 针对以上问题，**提出后端无关的MathIR逻辑表示**与**基于MathIR的multi kernel gen方案**，解决：
    - 缺少后端无关逻辑表示的问题。通过将逻辑表示和程序表示分离
        - Coder 不再需要直接从复杂 PyTorch API 中推理算子语义，而是可以基于更稳定、更通用的数学表达进行代码生成。
        - Coder可以基于同一mathIR结果，为不同并行加速器后端skills进行跨后端生成。
        - 提供逻辑错误检索能力：端到端测试通常只能告诉 Agent “最终输出不一致”，却无法直接指出错误来自 QK、mask、softmax、PV accumulation 还是 combine 阶段。因此，Agent 容易陷入反复局部搜索和盲目修补。基于 MathIR 的 TDD 设计可以将测试下沉到中间表示和子 kernel 层面。
    - 单次生成大 kernel 的 prompt 冗余问题。对于 FlashAttention、paged attention、fused MLP、复杂 reduction 等算子，一个完整 kernel 往往同时包含多种逻辑。如果将完整 PyTorch 代码、后端约束、shape 信息、性能要求、历史错误和修复记录全部放入一次 prompt，容易造成上下文过长、语义约束分散、弱模型幻觉增多、已修复错误被重新引入等问题。基于 MathIR 将复杂算子拆分为多个独立子任务，每个子任务只携带局部 MathIR 和局部 TDD 要求。通过子任务拆分，Coder 面对的是边界清晰、输入输出明确、测试目标明确的局部生成任务，而不是一次性生成完整复杂 kernel。
- 为什么可行：
    - 强弱模型协同：MathIR和TDD的正确性由强模型/人工/已有数据库进行保证，即使用强模型+输入代码得到能够完善表示模型代码变换的MathIR中间表示与TDD要求。Coder接受这些额外信息进行代码生成。测试表示，强模型更适合完成语义理解、MathIR 抽取、TDD 生成和任务拆分；在明确规格下，使用弱 Coder 模型也可以局部代码生成，降低模型使用成本。
    - 领域转换：将目标程序从复杂的 PyTorch API 语义空间转换到模型更通用、更稳定的数学语义空间。对于LLM来说，PyTorch API同时引入了许多程序层面的偶然复杂性，如依赖 shape、stride、broadcast 或 dtype时，Coder 容易把程序实现细节和数学语义混淆。
    - TDD 将复杂 kernel 的端到端正确性问题拆成局部可验证问题：生成记录显示，如果只是生成子kernel然后进行组合，子kernel的幻觉会被一并引入。强模型生成的MathIR的TDD要求可以大幅增强每个独立子任务的潜在正确性。每个正确性得到一定保证的子kernel进行combine得到原始kernel的成功率，比不进行测试验证直接进行combine的子kernel的成功率要高很多。
- 职责：
    - MathIR 负责回答：这个算子在数学上到底做什么？
    - TDD 负责回答：如何证明每个子任务和最终 kernel 是正确的？
    - Multi-kernel generation 负责回答：如何将一个超出单次 prompt 和单次生成能力的大算子，拆成多个可生成、可测试、可组合的局部任务？

通过这种方式，可以显著降低复杂算子生成对单次 prompt 长度、模型推理能力和黑盒执行反馈的依赖，提高复杂 kernel 的生成正确率、可调试性、可迁移性和最终组合成功率

### 总览

- workflow
    - MathIR node 只在入口执行一次，当前使用`deepseek_v4_pro_thinking_high`进行生成。不会被反复重算。
        
        ```python
            "deepseek_v4_pro_thinking_high": {
              "base_url": "https://api.deepseek.com",
              "api_key": "",
              "model_name": "deepseek-v4-pro",
              "temperature": 0.1,
              "max_tokens": 393216,
              "top_p": 0.9,
              "frequency_penalty": 0.0,
              "presence_penalty": 0.0,
              "thinking_mode": "enabled",
              "extra_body": {
                "reasoning_effort": "max"
              }
            }
        ```
        
    - MathIR workflow 在普通 Coder loop 前增加一个只执行一次的 `mathIR` 节点。
        
        ```
        mathIR -> api_recall -> coder -> code_checker -> verifier
                                      ^                    |
                                      +--------------------+
        ```
        
    - `mathir_multi_kernel_gen_workflow` 在 `api_recall` 后根据 MathIR expression 数量和 `multi_kernel_gen` 配置分支
        
        ```python
        mathIR -> api_recall
                  |
                  +-- expression 数量 <= 1 或 multi_kernel_gen=false
                  |       |
                  |       v
                  |     coder -> code_checker -> verifier
                  |        ^                        |
                  |        +------------------------+
                  |
                  +-- expression 数量 > 1 且 multi_kernel_gen=true
                          |
                          v
                      multi_expr_coder
                          |
                          +-- 子 expression 生成/修复
                          +-- 子 CodeChecker 静态检查
                          +-- 子 verifier 串行验证
                          +-- 全部通过后 combine
                          |
                          v
                      code_checker -> verifier
        ```
        
        - `multi_expr_coder` 从 `mathIR_code["expressions"]` 读取子表达式。
        - 每个 expression 构造独立子任务，先生成/修复子 kernel，再对子 kernel 验证。
        - 所有子 expression 共享 `multi_kernel_max_retries` 总预算，不是每个 expression 各自一份预算。
        - 全部子 kernel 通过后才 combine 成最终 `ModelNew`。
        - 任一子 kernel 耗尽预算后写入 `multi_expr_error` / `verifier_error`，workflow 直接失败，不再兜底 combine。
- MathIR 字段设计：详细参考akg_agents/python/akg_agents/op/resources/prompts/mathIR/mathIRgen_multi.j2。这里进行字段归类简介
    - 多expr图语义：`name`，`next_expressions`，`consumes`，`produces` 表述多expr组合DAG图的消费关系。
    - 逻辑表述相关：`formula` 。要求强模型输出为自然语言 + 数学表达的混合写法。通过强模型自身推理能力/人工保证逻辑正确
    - 程序语义相关：`boundary_treatment`，`axis_mapping`，`symbol_binding` 。表述数学表达式到具体并行程序的规约。
    - TDD相关：`TDD_requirement` 。在单kernel生成时仅进行测试要求的展示。在多kernel生成时要求每个子kernel在生成时同时生成`if __name__=="__main__"` 的相关验证，只有通过验证才允许combine。
- MathIR lowering doc设计：详细参考akg_agents/python/akg_agents/op/resources/docs/triton_cuda_docs/mathir_lowering_docs.md和ascend下同名文档。具体内容为指导LLM如何从MathIR的数学表达式生成到Triton的并行语义。
- 输出会写入 state，Coder prompt 可读取这些字段：
    - `mathIR_code`：结构化 IR，通常包含 inputs、outputs、expressions 等。
    - `mathIR_prompt` / `mathIR_reasoning`：MathIR LLM 调用的提示词和推理记录。
    - `mathIR_error`：MathIR 解析失败时的错误，失败不会阻断 Coder。
    - `mlir` / `mlir_compile_code`：可选 MLIR 导出结果。
    - `pytorch_doc_string` / `standard_formula`：辅助 Coder 理解 PyTorch API 语义。
    - `preset_ir_json` / `preset_ir_path`：命中 preset 时的来源。

### 使用细节

- 当前 YAML 配置：
    
    
    | YAML 配置项 | 当前/推荐值 | 作用 |
    | --- | --- | --- |
    | `agent_model_config.mathIR` | `deepseek_v4_pro_thinking_high` | MathIR LLM 使用的模型档位。 |
    | `mathir_use_preset` | `True` | 优先查找并复用 preset IR。命中时视为 cache hit。 |
    | `mathir_save` | `False` | 是否把新生成的 MathIR 保存为 preset。 |
    | `mathir_overwrite` | `False` | 保存 preset 时是否覆盖已有文件。 |
    | `mathir_mlir_export` | `True` | 是否尝试导出 torch-mlir/MLIR 辅助信息。不可用时应降级，不阻断主流程。 |
    | `mathir_doc_string` | `True` | 是否补充 PyTorch API doc string / formula 信息。 |
    | `mathir_db_math_query` | `False` | 是否启用额外数学语义数据库查询（未启用）。 |
    | `multi_kernel_gen` | `True` | 是否允许多 expression 进入 `multi_expr_coder`。仅 `mathir_multi_kernel_gen_workflow` 使用该分支。 |
    | `multi_kernel_max_retries` | `15` | 多 expression 子 kernel 生成/修复共享总预算。 |

### 技术细节

主要实现文件：

| 文件 | 作用 |
| --- | --- |
| `python/akg_agents/core/agent/mathIR.py` | MathIR agent 主实现；按 preset、MLIR、doc string、db query、standard formula、LLM 的策略顺序准备 MathIR 输入并生成结果。 |
| `python/akg_agents/op/workflows/mathir_coder_workflow.py` | `mathIR -> api_recall -> coder -> code_checker -> verifier` workflow 定义。 |
| `python/akg_agents/op/workflows/mathir_multi_kernel_gen_workflow.py` | MathIR 多 expression 分支、`multi_expr_coder` 路由、combine 后验证 workflow 定义。 |
| `python/akg_agents/op/langgraph_op/nodes.py` | MathIR、api_recall、coder、multi_expr_coder、code_checker、verifier 节点实现。 |
| `python/akg_agents/op/langgraph_op/routers.py` | verifier/code_checker/codegen 路由；MathIR workflows 中 verifier 失败直接回 Coder。 |
| `python/akg_agents/op/langgraph_op/state.py` | MathIR、API recall、multi-kernel、CodeChecker 相关 state 字段。 |
| `python/akg_agents/op/utils/mlir_export.py` | torch-mlir/MLIR 导出辅助。 |
| `python/akg_agents/op/resources/prompts/mathIR/defaultIR.j2` | 默认数学公式和高频算子语义提示。 |
| `python/akg_agents/op/resources/prompts/mathIR/mathIRgen_multi.j2` | MathIR LLM 生成 prompt。 |
| `python/akg_agents/op/resources/docs/triton_cuda_docs/mathir_lowering_docs.md` | Coder 侧 MathIR lowering 指南。 |

MathIR agent 生成逻辑：

1. `run_preset`：若 `mathir_use_preset` 开启且找到 preset IR，直接返回 preset，跳过 LLM。
2. `run_mlir_export`：若开启 `mathir_mlir_export`，尝试从 PyTorch 代码导出 MLIR 辅助信息。
3. `run_doc_string`：若开启 `mathir_doc_string`，补充 PyTorch API 文档/公式信息。
4. `run_db_math_query`：若开启 `mathir_db_math_query`，补充额外数学语义查询结果。
5. `run_standard_formula`：加载 `mathIR/defaultIR.j2` 作为默认公式和语义规则。
6. `_run_with_llm`：调用 MathIR LLM，输出结构化 IR 文本并由 parser 解析为 `mathIR_code`。

## Triton API database

### 背景

- 本地知识库幻觉问题：由于目标架构为Triton，当前Triton的fork较多，API更新较快。而调用的LLM API 只能依靠训练时的Triton语料进行代码的生成，所以在测试中常见：生成的API由于版本不同，缺失参数，使用错误等问题。这实际上需要在线数据库进行召回指导LLM使用本地可用的Triton API进行代码生成。但当前的api召回功能仅为静态文档，难以满足多后端多Triton版本的需求，LLM 生成Triton API不对齐行为频发
- 针对本地知识库幻觉问题和静态文档无法满足Triton API多样化的需求，搭建Triton API召回数据库：在kernel生成前使用本地Triton+pytorch包进行本地API数据库的构建。通过本地API doc和Triton RAG向量数据库的配合，为任务生成提供环境实时可用API doc，有效的减少了弱模型的API幻觉问题。

### 总览

- 当前Triton API database支持下的API doc由两部分组成：
    1. 静态 `api/api.md` ：仍保留一组稳定 core API 约束，例如 `tl.load`、`tl.store`、`tl.arange`、`tl.dot`等；
    2. database retrieve 负责补充非 core API 或任务相关 API
    - 这样做的原因是database retrieve结果往往细节缺失，静态 `api/api.md`往往不能描述部分算子使用的冷门API。保留`api.md` 作为使用最广泛API如`tl.load`、`tl.store`的详细描述，使用database retrieve进行当前特点任务的专用API补充如`tl.cumsum`、`tl.extra.cuda.libdevice.tanh`等，是较好的行为。
- 全局视角：LangGraph 中，在Coder/MultiCoder前添加`api_recall_node` ，进行API的召回，持久化到本次任务的`log/api_recall`目录中。后续Coder生成的API doc部分直接填入`api_recall_node` 的结果。
- `api_recall_node` ：加载基础 API 文档，并通过 `extract_documented_triton_apis()` 抽取已展示 API。对于Triton召回的API，会调用`render_triton_recall(..., documented_apis=...)` 渲染 database recall block：
    - 已在 `api.md` 出现的 API 进入“已展示”。
    - 未在 `api.md` 出现的 API 进入“补充API”，并展开签名和文档。

### 使用细节

- 使用环境需求
    - `api_recall_node` 节点只对 `dsl` 包含 `triton` 且 `framework=torch` 的任务启用 API database。
    - `api_recall_node` 节点需要确保 Qdrant collection 可用。
- yaml设置：可以通过config的yaml配置改变api database行为
    
    
    | 配置项 | 默认值 / 示例 | 含义 | 调参影响 |
    | --- | --- | --- | --- |
    | `enabled` | `true` | 是否启用 Triton API database。 | 调试静态 API 文档路径时可设为 `false`；正常 Triton CUDA 生成建议开启。 |
    | `qdrant_host` | `"localhost"` | Qdrant 服务地址。 | 如果 Qdrant 不在本机，需要改成对应 host；不可达时会跳过 database recall 并记录状态。 |
    | `qdrant_port` | `6333` | Qdrant HTTP 端口。 | 需要和启动 Qdrant container / service 时暴露的端口一致。 |
    | `triton_collection` | `"triton_api"` | 存放 Triton API 的 Qdrant collection 名称。 | 多版本 Triton 或多 backend 共存时可改名隔离；改名后需要重新 bootstrap。 |
    | `torch_collection` | `"torch_api"` | 存放 PyTorch public API 文档的 Qdrant collection 名称 | 多版本 PyTorch 共存时可改名隔离；缺失时会影响 public API pivot，但 ATen dispatcher 仍可提供 fallback query。 |
    | `target_backend` | `"cuda"` | Triton API 扫描和过滤的目标后端。当前用于保留 CUDA 相关 namespace，并过滤 AMD/HIP/Ascend/NPU 等非目标 backend API。 | CUDA 任务保持 `cuda`；设为 `auto` 会尝试按当前 Triton runtime 检测 backend。 |
    | `embed_model` | `"sentence-transformers/all-MiniLM-L6-v2"` | 用于 query 和 API 文档向量化的 SentenceTransformer 模型。 | 改模型需要确保已安装/缓存，并通常需要重建 collection；模型变化会影响召回排序。 |
    | `embed_cache_folder` | `"python/akg_agents/database/api/embed_model_cache"` | embedding 模型缓存目录。相对路径会按 `akg_agents` repo 根目录解析。 | 离线环境需要提前准备该目录；路径错误会导致模型下载或加载失败。 |
    | `force_rebuild` | `false` | 是否强制同时重建 `triton_collection` 和 `torch_collection`。 | Triton/PyTorch 版本变化、扫描逻辑变化或 collection 污染时开启；会增加启动耗时。 |
    | `force_rebuild_triton` | `false` | 是否只强制重建 Triton API collection。 | 修改 Triton API scanner、切换 Triton 版本或 backend 时使用。 |
    | `force_rebuild_torch` | `false` | 是否只强制重建 PyTorch API collection。 | 修改 PyTorch API 抽取逻辑、切换 PyTorch 版本时使用。 |
    | `topk_per_query` | `32` | 每个 PyTorch/ATen source API 在 Triton collection 中做 embedding 检索时的候选上限。 | 增大可提高召回覆盖率但会增加噪声和渲染长度；减小可降低噪声但可能漏召回。 |
    | `filter_tags` | `["tl"]` | Qdrant 检索时按 tag 过滤候选，默认只保留 Triton language (`tl.*`) 相关 API。 | 放宽过滤可召回 `triton.*` 或 backend API，但也更容易引入不适合 Coder 使用的内部 API。 |
    | `enable_keyword_recall` | `true` | 是否启用 keyword-gated recall。对卷积、pooling、reduction、softmax 等常见 source API，优先保留命中 keyword map 的 Triton API。 | 开启后召回更稳定、噪声更低；关闭后主要依赖 embedding elbow cut。 |
    | `min_keep` | `2` | 每个 source API 至少保留的候选数量。 | 提高可减少漏召回，但会增加不相关候选；降低可让召回更精简。 |
    | `max_keep` | `10` | 每个 source API 最多保留的候选数量。 | 提高可扩大候选面；降低可控制 prompt 长度和噪声。 |
    | `elbow_min_gap_ratio` | `0.15` | embedding fallback 时的 elbow cut 阈值，用于根据相邻分数下降幅度截断候选。 | 值越大截断越激进；值越小保留更多相近候选。 |
    | `relative_decay` | `0.90` | embedding fallback 时的相对分数衰减阈值，低于首个候选分数一定比例后停止保留。 | 值越高越严格，召回更短；值越低越宽松，召回更长。 |
    | `keyword_fallback_qdrant` | `true` | keyword-gated recall 下，如果 embedding topK 没召回某个必要 keyword，是否用 Qdrant exact filter 按 API name 补齐。 | 开启可提高 `tl.arange`、`tl.load`、`tl.store` 等核心 API 的召回稳定性；关闭则完全依赖 embedding topK。 |
    | `keyword_fallback_limit_per_kw` | `4` | 每个 keyword 通过 Qdrant exact fallback 补齐的候选上限。 | 提高可覆盖同名/近名 API；降低可减少重复和 backend 噪声。 |

### 技术细节

- 核心特性：
    - 使用**ATen dispatcher**从 KernelBench 文件中提取出 PyTorch API
    - 通过构建双重数据库，从 PyTorch API 召回到 Triton API。
    - 对基础 API 文档和召回 API 文档去重。
    - 避免 API 文档 LLM 压缩导致关键 API 约束丢失
- **从 KernelBench 文件中提取出 PyTorch API：ATen dispatcher**
    - 实现代码：`aten_dispatch_code_to_doc_list(code, class_name="Model")`
    - PyTorch 的高层 API、`nn.Module`、Tensor method 和 Python operator 最终会落到 PyTorch 底层 dispatcher 中的 ATen operator。例如：
        
        
        | 用户代码 | 典型 ATen op |
        | --- | --- |
        | `nn.Conv2d(...)(x)` | `aten.convolution.default` |
        | `torch.tanh(x)` | `aten.tanh.default` |
        | `x * 2.0` | `aten.mul.Tensor` |
        | `x + bias` | `aten.add.Tensor` |
        | `nn.MaxPool2d(...)(x)` | `aten.max_pool2d_with_indices.default` |
    - 具体过程：
        - 解析 KernelBench 源码 AST并构造 `Model` 模拟执行
        
        ```python
        with torch.no_grad(), FakeTensorMode(allow_non_fake_inputs=True):
            with TorchDispatchMode():
                model(*fake_inputs)
        ```
        
        - `TorchDispatchMode.__torch_dispatch__()` 会拦截每个进入 dispatcher 的 op
    - KernelBench case 82 会得到：
    
    ```python
    torch.ops.aten.convolution.default
    torch.ops.aten.tanh.default
    torch.ops.aten.mul.Tensor
    torch.ops.aten.add.Tensor
    torch.ops.aten.max_pool2d_with_indices.default
    ```
    
- **如何通过 PyTorch API 召回到 Triton API：基于 Qdrant 向量数据库的 RAG**
    - 入口函数：`retrieve_triton_apis()`。
    - database 初始化入口：`ensure_qdrant_databases()`。
    - 当前使用两个 Qdrant collection：
    
    | Collection | 用途 |
    | --- | --- |
    | `triton_api` | 本地 Triton runtime 可导入的 Triton API 文档、签名、tag、source file 等。 |
    | `torch_api` | 本地 PyTorch public API 文档，用作 query pivot 和 AST fallback 的增强信息。 |
    - collection 初始化流程：
        - 连接 Qdrant。
        - 检查 `triton_api` 和 `torch_api` 是否存在且点数足够。
        - 如果 collection 缺失、点数过少，或开启 rebuild flag，则重新扫描并 upsert。
        - 使用 `_BOOTSTRAPPED_KEYS` 记录同一进程中已经检查过的配置，避免 batch 任务重复 rebuild。
    - yaml rebuild 开关：
    
    | 开关 | 含义 |
    | --- | --- |
    | `force_rebuild` | 同时 rebuild `triton_api` 和 `torch_api`。 |
    | `force_rebuild_triton` | 只 rebuild `triton_api`。 |
    | `force_rebuild_torch` | 只 rebuild `torch_api`。 |
    - Triton API database构建过程：
        - Triton API 扫描：
            - 扫描公共 Triton namespace，例如 `triton.language`、`triton.language.math`、`triton.language.standard`、`triton.runtime`、`triton.compiler`、`triton`。
            - 根据 `target_backend` 额外扫描 backend-specific namespace。CUDA 下会保留 `triton.language.extra.cuda.libdevice` 和 `triton.backends.nvidia`。
            - 跳过 generic `tl.extra.libdevice.*`，避免在 CUDA 任务里引入不稳定旧路径。
            - 跳过非目标 backend API，例如 CUDA 任务中不召回 AMD/HIP/Ascend/NPU API。
            - 将 `tl.core.load`、`tl.core.store`、`tl.core.dot` 等 public alias 归一成 `tl.load`、`tl.store`、`tl.dot`。
        - Query 构造：
            - 对每个 source PyTorch/ATen API，优先在 `torch_api` collection 中找到同 canonical 的 payload 作为 pivot。
            - 如果没有 pivot，则使用 ATen dispatcher 抽取出的 schema/doc。
            - query text 由 `redsig` 和 doc 前部文本组成。
            - 会弱化 `torch`、`tensor` 这类偏 PyTorch 表述的词，减少 embedding 偏移。
            - 使用 `sentence-transformers/all-MiniLM-L6-v2` 编码 query，再到 `triton_api` collection 中检索。
- **基础 API 文档与召回 API 去重：**
    - 基础 API 文档路径：`op/resources/docs/triton_cuda_docs/api/api.md`该文档保留稳定 core API 和关键使用约束。它不是完整 Triton API 手册，而是默认总会展示的基础 API block。
    - `extract_documented_triton_apis(doc_text)` 会从基础 API 文档的 markdown 标题中抽取已展示 API。
    - `render_triton_recall()` 接收 `documented_apis` 后，会将召回候选分成不同类别进行不同文本展示。对于不在基础 API 文档中的 API，会展示为补充 API。
        - `displayed_apis`：已在基础 API 文档出现，召回 block 只列名称。
            
            ```markdown
            ### 来源 PyTorch API: `torch.ops.aten.convolution.default`（共 6 条候选）
            已展示: `tl.load`, `tl.store`, `tl.dot`, `tl.arange`, `tl.program_id`, `tl.where`
            补充API: 无
            （本 PyTorch API 的召回候选已由基础 API 文档或前文召回覆盖）
            ```
            
        - `supplemental_apis`：未在基础 API 文档出现，召回 block 展开 tags、签名和文档。
            
            ```markdown
            ### 来源 PyTorch API: `torch.ops.aten.cumsum.default`（共 2 条候选）
            已展示: `tl.cumprod`
            补充API: `tl.cumsum`
            - 1. Triton API: `tl.cumsum` / `cumsum`
              - Tags: ['tl']
              - Triton 函数签名: tl.cumsum(*args, **kwargs)
              - Triton 函数文档: ...
            ```
            

## code checker

### 背景

在当前`coder -> code_checker -> verifier` 生成流中，主要行为为：如果代码运行时失败，则拦截本次运行时报错，交给下轮coder生成。经测试，在部分任务中，出现一次代码生成中出现多处API幻觉，但是运行时报错只暴露了最早期的一处。这会导致LLM调用次数骤增。

### 总览

针对上述问题，选择增强原有code checker node能力。其主要作用是将进入verify前就能发现的很多失败进行拦截，比如出现中文，import不可用等。如果拦截成功，则不进行verify而是直接回退到coder。

增强后CodeChecker 分两层：

- blocking 静态检查（原code checker 功能）：失败时直接回到 Coder。
- non-blocking Triton diagnostics：只写入 `code_diagnostic_*` 字段，不改变路由。和verify的fail log一起交给下轮coder。

这种设计让确定性错误快速修复，同时避免诊断误报导致本来可以通过的代码不能进入 verifier。

### 使用细节

workflow 中默认开启 CodeChecker。如果 `enable_code_checker` 缺省，当前 MathIR workflows 默认按开启处理；需要关闭时显式配置为 `false`。

| YAML 配置项 | 可选值 | 作用 |
| --- | --- | --- |
| `enable_code_checker` | `true` / `false` | 是否在 Coder 后、Verifier 前运行 CodeChecker。关闭后 workflow 直接走 `coder -> verifier`。 |
| `code_checker.base_checkers` | `all` / checker 名称列表 / `[]` | blocking base checker pipeline。任一 checker 返回错误时，workflow 直接回到 Coder，不进入 Verifier。 |
| `code_checker.triton_checkers` | `all` / checker 名称列表 / `[]` | non-blocking Triton diagnostic pipeline。诊断结果写入 `code_diagnostic_*`，但不阻止 Verifier。 |
| `code_diagnostic_checker.enabled` | `true` / `false` | 是否运行非阻塞 Triton 诊断检查。关闭后 `triton_checkers` 不产生诊断结果。 |
| `code_diagnostic_checker.only_errors` | `true` / `false` | 是否只保留 `ERROR` 级别诊断。 |
| `code_diagnostic_checker.dedup` | `true` / `false` | 是否对诊断结果按规则、位置和消息去重。 |

`base_checkers` 可选 checker：

| Checker 名称 | 功能 |
| --- | --- |
| `empty_code` | 检查生成代码是否为空。 |
| `python_syntax` | 使用 `ast.parse` 检查 Python 语法。 |
| `py_compile` | 使用 `py_compile` 检查编译期错误。 |
| `import_availability` | 检查 import 的顶层模块是否可用。 |
| `stray_chinese` | 检查注释和字符串外是否混入中文说明文本。 |
| `triton_dsl_compliance` | 检查 Triton DSL 合规性和明显 torch API 替代 kernel 的情况。 |

`triton_checkers` 可选 checker：

| Checker 名称 | 功能 |
| --- | --- |
| `api_signature` | 检查 Triton API 是否存在、关键字参数和位置参数数量是否匹配当前环境。 |
| `high_confidence_semantics` | 检查高置信 Triton 语义风险，例如动态 shape 分配、不支持控制流、重复 kernel 参数等。 |

Coder 会收到这些信息：

- `code_check_errors`：blocking 静态检查错误。
- `code_diagnostic_errors`：非阻塞 Triton 诊断信息。
- `verifier_error`：Verifier runtime/correctness/performance 错误。

### 技术细节

主要实现文件：

- `python/akg_agents/op/utils/code_checker/`
- `python/akg_agents/op/utils/code_checker/base.py`
- `python/akg_agents/op/utils/code_checker/registry.py`
- `python/akg_agents/op/utils/code_checker/base_checkers/`
- `python/akg_agents/op/utils/code_checker/triton_checkers/`
- `python/akg_agents/op/langgraph_op/nodes.py`

blocking 检查流程为原有code checker功能，基于当前代码进行了oop风格重构，保证功能不变。同时，在code checker路径下保留有skills.md，可以指导LLM直接按当前接口注册新的checker到yaml可选配置。

## Conductor

原CoderOnly算子生成中，Conductor会根据历史情况进行全局考虑，并选择不同错误处理类型。但是由于模型欠缺Triton能力提示并且只得到部分任务生成说明文本，选择不同错误处理类型提示简略，往往Conductor会误判，导致输出干扰提示或提前终止任务，因此，在当前workflow中，去掉了verify→ conductor流。但是将全局的历史错误交给生成过程这一行为也不可或缺，所以将这一行为移动到了coder的history_attempts字段。 

- 某次生成过程中的误判thinking log
    
    ```python
    我决定输出finish，因为超时属于B类错误。
    
    但是，需注意任务描述中还有：“如果选择继续生成代码，为下个Agent提供精确、合理的提示，要有具体的指导意见；” 如果我们选择finish，就不需要了。
    
    我选择finish。
    ```
    

## 其他

- Step 计数：
    
    
    | 节点 | step 消耗 | 说明 |
    | --- | --- | --- |
    | `mathIR` preset 命中 | `+0` | 使用 `write_record` 写日志，视为 cache hit。 |
    | `mathIR` LLM 生成 | `+1` | 使用 `log_record`，真实消耗一次生成预算。 |
    | `api_recall` 无 LLM 压缩 | `+0` | 本地检索、渲染、落盘。 |
    | `api_recall` 发生 LLM 压缩 | `+1` | 只有实际调用压缩 LLM 时计数。 |
    | `coder` | `+1` | 每次生成或修复代码。 |
    | `multi_expr_coder` | `+1` | 一次高层多 expression 生成/合并。 |
    | `code_checker` | `+0` | 静态检查和非阻塞诊断只写日志。 |
    | `verifier` | `+1` | 每次设备验证。 |

# 结果与分析

## TritonCuda

- Pass@1正确率：
    - 原benchmark：沿用reproduce/buaa-ir，为88
    - mathir_coder_workflow正确率：首测100，复测97
- attn生成：在akg_agents/thirdparty中新增了attention_bench。成功生成3种attn。
- 正确率分析：
    - 当前正确率的摆动主要在于生成质量/修复未收敛问题。当前的权重初始化代码极不稳定，复测失败代码的原因主要在于错误的进行了权重初始化，如多余初始化导致消耗了随机种子，device="cuda”在cuda侧进行初始化导致没有使用CPU侧的随机种子。

## TritonAscend

- Pass@1正确率：
    - 原benchmark：沿用reproduce/buaa-ir，为69
    - mathir_multi_kernel_gen_workflow正确率：85
- attn生成：在akg_agents/thirdparty中新增了attention_bench。共pass-k=1生成两次。第一次生成page&radix，第二次生成flash&page。成功生成3种attn。
- 正确率分析：在triton Ascend 3.2.0版本中，以下问题导致成功率下降
    - 错误有用信息少：部分算子生成结果的报错返回的错误信息少，导致模型难以提取出有效修复方案。比如只指向 /tmp/.../kernel.ttadapter.mlir:3:3；或者直接 EZ9999/EE9999 vector core exception、timeout or trap error。模型很难知道是哪个 tl.load、哪个 block shape、哪个 mask 或哪个 index 公式导致的。
        
        ```python
        [ERROR] 2026-06-14-19:45:32 (PID:1557974, Device:0, RankID:-1) ERR99999 UNKNOWN applicaiton exception
        
        /home/lutao/env/anaconda3/envs/aikg_fork/lib/python3.11/site-packages/torch_npu/utils/collect_env.py:58: UserWarning: Warning: The /usr/local/Ascend/cann-8.5.0 owner does not match the current owner.
          warnings.warn(f"Warning: The {path} owner does not match the current owner.")
        /home/lutao/env/anaconda3/envs/aikg_fork/lib/python3.11/site-packages/torch_npu/utils/collect_env.py:58: UserWarning: Warning: The /usr/local/Ascend/cann-8.5.0/aarch64-linux/ascend_ops_install.info owner does not match the current owner.
          warnings.warn(f"Warning: The {path} owner does not match the current owner.")
        /home/lutao/env/anaconda3/envs/aikg_fork/lib/python3.11/site-packages/torch_npu/utils/collect_env.py:58: UserWarning: Warning: The /usr/local/Ascend/cann-8.5.0 owner does not match the current owner.
          warnings.warn(f"Warning: The {path} owner does not match the current owner.")
        /home/lutao/env/anaconda3/envs/aikg_fork/lib/python3.11/site-packages/torch_npu/utils/collect_env.py:58: UserWarning: Warning: The /usr/local/Ascend/cann-8.5.0/aarch64-linux/ascend_ops_install.info owner does not match the current owner.
          warnings.warn(f"Warning: The {path} owner does not match the current owner.")
        [W614 19:45:30.581203610 compiler_depend.ts:57] Warning: EZ9999: Inner Error!
        EZ9999[PID: 1557974] 2026-06-14-19:45:29.794.287 (EZ9999):  The error from device(chipId:0, dieId:0), serial number is 67, there is an exception of aivec error, core id is 43, error code = 0x800000, dump info: pc start: 0x124000000000, current: 0x124000000784, vec error info: 0x100000037, mte error info: 0x103060f71, ifu error info: 0x212c17ffffe00, ccu error info: 0x41f0280400000000, cube error info: 0, biu error info: 0, aic error mask: 0x6500020bd00028c, para base: 0x12c100400000.[FUNC:PrintCoreInfo][FILE:device_error_core_proc.cc][LINE:347]
                TraceBack (most recent call last):
               The extend info: errcode:(0x800000, 0, 0) errorStr: The DDR address of the MTE instruction is out of range. fixp_error0 info: 0x3060f71, fixp_error1 info: 0x1, fsmId:1, tslot:4, thread:0, ctxid:0, blk:14, sublk:0, subErrType:4.[FUNC:PrintCoreInfo][FILE:device_error_core_proc.cc][LINE:360]
               The error from device(chipId:0, dieId:0), serial number is 67, there is an exception of aivec error, core id is 44, error code = 0x800000, dump info: pc start: 0x124000000000, current: 0x124000000784, vec error info: 0x100000037, mte error info: 0x103073f71, ifu error info: 0x212c1c0200000, ccu error info: 0x40c24d0800000000, cube error info: 0, biu error info: 0, aic error mask: 0x6500020bd00028c, para base: 0x12c100400000.[FUNC:PrintCoreInfo][FILE:device_error_core_proc.cc][LINE:347]
               The extend info: errcode:(0x800000, 0, 0) errorStr: The DDR address of the MTE instruction is out of range. fixp_error0 info: 0x3073f71, fixp_error1 info: 0x1, fsmId:1, tslot:4, thread:0, ctxid:0, blk:15, sublk:0, subErrType:4.[FUNC:PrintCoreInfo][FILE:device_error_core_proc.cc][LINE:360]
               Kernel task happen error, retCode=0x31, [vector core exception].[FUNC:PreCheckTaskErr][FILE:davinci_kernel_task.cc][LINE:1493]
               rtStreamSynchronize execution failed, reason=vector core exception[FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:61]
               synchronize stream failed, runtime result = 507035[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:148]
         (function copy_between_host_and_device_opapi)
        Traceback (most recent call last):
          File "/home/lutao/workspaces/aikg/aikg_fork/akg_agents/logs/Task_36c0szuf/aikg_kernelbench_35_GroupNorm_/Iteration35_attempt1_Step01_verify/verify_aikg_kernelbench_35_GroupNorm_.py", line 553, in <module>
            verify_implementations()
          File "/home/lutao/workspaces/aikg/aikg_fork/akg_agents/logs/Task_36c0szuf/aikg_kernelbench_35_GroupNorm_/Iteration35_attempt1_Step01_verify/verify_aikg_kernelbench_35_GroupNorm_.py", line 500, in verify_implementations
            verify_result, framework_output = verify_with_timeout(
                                              ^^^^^^^^^^^^^^^^^^^^
          File "/home/lutao/workspaces/aikg/aikg_fork/akg_agents/logs/Task_36c0szuf/aikg_kernelbench_35_GroupNorm_/Iteration35_attempt1_Step01_verify/verify_aikg_kernelbench_35_GroupNorm_.py", line 434, in verify_with_timeout
            verify_result, framework_output = verify_case()
                                              ^^^^^^^^^^^^^
          File "/home/lutao/workspaces/aikg/aikg_fork/akg_agents/logs/Task_36c0szuf/aikg_kernelbench_35_GroupNorm_/Iteration35_attempt1_Step01_verify/verify_aikg_kernelbench_35_GroupNorm_.py", line 70, in wrapper
            result = func(*args, **kwargs)
                     ^^^^^^^^^^^^^^^^^^^^^
          File "/home/lutao/workspaces/aikg/aikg_fork/akg_agents/logs/Task_36c0szuf/aikg_kernelbench_35_GroupNorm_/Iteration35_attempt1_Step01_verify/verify_aikg_kernelbench_35_GroupNorm_.py", line 431, in verify_case
            return verify_single_case(inputs_for_framework, inputs_for_impl, reference_output)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          File "/home/lutao/workspaces/aikg/aikg_fork/akg_agents/logs/Task_36c0szuf/aikg_kernelbench_35_GroupNorm_/Iteration35_attempt1_Step01_verify/verify_aikg_kernelbench_35_GroupNorm_.py", line 422, in verify_single_case
            compare(fw_out, impl_out, limit, data_type)
          File "/home/lutao/workspaces/aikg/aikg_fork/akg_agents/logs/Task_36c0szuf/aikg_kernelbench_35_GroupNorm_/Iteration35_attempt1_Step01_verify/verify_aikg_kernelbench_35_GroupNorm_.py", line 112, in compare
            fw_flat = fw_out.flatten().detach().cpu()
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        RuntimeError: ACL stream synchronize failed, error code:507035
        [W614 19:45:30.602900611 compiler_depend.ts:545] Warning: NPU warning, error code is 507035[Error]: 
        [Error]: The vector core execution is abnormal. 
                Rectify the fault based on the error information in the ascend log.
        EE9999: Inner Error!
        EE9999[PID: 1557974] 2026-06-14-19:45:30.818.274 (EE9999):  rtDeviceSynchronizeWithTimeout execution failed, reason=vector core exception[FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:61]
                TraceBack (most recent call last):
               wait for compute device to finish failed, runtime result = 507035.[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:148]
         (function npuSynchronizeUsedDevices)
        [W614 19:45:30.603659446 compiler_depend.ts:527] Warning: NPU warning, error code is 507035[Error]: 
        [Error]: The vector core execution is abnormal. 
                Rectify the fault based on the error information in the ascend log.
        EE9999: Inner Error!
        EE9999[PID: 1557974] 2026-06-14-19:45:30.819.320 (EE9999):  rtDeviceSynchronizeWithTimeout execution failed, reason=vector core exception[FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:61]
                TraceBack (most recent call last):
               wait for compute device to finish failed, runtime result = 507035.[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:148]
         (function npuSynchronizeDevice)
        
        ```
        
    - 文档的限制严苛：在原始文档中，grid数量被严格说明限制在65535之内，对于大规模程序，每个grid处理的数据极易超过芯片core所拥有的UB buffer大小（KB级别）。普遍的修复是将程序派分给更多的grid进行处理，但是文档要求的65535 grid数量与之又产生了冲突，导致反复修复UB buffer overflow失败。
    - UB/L0/多缓冲资源约束是非线性的，模型难以估算日志里大量 ub overflow, requires ... bits while 1572864 bits available。但实际占用不只是输入/输出 tile，还包括 mask、中间 tensor、accumulator、auto multi-buffer、临时 reshape/broadcast。模型常按表面 BLOCK_SIZE * dtype_size 估，结果低估后端真实 buffer 占用。

# 总结

本工作把 kernel 生成从黑盒端到端试错，改造成“结构化语义表示 + 局部 TDD 验证 + API 环境召回 + 静态诊断修复”的闭环系统。MathIR 解决复杂算子语义表达和跨后端复用问题，multi-kernel generation 解决大 kernel 一次性生成困难问题，Triton API database 解决 API 版本和环境不对齐问题，CodeChecker 解决单次报错信息不足导致的低效修复问题。CUDA 结果已证明该路线能显著提升 Pass@1；Ascend 方向的主要后续工作则应集中在更精细的错误定位、UB/L0 资源建模、grid/tiling 策略约束，以及对 NPU Triton lowering 行为的专门知识库建设。