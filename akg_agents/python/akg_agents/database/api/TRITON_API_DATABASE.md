## Triton API database

### 背景

- Triton API 召回数据库：应对静态文档无法满足 Triton API 多样化的需求，在 kernel 生成前使用本地 Triton + PyTorch 包构建本地 API 数据库，之后的请求通过本地 API 数据库召回进行实现，有效减少弱模型的 API 幻觉问题。
- 静态 `api/api.md` 仍保留一组稳定 core API 约束，例如 `tl.load`、`tl.store`、`tl.arange`、`tl.dot`、`tl.constexpr` 等；database recall 负责补充非 core API 或任务相关 API。
- 当前 prompt 中 API 文档被拆成两层：
  - 基础 API 文档：来自 `op/resources/docs/triton_cuda_docs/api/api.md`。
  - Triton API database 召回：按当前 PyTorch/ATen source API 召回到 Triton API。
- 召回渲染时会识别基础 API 文档中已经展示过的 API，同名 API 不重复展开，只在召回 block 中标记为“已展示”；未在基础文档出现的 API 会作为“补充API”展开签名和文档。

### 总览

当前流程下的 Triton API database 如下：

- 设置 yaml 中的 enable：

```yaml
api_database:
  enabled: true
  qdrant_host: "localhost"
  qdrant_port: 6333
  triton_collection: "triton_api"
  torch_collection: "torch_api"
  target_backend: "cuda"
  embed_model: "sentence-transformers/all-MiniLM-L6-v2"
  embed_cache_folder: "python/akg_agents/database/api/embed_model_cache"
  force_rebuild: false
  force_rebuild_triton: false
  force_rebuild_torch: false
  topk_per_query: 32
  filter_tags: ["tl"]
  enable_keyword_recall: true
  min_keep: 2
  max_keep: 10
  elbow_min_gap_ratio: 0.15
  relative_decay: 0.90
  keyword_fallback_qdrant: true
  keyword_fallback_limit_per_kw: 4
```

| 配置项 | 默认值 / 示例 | 含义 | 调参影响 |
| --- | --- | --- | --- |
| `enabled` | `true` | 是否启用 Triton API database。关闭后不做 PyTorch/ATen API 抽取、Qdrant 检索和召回文档注入。 | 调试静态 API 文档路径时可设为 `false`；正常 Triton CUDA 生成建议开启。 |
| `qdrant_host` | `"localhost"` | Qdrant 服务地址。 | 如果 Qdrant 不在本机，需要改成对应 host；不可达时会跳过 database recall 并记录状态。 |
| `qdrant_port` | `6333` | Qdrant HTTP 端口。 | 需要和启动 Qdrant container / service 时暴露的端口一致。 |
| `triton_collection` | `"triton_api"` | 存放 Triton API 文档、签名、tag、source 信息的 Qdrant collection 名称。 | 多版本 Triton 或多 backend 共存时可改名隔离；改名后需要重新 bootstrap。 |
| `torch_collection` | `"torch_api"` | 存放 PyTorch public API 文档的 Qdrant collection 名称，用于从 PyTorch API 找 query pivot。 | 多版本 PyTorch 共存时可改名隔离；缺失时会影响 public API pivot，但 ATen dispatcher 仍可提供 fallback query。 |
| `target_backend` | `"cuda"` | Triton API 扫描和过滤的目标后端。当前用于保留 CUDA 相关 namespace，并过滤 AMD/HIP/Ascend/NPU 等非目标 backend API。 | CUDA 任务保持 `cuda`；设为 `auto` 会尝试按当前 Triton runtime 检测 backend。 |
| `embed_model` | `"sentence-transformers/all-MiniLM-L6-v2"` | 用于 query 和 API 文档向量化的 SentenceTransformer 模型。 | 改模型需要确保已安装/缓存，并通常需要重建 collection；模型变化会影响召回排序。 |
| `embed_cache_folder` | `"python/akg_agents/database/api/embed_model_cache"` | embedding 模型缓存目录。相对路径会按 `akg_agents` repo 根目录解析。 | 离线环境需要提前准备该目录；路径错误会导致模型下载或加载失败。 |
| `force_rebuild` | `false` | 是否强制同时重建 `triton_collection` 和 `torch_collection`。 | Triton/PyTorch 版本变化、扫描逻辑变化或 collection 污染时开启；会增加启动耗时。 |
| `force_rebuild_triton` | `false` | 是否只强制重建 Triton API collection。 | 修改 Triton API scanner、切换 Triton 版本或 backend 时使用。 |
| `force_rebuild_torch` | `false` | 是否只强制重建 PyTorch API collection。 | 修改 PyTorch API 抽取逻辑、切换 PyTorch 版本时使用。 |
| `topk_per_query` | `32` | 每个 PyTorch/ATen source API 在 Triton collection 中做 embedding 检索时的候选上限。 | 增大可提高召回覆盖率但会增加噪声和渲染长度；减小可降低噪声但可能漏召回。 |
| `filter_tags` | `["tl"]` | Qdrant 检索时按 tag 过滤候选，默认只保留 Triton language (`tl.*`) 相关 API。 | 放宽过滤可召回 `triton.*` 或 backend API，但也更容易引入不适合 Coder 使用的内部 API。 |
| `enable_keyword_recall` | `true` | 是否启用 keyword-gated recall。对卷积、pooling、reduction、softmax 等常见 source API，优先保留命中 keyword map 的 Triton API。 | 开启后召回更稳定、噪声更低；关闭后主要依赖 embedding elbow cut。 |
| `min_keep` | `2` | 每个 source API 至少保留的候选数量。 | 提高可减少漏召回，但会增加不相关候选；降低可让召回更精简。 |
| `max_keep` | `10` | 每个 source API 最多保留的候选数量。 | 提高可扩大候选面；降低可控制 prompt 长度和噪声。 |
| `elbow_min_gap_ratio` | `0.15` | embedding fallback 时的 elbow cut 阈值，用于根据相邻分数下降幅度截断候选。 | 值越大截断越激进；值越小保留更多相近候选。 |
| `relative_decay` | `0.90` | embedding fallback 时的相对分数衰减阈值，低于首个候选分数一定比例后停止保留。 | 值越高越严格，召回更短；值越低越宽松，召回更长。 |
| `keyword_fallback_qdrant` | `true` | keyword-gated recall 下，如果 embedding topK 没召回某个必要 keyword，是否用 Qdrant exact filter 按 API name 补齐。 | 开启可提高 `tl.arange`、`tl.load`、`tl.store` 等核心 API 的召回稳定性；关闭则完全依赖 embedding topK。 |
| `keyword_fallback_limit_per_kw` | `4` | 每个 keyword 通过 Qdrant exact fallback 补齐的候选上限。 | 提高可覆盖同名/近名 API；降低可减少重复和 backend 噪声。 |

- 设置 API 文档压缩开关。默认关闭 LLM 压缩，避免 core API 约束被压缩丢失：

```yaml
api_docs:
  llm_compress_enabled: false
  llm_compress_threshold: 24000
```

- LangGraph 中的 `api_recall_node` 会在 Coder 之前运行。该节点只对 `dsl` 包含 `triton` 且 `framework=torch` 的任务启用 API database。
- `api_recall_node` 调用 `maybe_retrieve_and_store_triton_apis()`：
  - 确保 Qdrant collection 可用。
  - 从 KernelBench 代码中抽取 PyTorch/ATen API。
  - 根据 PyTorch/ATen API 召回 Triton API。
  - 写入 `task_info["triton_api_recall_by_source"]` 和 `task_info["triton_api_recall"]`。
- `api_recall_node` 加载基础 API 文档，并通过 `extract_documented_triton_apis()` 抽取已展示 API。
- `render_triton_recall(..., documented_apis=...)` 渲染 database recall block：
  - 已在 `api.md` 出现的 API 进入“已展示”。
  - 未在 `api.md` 出现的 API 进入“补充API”，并展开签名和文档。
- `compose_api_docs_block(base_api_docs, recall_block)` 拼接最终 API 文档。
- 最终文档落盘到：

```text
<log_dir>/<op_name>/api_recall/api_recall_rendered.md
<log_dir>/<op_name>/api_recall/api_recall_structured.json
```

- Coder 使用 `task_info["api_recall_docs_path"]` 直接读取 `api_recall_rendered.md`，因此主流程不会再触发 `ApiDocsAgent` 的 LLM 压缩。
- `ApiDocsAgent` 仍保留兼容路径；只有 `api_docs.llm_compress_enabled=true` 且文档长度超过 `llm_compress_threshold` 时才会进行 LLM 压缩。

### 技术细节

- 需要解决以下问题：
  - 如何从 KernelBench 文件中提取出 PyTorch API。
  - 如何通过 PyTorch API 召回到 Triton API。
  - 如何让基础 API 文档和召回 API 文档去重。
  - 如何避免 API 文档 LLM 压缩导致关键 API 约束丢失。

#### 从 KernelBench 文件中提取出 PyTorch API：ATen dispatcher

- 实现代码：`aten_dispatch_code_to_doc_list(code, class_name="Model")`。
- PyTorch 的高层 API、`nn.Module`、Tensor method 和 Python operator 最终会落到 PyTorch 底层 dispatcher 中的 ATen operator。例如：

| 用户代码 | 典型 ATen op |
| --- | --- |
| `nn.Conv2d(...)(x)` | `aten.convolution.default` |
| `torch.tanh(x)` | `aten.tanh.default` |
| `x * 2.0` | `aten.mul.Tensor` |
| `x + bias` | `aten.add.Tensor` |
| `nn.MaxPool2d(...)(x)` | `aten.max_pool2d_with_indices.default` |

- 具体过程：
  - 解析 KernelBench 源码 AST。
  - 在受限 namespace 中执行源码，找到 `Model`、`get_init_inputs()` 和 `get_inputs()`。
  - 构造 `Model` 实例，并用真实输入的 shape、stride、dtype 构造 meta/fake tensor。
  - 在 `FakeTensorMode` 和 `TorchDispatchMode` 下模拟执行 `Model.forward()`：

```python
with torch.no_grad(), FakeTensorMode(allow_non_fake_inputs=True):
    with TorchDispatchMode():
        model(*fake_inputs)
```

  - `TorchDispatchMode.__torch_dispatch__()` 会拦截每个进入 dispatcher 的 op：

```python
class _RecorderMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        recorder.ops.append(func)
        return func(*args, **(kwargs or {}))
```

- KernelBench case 82 会得到：

```python
torch.ops.aten.convolution.default
torch.ops.aten.tanh.default
torch.ops.aten.mul.Tensor
torch.ops.aten.add.Tensor
torch.ops.aten.max_pool2d_with_indices.default
```

- ATen dispatcher 抽取的优势：
  - 能覆盖 `nn.Module`、functional API、Tensor method、Python operator。
  - 能看到 forward 实际执行到的底层 op。
  - 比 AST public API 抽取更接近真实计算语义。

- ATen dispatcher 抽取的限制：
  - 只覆盖 fake/meta forward 实际执行路径，不覆盖未执行到的 data-dependent branch。
  - 某些 PyTorch op 如果没有 fake/meta kernel，可能抽取失败。
  - 抽取失败时 fallback 到 `pytorch_code_to_doc_list(code, only_in_forward=True, class_name="Model")`。

#### 如何通过 PyTorch API 召回到 Triton API：基于 Qdrant 向量数据库的 RAG

- 入口函数：`retrieve_triton_apis()`。
- database 初始化入口：`ensure_qdrant_databases()`。
- 当前使用两个 Qdrant collection：

| Collection | 用途 |
| --- | --- |
| `triton_api` | 本地 Triton runtime 可导入的 Triton API 文档、签名、tag、source file 等。 |
| `torch_api` | 本地 PyTorch public API 文档，用作 query pivot 和 AST fallback 的增强信息。 |

- collection 初始化流程：
  - 连接 Qdrant。
  - 检查 `triton_api` 和 `torch_api` 是否存在且点数足够。
  - 如果 collection 缺失、点数过少，或开启 rebuild flag，则重新扫描并 upsert。
  - 使用 `_BOOTSTRAPPED_KEYS` 记录同一进程中已经检查过的配置，避免 batch 任务重复 rebuild。

- rebuild 开关：

| 开关 | 含义 |
| --- | --- |
| `force_rebuild` | 同时 rebuild `triton_api` 和 `torch_api`。 |
| `force_rebuild_triton` | 只 rebuild `triton_api`。 |
| `force_rebuild_torch` | 只 rebuild `torch_api`。 |

- 只 rebuild Triton API collection：

```bash
cd <workspace>/aikg_fork/akg_agents

PYTHONPATH=python \
python \
  -m akg_agents.database.api.api_db_bootstrap \
  --force-rebuild-triton \
  --target-backend cuda
```

- Triton API 扫描：
  - 扫描公共 Triton namespace，例如 `triton.language`、`triton.language.math`、`triton.language.standard`、`triton.runtime`、`triton.compiler`、`triton`。
  - 根据 `target_backend` 额外扫描 backend-specific namespace。CUDA 下会保留 `triton.language.extra.cuda.libdevice` 和 `triton.backends.nvidia`。
  - 跳过 generic `tl.extra.libdevice.*`，避免在 CUDA 任务里引入不稳定旧路径。
  - 跳过非目标 backend API，例如 CUDA 任务中不召回 AMD/HIP/Ascend/NPU API。
  - 将 `tl.core.load`、`tl.core.store`、`tl.core.dot` 等 public alias 归一成 `tl.load`、`tl.store`、`tl.dot`。

- Query 构造：
  - 对每个 source PyTorch/ATen API，优先在 `torch_api` collection 中找到同 canonical 的 payload 作为 pivot。
  - 如果没有 pivot，则使用 ATen dispatcher 抽取出的 schema/doc。
  - query text 由 `redsig` 和 doc 前部文本组成。
  - 会弱化 `torch`、`tensor` 这类偏 PyTorch 表述的词，减少 embedding 偏移。
  - 使用 `sentence-transformers/all-MiniLM-L6-v2` 编码 query，再到 `triton_api` collection 中检索。

- Keyword-gated recall：
  - 纯 embedding topK 容易召回相近但无用的 API，因此当前实现使用 keyword map 约束高频 op。
  - 例如：

```python
"convolution": ["program_id", "arange", "load", "store", "dot", "where"]
"max_pool2d_with_indices": ["program_id", "arange", "load", "store", "max", "maximum", "where"]
"add": ["add"]
"mul": ["mul"]
"tanh": ["exp", "tanh"]
"softmax": ["exp", "max", "sum"]
"layernorm": ["rsqrt", "sqrt", "sum"]
```

  - 如果 source leaf 有 keyword map，则优先保留命中 keyword 的候选。
  - 如果某个 keyword 没被 embedding 召回到，会用 Qdrant exact filter 按 `name` 补齐。
  - 如果 source 没有 keyword map 或没有 keyword 命中，才 fallback 到 embedding elbow cut。
  - 同一个 keyword 下优先级为：`tl.* public API > tl.core.* > backend libdevice > other`。

- case 82 的典型召回：

```text
torch.ops.aten.convolution.default:
  - tl.load
  - tl.store
  - tl.dot
  - tl.arange
  - tl.program_id
  - tl.where

torch.ops.aten.max_pool2d_with_indices.default:
  - tl.store
  - tl.load
  - tl.maximum
  - tl.where
  - tl.arange
  - tl.max
  - tl.program_id

torch.ops.aten.mul.Tensor:
  - tl.core.mul

torch.ops.aten.tanh.default:
  - tl.exp
  - tl.extra.cuda.libdevice.tanh
```

#### 基础 API 文档与召回 API 去重

- 基础 API 文档路径：

```text
op/resources/docs/triton_cuda_docs/api/api.md
```

- 该文档保留稳定 core API 和关键使用约束。它不是完整 Triton API 手册，而是默认总会展示的基础 API block。
- `extract_documented_triton_apis(doc_text)` 会从基础 API 文档的 markdown 标题中抽取已展示 API，例如：

```text
### tl.load(...)
### tl.store(...)
### tl.arange(...)
### tl.dot(...)
```

- `render_triton_recall()` 接收 `documented_apis` 后，会将召回候选分成：
  - `displayed_apis`：已在基础 API 文档出现，召回 block 只列名称。
  - `supplemental_apis`：未在基础 API 文档出现，召回 block 展开 tags、签名和文档。

- 当前 `triton_doc_gen_by_source.j2` 的展示形式：

```markdown
### 来源 PyTorch API: `torch.ops.aten.convolution.default`（共 6 条候选）
已展示: `tl.load`, `tl.store`, `tl.dot`, `tl.arange`, `tl.program_id`, `tl.where`
补充API: 无
（本 PyTorch API 的召回候选已由基础 API 文档或前文召回覆盖）
```

- 对于不在基础 API 文档中的 API，会展示为补充 API。例如 `tl.cumsum` 已从 base API 文档移除，`torch.ops.aten.cumsum.default` 的 recall 会渲染为：

```markdown
### 来源 PyTorch API: `torch.ops.aten.cumsum.default`（共 2 条候选）
已展示: `tl.cumprod`
补充API: `tl.cumsum`
- 1. Triton API: `tl.cumsum` / `cumsum`
  - Tags: ['tl']
  - Triton 函数签名: tl.cumsum(*args, **kwargs)
  - Triton 函数文档: ...
```

#### Prompt 接入与日志

- `api_recall_node` 中的核心流程：

```python
maybe_retrieve_and_store_triton_apis(...)
base_api_docs = await NodeFactory._load_coder_api_docs_for_recall(...)
filtered_base_api_docs = filter_api_doc_blocks(base_api_docs)
documented_triton_apis = extract_documented_triton_apis(filtered_base_api_docs)
recall_block = render_triton_recall(
    task_info,
    verify_runtime=True,
    documented_apis=documented_triton_apis,
)
rendered = compose_api_docs_block(filtered_base_api_docs, recall_block)
persist_api_recall_artifacts(...)
```

- `api_recall_structured.json` 保存结构化数据：
  - `api_database_enabled`
  - `api_database_status`
  - `api_database_source_kind`
  - `api_database_source_apis`
  - `documented_triton_apis`
  - `triton_api_recall`
  - `triton_api_recall_by_source`
  - `rendered_docs_path`

- `api_recall_rendered.md` 保存最终注入 prompt 的 API 文档。
- `Coder._generate_api_docs()` 优先读取 `task_info["api_recall_docs_path"]` 指向的 rendered markdown，因此主流程直接使用落盘文档，不再二次生成或压缩。
- `KernelGen` 路径会加载 `aggregated_api_docs`，然后同样用 `extract_documented_triton_apis()` 对 database recall 做“已展示/补充API”去重。

#### API 文档压缩控制

- 兼容路径 `ApiDocsAgent.generate()` 仍保留 LLM 压缩逻辑，但默认关闭。
- 配置项：

```yaml
api_docs:
  llm_compress_enabled: false
  llm_compress_threshold: 24000
```

- 只有同时满足以下条件才会触发压缩：
  - `api_docs.llm_compress_enabled=true`
  - `len(combined_api_docs) > api_docs.llm_compress_threshold`
  - `base_for_prompt` 非空

- 关闭压缩的原因：
  - 基础 API 文档中包含 core API 使用约束。
  - LLM 压缩可能删掉 `tl.arange` 2 的幂、`tl.dot input_precision`、`tl.static_range constexpr` 等关键约束。
  - 当前主流程通过 `api_recall_rendered.md` 直接注入，通常不需要这一步压缩。

#### 运行与调试

- 当前环境推荐在运行前加载：

```bash
source <workspace>/aikg_fork/akg_agents/env.sh
```

- 如果不希望 SentenceTransformer 访问网络，可设置：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

- 单 task 运行后可检查：

```text
<log_dir>/<op_name>/api_recall/api_recall_structured.json
<log_dir>/<op_name>/api_recall/api_recall_rendered.md
Iteration*_api_recall_summary.txt
Iteration*_api_recall_status.txt
```

- 典型状态：

```json
{
  "enabled": true,
  "source_kind": "aten_dispatch",
  "source_apis": ["torch.ops.aten.cumsum.default"],
  "recall_sources": ["torch.ops.aten.cumsum.default"],
  "recall_flat_count": 2
}
```

#### 已知限制

- ATen dispatcher 只能记录当前 fake/meta forward 执行路径。
- 某些 PyTorch op 没有 fake/meta kernel 时会 fallback 到 AST 抽取。
- Keyword map 是工程约束，不是完整 compiler lowering 规则，新 op 类型需要逐步补充。
- Database recall 是提示信息，不是强制 API whitelist；Coder 仍需要结合 MathIR、shape、硬件和 verifier 反馈判断是否使用。
- 基础 API 文档应只保留高频稳定 core API；可以通过历史 log 证明稳定召回的 API，应逐步迁移到 database recall 的“补充API”中。
