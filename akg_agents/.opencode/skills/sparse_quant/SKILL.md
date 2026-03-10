---
name: sparse-quantization
description: "【仅适用于 vLLM-MindSpore，不适用于 PTA/vLLM-Ascend 等其他框架】稀疏量化模型加载与适配 - 在 vLLM-MindSpore 中加载稀疏量化模型、适配代码实现或修复常见 Bug。Triggers: '稀疏量化', '加载稀疏量化模型', 'W8A8SC', 'sparse quantization', '稀疏量化适配', '稀疏量化加载', 'sparse-quant'。"
---

# 稀疏量化模型加载与适配 Skill

<role>
你是一个专业的推理框架工程师，擅长在 vLLM-MindSpore 框架中调通稀疏量化（W8A8SC）模型，包括加载流程梳理、代码实现先验知识应用和常见 Bug 修复。
</role>

---

## 背景知识

vLLM-MindSpore 基于 vLLM 二次开发适配华为昇腾 NPU，稀疏量化（W8A8SC）是一种在 **310P** 推理机上支持的量化方案，使用 **Golden Stick** 量化工具链生成量化模型。

**实现依据**：vllm_mindspore PR !1428（W8A8SC 基础设施）、!1490（310P/910 兼容、权重加载优化、测试样例）

| Rule | Value |
|------|-------|
| 量化格式 | W8A8SC（weight 8bit, activation 8bit, sparse compressed） |
| 支持推理机 | 310P（910 不支持） |
| 核心文件 | `vllm_mindspore/.../golden_stick/` |
| 量化工具 | Golden Stick（msModelSlim） |

**设计方案**：vLLM-MindSpore 官方稀疏量化设计文档 → [Native 模型支持稀疏量化设计方案与代码改动说明](https://gitcode.com/mindspore/vllm-mindspore/wiki/Design%20Documents%2FNative%E6%A8%A1%E5%9E%8B%E6%94%AF%E6%8C%81%E7%A8%80%E7%96%8F%E9%87%8F%E5%8C%96%E8%AE%BE%E8%AE%A1%E6%96%B9%E6%A1%88%E4%B8%8E%E4%BB%A3%E7%A0%81%E6%94%B9%E5%8A%A8%E8%AF%B4%E6%98%8E.md)

---

## 加载流程架构

<fetch>
核心文件清单：

```
文件路径                                                    | 作用                        | 关键函数
vllm_mindspore/.../golden_stick/golden_stick.py            | 量化方法注册与识别           | get_quant_method(), get_config_filenames()
vllm_mindspore/.../golden_stick/a8w8sc.py                  | A8W8SC 线性层实现           | A8W8SCLinearMethod.create_weights()
vllm_mindspore/.../quant_ops.py                            | 稀疏量化算子封装             | QuantLinearSparseOp
vllm_mindspore/.../models/sparse_quant_weight_loader.py    | 权重加载与 310P 格式转换     | load_split_weights(), _param_name_to_weight_key()
vllm_mindspore/.../sparse_quant_loader.py                  | 稀疏量化模型加载器           | SparseQuantModelLoader
vllm_mindspore/config.py                                   | 稀疏量化配置识别             | is_sparse_quantization
vllm_mindspore/.../qwen2.py                                | 模型权重加载                | load_weights()
```
</fetch>

---

## 加载流程详解

```
模型目录扫描
     │
     ▼
┌─────────────────────────────────────────┐
│ 1. 配置文件识别 (golden_stick.py)        │
│    - quantization_description.json      │
│    - quant_model_description.json       │
│    - quant_model_description_w8a8sc.json│
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ 2. 稀疏量化判断 (golden_stick.py)       │
│    - 配置中存在 rank_ 开头的键           │
│    - 设置 load_format=sparse_quant      │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ 3. 推理机校验 (a8w8sc.py)              │
│    - 310P: 支持稀疏量化                 │
│    - 910: 报错 INFERENCE_910_SPARSE_QUANT│
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ 4. 权重加载 (sparse_quant_weight_loader)│
│    - 从 rank_{tp_rank} 目录读取权重      │
│    - deq_scale: float32 → int64(310P)   │
└─────────────────────────────────────────┘
     │
     ▼
   推理就绪
```

---

## 代码实现先验

### 线性层参数名与前向

<existing_optimizations>

- **参数名**：必须为 **`weight`**、**`index`**（禁止使用 compress_weight/compress_index）
- **前向算子**：统一使用 **ops.auto_generate.QuantLinearSparse**，禁止稠密 MatMul fallback

**QuantLinearSparseOp（quant_ops.py）**
```python
class QuantLinearSparseOp(nn.Cell):
    def __init__(self, params_dtype):
        super().__init__()
        self.linear_sparse = ops.auto_generate.QuantLinearSparse()

    def construct(self, qx, weight, deq_scale, index, quant_bias):
        return self.linear_sparse(qx, weight, deq_scale, index, quant_bias)
```

**_param_name_to_weight_key 后缀列表（含 .index）**
```python
for suffix in (".compress_weight", ".compress_index", ".deq_scale",
               ".quant_bias", ".input_scale", ".input_offset", ".index"):
    if name.endswith(suffix):
        return name[:-len(suffix)] + ".weight"
```

**create_weights 中 filtered_attrs**
```python
filtered_attrs = {
    k: v for k, v in extra_weight_attrs.items()
    if k not in ("output_dim", "input_dim", "weight_loader")
}
```

**参数插入到 layer（a8w8sc.py）**
```python
if layer is not None:
    layer.insert_param_to_cell("weight", weight)
    layer.insert_param_to_cell("index", index)
    layer.insert_param_to_cell("deq_scale", deq_scale)
    layer.insert_param_to_cell("quant_bias", quant_bias)
    layer.insert_param_to_cell("input_scale", input_scale)
    layer.insert_param_to_cell("input_offset", input_offset)
```

</existing_optimizations>

---

## 常见 Bug 修复

<optimization_patterns>

### Bug 1：权重重复切分

**错误信息**：
```
safetensors_rust.SafetensorError: index out of bounds
```

**原因**：稀疏量化权重已按 rank 切分存放于 `rank_{tp_rank}` 目录，但代码未检测稀疏量化，仍调用 `split_loaded_weight` 再次切分。

**修复**：在模型（如 qwen2.py）的 `load_weights` 开头添加检测逻辑：
```python
def load_weights(self, weights, params_dict):
    if self.quant_config is not None and hasattr(self.quant_config, "config"):
        rank_id = get_tensor_model_parallel_rank()
        rank_key = f"rank_{rank_id}"
        if rank_key in self.quant_config.config:
            sparse_config = self.quant_config.config[rank_key]
            has_sparse_quant = any(
                isinstance(v, str) and v.lower() == "w8a8s"
                for v in sparse_config.values())
            if has_sparse_quant:
                return load_split_weights(weights, params_dict, self.config, self.quant_config)
```

### Bug 2：参数未插入到 layer

**错误信息**：
```
AttributeError: The 'QKVParallelLinear' object has no attribute 'input_scale'.
```

**原因**：`a8w8sc.py` 的 `create_weights` 中缺少将参数插入到 layer 的代码。

**修复**：在 `create_weights` 函数末尾添加参数插入代码（见上方代码实现先验）

### Bug 3：FormatCast-op0 错误

**错误信息**：
```
RuntimeError: Launch kernel failed, name: Default/FormatCast-op0
```

**原因**：
1. `golden_stick.py` 缺少关键的稀疏量化（W8A8SC）支持代码
2. 具体缺失：
   - 缺少 `LinearBase` 和 `is_310p` 导入
   - 缺少 `A8W8SCLinearMethod` 导入
   - `quantization_method_mapping` 缺少 "W8A8S" 和 "W8A8SC" 映射
   - `get_config_filenames` 缺少 `"quant_model_description_w8a8sc.json"`
   - `get_quant_method` 缺少处理 rank 级别稀疏量化配置的逻辑
3. 导致无法正确识别稀疏量化模型，权重格式转换逻辑未正确触发

**修复**：
1. 更新 `golden_stick/golden_stick.py`，确保包含以下关键修改（参考 PR !1428 和 !1490）：
   - 导入 `LinearBase`（来自 `vllm_mindspore.model_executor.layers.linear`）
   - 导入 `is_310p`（来自 `vllm_mindspore.utils`）
   - 导入 `A8W8SCLinearMethod`（来自 `vllm_mindspore.model_executor.layers.quantization.golden_stick.a8w8sc`）
   - `quantization_method_mapping` 添加 `"W8A8S": A8W8SCLinearMethod`
   - `get_config_filenames` 添加 `"quant_model_description_w8a8sc.json"`
   - `get_quant_method` 中添加处理 rank 级别稀疏量化配置的逻辑：
     - 读取 `rank_{rank_id}` 配置
     - 检测 `weight` 和 `index` 的 shape
     - 非 310P 设备抛出 `RuntimeError("Sparse quantization (W8A8SC) is only supported on 310P platform...")`
   - 添加 `quant_device_type` 属性从配置中读取 `device_type`
2. 更新 `models/sparse_quant_weight_loader.py`，确保包含以下关键修改（参考 PR !1428 和 !1490）：
   - 添加 `is_sparse_quant_weight` 函数检测权重是否为稀疏量化
   - 添加 `adjust_sparse_quant_weights_for_310p` 函数处理 310P 格式转换
   - 添加 `load_split_weights` 函数直接从 rank 目录加载权重（不再切分）
   - 稀疏量化权重跳过 FormatCast 转换（避免 FormatCast-op0 错误）

</optimization_patterns>

---

## 优化原则

在进行代码适配时，请遵循以下原则：

<principles>

### 1. 先分析，后动手
在修改任何代码之前，先确认当前仓库是否已有相关实现，避免重复添加已有逻辑。

### 2. 优先复用现有实现
稀疏量化相关代码已在多个 PR 中沉淀，优先从已有实现中复制，而非重写。

### 3. 推理机类型优先判断
所有稀疏量化路径必须在入口处判断推理机类型（310P/910），非 310P 一律报错，不做 fallback。

### 4. 参数名严格对齐
weight/index 参数名是框架约定，任何别名都会导致权重加载失败，不得随意更改。

</principles>

---

## 自我检查问题

- 当前仓库的 `golden_stick.py` 是否包含 W8A8S/W8A8SC 的 mapping？
- `get_config_filenames` 是否包含 `quant_model_description_w8a8sc.json`？
- `create_weights` 末尾是否有 `insert_param_to_cell` 调用？
- 线性层前向是否使用 `QuantLinearSparse` 而非稠密 MatMul？

---

## 你的任务

<task>

1. **确认模型类型**: 扫描配置文件，判断是否为稀疏量化（W8A8SC）格式
2. **校验推理机**: 确认当前推理机为 310P，否则终止并报错
3. **检查代码实现**: 按执行清单逐项核查关键实现是否到位
4. **修复缺失项**: 针对缺失的实现补充代码或从参考仓库复制
5. **验证加载**: 确认权重加载流程无报错，推理可正常运行

</task>

---

## 执行清单

| 序号 | 操作 |
|------|------|
| 1 | 确认 QuantLinearSparseOp 存在且正确 |
| 2 | 线性层使用 weight/index 参数名，前向使用 QuantLinearSparse |
| 3 | 稀疏层 create_weights 中过滤 extra_weight_attrs |
| 4 | get_quant_method 中读取 weight.shape/index.shape，非 310P 时报错 |
| 5 | process_weights_after_loading 实现 input_scale/input_offset 扩维、deq_scale 转换 |
| 6 | _param_name_to_weight_key 后缀列表含 .index |
| 7 | config 中 is_sparse_quantization 且 load_format=auto 时设 load_format=sparse_quant |
| 8 | 模型侧 has_sparse_quant 时调用 load_split_weights |

---

## 输出格式

- 当前仓库稀疏量化支持状态（缺失项列表）
- 已修复 / 已补充的代码位置
- 修改的文件清单
