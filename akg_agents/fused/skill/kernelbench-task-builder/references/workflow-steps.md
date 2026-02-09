# 工作流步骤详细说明

## 依赖追踪详解

`trace_dependencies` 工具的工作原理：

1. **解析文件 import**: 自动构建 `{别名: 来源模块}` 映射
   - `import torch._prims_common as utils` → `{"utils": "torch._prims_common"}`
   - `aten = torch._ops.ops.aten` → `{"aten": "torch._ops.ops.aten"}`

2. **区分外部调用类型**:
   - `tensor.size()` → 局部变量的方法调用 → 跳过
   - `torch.cat()` → 公共 API → 跳过
   - `utils.canonicalize_dim()` → 来源含私有段 `_prims_common` → 需要内联
   - `aten.constant_pad_nd()` → 来源含私有段 `_ops` → 需要内联

3. **处理建议**:
   - 同文件依赖 → 直接在 `assemble_task` 的 `functions` 列表中包含
   - 外部调用 → 用 `read_function` 查看原始签名 → 在 `helper_code` 中内联

## 任务装配策略选择

| 场景 | 推荐策略 | 示例 |
|------|---------|------|
| 目标函数依赖文件中大部分函数 | 排除式嵌入 | `exclude_functions=["unused1"]` |
| 目标函数只依赖少数函数 | 选择性提取 | `functions=["f1", "f2"]` |
| 需要整个文件 | 完整嵌入 | `source_files=["file.py"]` |
| 需要修改源函数 | 分段追加 | `write_file` + `append_to_file` |

## 验证流程

1. **格式验证** (`validate_task`):
   - Model 类能实例化
   - forward 能执行
   - 输出无 NaN/Inf
   - 两次运行结果一致

2. **正确性验证** (`test_with_reference`):
   - 与原始 torch 函数对比输出
   - 多组输入覆盖边界
   - per-case init_inputs 支持不同参数组合
