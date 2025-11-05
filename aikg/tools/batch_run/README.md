# Batch Run 批量运行工具

这个工具可以批量运行目录中的 PyTorch NPU Triton 任务文件。

## 功能特性

- 自动扫描指定目录中的所有 `.py` 文件
- 检查每个文件是否包含必需的组件：
  - `class Model(nn.Module)`
  - `def get_inputs()`
  - `def get_init_inputs()`
- 只运行包含所有必需组件的文件
- 提供详细的运行日志和总结报告

## 使用方法

```bash
python batch_run_torch_npu_triton.py <目录路径> [--devices DEVICE_IDS]
```

### 参数说明

- `目录路径`: 包含要运行的 `.py` 文件的目录路径（必填）
- `--devices`: 设备ID列表，默认为 `[0]`（可选）

### 使用示例

```bash
# 使用默认设备 [0] 运行目录中的所有有效文件
python batch_run_torch_npu_triton.py /path/to/your/directory

# 指定多个设备
python batch_run_torch_npu_triton.py /path/to/your/directory --devices 0 1 2
```

## 文件要求

要运行的文件必须包含以下组件：

1. **Model 类定义**：
   ```python
   class Model(nn.Module):
       # ... 实现 ...
   ```

2. **get_inputs 函数**：
   ```python
   def get_inputs():
       # ... 返回输入列表 ...
   ```

3. **get_init_inputs 函数**：
   ```python
   def get_init_inputs():
       # ... 返回初始化输入列表 ...
   ```

## 输出说明

脚本会输出：
- 扫描到的文件列表（标记为有效或无效）
- 每个任务的运行进度和结果
- 最终的批量运行总结（总数、通过数、失败数）

## 注意事项

- 脚本会递归扫描目录中的所有子目录
- 每个文件的内容会被完整读取作为 `task_desc` 传递给任务
- 算子名称（op_name）从文件名自动提取（去掉路径和 .py 扩展名）
- 任务按顺序运行，不会并行执行

