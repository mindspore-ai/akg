# Benchmark 选择功能

## 功能说明

支持选择 **KernelBench** 或 **MultiKernelBench** 作为基准测试数据集。

### 支持的 Benchmark

- **KernelBench**: 100 个测试用例，位于 `aikg/thirdparty/KernelBench/KernelBench/level1/`
- **MultiKernelBench**: 286 个测试用例，按分类组织在 `aikg/thirdparty/multiKernelBench/reference/`

### 分类目录

MultiKernelBench 包含以下分类：
- `activation`: 激活函数 (relu, sigmoid, tanh, softmax 等)
- `convolution`: 卷积操作
- `pooling`: 池化操作
- `normalization`: 归一化操作
- `math`: 数学运算
- `reduce`: 归约操作
- `index`: 索引操作
- `loss`: 损失函数
- `broadcast`: 广播操作
- `resize`: 尺寸调整操作
- `optimizer`: 优化器
- `fuse`: 融合操作
- `arch`: 架构相关

## 使用示例

```python
from aikg.tests.utils import get_benchmark_name, get_benchmark_task

# 方式1: 按分类获取 MultiKernelBench case
all_cases = get_benchmark_name(category="all", framework="torch", benchmark="multiKernelBench")
activation_cases = get_benchmark_name(category="activation", framework="torch", benchmark="multiKernelBench")

# 方式2: 按序号获取 KernelBench case
kernelbench_cases = get_benchmark_name(task_index_list=[19, 20, 21], framework="torch", benchmark="KernelBench")

# 方式3: 在指定类别中按 op_name 获取单个 case (仅 MultiKernelBench)
single_case = get_benchmark_name(op_name="relu", category="activation", framework="torch", benchmark="multiKernelBench")
specific_conv_case = get_benchmark_name(op_name="conv_2d", category="convolution", framework="torch", benchmark="multiKernelBench")

# 获取任务内容
task_desc = get_benchmark_task("relu", framework="torch", benchmark="multiKernelBench")
```

## 参数说明

- `task_index_list`: 序号列表 (用于 KernelBench)
- `category`: 分类名称，`"all"` 表示所有分类 (用于 MultiKernelBench)
- `op_name`: 操作名称，用于在指定类别中获取单个 case (仅用于 MultiKernelBench)
- `framework`: 框架名称 (`"torch"`, `"mindspore", "numpy"`)
- `benchmark`: benchmark 类型 (`"KernelBench"`, `"multiKernelBench"`)

## 使用规则

- **MultiKernelBench**: 
  - 使用 `category` 参数按分类读取所有 case
  - 使用 `op_name` + `category` 参数在指定分类中获取单个 case
- **KernelBench**: 使用 `task_index_list` 参数按序号读取

## 在测试中的统一调用方式

```python
# 根据 benchmark 类型选择调用方式
if benchmark == "multiKernelBench":
    # MultiKernelBench: 按分类读取，支持指定 op_name 获取单个 case
    if category != "all":
        # 可以指定具体的 op_name 来获取单个 case
        benchmark_name = get_benchmark_name(op_name="relu", category=category, framework=framework, benchmark=benchmark)
        # 或者获取该分类的所有 case
        # benchmark_name = get_benchmark_name(category=category, framework=framework, benchmark=benchmark)
    else:
        benchmark_name = get_benchmark_name(category=category, framework=framework, benchmark=benchmark)
elif benchmark == "KernelBench":
    # KernelBench: 按序号读取
    benchmark_name = get_benchmark_name(task_index_list=[19, ], framework=framework, benchmark=benchmark)
else:
    # 不支持的 benchmark 类型
    print(f"警告: 不支持的 benchmark 类型 '{benchmark}'")
    print(f"当前支持的 benchmark 类型: KernelBench, multiKernelBench")
    benchmark_name = None

if benchmark_name is None:
    print(f"跳过测试: benchmark '{benchmark}' 不支持")
    return
```

## 当前支持的 Benchmark 类型

- **KernelBench**: 按序号读取，支持 `task_index_list` 参数
- **multiKernelBench**: 按分类读取，支持 `category` 参数

## 错误处理

当传入不支持的 benchmark 类型时：
- `get_benchmark_name` 函数会返回 `None`
- 测试函数会打印警告信息并跳过测试

