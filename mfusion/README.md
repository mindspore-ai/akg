# mfusion

基于MLIR的图层算子融合工具。

## 依赖组件

- ***LLVM/MLIR***:
  地址：https://github.com/llvm/llvm-project
  当前使用的 commit id：cd708029e0b2869e80abe31ddb175f7c35361f90

- ***Torch-MLIR***:
  地址：https://gitee.com/mirrors_llvm/torch-mlir
  当前使用的 commit id：155680c08e08bff6d2e6883415e3f5a1b474d96e
  编译时会应用 `third_party/patch/torch-mlir/` 下的补丁。

- ***SymEngine***:
  地址：https://gitee.com/mirrors/SymEngine

- ***Python 构建依赖***:
  Python >= 3.9，`setuptools>=61.0`，`build`，`packaging>=24.2`，`wheel>=0.34.2`，`pybind11==2.9.0`。

- ***测试依赖***:
  `lit` 和 `pytest`。

## 编译和安装

### 安装 Python 构建依赖

```shell
cd PATH_TO_MFUSION_ROOT_PATH
pip install -r requirements-build.txt
```

### 使用 build.sh 脚本构建

```shell
cd PATH_TO_MFUSION_ROOT_PATH

print usage by -h (bash build.sh -h):
Usage:
bash build.sh [-a on|off] [-d] [-h] [-i] [-j[n]] [-s path] [-t]

Options:
    -a on|off Enable AddressSanitizer (implies Debug mode), default off
    -d Debug mode, default release mode
    -h Print usage
    -i Incremental build
    -j[n] Set the threads when building (Default: the number of cpu)
    -s Specifies the CMAKE_PREFIX_PATH for dependencies
    -t Enable unit test (Default: disable)

Command Example:
    # First time build
    bash build.sh -j32

    # Reuse prebuilt dependencies through CMAKE_PREFIX_PATH
    bash build.sh -s /path/to/dependency/prefix -j32

    # Incremental build
    bash build.sh -i -j32

    # Build with unit tests
    bash build.sh -t -j32

    # Debug / ASAN build
    bash build.sh -d -j32
    bash build.sh -a on -j32
```

构建脚本通过 `python -m build --wheel --no-isolation` 触发 CMake 编译。生成的 wheel 会复制到 `output/` 目录。

```shell
pip install output/mfusion-*.whl
```

### 手动分步构建

```shell
cd PATH_TO_MFUSION_ROOT_PATH

export BUILD_JOBS=32
export BUILD_TYPE=Release
export BUILD_TESTS=OFF
export ENABLE_ASAN=OFF

# 可选：通过 CMAKE_PREFIX_PATH 提供已有依赖前缀
export CMAKE_PREFIX_PATH=/path/to/dependency/prefix

python -m build --wheel --no-isolation
mkdir -p output
cp dist/*.whl output/
```

## 运行示例

### 使用 mfusion-opt

安装 wheel 后可以直接使用 Python console script：

```shell
mfusion-opt tests/ut/lit/Pipeline/test_mfuse_cluster_then_outline.mlir \
    --mfuse-dvm-cluster \
    --outline-mfuse-fused-subgraphs
```

也可以直接使用构建目录中的二进制：

```shell
build/bin/mfusion-opt tests/ut/lit/Conversion/TorchToMfuse/test_convert_torch_relu.mlir \
    --convert-torch-to-mfuse \
    --canonicalize
```

### 使用 Python pipeline

当前支持接入torch inductor后端：

```python
from mfusion.torch.inductor import fuse_and_optimize

optimized_mlir = fuse_and_optimize(torch_mlir_text, kernel_generator="dvm")
```

`kernel_generator` 支持 `dvm`、`akg` 和 `bisheng`，默认值为 `dvm`。

## 运行测试

构建测试目标：

```shell
bash build.sh -t -j32
```

运行全部单元测试：

```shell
bash tests/run_test.sh -t ut -u all
```

只运行 lit 或 Python 单元测试：

```shell
bash tests/run_test.sh -t ut -u lit
bash tests/run_test.sh -t ut -u python
```

## IR 调试

Python pipeline 支持如下环境变量：

```shell
# 打印每个 pipeline stage 的 IR
export MFUSION_PRINT_IR=1

# 保存每个 pipeline stage 的 IR，默认写入当前目录下的 graphs/
export MFUSION_SAVE_IR=1

# 额外保存 torch-fusion / mfuse-fusion 内部子 pass 的 IR
export MFUSION_SAVE_IR=2

# 指定 IR 保存目录
export MFUSION_SAVE_IR_PATH=/path/to/graphs

# level 2 下只保存发生变化的内部子 pass IR
export MFUSION_VERBOSE_IR_DUMP_ON_CHANGE=1
```

## 代码格式化

项目上级目录包含 `.clang-format`，C/C++ 代码按该配置格式化。例如：

```shell
git diff --name-only -- '*.cc' '*.h' | xargs -r clang-format -i
```
