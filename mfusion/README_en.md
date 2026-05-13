# mfusion

A graph-level operator fusion tool based on MLIR.

## Dependencies

- ***LLVM/MLIR***:
  URL: https://github.com/llvm/llvm-project
  Current commit id: cd708029e0b2869e80abe31ddb175f7c35361f90

- ***Torch-MLIR***:
  URL: https://gitee.com/mirrors_llvm/torch-mlir
  Current commit id: 155680c08e08bff6d2e6883415e3f5a1b474d96e
  Patches under `third_party/patch/torch-mlir/` are applied during build.

- ***SymEngine***:
  URL: https://gitee.com/mirrors/SymEngine

- ***Python build dependencies***:
  Python >= 3.9, `setuptools>=61.0`, `build`, `packaging>=24.2`, `wheel>=0.34.2`, and `pybind11==2.9.0`.

- ***Test dependencies***:
  `lit` and `pytest`.

## Build and Install

### Install Python build dependencies

```shell
cd PATH_TO_MFUSION_ROOT_PATH
pip install -r requirements-build.txt
```

### Build with build.sh

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

The build script triggers the CMake build through `python -m build --wheel --no-isolation`. The generated wheel is copied to the `output/` directory.

```shell
pip install output/mfusion-*.whl
```

### Manual Step-by-Step Build

```shell
cd PATH_TO_MFUSION_ROOT_PATH

export BUILD_JOBS=32
export BUILD_TYPE=Release
export BUILD_TESTS=OFF
export ENABLE_ASAN=OFF

# Optional: provide existing dependency prefixes through CMAKE_PREFIX_PATH
export CMAKE_PREFIX_PATH=/path/to/dependency/prefix

python -m build --wheel --no-isolation
mkdir -p output
cp dist/*.whl output/
```

## Run Examples

### Use mfusion-opt

After installing the wheel, you can directly use the Python console script:

```shell
mfusion-opt tests/ut/lit/Pipeline/test_mfuse_cluster_then_outline.mlir \
    --mfuse-dvm-cluster \
    --outline-mfuse-fused-subgraphs
```

You can also directly use the binary in the build directory:

```shell
build/bin/mfusion-opt tests/ut/lit/Conversion/TorchToMfuse/test_convert_torch_relu.mlir \
    --convert-torch-to-mfuse \
    --canonicalize
```

### Use Python pipeline

Currently, integration with the torch inductor backend is supported:

```python
from mfusion.torch.inductor import fuse_and_optimize

optimized_mlir = fuse_and_optimize(torch_mlir_text, kernel_generator="dvm")
```

`kernel_generator` supports `dvm`, `akg`, and `bisheng`; the default value is `dvm`.

## Run Tests

Build test targets:

```shell
bash build.sh -t -j32
```

Run all unit tests:

```shell
bash tests/run_test.sh -t ut -u all
```

Run only lit or Python unit tests:

```shell
bash tests/run_test.sh -t ut -u lit
bash tests/run_test.sh -t ut -u python
```

## IR Debugging

The Python pipeline supports the following environment variables:

```shell
# Print the IR for each pipeline stage
export MFUSION_PRINT_IR=1

# Save the IR for each pipeline stage; by default, it is written to graphs/ under the current directory
export MFUSION_SAVE_IR=1

# Additionally save the IR for internal sub-passes of torch-fusion / mfuse-fusion
export MFUSION_SAVE_IR=2

# Specify the IR save directory
export MFUSION_SAVE_IR_PATH=/path/to/graphs

# At level 2, save internal sub-pass IR only when changes occur
export MFUSION_VERBOSE_IR_DUMP_ON_CHANGE=1
```

## Code Formatting

The parent directory of the project contains `.clang-format`; C/C++ code is formatted according to that configuration. For example:

```shell
git diff --name-only -- '*.cc' '*.h' | xargs -r clang-format -i
```
