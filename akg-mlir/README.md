# akg-mlir

基于MLIR的自动算子生成器（Auto kernel generator(AKG) based on MLIR）

## 依赖组件

- ***LLVM/MLIR***:
  地址：https://github.com/llvm/llvm-project
  当前使用的 commit id：cd708029e0b2869e80abe31ddb175f7c35361f90

- ***Polytops***:
  地址：https://gitee.com/ms-incubator/polytops
  推荐版本 commit SHA：ca3df32829ff81869ba0f209c7fca24d9710a89e
  编译时会自动构建
  镜像仓库：https://codehub-y.huawei.com/DPSL-Paris/MLScheduler

- ***Symengine***:
  地址：https://github.com/symengine/symengine
  当前使用的 commit id：7b1880824c2cce98787ae29a317682ba6c294484
  编译时会自动构建

- ***AscendNPU IR***:
  地址：https://gitcode.com/Ascend/AscendNPU-IR
  当前使用的 commit id：e4633e70f812b7c483768fdcc850c6077a3727e1
  编译时会自动构建

## 编译和安装

### 使用 build.sh 脚本构建

```shell
cd PATH_TO_AKG_MLIR_ROOT_PATH

print usage by -h (bash build.sh -h):
Usage:
bash build.sh [-e cpu|gpu|ascend|all] [-j[n]] [-t] [-b] [-u] [-s path] [-c] [-h]

Options:
    -b enable binds python (Default: disable)
    -c Clean built files, default: off
    -d Debug mode
    -e Hardware environment: cpu, gpu, ascend or all
    -h Print usage
    -j[n] Set the threads when building (Default: the number of cpu)
    -s Specifies the source path of third-party, default: none
    -t Enable unit test (Default: disable)
    -u Enable auto tune (Default: disable)

Options:
    -h Print usage
    -c Clean built files, default: off
    -d Enable debug mode, default: off
    -t Unit test: on or off, default: off
    -m Compile mode: akg-mlir-only or all, default: all
    -e Hardware environment: cpu, gpu, ascend or all
    -s Specifies the source path of third-party, default: none \n\tllvm-project
    -u Enable auto tune
    -j[n] Set the threads when building, Default: -j8

Command Example:
    # First time build, full compile
    bash build.sh -e ascend -j32 -s /path/to/llvm
```

### 构建llvm

```shell
# build llvm/mlir
cmake ../llvm \
    -G Ninja  \
    -DPython3_FIND_STRATEGY=LOCATION \
    -DLLVM_BUILD_EXAMPLES=ON \
    -DLLVM_ENABLE_PROJECTS="llvm;mlir;clang;openmp" \
    -DLLVM_ENABLE_RUNTIMES="compiler-rt" \
    -DLLVM_OPTIMIZED_TABLEGEN=ON \
    -DLLVM_ENABLE_OCAMLDOC=OFF \
    -DLLVM_ENABLE_BINDINGS=OFF \
    -DLLVM_INSTALL_UTILS=ON \
    -DCMAKE_BUILD_TYPE="Release" \
    -DLLVM_ENABLE_RTTI=ON \
    -DCMAKE_C_COMPILER=${C_COMPILER_PATH} \
    -DCMAKE_CXX_COMPILER=${CXX_COMPILER_PATH} \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON

export PATH_TO_BUILT_LLVM=${PWD}
cmake --build . --config Release -j32
```

**说明**：GPU后端需要 -DLLVM_TARGETS_TO_BUILD=NVPTX，SymEngine项目需要 -DLLVM_ENABLE_RTTI=ON

## 运行示例

```shell
cd compile/lib/test/
PATH_TO_BUILD/bin/akg-opt akg_loop_tiling.mlir -allow-unregistered-dialect -split-input-file -akg-affine-loop-tile="tile-size=2" | FileCheck akg_loop_tiling.mlir
```

## 代码格式化

```shell
git diff -U0 HEAD^ | ./third-party/llvm-project/clang/tools/clang-format/clang-format-diff.py -i -p1
```
