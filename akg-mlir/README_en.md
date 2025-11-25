# akg-mlir

Auto kernel generator(AKG) based on MLIR

## Dependency 
- ***LLVM/MLIR***: https://github.com/llvm/llvm-project, current llvm commit id: cd708029e0b2869e80abe31ddb175f7c35361f90

- ***Polytops*** : https://gitee.com/ms-incubator/polytops, last know working polytops v0.24.2 commit SHA: ca3df32829ff81869ba0f209c7fca24d9710a89e
automattically build during make/ninja
mirror git https://codehub-y.huawei.com/DPSL-Paris/MLScheduler

- ***Symengine*** : https://github.com/symengine/symengine, current symengine commit id: 7b1880824c2cce98787ae29a317682ba6c294484

- ***AscendNPU IR***: https://gitee.com/ascend/ascendnpu-ir, current bishengir commit id: f4bb879a22c56c591b163f397eeb3b82794863f9

## Build and Install

### Install pre-builts thats are required to build BiShengIR

1. Extract the package (**Verison 0.4**, available in the [release page](https://gitee.com/ascend/ascendnpu-ir/releases)) containing the pre-builts corresponding to your target machine to any location. After install, it should contain the following contents:

   ```bash
   ├── lib
     └── libBiShengIR.so     // used to build bishengir dialects
   └── bin
     └── bishengir-compile   // used to compile `.mlir` to binary
     └── bishengir-yaml-gen  // used to generate files from yaml
   ```

2. Set environment variable to the installed path:

  ```bash
  export BISHENG_IR_INSTALL_PATH= ...
  ```

### Via build.sh
```shell
cd PATH_TO_AKG_MLIR_ROOT_PATH

print usage by -h (bash build.sh -h):
Usage:
bash build.sh [-e cpu|gpu|ascend|all] [-j[n]] [-t on|off] [-o] [-u] [-m akg-mlir-only|all] [-s] [-c] [-h]

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
    bash build.sh -e ascend -j32
    # Non-First time build, akg-mlir-only
    bash build.sh -e ascend akg-mlir-only -j32
```

### Step-by-Step
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
    -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
    -DLLVM_EXTERNAL_PROJECTS="bishengir" \
    -DLLVM_EXTERNAL_BISHENGIR_SOURCE_DIR=${LLVM_BASE_PATH}/third-party/bishengir \
    -DBISHENG_IR_INSTALL_PATH="${BISHENG_IR_INSTALL_PATH}"

export PATH_TO_BUILT_LLVM=${PWD}
cmake --build . --config Release -j32
```
-DLLVM_TARGETS_TO_BUILD=NVPTX required for GPU backend, Target/PTX
-DLLVM_ENABLE_RTTI=ON required with symengine project

```shell
# build symengine
cmake .. \
    -DHAVE_SYMENGINE_NOEXCEPT=OFF \
    -DCMAKE_BUILD_TYPE:STRING="Release" \
    -DWITH_BFD:BOOL=OFF \
    -DWITH_SYMENGINE_ASSERT:BOOL=OFF \
    -DWITH_SYMENGINE_RCP:BOOL=ON \
    -DWITH_SYMENGINE_THREAD_SAFE:BOOL=OFF \
    -DWITH_ECM:BOOL=OFF \
    -DBUILD_TESTS:BOOL=OFF \
    -DBUILD_BENCHMARKS:BOOL=OFF \
    -DBUILD_BENCHMARKS_GOOGLE:BOOL=OFF \
    -DBUILD_SHARED_LIBS:BOOL=ON \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=ON
make -j16
export PATH_TO_BUILT_SYMENGINE=${PWD}
```

```shell
# build akg-mlir
#    -G Ninja \
cmake ../cmake/ \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_BUILD_PATH=${PATH_TO_BUILT_LLVM} \
    -DSYMENGINE_BUILD_PATH=${PATH_TO_BUILT_SYMENGINE} \
    -DCMAKE_C_COMPILER=${C_COMPILER_PATH} \
    -DCMAKE_CXX_COMPILER=${CXX_COMPILER_PATH}
cmake --build . --config Release -j32
```
ninja doesn't work due to issue when defining project dependencies with cloog.


## Run example

```shell
cd compile/lib/test/
PATH_TO_BUILD/bin/akg-opt akg_loop_tiling.mlir -allow-unregistered-dialect -split-input-file -akg-affine-loop-tile="tile-size=2" | FileCheck akg_loop_tiling.mlir

```

## Code formatting

```shell
git diff -U0 HEAD^ | ./third-party/llvm-project/clang/tools/clang-format/clang-format-diff.py -i -p1
```