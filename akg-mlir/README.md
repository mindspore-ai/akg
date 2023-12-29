# akg-mlir

Auto kernel generator(AKG) based on MLIR

## Dependency 
***LLVM/MLIR***: https://github.com/llvm/llvm-project, current llvm commit id: ac07911b455ed0b29c201b1d59d3f33105777af9

***Polytops*** : https://gitee.com/ms-incubator/polytops, last know working polytops v0.24.2 commit SHA: ca3df32829ff81869ba0f209c7fca24d9710a89e
automattically build during make/ninja
mirror git https://codehub-y.huawei.com/DPSL-Paris/MLScheduler

***Symengine*** : https://github.com/symengine/symengine, current symengine commit id: 7b1880824c2cce98787ae29a317682ba6c294484


## Build and Install

### Via build.sh
```shell
cd PATH_TO_AKG_MLIR_ROOT_PATH

print usage by -h (bash build.sh -h):
Usage:
bash build.sh [-h] [-c] [-d] [-t] [-m] [-e] [-S] [-s]
              [-u] [-j[n]]

Options:
    -h Print usage
    -c Clean built files, default: off
    -d Enable debug mode, default: off
    -t Unit test: on or off, default: off
    -m Compile mode: akg-mlir-only or all, default: all
    -e Backend Environment: cpu, gpu, or auto, default: auto
    -S Specifies the build path of third-partys, default: none
        [0]llvm-project
        [1]symengine
        [2]polytops
    -s Specifies the source path of third-partys, default: none
        [0]llvm-project
        [1]symengine
        [2]polytops
    -u Update submodule, default: off
    -j[n] Set the threads when building, default: -j8

Command Example:
    # First time build, full compile; debug mode; enable Unit tests
    bash build.sh -d -t -j32
    # Non-First time build, akg-mlir-only; debug mode; enable Unit tests
    bash build.sh -d -m akg-mlir-only -t -j32
```

### Step-by-Step
```shell
# build llvm/mlir
cmake ../llvm \
    -G Ninja \
    -DLLVM_BUILD_EXAMPLES=ON \
    -DLLVM_TARGETS_TO_BUILD="host;Native;NVPTX" \
    -DLLVM_ENABLE_PROJECTS="llvm;mlir" \
    -DLLVM_OPTIMIZED_TABLEGEN=ON \
    -DLLVM_ENABLE_OCAMLDOC=OFF \
    -DLLVM_ENABLE_BINDINGS=OFF \
    -DLLVM_INSTALL_UTILS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_RTTI=ON \
    -DMLIR_ENABLE_CUDA_RUNNER=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON
ninja
export PATH_TO_BUILT_LLVM=${PWD}
# cmake --build . --target check-mlir
# cmake --build . --config Release
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
    -DSYMENGINE_BUILD_PATH=${PATH_TO_BUILT_SYMENGINE}
cmake --build . --config Release -j16
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

