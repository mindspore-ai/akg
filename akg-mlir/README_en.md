# akg-mlir

Auto kernel generator(AKG) based on MLIR

## Dependency

- ***LLVM/MLIR***: https://github.com/llvm/llvm-project, current llvm commit id: cd708029e0b2869e80abe31ddb175f7c35361f90
  automatically build during build script
  mirror git https://gitee.com/mirrors/LLVM

- ***Symengine***: https://github.com/symengine/symengine, current symengine commit id: 7b1880824c2cce98787ae29a317682ba6c294484
  automatically build during build script
  mirror git https://gitee.com/mirrors/SymEngine

- ***AscendNPU IR***: https://gitcode.com/Ascend/AscendNPU-IR, current AscendNPU IR commit id: e4633e70f812b7c483768fdcc850c6077a3727e1
  automatically build during build script

- ***gmp***: Install prerequisites. For Debian based systems (Ubuntu etc.):

```shell
apt-get install cmake libgmp-dev
```

For RPM based systems (Fedora etc.):

```shell
yum install cmake gmp-devel
```

## Build and Install

### Via build.sh

```shell
cd PATH_TO_AKG_MLIR_ROOT_PATH

print usage by -h (bash build.sh -h):
Usage:
bash build.sh [-e cpu|gpu|ascend|all] [-j[n]] [-t] [-b] [-u] [-s path] [-c] [-m] [-h]

Options:
    -b enable binds python (Default: disable)
    -c Clean built files, default: off
    -d Debug mode
    -e Hardware environment: cpu, gpu, ascend or all
    -h Print usage
    -j[n] Set the threads when building (Default: auto)
    -m Enable auto build mlir (Default: disable)
    -s Specifies the source path of third-party, default: none
    -t Enable unit test (Default: disable)
    -u Enable auto tune (Default: disable)
Command Example:
    # First time build, full compile
    bash build.sh -e ascend -j32 -s /path/to/llvm
```

### build llvm

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

**Note**: -DLLVM_TARGETS_TO_BUILD=NVPTX required for GPU backend, Target/PTX. -DLLVM_ENABLE_RTTI=ON required with symengine project.

## Run example

```shell
cd tests/ut/Dialect/Affine
PATH_TO_BUILD/bin/akg-opt akg_loop_tiling.mlir -allow-unregistered-dialect -split-input-file -akg-affine-loop-tile="tile-size=2" | FileCheck akg_loop_tiling.mlir
```
