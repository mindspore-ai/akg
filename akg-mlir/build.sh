#!/bin/bash
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

AKG_MLIR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/" && pwd )"
BUILD_DIR="${AKG_MLIR_DIR}/build"
# Parse arguments
THREAD_NUM=$(nproc)
CMAKE_ARGS="-DENABLE_GITEE=ON"
AKG_MLIR_CMAKE_ARGS=""
AKG_MLIR_ARGS=""
BUILD_TYPE="Release"
ENABLE_BINDINGS_PYTHON="OFF"

usage()
{
    echo "Usage:"
    echo "bash build.sh [-e cpu|gpu|ascend|all] [-j[n]] [-t] [-b] [-u] [-s path] [-c] [-h]"
    echo ""
    echo "Options:"
    echo "    -b enable binds python (Default: disable)"
    echo "    -c Clean built files, default: off"
    echo "    -d Debug mode"
    echo "    -e Hardware environment: cpu, gpu, ascend or all"
    echo "    -h Print usage"
    echo "    -j[n] Set the threads when building (Default: the number of cpu)"
    echo "    -s Specifies the source path of third-party, default: none"
    echo "    -t Enable unit test (Default: disable)"
    echo "    -u Enable auto tune (Default: disable)"
}

make_clean()
{
  echo "enable make clean"
  cd "${BUILD_PATH}"
  cmake --build . --target clean
}

while getopts 'bcde:hj:s:tu' opt
do
    case "${opt}" in
        b)
            ENABLE_BINDINGS_PYTHON="ON"
            ;;
        c)
            CLEAN_BUILT="on"
            ;;
        d)
            CMAKE_ARGS="${CMAKE_ARGS} -DCMAKE_BUILD_TYPE=Debug -DUSE_AKG_LOG=1"
            BUILD_TYPE=Debug
            ;;
        e)
            if [[ "${OPTARG}" == "gpu" ]]; then
                CMAKE_ARGS="${CMAKE_ARGS} -DUSE_CUDA=ON"
            elif [[ "${OPTARG}" == "ascend" ]]; then
                CMAKE_ARGS="${CMAKE_ARGS} -DUSE_LLVM=ON"
            elif [[ "${OPTARG}" == "cpu" ]]; then
                # AKG requires LLVM on CPU, the optimal version is 12.xx.xx.
                # if not found in the environment, it will find another existing version to use.
                CMAKE_ARGS="${CMAKE_ARGS} -DUSE_LLVM=ON"
            elif [[ "${OPTARG}" == "all" ]]; then
                CMAKE_ARGS="${CMAKE_ARGS} -DUSE_CUDA=ON -DENABLE_D=ON -DUSE_LLVM=ON"
            else
                echo "Unknown parameter ${OPTARG}!"
                usage
                exit 1
            fi
            ;;
        h)
            usage
            exit 0
            ;;
        j)
            THREAD_NUM=${OPTARG}
            ;;
        s)
            PREFIX_PATH=${OPTARG}
            ;;
        t)
            AKG_MLIR_ARGS="${AKG_MLIR_ARGS} --target check-akg-mlir"
            ;;
        u)
            CMAKE_ARGS="${CMAKE_ARGS} -DUSE_AUTO_TUNE=1"
            ;;
        *)
            echo "Unknown option ${opt}!"
            usage
            exit 1
    esac
done

echo "CMAKE_ARGS: ${CMAKE_ARGS}"

# Create directories
mkdir -pv "${BUILD_DIR}"

echo "---------------- AKG: build start ----------------"

if [[ "X$CLEAN_BUILT" = "Xon" ]]; then
    make_clean
fi

# Build akg target
cd $BUILD_DIR
set -x
cmake .. ${CMAKE_ARGS} ${AKG_MLIR_CMAKE_ARGS} \
    -DAKG_ENABLE_BINDINGS_PYTHON=${ENABLE_BINDINGS_PYTHON} \
    -DCMAKE_PREFIX_PATH=${PREFIX_PATH}
cmake --build . --config ${BUILD_TYPE} -j${THREAD_NUM} ${AKG_MLIR_ARGS}

cd $AKG_MLIR_DIR
AKG_CMAKE_ALREADY_BUILD=1 \
  AKG_CMAKE_BUILD_DIR=${BUILD_DIR} \
  AKG_ENABLE_BINDINGS_PYTHON=${ENABLE_BINDINGS_PYTHON} \
  python3 setup.py bdist_wheel
set -

echo "---------------- AKG: build end ----------------"
