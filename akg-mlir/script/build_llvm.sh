#!/bin/bash
# Copyright 2019-2023 Huawei Technologies Co., Ltd
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

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/" && pwd )"
AKG_DIR="${CUR_DIR}/../../"

usage()
{
    echo "Usage:"
    echo "bash build.sh [-j[n]] [-e[gpu|cpu]] [-t] [-s] [-h]"
    echo ""
    echo "Options:"
    echo "    -h Print usage"
    echo "    -d Debug mode"
    echo "    -e Hardware environment: cpu, gpu, ascend or all"
    echo "    -j[n] Set the threads when building (Default: -j8)"
    echo "    -t install target dir"
    echo "    -s Specifies the source path of llvm-project, default: none "
}

mk_new_dir()
{
    local create_dir="$1"

    if [[ -d "${create_dir}" ]]; then
        rm -rf "${create_dir}"
    fi

    mkdir -pv "${create_dir}"
}


if [ ! -n "$1" ]; then
    echo "Must input parameter!"
    usage
    exit 1
fi

# Parse arguments
THREAD_NUM=32
SIMD_SET=off
CMAKE_ARGS=""
PATH_TO_SOURCE_LLVM=${AKG_DIR}/third-party/llvm-project/
_BUILD_TYPE="Release"
BACKEND_ENV="CPU"

while getopts 'h:e:j:s:t:d' opt
do
    echo "${opt} ${OPTARG}"
    case "${opt}" in
        h)
            usage
            exit 0
                ;;
        e)
            if [[ "${OPTARG}" == "gpu" ]]; then
                CMAKE_ARGS="${CMAKE_ARGS} -DUSE_CUDA=ON"
                BACKEND_ENV="GPU"
            elif [[ "${OPTARG}" == "ascend" ]]; then
                CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_D=ON"
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
        j)
            THREAD_NUM=${OPTARG}
            ;;
        d)
            CMAKE_ARGS="${CMAKE_ARGS} -DCMAKE_BUILD_TYPE=Debug -DUSE_AKG_LOG=1"
            _BUILD_TYPE=Debug
            ;;
        s)
            echo "path_to source"
            echo "${OPTARG}"
            PATH_TO_SOURCE_LLVM=${OPTARG}
            ;;
        t)
            LLVM_OUTPUT_PATH="${OPTARG}"
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


third_party_patch() {
  echo "Start patching to llvm."
  local FILE=${AKG_DIR}/third-party/llvm_patch_7cbf1a2591520c2491aa35339f227775f4d3adf6.patch
  if [ -f "$FILE" ]; then
    cd ${PATH_TO_SOURCE_LLVM}
    local LLVM_CUR_COMMIT_ID=$(echo `git rev-parse HEAD`)
    if [[ "X${LLVM_CUR_COMMIT_ID}" != "X7cbf1a2591520c2491aa35339f227775f4d3adf6" ]]; then
        git checkout main
        git checkout .
        git clean -df
        git pull
        git reset --hard 7cbf1a2591520c2491aa35339f227775f4d3adf6
        echo "set llvm to commit: 7cbf1a2591520c2491aa35339f227775f4d3adf6"
    fi
    cd ${PATH_TO_SOURCE_LLVM}
    git checkout .
    git clean -df
    patch -p1 -i ${FILE}
    echo "Success patch to llvm!"
  fi
}

build_llvm() {
    echo "Start building llvm project."
    LLVM_BASE_PATH=${PATH_TO_SOURCE_LLVM}
    echo "LLVM_BASE_PATH = ${PATH_TO_SOURCE_LLVM}"
    cd ${LLVM_BASE_PATH}
    if [ ! -d "./_build" ]; then
        mkdir -pv _build
    fi
    LLVM_BUILD_PATH=${LLVM_BASE_PATH}/_build
    echo "LLVM_BUILD_PATH = ${LLVM_BUILD_PATH}"
    cd ${LLVM_BUILD_PATH}
    local LLVM_CMAKE_ARGS="-G Ninja "
    if [[ "X${BACKEND_ENV}" = "XGPU" ]]; then
        LLVM_CMAKE_ARGS="${LLVM_CMAKE_ARGS} -DLLVM_TARGETS_TO_BUILD='host;Native;NVPTX'"
        LLVM_CMAKE_ARGS="${LLVM_CMAKE_ARGS} -DMLIR_ENABLE_CUDA_RUNNER=ON"
    else
        LLVM_CMAKE_ARGS="${LLVM_CMAKE_ARGS} -DLLVM_TARGETS_TO_BUILD='host'"
    fi

    cmake ../llvm \
    ${LLVM_CMAKE_ARGS} \
    -DPython3_FIND_STRATEGY=LOCATION \
    -DLLVM_BUILD_EXAMPLES=ON \
    -DLLVM_ENABLE_PROJECTS="llvm;mlir;clang;openmp" \
    -DLLVM_ENABLE_RUNTIMES="compiler-rt" \
    -DLLVM_OPTIMIZED_TABLEGEN=ON \
    -DLLVM_ENABLE_OCAMLDOC=OFF \
    -DLLVM_ENABLE_BINDINGS=OFF \
    -DLLVM_INSTALL_UTILS=ON \
    -DCMAKE_BUILD_TYPE=${_BUILD_TYPE} \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_RTTI=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
    -DCMAKE_INSTALL_PREFIX=${LLVM_OUTPUT_PATH}

    cmake --build . --config ${_BUILD_TYPE} -j${THREAD_NUM} 
    cmake --build . --target install 
    echo "Success to build llvm project!"
}

#third_party_patch
build_llvm
