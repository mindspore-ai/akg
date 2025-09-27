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
export AKG_MLIR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/" && pwd )"
BUILD_DIR="${AKG_MLIR_DIR}/build"
C_COMPILER_PATH=$(which gcc)
CXX_COMPILER_PATH=$(which g++)

usage()
{
    echo "Usage:"
    echo "bash build.sh [-e cpu|gpu|ascend|all] [-j[n]] [-t on|off] [-b] [-o] [-u] [-s] [-c] [-h]"
    echo ""
    echo "Options:"
    echo "    -h Print usage"
    echo "    -b enable binds python (Default: off)"
    echo "    -d Debug mode"
    echo "    -e Hardware environment: cpu, gpu, ascend or all"
    echo "    -j[n] Set the threads when building (Default: -j8)"
    echo "    -t Unit test: on or off (Default: off)"
    echo "    -o Output .o file directory"
    echo "    -u Enable auto tune"
    echo "    -s Specifies the source path of third-party, default: none \n\tllvm-project"
    echo "    -c Clean built files, default: off"
}

mk_new_dir()
{
    local create_dir="$1"

    if [[ -d "${create_dir}" ]]; then
        rm -rf "${create_dir}"
    fi

    mkdir -pv "${create_dir}"
}

check_binary_file()
{
  local binary_dir="$1"
  for cur_file in `ls "${binary_dir}"/*.o`
  do
    file_lines=`cat "${cur_file}" | wc -l`
    if [ ${file_lines} -eq 3 ]; then
        check_sha=`cat ${cur_file} | grep "oid sha256"`
        if [ $? -eq 0 ]; then
            echo "-- Warning: ${cur_file} is not a valid binary file."
            return 1
        fi
    fi
  done
  return 0
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
COMPILE_AKG_MLIR="off"
AKG_MLIR_CMAKE_ARGS=""
AKG_MLIR_ARGS=""
_BUILD_TYPE="Release"
BACKEND_ENV="CPU"
ENABLE_BINDS_PYTHON="OFF"

while getopts 'bhe:j:u:t:o:d:m:s:c:r' opt
do
    case "${opt}" in
        b)
            ENABLE_BINDINGS_PYTHON="ON"
            ;;
        h)
            usage
            exit 0
                ;;
        e)
            if [[ "${OPTARG}" == "gpu" ]]; then
                CMAKE_ARGS="${CMAKE_ARGS} -DUSE_CUDA=ON"
                BACKEND_ENV="GPU"
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
        j)
            THREAD_NUM=${OPTARG}
            ;;
        t)
            ENABLE_UNIT_TEST="on"
            ;;
        u)
            CMAKE_ARGS="${CMAKE_ARGS} -DUSE_AUTO_TUNE=1"
            ;;
        d)
            CMAKE_ARGS="${CMAKE_ARGS} -DCMAKE_BUILD_TYPE=Debug -DUSE_AKG_LOG=1"
            _BUILD_TYPE=Debug
            ;;
        o)
            arch_info=`arch | tr '[A-Z]' '[a-z]'`
            arch_name=""
            if [[ "${arch_info}" =~ "aarch64" ]]; then
              arch_name="aarch64"
            elif [[ "${arch_info}" =~ "x86_64" ]]; then
              arch_name="x86_64"
            else
              echo "-- Warning: Only supports aarch64 and x86_64, but current is ${arch_info}"
              exit 1
            fi

            akg_extend_dir="${AKG_MLIR_DIR}/prebuild/${arch_name}"
            if [ ! -d "${akg_extend_dir}" ]; then
              echo "-- Warning: Prebuild binary file directory ${akg_extend_dir} not exits"
              exit 1
            fi

            check_binary_file "${akg_extend_dir}"
            if [ $? -ne 0 ]; then
              GIT_LFS=`which git-lfs`
              if [ $? -ne 0 ]; then
                echo "-- Warning: git lfs not found, you can perform the following steps:"
                echo "            1. Install git lfs, refer https://github.com/git-lfs/git-lfs/wiki/installation"
                echo "            2. After installing git lfs, do not forget executing the following command:"
                echo "               git lfs install"
                echo "            3. Download the files tracked by git lfs, executing the following commands:"
                echo "               cd ${AKG_MLIR_DIR}"
                echo "               git lfs pull"
                echo "            4. Re-compile the source codes"
                exit 1
              else
                echo "-- Warning: git lfs found, but lfs files are not downloaded, you can perform the following steps:"
                echo "            1. After installing git lfs, do not forget executing the following command:"
                echo "               git lfs install"
                echo "            2. Download the files tracked by git lfs, executing the following commands:"
                echo "               cd ${AKG_MLIR_DIR}"
                echo "               git lfs pull"
                echo "            3. Re-compile the source codes"
                exit 1
              fi
            fi
            echo "${akg_extend_dir}"
            exit 0
            ;;
        s)
            LLVM_INSTALL_PATH=${OPTARG}
            ;;
        c)
            CLEAN_BUILT="on"
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

make_clean()
{
  echo "enable make clean"
  cd "${BUILD_PATH}"
  cmake --build . --target clean
}

get_akg_mlir_cmake_args() {
  if [[ "X${ENABLE_UNIT_TEST}" = "Xon" ]]; then
    AKG_MLIR_ARGS="${AKG_MLIR_ARGS} --target check-akg-mlir"
  fi
}

echo "---------------- AKG: build start ----------------"

get_akg_mlir_cmake_args

if [[ "X$CLEAN_BUILT" = "Xon" ]]; then
    make_clean
fi

# Build akg target
cd $BUILD_DIR
set -x
cmake .. ${CMAKE_ARGS} ${AKG_MLIR_CMAKE_ARGS} \
    -DAKG_ENABLE_BINDINGS_PYTHON=${ENABLE_BINDINGS_PYTHON} \
    -DCMAKE_C_COMPILER=${C_COMPILER_PATH} \
    -DCMAKE_CXX_COMPILER=${CXX_COMPILER_PATH} \
    -DCMAKE_PREFIX_PATH=${LLVM_INSTALL_PATH}
cmake --build . --config ${_BUILD_TYPE} -j${THREAD_NUM} ${AKG_MLIR_ARGS}

cd $AKG_MLIR_DIR
AKG_CMAKE_ALREADY_BUILD=1 \
  AKG_CMAKE_BUILD_DIR=${BUILD_DIR} \
  AKG_ENABLE_BINDINGS_PYTHON=${ENABLE_BINDINGS_PYTHON} \
  python3 setup.py bdist_wheel
set -

echo "---------------- AKG: build end ----------------"
