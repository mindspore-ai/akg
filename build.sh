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

export AKG_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/" && pwd )"
BUILD_DIR="${AKG_DIR}/build"
OUTPUT_PATH="${AKG_DIR}/output"

usage()
{
    echo "Usage:"
    echo "bash build.sh [-e cpu|gpu|ascend|all] [-j[n]] [-t on|off] [-o] [-u] [-m akg-mlir-only|all] [-s] [-c] [-h]"
    echo ""
    echo "Options:"
    echo "    -h Print usage"
    echo "    -d Debug mode"
    echo "    -e Hardware environment: cpu, gpu, ascend or all"
    echo "    -j[n] Set the threads when building (Default: -j8)"
    echo "    -t Unit test: on or off (Default: off)"
    echo "    -o Output .o file directory"
    echo "    -u Enable auto tune"
    echo "    -m Compile mode: akg-mlir-only or all, default: all"
    echo "    -s Specifies the source path of third-party, default: none \n\tllvm-project"
    echo "    -c Clean built files, default: off"
    echo "    -r Enable akg-mlir, default: off"
}

mk_new_dir()
{
    local create_dir="$1"

    if [[ -d "${create_dir}" ]]; then
        rm -rf "${create_dir}"
    fi

    mkdir -pv "${create_dir}"
}

write_checksum_tar()
{
    cd "$OUTPUT_PATH" || exit
    PACKAGE_LIST=$(ls lib*.tar.gz) || exit
    for PACKAGE_NAME in $PACKAGE_LIST; do
        echo $PACKAGE_NAME
        sha256sum -b "$PACKAGE_NAME" >"$PACKAGE_NAME.sha256"
    done
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
PATH_TO_SOURCE_LLVM=${AKG_DIR}/third-party/llvm-project/
_BUILD_TYPE="Release"
BACKEND_ENV="CPU"

while getopts 'h:e:j:u:t:o:d:m:s:c:r' opt
do
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
        r)
            CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_AKG_MLIR=1"
            COMPILE_AKG_MLIR="on"
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

            akg_extend_dir="${AKG_DIR}/prebuild/${arch_name}"
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
                echo "               cd ${AKG_DIR}"
                echo "               git lfs pull"
                echo "            4. Re-compile the source codes"
                exit 1
              else
                echo "-- Warning: git lfs found, but lfs files are not downloaded, you can perform the following steps:"
                echo "            1. After installing git lfs, do not forget executing the following command:"
                echo "               git lfs install"
                echo "            2. Download the files tracked by git lfs, executing the following commands:"
                echo "               cd ${AKG_DIR}"
                echo "               git lfs pull"
                echo "            3. Re-compile the source codes"
                exit 1
              fi
            fi
            echo "${akg_extend_dir}"
            exit 0
            ;;
        s)
            LLVM_BUILD_PATH=${OPTARG}
            ;;
        m)
            COMPILE_MODE=${OPTARG}
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
mkdir -pv "${OUTPUT_PATH}/akg"


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
    if [ ! -d "./build" ]; then
        mkdir -pv build
    fi
    LLVM_BUILD_PATH=${LLVM_BASE_PATH}/build
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
    -DMLIR_ENABLE_BINDINGS_PYTHON=OFF

    export PATH_TO_BUILT_LLVM=${PWD}
    cmake --build . --config ${_BUILD_TYPE} -j${THREAD_NUM}
    cmake --install ${LLVM_BUILD_PATH} --component clang --prefix ${OUTPUT_PATH}/akg
    cmake --install ${LLVM_BUILD_PATH} --component llc --prefix ${OUTPUT_PATH}/akg
    echo "Success to build llvm project!"
}

make_clean()
{
  echo "enable make clean"
  cd "${BUILD_PATH}"
  cmake --build . --target clean
}

get_akg_mlir_cmake_args() {
  AKG_MLIR_CMAKE_ARGS="${AKG_MLIR_CMAKE_ARGS} -DLLVM_BUILD_PATH=${PATH_TO_BUILT_LLVM} 
  -DLLVM_EXTERNAL_LIT=${PATH_TO_BUILT_LLVM}/bin/llvm-lit"
  if [[ "X${ENABLE_UNIT_TEST}" = "Xon" ]]; then
    AKG_MLIR_ARGS="${AKG_MLIR_ARGS} --target check-akg-mlir"
  fi
}

update_submodule(){
  git submodule update --init --depth 1
}


echo "---------------- AKG: build start ----------------"

if [[ "X$COMPILE_AKG_MLIR" = "Xon" ]]; then
  if [[ "X${COMPILE_MODE}" = "Xakg-mlir-only" ]]; then
    PATH_TO_BUILT_LLVM=${PATH_TO_SOURCE_LLVM}/build
    get_akg_mlir_cmake_args
  else
    update_submodule
    third_party_patch
    build_llvm
    get_akg_mlir_cmake_args
  fi
fi


if [[ "X$CLEAN_BUILT" = "Xon" ]]; then
    make_clean
fi

# Build akg target
cd $BUILD_DIR
cmake .. ${CMAKE_ARGS} ${AKG_MLIR_CMAKE_ARGS}
cmake --build . --config ${_BUILD_TYPE} -j${THREAD_NUM} ${AKG_MLIR_ARGS}
cmake --build . --target install


if [ ! -f "akg-tvm/akg/lib/libakg.so" ];then
  echo "[ERROR] libakg.so not exist!"
  exit 1
fi
if [[ "X$COMPILE_AKG_MLIR" = "Xon" ]]; then
  if [ ! -f "akg-mlir/akg/bin/akg-opt" ];then
    echo "[ERROR] akg-opt not exist!"
    exit 1
  fi
  cp -r akg-mlir/akg/bin ${OUTPUT_PATH}/akg/
  cp -r akg-mlir/akg/lib ${OUTPUT_PATH}/akg/
fi

# Copy target to output/ directory
cp akg-tvm/akg/lib/libakg.so ${OUTPUT_PATH}/akg/lib
cd ${OUTPUT_PATH}
tar czvf libakg.tar.gz akg/
write_checksum_tar
#bash ${AKG_DIR}/scripts/package.sh
cd -
echo "---------------- AKG: build end ----------------"
