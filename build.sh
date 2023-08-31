#!/bin/bash
# Copyright 2019-2021 Huawei Technologies Co., Ltd
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
    echo "bash build.sh [-e cpu|gpu|ascend|all] [-j[n]] [-t on|off] [-o] [-u]"
    echo ""
    echo "Options:"
    echo "    -d Debug mode"
    echo "    -e Hardware environment: cpu, gpu, ascend or all"
    echo "    -j[n] Set the threads when building (Default: -j8)"
    echo "    -t Unit test: on or off (Default: off)"
    echo "    -o Output .o file directory"
    echo "    -u Enable auto tune"
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
while getopts 'e:j:u:t:od' opt
do
    case "${opt}" in
        e)
            if [[ "${OPTARG}" == "gpu" ]]; then
                CMAKE_ARGS="${CMAKE_ARGS} -DUSE_CUDA=ON -DUSE_LLVM=ON -DUSE_RPC=ON"
            elif [[ "${OPTARG}" == "ascend" ]]; then
                CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_D=ON -DUSE_LLVM=ON"
            elif [[ "${OPTARG}" == "cpu" ]]; then
                # AKG requires LLVM on CPU, the optimal version is 12.xx.xx.
                # if not found in the environment, it will find another existing version to use.
                CMAKE_ARGS="${CMAKE_ARGS} -DUSE_LLVM=ON -DUSE_RPC=ON"
            elif [[ "${OPTARG}" == "all" ]]; then
                CMAKE_ARGS="${CMAKE_ARGS} -DUSE_CUDA=ON -DENABLE_D=ON -DUSE_LLVM=ON -DUSE_RPC=ON"
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
            ;;
        u)
            CMAKE_ARGS="${CMAKE_ARGS} -DUSE_AUTO_TUNE=1"
            ;;
        d)
            CMAKE_ARGS="${CMAKE_ARGS} -DCMAKE_BUILD_TYPE=Debug -DUSE_AKG_LOG=1"
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
        *)
            echo "Unknown option ${opt}!"
            usage
            exit 1
    esac
done
echo "CMAKE_ARGS: ${CMAKE_ARGS}"

# Create directories
mkdir -pv "${BUILD_DIR}"
mkdir -pv "${OUTPUT_PATH}"

echo "---------------- AKG: build start ----------------"

# Build target
cd $BUILD_DIR
cmake .. ${CMAKE_ARGS}
make -j$THREAD_NUM
make install

if [ ! -f "libakg.so" ];then
  echo "[ERROR] libakg.so not exist!"
  exit 1
fi

# Copy target to output/ directory
cp libakg.so ${OUTPUT_PATH}
cd ${OUTPUT_PATH}
tar czvf libakg.tar.gz libakg.so
rm -rf libakg.so
write_checksum_tar
bash ${AKG_DIR}/scripts/package.sh

cd -
echo "---------------- AKG: build end ----------------"
