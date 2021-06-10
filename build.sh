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
    echo "bash build.sh [-e gpu|ascend] [-j[n]] [-t on|off] [-a]"
    echo ""
    echo "Options:"
    echo "    -e Hardware environment: gpu or ascend"
    echo "    -j[n] Set the threads when building (Default: -j8)"
    echo "    -t Unit test: on or off (Default: off)"
    echo "    -a Download libakg_ext.a"
}

mk_new_dir()
{
    local create_dir="$1"

    if [[ -d "${create_dir}" ]]; then
        rm -rf "${create_dir}"
    fi

    mkdir -pv "${create_dir}"
}

write_checksum()
{
    cd "$OUTPUT_PATH" || exit
    PACKAGE_LIST=$(ls lib*.tar.gz) || exit
    for PACKAGE_NAME in $PACKAGE_LIST; do
        echo $PACKAGE_NAME
        sha256sum -b "$PACKAGE_NAME" >"$PACKAGE_NAME.sha256"
    done
}

acquire_lib_url()
{
    arch_info=`arch | tr '[A-Z]' '[a-z]'`
    arch_name=""
    if [[ "${arch_info}" =~ "aarch64" ]]; then
        arch_name="aarch64"
    elif [[ "${arch_info}" =~ "x86_64" ]]; then
        arch_name="x86_64"
    fi
    url_prefix="https://repo.mindspore.cn/public/ms-incubator/akg-binary/version"
    lib_mark="202106/20210610/master_20210610165225_fc8a1d1eef69e2d828b98701a771fc4268b551db"
    lib_url="${url_prefix}/${lib_mark}/lib/${arch_name}/libakg_ext.a"
    echo "${lib_url}"
}

if [ ! -n "$1" ]; then
    echo "Must input parameter!"
    usage
    exit 1
fi

# Parse arguments
THREAD_NUM=32
CMAKE_ARGS=""
while getopts 'e:j:t:a' opt
do
    case "${opt}" in
        e)
            if [[ "${OPTARG}" == "gpu" ]]; then
                CMAKE_ARGS="${CMAKE_ARGS} -DUSE_CUDA=ON -DUSE_RPC=ON"
            elif [[ "${OPTARG}" == "ascend" ]]; then
                CMAKE_ARGS="${CMAKE_ARGS} -DUSE_CCE_RT=1"
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
        a)
            acquire_lib_url
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
mk_new_dir "${BUILD_DIR}"
mk_new_dir "${OUTPUT_PATH}"

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
write_checksum

cd -
echo "---------------- AKG: build end ----------------"
