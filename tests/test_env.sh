#!/bin/bash

# Copyright 2019 Huawei Technologies Co., Ltd
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

usage()
{
    echo "Usage: . ./test_env.sh [option]"
    echo "       option can be one of [arm64, amd64, camodel, kc_air]."
    echo "       use arm64   if compiled with -DUSE_CCE_RT in arm64 env"
    echo "       use amd64   if compiled with -DUSE_CCE_RT in amd64 env"
    echo "       use camodel if compiled with -DUSE_CCE_RT_SIM"
    echo "       use kc_air  if compiled with -DUSE_KC_AIR"
    echo "       use gpu if compiled with -DUSE_CUDA"
}

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
AKG_DIR="${CUR_DIR}/.."
AD_BUILD_DIR="${AKG_DIR}/build"
TVM_ROOT="${AKG_DIR}/third_party/incubator-tvm"

export LD_LIBRARY_PATH=${AD_BUILD_DIR}:${LD_LIBRARY_PATH}
export PYTHONPATH=${TVM_ROOT}/python:${TVM_ROOT}/topi:${TVM_ROOT}/topi/python:${AKG_DIR}/tests/common:${AKG_DIR}/python:${AKG_DIR}/tests/fuzz/tune:${PYTHONPATH}

if [ $# -eq 1 ]; then
    case "$1" in
        "arm64")
            echo "Configuration setting in arm64 successfully."
            export LD_LIBRARY_PATH=${AD_BUILD_DIR}/aarch64:${LD_LIBRARY_PATH}
            ;;
        "amd64")
            echo "Configuration setting in amd64 successfully."
            export LD_LIBRARY_PATH=${AD_BUILD_DIR}/x86_64:${LD_LIBRARY_PATH}
            ;;
        "camodel")
            echo "Configuration setting in camodel successfully."
            export LD_LIBRARY_PATH=${AD_BUILD_DIR}/camodel:${LD_LIBRARY_PATH}
            ;;
        "kc_air")
            echo "Configuration setting in kc_air successfully."
            if [ -d ${AD_BUILD_DIR}/x86_64 ]; then
                export LD_LIBRARY_PATH=${AD_BUILD_DIR}/x86_64:${LD_LIBRARY_PATH}
            fi
            if [ -d ${AD_BUILD_DIR}/aarch64 ]; then
                export LD_LIBRARY_PATH=${AD_BUILD_DIR}/aarch64:${LD_LIBRARY_PATH}
            fi
            ;;
        "gpu")
            echo "Configuration setting in gpu successfully."
            ;;
	*)
	    echo "Configuration not set."
            usage
	    ;;
    esac
else
    usage
fi

export AIC_MODEL_PATH=/var/model/model_exe
[[ "${PATH}" =~ "aic_model" ]] || export PATH=$PATH:/opt/toolchain/artifacts/bin
