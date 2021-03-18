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

export RUNTIME_MODE="air_cloud"

if [ $# -eq 1 ] && [ $1 = "gpu_ci" ]; then
  echo "Argument gpu_ci is used in CI and will be deprecated."
else
  CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
  AKG_DIR="${CUR_DIR}/.."
  AKG_BUILD_DIR="${AKG_DIR}/build"
  TVM_ROOT="${AKG_DIR}/third_party/incubator-tvm"

  export LD_LIBRARY_PATH=${AKG_BUILD_DIR}:${LD_LIBRARY_PATH}
  export PYTHONPATH=${TVM_ROOT}/python:${TVM_ROOT}/topi:${TVM_ROOT}/topi/python:${AKG_DIR}:${AKG_DIR}/python:${PYTHONPATH}
  if [ $# -eq 1 ] && [ $1 = "gpu" ]; then
    export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}
  fi

  echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
  echo "PYTHONPATH: ${PYTHONPATH}"
fi