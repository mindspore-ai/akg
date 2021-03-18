#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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
# ============================================================================

set -e

CURRPATH=$(cd $(dirname $0); pwd)

if [ $# -gt 0 ]; then
    if [ $1 == "python" ]; then
        echo "run python ut"
        if [ -f ${CURRPATH}/python/runtest.sh ]; then
            bash ${CURRPATH}/python/runtest.sh $2
        fi
    elif [ $1 == "cpp" ]; then
        echo "run cpp ut"
        if [ -f ${CURRPATH}/cpp/runtest.sh ]; then
            bash ${CURRPATH}/cpp/runtest.sh
        fi
    fi
else
    echo "run all ut"
    # run python testcases
    if [ -f ${CURRPATH}/python/runtest.sh ]; then
        bash ${CURRPATH}/python/runtest.sh $2
    fi

    # run c++ ut testcases
    if [ -f ${CURRPATH}/cpp/runtest.sh ]; then
        bash ${CURRPATH}/cpp/runtest.sh
    fi
fi
