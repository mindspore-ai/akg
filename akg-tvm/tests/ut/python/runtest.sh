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

CURRPATH=$(cd $(dirname $0); pwd)
TESTS_DIR=${CURRPATH}/../..
. ${TESTS_DIR}/test_env.sh
cd ${CURRPATH}

casefiles=(
"${CURRPATH}/pass/test_promote_if.py"
"${CURRPATH}/pass/test_sink_if.py"
"${CURRPATH}/pass/test_copy_propagation.py"
)

for case in ${casefiles[@]}
do
    echo "start run unit case ${case}!!!"
    python3 ${case}
    if [[ $? -ne "0" ]]; then
        echo "run unit case ${case} failed!!!"
        exit 1
    else
        echo "run unit case ${case} success!!!"
    fi
done