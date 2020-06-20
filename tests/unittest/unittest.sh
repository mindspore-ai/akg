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

. ../test_env.sh kc_air

casefiles=(
"cce/test_bias_add.py"
"pass/test_helptiling.py"
"pass/test_promote_if.py"
"pass/test_sink_if.py"
"pass/test_ir_parser.py"
"pass/test_elim_vector_mask.py"
"pass/test_copy_propagation.py"
"pass/test_utils_detect_non_linear_index.py"
"pass/test_insn_info.py"
"pass/test_buffer_align.py")

for case in ${casefiles[@]}
do
	echo "start run unit case ${case}!!!"
	python3 ${case}
	if [[ $? -ne "0" ]]; then
	        echo "run unit case ${case} failed!!!"
	        exit 1
	fi
	echo "run unit case ${case} success!!!"
done
