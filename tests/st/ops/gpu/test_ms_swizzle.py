#!/usr/bin/env python3
# coding: utf-8
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
# limitations under the License

# To use MindTricks
from tests.st.ops.gpu.test_ms_mindtricks import tricks_dir
from tests.st.ops.gpu.test_ms_mindtricks import composite_operators_dir
from tests.st.ops.gpu.test_ms_mindtricks import test_mindtrick
# Specific tests
from tests.st.ops.gpu.test_ms_reduce_sum import test_ms_reduce_sum

########################################################################################################################

def test_swizzle_pass_should_do_nothing():
    test_ms_reduce_sum((21, 4, 28), 'float16', axis=2, keepdims=False, poly_sch=True)

def test_mindtricks_with_swizzle():
    targets = [
        "Fused_Transpose_split_18185609042134105765"
    ]
    for target in targets:
        operator = composite_operators_dir + "/" + target + ".info"
        trick = tricks_dir + "/" + target + ".json"
        test_mindtrick(operator, trick)

def test_swizzle():
    test_swizzle_pass_should_do_nothing()
    test_mindtricks_with_swizzle()

    return True

########################################################################################################################

if __name__ == '__main__':
    test_swizzle()
