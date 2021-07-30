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

# Basic libraries
import os
import logging
# To use MindTricks
from tests.st.ops.gpu.test_ms_mindtricks import tricks_dir
from tests.st.ops.gpu.test_ms_mindtricks import composite_operators_dir
from tests.st.ops.gpu.test_ms_mindtricks import test_mindtrick
from tests.st.ops.gpu.test_ms_mindtricks import test_single_composite_file
from tests.st.ops.gpu.test_ms_mindtricks import test_composite_operator
from tests.st.ops.gpu.test_ms_mindtricks import batch_test_targets
# Specific tests
from tests.st.ops.gpu.test_ms_reduce_sum import test_ms_reduce_sum

########################################################################################################################

composite_targets = {
    "undefined variables": [
        "Fused_GkDropout_2353362030752466006",
        "Fused_Cast_ReduceSum_Cast_RealDiv_split_16855420756764862693",
    ],
    "precision": [
        "Fused_Cast_BiasAdd_Gelu_fusion_7719078727474100806",
        "Fused_Cast_BiasAdd_Gelu_fusion_7971173442909348882",
        "Fused_Cast_BiasAdd_GkDropout_tuple_getitem_TensorAdd_fusion_13282325956852925231",
        "Fused_Cast_Reshape_BiasAdd_GkDropout_tuple_getitem_TensorAdd_fusion_3191929972328038399",
    ],
    "miscellaneous": [
        "Fused_Transpose_split_18185609042134105765",
    ],
}

########################################################################################################################

def miscellaneous_tests():
    # Previous bug for ms_reduce_sum: allocate var commented out
    logging.info("\033[1mTesting ms_reduce_sum\033[0m")
    test_ms_reduce_sum((21, 4, 28), 'float16', axis=2, keepdims=False, poly_sch=True)
    logging.info("")

def test_swizzle():
    miscellaneous_tests()
    for reason in composite_targets:
        batch_test_targets(reason, composite_targets[reason], with_autogen=True, with_trick=False, without_trick=False)
        batch_test_targets(reason, composite_targets[reason], with_autogen=False, with_trick=True, without_trick=False)

    return True

########################################################################################################################

if __name__ == '__main__':
    test_swizzle()
