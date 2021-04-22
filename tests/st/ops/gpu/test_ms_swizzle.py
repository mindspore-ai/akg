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
    "miscellaneous tricks": [
        "Fused_Transpose_split_18185609042134105765",
    ],
}

########################################################################################################################

def test_composite_operator(target, with_trick=True, without_trick=False, attrs=None):
    operator = composite_operators_dir + "/" + target + ".info"
    trick = tricks_dir + "/" + target + ".json"

    # Depending on the operator, we may want to test it both with and without tricks
    # or just one of the two.
    # We also need to be sure a trick exists
    test_with_trick = with_trick and os.path.isfile(trick)
    test_without_trick = without_trick or not os.path.isfile(trick)

    if test_with_trick:
        test_mindtrick(operator, trick);
    if test_without_trick:
        if attrs is None:
            attrs = {}
        attrs["target"] = "cuda"
        test_single_composite_file(operator, attrs, poly=True);

def miscellaneous_tests():
    # Previous bug for ms_reduce_sum: allocate var commented out
    logging.info("\033[1mTesting ms_reduce_sum\033[0m")
    test_ms_reduce_sum((21, 4, 28), 'float16', axis=2, keepdims=False, poly_sch=True)
    logging.info("")

def batch_test_targets(targets, with_trick=True, without_trick=False, attrs=None):
    """Quickly test multiple targets using test_composite_operator()"""
    log_header = "\033[1m\033[7m " + str(targets) + " \033[0m "
    log_header += "with_trick=" + str(with_trick) + ", without_trick=" + str(without_trick)
    logging.info(log_header)

    for target in composite_targets[targets]:
        test_composite_operator(target, with_trick, without_trick, attrs)
    logging.info("")

    return True

def test_swizzle():
    miscellaneous_tests()
    for reason in composite_targets:
        batch_test_targets(reason, with_trick=True, without_trick=False)
        batch_test_targets(reason, with_trick=False, without_trick=True)

    return True

########################################################################################################################

if __name__ == '__main__':
    test_swizzle()
