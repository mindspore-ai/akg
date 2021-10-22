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


########################################################################################################################

# Basic libraries
import os
import sys
import json
import pytest
import logging

# For composite cases
from akg import composite
from akg.utils import kernel_exec as utils
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array
from tests.common.gen_json_data import gen_json_data
from tests.common.base import get_rtol_atol
from tests.common.tensorio import compare_tensor

# Specific GPU tests
from tests.st.ops.gpu.test_fused_bn_update import test_fused_bn_update
from tests.st.ops.gpu.test_fused_relu_grad import test_fused_relu_grad

########################################################################################################################

mind_trick_cases_dir = "./mind-trick_cases"

# Note: no need to hardcode trick paths for composite_operators unless the trick's name differs from the operator name
tricks_dir = mind_trick_cases_dir + "/tricks"
tricks_filenames = {
    "fused_bn_update": "fused_bn_update.json",
    "fused_relu_grad": "fused_relu_grad.json",
}

composite_operators_dir = mind_trick_cases_dir + "/operators"
composite_targets = {
    "miscellaneous": [
        "Fused_AddN_fusion_9584919353229493170",
        "Fused_Cast_BiasAdd_Gelu_fusion_7719078727474100806",
        "Fused_Cast_BiasAdd_GkDropout_tuple_getitem_TensorAdd_fusion_13282325956852925231",
        "Fused_Cast_RealDiv_Reshape_FusedAdamWeightDecay_fusion_1039082044534023692",
        "Fused_Cast_RealDiv_Reshape_FusedAdamWeightDecay_fusion_1545859458890067484",
        "Fused_Cast_RealDiv_Reshape_FusedAdamWeightDecay_fusion_1976850843332086880",
        "Fused_GkDropout_2353362030752466006",
        "Fused_Transpose_split_18185609042134105765",
    ],
}

########################################################################################################################

def _get_json_dict(desc):
    return json.loads(desc) if isinstance(desc, str) else desc

def _get_backend(desc):
    json_obj = _get_json_dict(desc)
    if "process" not in json_obj.keys():
        logging.info("Can't identify the backend.")
        return None
    return json_obj["process"]

def _compare_func(output, expect):
    rtol, atol = get_rtol_atol("FUSED", str(output.dtype))
    return compare_tensor(output, expect, rtol=rtol, atol=atol)

def get_result(desc, poly, attrs=None):
    backend = _get_backend(desc)
    if attrs is None:
        attrs = {}

    build_attrs = attrs if attrs else None
    mod = composite.build(desc, build_attrs, poly=poly)

    input_for_mod, expect, output_indexes = gen_json_data(desc)
    output = utils.mod_launch(mod, input_for_mod, output_indexes)

    if not all(map(_compare_func, output if isinstance(output, (list, tuple)) else [output],
                   expect if isinstance(expect, (list, tuple)) else [expect])):
        logging.info(mod.imported_modules[0].get_source())
        return False
    if backend == "cuda":
        inputs = to_tvm_nd_array(input_for_mod)
        expect = to_tvm_nd_array(expect)
        target_profiling(mod, *inputs, *expect, repeat_time=400)
    return True

def test_gpu_cases():
    # Check some GPU cases
    with open(tricks_dir + "/" + tricks_filenames["fused_bn_update"], "r") as trick:
        test_fused_bn_update((2048,), poly_sch=True, mind_trick=trick.read())
    with open(tricks_dir + "/" + tricks_filenames["fused_relu_grad"], "r") as trick:
        test_fused_relu_grad((256, 56, 56, 256), poly_sch=True, mind_trick=trick.read())

    return True

def test_single_composite_file(input_file, attrs, poly):
    with open(input_file, 'r') as f:
        desc = f.read()
        operator_name = json.loads(desc)["op"]

        logging.info("\033[1mTesting %s\033[0m", operator_name)
        if get_result(desc, poly, attrs):
            logging.info("\033[1m\033[32m%s: Success\033[0m", operator_name)
        else:
            logging.info("\033[1m\033[33m%s: Precision Error\033[0m", operator_name)
            logging.info("input: %s", str(input_file))
            logging.info("attrs: %s", str(attrs))
            logging.info("poly: %s", str(poly))
            raise ValueError("Precision Error")

def test_mindtrick(operator_path, trick_path):
    if os.path.isfile(operator_path) and os.path.isfile(trick_path):
        trick = open(trick_path, "r")
        attrs = {
            "target": "cuda",
            "mind_trick": trick.read(),
        }
        test_single_composite_file(operator_path, attrs, poly=True)
    return True

def test_composite_operator(target, with_autogen=True, with_trick=False, without_trick=False, attrs=None):
    operator = composite_operators_dir + "/" + target + ".info"
    trick = tricks_dir + "/" + target + ".json"

    # Depending on the operator, we may want to test it both with and without tricks
    # or just one of the two.
    # We also need to be sure a trick exists
    test_with_trick = with_trick and os.path.isfile(trick)
    test_without_trick = without_trick or not os.path.isfile(trick)

    if with_autogen:
        current_attrs = {} if attrs is None else attrs
        current_attrs["enable_mind_trick_autogen"] = 1
        test_single_composite_file(operator, current_attrs, poly=True)
    if test_with_trick:
        test_mindtrick(operator, trick);
    if test_without_trick:
        test_single_composite_file(operator, attrs, poly=True);

def batch_test_targets(reason, targets, with_autogen=True, with_trick=False, without_trick=False, attrs=None):
    """Quickly test multiple targets using test_composite_operator()"""
    log_header = "\033[1m\033[7m " + str(reason) + " \033[0m "
    log_header += "with_autogen=" + str(with_autogen) + ", with_trick=" + str(with_trick) + ", without_trick=" + str(without_trick)
    logging.info(log_header)

    for target in targets:
        test_composite_operator(target, with_autogen, with_trick, without_trick, attrs)
    logging.info("")

    return True

def test_composite_cases(operators):
    for reason in operators:
        batch_test_targets(reason, operators[reason], with_autogen=False, with_trick=True, without_trick=False)
        batch_test_targets(reason, operators[reason], with_autogen=True, with_trick=False, without_trick=False)

    return True

########################################################################################################################

def test_mindtricks(cases):
    if "gpu" in cases:
        test_gpu_cases()
    if "composite" in cases:
        test_composite_cases(composite_targets)

    return True

########################################################################################################################

if __name__ == '__main__':
    test_mindtricks(["gpu", "composite"])
