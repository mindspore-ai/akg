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

"""test"""
import time
from autotuning.job import launch
from akg.utils import kernel_exec
from akg.ops.math_gpu import reduce_sum
from autotuning.type_definitions import ReduceGpuDesc
import numpy as np
import sys
import argparse
from autotuning.tuning_utils import get_skip_configs_from_log, get_tuning_attrs_from_json


def reduce_sum_gpu_execute(in_shape, dtype, axis=None, keepdims=False, attrs=False):
    mod = utils.op_build_test(reduce_sum, (in_shape, ), (in_dtype, ),
                              kernel_name="reduce_sum_gpu", op_attrs=[axis, keepdims],
                              attrs={"target": "cuda", "enable_akg_reduce_lib": True})
    return mod

def run_test_reduce_sum(in_shape, in_dtype, axis=None, keepdims=False, skip_config_set=None, tuning_attrs_info=None):
    time_start = time.time()
    op_type_ = 'reduce_sum_gpu'
    debug_mode_ = True
    save_res_ = True
    all_space_ = True
    op_config = [in_shape, in_dtype, axis, keepdims,
                 "", "", "",
                 True, True, True]
    op_config = ReduceGpuDesc(*op_config)
    desc_ = ('reduce_sum_gpu', reduce_sum_gpu_execute,
             op_config, tuning_attrs_info)
    launch(op_type=op_type_, debug_mode=debug_mode_,
           save_res=save_res_, desc=desc_, all_space=all_space_,
           from_json=False, skip_config_set=skip_config_set,
           tuning_attrs_info=tuning_attrs_info)
    time_end = time.time()
    print("total tuning time: ", time_end - time_start)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_configs_log", type=str,
                        default="", help="skip those configs in .log file")
    parser.add_argument("--tuning_attrs_json", type=str, default="",
                        help="the json file to describe the tuning atttrs")
    args = parser.parse_args()

    # check whether have configs need to skip
    skip_config_set = get_skip_configs_from_log(args.skip_configs_log)

    # add tuning_attrs from json file
    tuning_attrs_info = get_tuning_attrs_from_json(args.tuning_attrs_json)

    run_test_reduce_sum((1024, 1024), "float32", (1,),
                        False, skip_config_set=skip_config_set, tuning_attrs_info=tuning_attrs_info)
