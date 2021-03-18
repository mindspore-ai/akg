# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
import numpy as np
from tests.common.gen_random import random_gaussian
from akg.utils import kernel_exec as utils
from akg.utils.result_analysis import gpu_profiling
from akg.utils.format_transform import to_tvm_nd_array
from tests.common.tensorio import compare_tensor
from akg.ops.array_gpu.trans_data import trans_data

def gen_data(shape, axes, dtype):
    support_list = {"float16": np.float16, "float32": np.float32}
    data = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
    expect = data.transpose(axes)
    output = np.full(expect.shape, np.nan, dtype)
    return data, output, expect


def test_ms_trans_data(shape, axes, dtype, poly_sch=False):
    if poly_sch:
        mod = utils.op_build_test(trans_data, [shape], [dtype], op_attrs=[axes], kernel_name="trans_data", attrs={"target": "cuda"})

    data, output, expect = gen_data(shape, axes, dtype)
    output = utils.mod_launch(mod, (data, output), expect = expect)
    ret = compare_tensor(output, expect, rtol=5e-03, atol=1.e-8, equal_nan=True)
    print("Test {}".format("Pass" if ret else "Failed"))
    data, expect = to_tvm_nd_array([data, expect])
    gpu_profiling(mod, data, expect, 400)

if __name__ == '__main__':
    try:
        test_ms_trans_data((8, 24, 38, 38), (0,2,1,3), 'float32')
    except:
        print("compile trans data manual schedule FP32 failed")

    try:
        test_ms_trans_data((8, 24, 38, 38), (0,2,1,3), 'float16')
    except:
        print("compile trans data manual schedule FP16 failed")

    try:
        test_ms_trans_data((8, 24, 38, 38), (0,2,1,3), 'float32', True)
    except:
        print("compile trans data poly auto schedule FP32 failed")

    try:
        test_ms_trans_data((8, 24, 38, 38), (0,2,1,3), 'float16', True)
    except:
        print("compile trans data poly auto schedule FP16 failed")