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
import akg
import numpy as np
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from akg.ops.math import Assign
from tests.common.gen_random import random_gaussian
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array

def assign_run(ref_shape, val_shape, dtype, kernel_name="assign", attrs_op={}, cce_path="./", attrs={}):
    attrs.update(attrs_op)
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(Assign, [ref_shape, val_shape], [dtype, dtype], kernel_name=kernel_name,
                                  attrs=attrs, tuning=t)
        if t:
            ref, val, expect = gen_data(dtype, ref_shape, val_shape)
            return mod, expect, (ref, val)
        else:
            return mod
    else:
        ref, val, expect = gen_data(dtype, ref_shape, val_shape)
        mod = utils.op_build_test(Assign, [ref_shape, val_shape], [dtype, dtype], kernel_name=kernel_name,
                                  attrs=attrs)
        fake_output = np.full(val_shape, np.nan, dtype)
        result, _ = utils.mod_launch(mod, (ref, val, fake_output), outputs=(0, -1), expect=expect)
        if attrs.get("profiling", False):
            target_name = attrs["target"].split()[0]
            ref, val, output = to_tvm_nd_array([ref, val, fake_output], akg.tvm.context(target_name, 0))
            target_profiling(mod, ref, val, output, target=target_name, repeat_time=attrs["repeat_times"])
        return (ref, val), result, expect, compare_tensor(result, expect, atol=5e-01, rtol=5e-03, equal_nan=True)

def gen_data(dtype, ref_shape, val_shape):
    if dtype == "float16":
        ref = random_gaussian(ref_shape, miu=1, sigma=0.1).astype(np.float16)
        val = random_gaussian(val_shape, miu=1, sigma=0.1).astype(np.float16)
    elif dtype == "int32":
        ref = np.random.randint(2, size=ref_shape).astype(np.int32)
        val = np.random.randint(2, size=val_shape).astype(np.int32)
    else:
        ref = random_gaussian(ref_shape, miu=1, sigma=0.1).astype(np.float32)
        val = random_gaussian(val_shape, miu=1, sigma=0.1).astype(np.float32)

    expect = val

    return ref, val, expect
