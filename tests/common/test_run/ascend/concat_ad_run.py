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

import numpy as np
import akg
import akg.tvm
from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend import concat_ad
from akg.utils import validation_check as vc_util
from tests.common.gen_random import random_gaussian
from tests.common.tensorio import compare_tensor


def concat_ad_run(shapes, dtype, axis, attrs):
    # prepare inputs placeholder
    inp_dtype = dtype.lower()
    data = []
    for i in range(len(shapes)):
        shape = shapes[i]
        data.append(akg.tvm.placeholder(shape, name="data_%d" % i, dtype=inp_dtype))

    kernel_name = utils.genKernelName("concat", inp_dtype, shapes)
    res, head = concat_ad.concat_ad(data, axis)

    opvars = [head] + data + [res]
    s = akg.tvm.create_schedule(res.op)
    op_attrs = [axis]

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(concat_ad.concat_ad, [shapes], [dtype.lower()], op_attrs,
                                  kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            args, expect, head_data, inputs = gen_data(dtype, head, shapes)
            return mod, expect, tuple(args)
        else:
            return mod
    else:
        # build the cce kernel
        with akg.build_config(add_lower_pass=utils.debug_mode(0), dump_pass_ir=True):
            mod = akg.build(s, opvars, "cce", name=kernel_name, attrs=attrs, polyhedral=True)
        print(mod.imported_modules[0].get_source())

        args, expect, head_data, inputs = gen_data(dtype, head, shapes)
        output = utils.mod_launch(mod, tuple(args), expect=expect)
        return tuple(inputs) + (head_data, ), output, expect, compare_tensor(output, expect, rtol=5e-03, equal_nan=True)


def gen_data(dtype, head, shapes):
    # check shapes
    for i in range(len(shapes)):
        vc_util.check_shape(shapes[i])
    # check types
    support_list = {"float16": np.float16, "int32": np.int32}
    if not (dtype.lower() in support_list):
        raise RuntimeError("concat_cce only support %s while dtype is %s" % (",".join(support_list.keys()), dtype))
    inputs = []
    for i in range(len(shapes)):
        shape = shapes[i]
        input = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
        inputs.append(input)
    head_data = random_gaussian([x.value for x in head.shape], miu=1, sigma=0.1).astype(support_list[dtype])
    expect = head_data[[slice(0, x) for x in inputs[0].shape]]
    output_shape = shapes[0]
    output = np.full(output_shape, np.nan, dtype)
    args = inputs + [head_data]
    args.append(output)
    return args, expect, head_data, inputs
