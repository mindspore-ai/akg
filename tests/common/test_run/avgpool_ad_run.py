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

from tensorio import compare_tensor
import numpy as np
from akg.utils import kernel_exec as utils
from . import avgpool_run
from . import avgpool_grad_run
from test_op.avgpool_ad import avgpool_ad
from test_op.avgpool_ad import avgpool_ad_no_custom_diff_manual_schedule
from gen_random import random_gaussian

def avgpool_ad_run(shape, kernel, stride, pad, dtype, polyhedral=False, attrs=None):
    support_list = {"float16": np.float16, "float32": np.float32}
    if attrs is None:
        attrs = {'loop_partition_unroll': True}
    else:
        attrs['loop_partition_unroll'] = True

    kernel_name = 'avgpool_ad'
    if polyhedral:
        avgpool = avgpool_ad
    else:
        kernel_name = kernel_name + "_manual_schedule"
        avgpool = avgpool_ad_no_custom_diff_manual_schedule

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        input = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
        y = avgpool_run.benchmark(input, kernel, stride, pad)
        mod = utils.op_build_test(avgpool, [y.shape, shape], [dtype, dtype], op_attrs=[kernel, stride, pad],
                                  kernel_name=kernel_name, attrs=attrs, log_cce=True, dump_code=True, tuning=t)
        if t:
            expect, head, output = gen_data(dtype, input, kernel, pad, stride, support_list, y)
            return mod, expect, (head, input, output)
        else:
            return mod
    else:
        input = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
        y = avgpool_run.benchmark(input, kernel, stride, pad)
        mod = utils.op_build_test(avgpool, [y.shape, shape], [dtype, dtype], op_attrs=[kernel, stride, pad],
                                  kernel_name=kernel_name, attrs=attrs, log_cce=True, dump_code=True)
        expect, head, output = gen_data(dtype, input, kernel, pad, stride, support_list, y)
        output = utils.mod_launch(mod, [head, input, output], expect=expect)

        return [head, input], output, expect, compare_tensor(output, expect, rtol=5e-03, atol=5e-03, equal_nan=True)


def gen_data(dtype, input, kernel, pad, stride, support_list, y):
    head = random_gaussian(y.shape, miu=1, sigma=0.1).astype(support_list[dtype])
    expect = avgpool_grad_run.benchmark(dtype, input, y, head, kernel, stride, pad)
    out_shape = expect.shape
    output = np.full(out_shape, 0, dtype)
    return expect, head, output
