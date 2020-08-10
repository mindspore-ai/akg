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
from akg.utils import kernel_exec as utils
from test_run.maxpool_ad_run import gen_data
from akg.ops.nn.maxpool_grad_with_argmax import maxpool_grad_with_argmax
from base import get_rtol_atol


def maxpool_grad_with_argmax_run(shape, kernel, stride, pad, dtype, polyhedral=False, attrs=None):
    expect, head, input, output, forward, mask = gen_data(dtype, kernel, pad, shape, stride, True)

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        if polyhedral:
            raise Exception("ERROR: no poly support for maxpool_grad_with_argmax, please select the mansch version")
        else:
            mod = utils.op_build_test(maxpool_grad_with_argmax, [head.shape, mask.shape],
                                      [dtype, dtype], kernel_name="maxpool_grad_with_argmax",
                                      op_attrs=[shape, kernel, stride, pad], attrs=attrs,
                                      log_cce=False, dump_code=True, polyhedral=polyhedral)
        if t:
            return mod, expect, (head, mask, output)
        else:
            return mod
    else:
        if polyhedral:
            raise Exception("ERROR: no poly support for maxpool_grad_with_argmax, please select the mansch version")
        else:
            mod = utils.op_build_test(maxpool_grad_with_argmax, [head.shape, mask.shape],
                                      [dtype, dtype], kernel_name="maxpool_grad_with_argmax",
                                      op_attrs=[shape, kernel, stride, pad], attrs=attrs,
                                      log_cce=False, dump_code=True, polyhedral=polyhedral)
            output = utils.mod_launch(mod, [head, mask, output], expect=expect)

        rtol, atol = get_rtol_atol("maxpool_grad_with_argmax", dtype)
        return [head, mask], output, expect, compare_tensor(output, expect, rtol=rtol, atol=atol, equal_nan=True)
