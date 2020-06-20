# Copyright 2020 Huawei Technologies Co., Ltd
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

"""run function for pooling"""

from base import get_rtol_atol
from tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from test_op.pooling import pooling
from test_run.maxpool_run import gen_data as maxpool_gen_data
from test_run.avgpool_run import gen_data as avgpool_gen_data

def pooling_run(shape, dtype, window, stride, mode, pad_mode, pad,
                global_pooling, ceil_mode, attrs):
    """run function"""
    mod = utils.op_build_test(pooling, [shape], [dtype],
                              op_attrs=[window, stride, mode, pad_mode, pad,
                                        global_pooling, ceil_mode],
                              kernel_name="pooling", attrs=attrs)
    pad_mode = "VALID" if pad_mode == 5 else "SAME"
    if window[0] >= shape[2] and window[1] >= shape[3]:
        window = shape[2:4]
        pad_mode = "VALID"
        stride = [1, 1]

    if mode == 0:
        expect, input_, _, out_buf = maxpool_gen_data(dtype, window, pad_mode,
                                                      shape, stride)
    else:
        expect, input_, out_buf = avgpool_gen_data(dtype, window, shape,
                                                   pad_mode, stride)
    output = utils.mod_launch(mod, (input_, out_buf), expect=expect)
    rtol, atol = get_rtol_atol("pooling", dtype)
    cmp_res = compare_tensor(output, expect, rtol=rtol, atol=atol)
    return input_, output, expect, cmp_res
