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

from akg.utils import kernel_exec as utils
from tensorio import compare_tensor
from test_op.apply_ftrl_v2 import apply_ftrl_v2
from base import get_rtol_atol

from .apply_ftrl_run import gen_data as ftrl_gen_data


def apply_ftrl_v2_run(shape, dtype, attrs=None):
    """run function for dsl function apply_ftrl_v2."""
    scalar_shape = (1,)
    var_shape, accum_shape, linear_shape, grad_shape = [shape] * 4
    lr_shape, l1_shape, l2_shape, l2_shrinkage_shape, lr_power_shape = [scalar_shape] * 5
    shapes = [var_shape, accum_shape, linear_shape, grad_shape, lr_shape, l1_shape, l2_shape,
              l2_shrinkage_shape, lr_power_shape]
    dtypes = [dtype] * 9
    mod = utils.op_build_test(apply_ftrl_v2, shapes, dtypes, kernel_name='apply_ftrl_v2', attrs=attrs)
    expects, (var, accum, linear, grad), (lr, l1, l2, l2_shrinkage, lr_power) = ftrl_gen_data(dtype, shape,
                                                                                              with_l2_shrinkage=True)
    outputs = utils.mod_launch(mod, (var, accum, linear, grad, lr, l1, l2, l2_shrinkage, lr_power),
                               outputs=(0, 1, 2))
    rtol, atol = get_rtol_atol("apply_ftrl_v2", dtype)
    compare_result = list(map(lambda x, y: compare_tensor(x, y, rtol=rtol, atol=atol), outputs, expects))
    inputs = (var, accum, linear, grad, lr, l1, l2, l2_shrinkage)
    return inputs, outputs, expects, all(compare_result)
