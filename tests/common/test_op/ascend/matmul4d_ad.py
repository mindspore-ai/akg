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

"""operator dsl function: matmul4d_ad"""
import akg.tvm
import akg
from akg.ops.math.ascend import MatMul
from akg.utils import custom_tiling as ct_util


def get_shape(pld): return [d.value for d in pld.shape]


matmul4d_ad_set_dim_map = {
    str(([1, 4, 64, 16, 16], [1, 64, 2, 16, 16], False, False)): ((4, 4), (16, 16), (128, 128), (16, 16), (16, 16)),
    str(([1, 64, 4, 16, 16], [1, 64, 2, 16, 16], True, False)): ((4, 4), (16, 16), (128, 128), (16, 16), (16, 16)),
    str(([1, 4, 64, 16, 16], [1, 2, 64, 16, 16], False, True)): ((4, 4), (16, 16), (128, 128), (16, 16), (16, 16)),
    str(([1, 64, 4, 16, 16], [1, 2, 64, 16, 16], True, True)): ((4, 4), (16, 16), (128, 128), (16, 16), (16, 16)),
    str(([1, 4, 8, 16, 16], [1, 8, 2, 16, 16], False, False)): ((65536, 65536), (65536, 65536), (65536, 65536), (65536, 65536), (65536, 65536)),

}


def matmul4d_ad_set_dim_func(head, x, y, b, out_dtype, adj_x=False, adj_y=False):
    key = []
    key.append(get_shape(x))
    key.append(get_shape(y))
    key.append(adj_x)
    key.append(adj_y)

    hash_key = str(tuple(key))

    if hash_key in matmul4d_ad_set_dim_map.keys():
        return ct_util.set_dims(matmul4d_ad_set_dim_map[hash_key])
    else:
        return ""


@ct_util.reg_set_dim_func(matmul4d_ad_set_dim_func)
def matmul4d_ad(head, x, y, b, out_dtype, adj_x=False, adj_y=False):
    """compute 4d format mat shape from shape inputs."""
    shape_xx = get_shape(x)

    if adj_x:  # no need to change in this case
        shape_xx_forward = shape_xx

    else:
        batch_num, m_o, k_o, m_i, k_i = shape_xx
        shape_xx_forward = (batch_num, k_o, m_o, k_i, m_i)

    ########################################
    #  compute the forward kernel          #
    ########################################

    x_temp = akg.tvm.placeholder(shape_xx_forward, name="input_1", dtype=x.dtype)

    # we transfer all cases to that of adj_x=False
    out = MatMul(x_temp, y, b, out_dtype, "zN", "nZ", "zN", False, adj_y)[0]

    ########################################
    #  compute the backward kernel         #
    ########################################

    _jacs = list(akg.differentiate(out, [x_temp], head))

    if adj_x:
        grad = akg.tvm.compute(shape_xx, lambda n, ko, mo, ki, mi: _jacs[0][n, ko, mo, mi, ki])
    else:
        grad = akg.tvm.compute(shape_xx, lambda n, mo, ko, mi, ki: _jacs[0][n, ko, mo, mi, ki])

    sjacs = akg.tvm.create_schedule([grad.op])

    attrs = dict()

    attrs["pragma_data_transpose"] = "Y"
    attrs["pragma_data_transpose_block"] = "Y"
    if not adj_y:
        attrs["pragma_weight_transpose"] = "Y"

    return grad, attrs
