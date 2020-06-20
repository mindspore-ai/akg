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

"""two2fractal"""
import akg
import akg.tvm
from akg.tvm.hybrid import script
from test_op import pad
from akg.utils import custom_tiling as ct_util
from akg.utils.format_transform import get_const


two2fractal_set_dim_map = {
    str(((128, 768, 128), 'zN', 'float32')): ((1, 1), (1, 1), (1, 1), (16, 1), (16, 1)),
    str(((125, 256), 'nZ', 'float16')) : ((0,0,1,1),(0,1,16,1),(0,2,16,1),(0,3,16,1),
                                         (1,0,1,1),(1,1,16,1),(1,2,13,1),
                                         (2,0,16,1),(2,1,16,1),(2,2,3,1)),
}


def two2fractal_set_dim_func(data, format_):
    shape = [x for x in data.shape]

    hash_key = str((tuple(shape), format_, data.dtype))
    return ct_util.set_dims_by_key(hash_key, two2fractal_set_dim_map), hash_key


@ct_util.reg_set_dim_func(two2fractal_set_dim_func)
def two2fractal(data, format_):
    support_formats = ['zN', 'zZ', 'nZ']
    shape = [get_const(x) for x in data.shape]

    assert format_ in support_formats
    assert len(shape) >= 2 and len(shape) <= 4

    m, n = shape[-2:]
    if len(shape) == 3:
        b = shape[0]
    if len(shape) == 4:
        b, s = shape[:2]
    pad_m, pad_n = m, n
    if m % 16 != 0:
        pad_m = (m + 15) // 16 * 16
    if n % 16 != 0:
        pad_n = (n + 15) // 16 * 16
    m1, n1 = pad_m // 16, pad_n // 16
    m0, n0 = 16, 16

    @script(capture=locals())
    def reshape_zn_2d(inputs, zero):
        output = allocate((n1, m1, m0, n0), inputs.dtype, 'local')
        for n_i in range(n1):
            for m_i in range(m1):
                for m_i0 in range(m0):
                    for n_i0 in range(n0):
                        if (m_i * 16 + m_i0 >= m):
                            output[n_i, m_i, m_i0, n_i0] = zero
                        else:
                            output[n_i, m_i, m_i0, n_i0] = inputs[m_i * 16 + m_i0, n_i * 16 + n_i0]
        return output

    @script(capture=locals())
    def reshape_zn_3d(inputs, zero):
        output = allocate((b, n1, m1, m0, n0), inputs.dtype, 'local')
        for b_i in range(b):
            for n_i in range(n1):
                for m_i in range(m1):
                    for m_i0 in range(m0):
                        for n_i0 in range(n0):
                            if (m_i * 16 + m_i0 >= m):
                                output[b_i, n_i, m_i, m_i0, n_i0] = zero
                            else:
                                output[b_i, n_i, m_i, m_i0, n_i0] = inputs[b_i, m_i * 16 + m_i0, n_i * 16 + n_i0]
        return output

    @script(capture=locals())
    def reshape_zn_4d(inputs, zero):
        output = allocate((b, s, n1, m1, m0, n0), inputs.dtype, 'local')
        for b_i in range(b):
            for s_i in range(s):
                for n_i in range(n1):
                    for m_i in range(m1):
                        for m_i0 in range(m0):
                            for n_i0 in range(n0):
                                if (m_i * 16 + m_i0 >= m):
                                    output[b_i, s_i, n_i, m_i, m_i0, n_i0] = zero
                                else:
                                    output[b_i, s_i, n_i, m_i, m_i0, n_i0] = inputs[b_i, s_i, m_i * 16 + m_i0, n_i * 16 + n_i0]
        return output

    @script(capture=locals())
    def reshape_nz_2d(inputs, zero):
        output = allocate((m1, n1, n0, m0), inputs.dtype, 'local')
        for m_i in range(m1):
            for n_i in range(n1):
                for n_i0 in range(n0):
                    for m_i0 in range(m0):
                        if (m_i * 16 + m_i0 >= m):
                            output[m_i, n_i, n_i0, m_i0] = zero
                        else:
                            output[m_i, n_i, n_i0, m_i0] = inputs[m_i * 16 + m_i0, n_i * 16 + n_i0]
        return output

    @script(capture=locals())
    def reshape_nz_3d(inputs, zero):
        output = allocate((b, m1, n1, n0, m0), inputs.dtype, 'local')
        for b_i in range(b):
            for m_i in range(m1):
                for n_i in range(n1):
                    for n_i0 in range(n0):
                        for m_i0 in range(m0):
                            if (m_i * 16 + m_i0 >= m):
                                output[b_i, m_i, n_i, n_i0, m_i0] = zero
                            else:
                                output[b_i, m_i, n_i, n_i0, m_i0] = inputs[b_i, m_i * 16 + m_i0, n_i * 16 + n_i0]
        return output

    @script(capture=locals())
    def reshape_nz_4d(inputs, zero):
        output = allocate((b, s, m1, n1, n0, m0), inputs.dtype, 'local')
        for b_i in range(b):
            for s_i in range(s):
                for m_i in range(m1):
                    for n_i in range(n1):
                        for n_i0 in range(n0):
                            for m_i0 in range(m0):
                                if (m_i * 16 + m_i0 >= m):
                                    output[b_i, s_i, m_i, n_i, n_i0, m_i0] = zero
                                else:
                                    output[b_i, s_i, m_i, n_i, n_i0, m_i0] = inputs[b_i, s_i, m_i * 16 + m_i0, n_i * 16 + n_i0]
        return output

    @script(capture=locals())
    def reshape_zz_2d(inputs, zero):
        output = allocate((m1, n1, m0, n0), inputs.dtype, 'local')
        for m_i in range(m1):
            for n_i in range(n1):
                for m_i0 in range(m0):
                    for n_i0 in range(n0):
                        if (m_i * 16 + m_i0 >= m):
                            output[m_i, n_i, m_i0, n_i0] = zero
                        else:
                            output[m_i, n_i, m_i0, n_i0] = inputs[m_i * 16 + m_i0, n_i * 16 + n_i0]
        return output

    @script(capture=locals())
    def reshape_zz_3d(inputs, zero):
        output = allocate((b, m1, n1, m0, n0), inputs.dtype, 'local')
        for b_i in range(b):
            for m_i in range(m1):
                for n_i in range(n1):
                    for m_i0 in range(m0):
                        for n_i0 in range(n0):
                            if (m_i * 16 + m_i0 >= m):
                                output[b_i, m_i, n_i, m_i0, n_i0] = zero
                            else:
                                output[b_i, m_i, n_i, m_i0, n_i0] = inputs[b_i, m_i * 16 + m_i0, n_i * 16 + n_i0]
        return output

    @script(capture=locals())
    def reshape_zz_4d(inputs, zero):
        output = allocate((b, s, m1, n1, m0, n0), inputs.dtype, 'local')
        for b_i in range(b):
            for s_i in range(s):
                for m_i in range(m1):
                    for n_i in range(n1):
                        for m_i0 in range(m0):
                            for n_i0 in range(n0):
                                if (m_i * 16 + m_i0 >= m):
                                    output[b_i, s_i, m_i, n_i, m_i0, n_i0] = zero
                                else:
                                    output[b_i, s_i, m_i, n_i, m_i0, n_i0] = inputs[b_i, s_i, m_i * 16 + m_i0, n_i * 16 + n_i0]
        return output

    cast_data = data
    if data.dtype == 'float32':
        cast_data = akg.lang.cce.cast_to(data, 'float16')
    zero = akg.tvm.const(0.0, cast_data.dtype)
    pad_data = cast_data
    # n padding is not support now because of alignment issue
    if n % 16 != 0:
        paddings = [[0, 0] for _ in range(len(shape))]
        paddings[-1] = [0, pad_n - n]
        pad_data = pad.pad(cast_data, paddings, 'constant')
    if format_ == 'zN':
        if len(shape) == 2:
            output = reshape_zn_2d(pad_data, zero)
        if len(shape) == 3:
            output = reshape_zn_3d(pad_data, zero)
        if len(shape) == 4:
            output = reshape_zn_4d(pad_data, zero)
    elif format_ == 'zZ':
        if len(shape) == 2:
            output = reshape_zz_2d(pad_data, zero)
        if len(shape) == 3:
            output = reshape_zz_3d(pad_data, zero)
        if len(shape) == 4:
            output = reshape_zz_4d(pad_data, zero)
    elif format_ == 'nZ':
        if len(shape) == 2:
            output = reshape_nz_2d(pad_data, zero)
        if len(shape) == 3:
            output = reshape_nz_3d(pad_data, zero)
        if len(shape) == 4:
            output = reshape_nz_4d(pad_data, zero)

    return output
