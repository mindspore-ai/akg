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

"""fractal2two"""
from akg.utils import kernel_exec as utils
from akg.tvm.hybrid import script
from akg.utils import custom_tiling as ct_util
from akg.utils.format_transform import get_const

fractal2two_set_dim_map = {
}


def fractal2two_set_dim_func(data, out_dtype, shape_original, format_):
    shape = [x for x in data.shape]

    hash_key = str((tuple(shape), format_, data.dtype))
    return ct_util.set_dims_by_key(hash_key, fractal2two_set_dim_map), hash_key


def fractalzn2two(data, out_dtype, shape_original):
    """zN change"""
    shape = [get_const(x) for x in data.shape]
    assert len(shape) >= 4

    n1, m1, m0, n0 = shape[-4:]
    if len(shape) == 5:
        b = shape[0]
    elif len(shape) == 6:
        b, s = shape[:2]
    m, n = m1 * m0, n1 * n0
    @script(capture=locals())
    def reshape_zn_2d(inputs):
        output = allocate((m, n), inputs.dtype, 'local')
        for n_i1 in range(n1):
            for m_i1 in range(m1):
                for m_i0 in range(m0):
                    for n_i0 in range(n0):
                        output[m_i1 * 16 + m_i0, n_i1 * 16 + n_i0] = inputs[n_i1, m_i1, m_i0, n_i0]
        return output

    @script(capture=locals())
    def reshape_zn_3d(inputs):
        output = allocate((b, m, n), inputs.dtype, 'local')
        for b_i in range(b):
            for n_i1 in range(n1):
                for m_i1 in range(m1):
                    for m_i0 in range(m0):
                        for n_i0 in range(n0):
                            output[b_i, m_i1 * 16 + m_i0, n_i1 * 16 + n_i0] = inputs[b_i, n_i1, m_i1, m_i0, n_i0]
        return output

    @script(capture=locals())
    def reshape_zn_4d(inputs):
        output = allocate((b, s, m, n), inputs.dtype, 'local')
        for b_i in range(b):
            for s_i in range(s):
                for n_i1 in range(n1):
                    for m_i1 in range(m1):
                        for m_i0 in range(m0):
                            for n_i0 in range(n0):
                                output[b_i, s_i, m_i1 * 16 + m_i0, n_i1 * 16 + n_i0] = \
                                inputs[b_i, s_i, n_i1, m_i1, m_i0, n_i0]
        return output

    if len(shape_original) == 2:
        output = reshape_zn_2d(data)
    elif len(shape_original) == 3:
        output = reshape_zn_3d(data)
    elif len(shape_original) == 4:
        output = reshape_zn_4d(data)
    finalShape = shape[:-4] + [m, n]
    assert finalShape == shape_original
    assert out_dtype == data.dtype
    # if finalShape != shape_original:
    # output = akg.tvm.compute(shape_original, lambda *indice: output(*indice), name="slice_output")
    # if out_dtype != data.dtype:
    # output = akg.lang.cce.cast_to(output, out_dtype)

    return output


def fractalzz2two(data, out_dtype, shape_original):
    """zZ change"""
    shape = [get_const(x) for x in data.shape]
    assert len(shape) >= 4

    m1, n1, m0, n0 = shape[-4:]
    if len(shape) == 5:
        b = shape[0]
    elif len(shape) == 6:
        b, s = shape[:2]
    m, n = m1 * m0, n1 * n0
    @script(capture=locals())
    def reshape_zz_2d(inputs):
        output = allocate((m, n), inputs.dtype, 'local')
        for m_i1 in range(m1):
            for n_i1 in range(n1):
                for m_i0 in range(m0):
                    for n_i0 in range(n0):
                        output[m_i1 * 16 + m_i0, n_i1 * 16 + n_i0] = inputs[m_i1, n_i1, m_i0, n_i0]
        return output

    @script(capture=locals())
    def reshape_zz_3d(inputs):
        output = allocate((b, m, n), inputs.dtype, 'local')
        for b_i in range(b):
            for m_i1 in range(m1):
                for n_i1 in range(n1):
                    for m_i0 in range(m0):
                        for n_i0 in range(n0):
                            output[b_i, m_i1 * 16 + m_i0, n_i1 * 16 + n_i0] = inputs[b_i, m_i1, n_i1, m_i0, n_i0]
        return output

    @script(capture=locals())
    def reshape_zz_4d(inputs):
        output = allocate((b, s, m, n), inputs.dtype, 'local')
        for b_i in range(b):
            for s_i in range(s):
                for m_i1 in range(m1):
                    for n_i1 in range(n1):
                        for m_i0 in range(m0):
                            for n_i0 in range(n0):
                                output[b_i, s_i, m_i1 * 16 + m_i0, n_i1 * 16 + n_i0] = \
                                inputs[b_i, s_i, m_i1, n_i1, m_i0, n_i0]
        return output

    if len(shape_original) == 2:
        output = reshape_zz_2d(data)
    elif len(shape_original) == 3:
        output = reshape_zz_3d(data)
    elif len(shape_original) == 4:
        output = reshape_zz_4d(data)
    final_shape = shape[:-4] + [m, n]
    assert final_shape == shape_original
    assert out_dtype == data.dtype
    # if finalShape != shape_original:
    # output = akg.tvm.compute(shape_original, lambda *indice: output(*indice), name="slice_output")
    # if out_dtype != data.dtype:
    # output = akg.lang.cce.cast_to(output, out_dtype)

    return output


@ct_util.reg_set_dim_func(fractal2two_set_dim_func)
def fractal2two(data, out_dtype, shape_original, format_):
    """
    Change fractal to 2-D.

    Args:
        data: Tensor.
        out_dtype: String. Dtype of output.
        shape_original: Tuple or list. Shape of original tensor.
        format_: String. Can be "zN" or "zZ".

    Returns:
        Tensor, has the same type and shape as data.
    """
    if format_ == 'zN':
        return fractalzn2two(data, out_dtype, shape_original)
    if format_ == 'zZ':
        return fractalzz2two(data, out_dtype, shape_original)
    return None
