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

"""dsl: distinguish_between_pn_samples"""
import akg.tvm
import akg.topi
from akg.lang import cce as dav
from akg.utils import validation_check as vc_util


def distinguish_between_pn_samples(data, threshold):

    shape = [x.value for x in data.shape]
    vc_util.check_shape(shape, length=3)
    dtype = data.dtype
    vc_util.ops_dtype_check(dtype, vc_util.DtypeForDavinci.ALL_FLOAT)
    if dtype == 'float32':
        dtype = 'float16'
        data_ = akg.tvm.compute(data.shape, lambda*i: data(*i).astype('float16'), name='data_f16')
    else:
        data_ = data

    thd = akg.tvm.const(threshold, 'float32')

    k = akg.tvm.reduce_axis((0, data.shape[2]), "k")
    reducer = akg.tvm.comm_reducer(lambda x, y: dav.fargmax(x, y), lambda t: akg.tvm.min_value(t))
    res_f16 = akg.tvm.compute((data.shape[0], data.shape[1]), lambda i, j: reducer(data_[i, j, k], axis=k), name='res_f16')

    res_int32 = akg.tvm.compute(res_f16.shape, lambda *indice: res_f16(*indice).astype("int32"), "cast")

    ini_value = akg.tvm.const(shape[-1], dtype='int32')
    res = akg.tvm.compute((data.shape[0], data.shape[1]), lambda x, y: ini_value, name='res')

    constOne_ = akg.tvm.compute((data.shape[0], data.shape[1]), lambda *indice: akg.tvm.const(1.0, dtype), name='one')
    constZero_ = akg.tvm.compute((data.shape[0], data.shape[1]), lambda *indice: akg.tvm.const(0.0, dtype='float16'), name='zero')

    idx = akg.tvm.reduce_axis((0, data.shape[2]), "idx")
    data_max = akg.tvm.compute((data.shape[0], data.shape[1]), lambda i, j: akg.tvm.max(data_[i, j, idx], axis=idx), name='data_max')
    data_max_f32 = akg.topi.cast(data_max, 'float32')

    tmp = akg.tvm.compute((data.shape[0], data.shape[1]), lambda *indice: data_max_f32(*indice) - thd, name='tmp')
    tmp_f16 = akg.tvm.compute((data.shape[0], data.shape[1]), lambda *indice: tmp(*indice).astype('float16'), name='tmp')

    data_sign = akg.tvm.compute((data.shape[0], data.shape[1]), lambda *i: akg.tvm.expr.Select(tmp_f16(*i) < constZero_(*i), constZero_(*i), constOne_(*i)), name="sign_data")

    data_sign_inv = akg.tvm.compute((data.shape[0], data.shape[1]), lambda *i: constOne_(*i) - data_sign(*i), name="sign_data_inv")

    res_value = akg.tvm.compute((data.shape[0], data.shape[1]), lambda *i: res_int32(*i) * data_sign(*i) + res(*i) * data_sign_inv(*i), name='res_value')
    Res_ = akg.tvm.compute((data.shape[0], data.shape[1]), lambda *i: res_value(*i).astype('int32'), name='Res_')

    return Res_
