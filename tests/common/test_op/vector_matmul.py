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

"""operator dsl function: vector_matmul"""
import akg.tvm
import akg
from akg.utils import kernel_exec as utils
from akg.tvm.hybrid import script



def get_shape(m, n, k, trans_a, trans_b):
    shape_x = (m, k)
    shape_y = (k, n)
    if trans_a:
        shape_x = (k, m)
    if trans_b:
        shape_y = (n, k)
    return shape_x, shape_y


def vector_matmul(data_m, data_n, data_k, trans_a, trans_b, dtype, kernel_name, attrs):
    check_list = ["float16", "float32"]
    if not dtype in check_list:
        raise TypeError("softmax test only support %s while dtype is %s" % (",".join(check_list), dtype))

    m = data_m
    n = data_n
    k = data_k
    data_shape, weight_shape = get_shape(m, n, k, trans_a, trans_b)
    output_shape = (m, n)

    A = akg.tvm.placeholder(data_shape, name='A', dtype=dtype)
    B = akg.tvm.placeholder(weight_shape, name='B', dtype=dtype)

    ZERO = akg.tvm.const(0.0, dtype=dtype)

    @script
    def matmul_hybrid_f_f(a, b, zero):
        t_1 = allocate((m, k, n), a.dtype, 'local')
        t_2 = output_tensor((m, n), a.dtype)
        for i_m in range(0, m):
            for i_k in range(0, k):
                for i_n in range(0, n):
                    t_1[i_m, i_k, i_n] = a[i_m, i_k] * b[i_k, i_n]
            for i1_n in range(0, n):
                t_2[i_m, i1_n] = zero
            for i1_k in range(0, k):
                for i1_n in range(0, n):
                    t_2[i_m, i1_n] = t_2[i_m, i1_n] + t_1[i_m, i1_k, i1_n]
        return t_2

    @script
    def matmul_hybrid_f_t(a, b, zero):
        t_1 = allocate((m, n, k), a.dtype, 'local')
        t_2 = output_tensor((m, n), a.dtype)
        for i_m in range(0, m):
            for i_n in range(0, n):
                t_2[i_m, i_n] = zero
                for i_k in range(0, k):
                    t_1[i_m, i_n, i_k] = a[i_m, i_k] * b[i_n, i_k]
                    t_2[i_m, i_n] = t_1[i_m, i_n, i_k] + t_2[i_m, i_n]
        return t_2

    @script
    def matmul_hybrid_t_f(a, b, zero):
        t_1 = allocate((m, k, n), a.dtype, 'local')
        t_2 = output_tensor((m, n), a.dtype)
        for i_m in range(0, m):
            for i_k in range(0, k):
                for i_n in range(0, n):
                    t_1[i_m, i_k, i_n] = a[i_k, i_m] * b[i_k, i_n]
            for i1_n in range(0, n):
                t_2[i_m, i1_n] = zero
            for i1_k in range(0, k):
                for i1_n in range(0, n):
                    t_2[i_m, i1_n] = t_2[i_m, i1_n] + t_1[i_m, i1_k, i1_n]
        return t_2

    C = ()

    if trans_a == False and trans_b == False:
        C = matmul_hybrid_f_f(A, B, ZERO)
    elif trans_a == False and trans_b == True:
        C = matmul_hybrid_f_t(A, B, ZERO)
    elif trans_a == True and trans_b == False:
        C = matmul_hybrid_t_f(A, B, ZERO)
    else:
        raise ValueError('Not support both transpose yet')

    forward_s = akg.tvm.create_schedule(C.op)
    op_vars = [A, B, C]

    with akg.build_config(add_lower_pass=utils.debug_mode(0), dump_pass_ir=True):
        mod = akg.build(forward_s, op_vars, "cce", name=kernel_name, attrs=attrs, polyhedral=True)
        source_code = mod.imported_modules[0].get_source()
        utils.create_code(kernel_name, "./", source_code)
        return mod, output_shape
