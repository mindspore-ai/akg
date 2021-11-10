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

"""operator dsl function: elemwise_chain"""
import akg.tvm
import akg
import akg.utils as utils
from akg.utils.kernel_exec import debug_mode, gen_name_kernel

def case_1(data_shape, dtype, kernel_name, attrs):
    """elemwise chain case 1"""
    utils.ops_dtype_check(dtype, utils.DtypeForDavinci.FLOAT16)
    utils.check_shape_length_equal("data", data_shape, 2)

    m, k = data_shape

    A = akg.tvm.placeholder((m, k), name='A', dtype=dtype)
    B = akg.tvm.placeholder((k,), name='B', dtype=dtype)
    C = akg.tvm.placeholder((m, k), name='C', dtype=dtype)

    E = akg.tvm.compute((m, k), lambda i, j: A[i, j] * (B[j] + C[i, j]), name="E")

    forward_s = akg.tvm.create_schedule(E.op)
    op_vars = [A, B, C, E]
    akg.lower(forward_s, op_vars, simple_mode=True, polyhedral=True)

    kernel_name = gen_name_kernel(kernel_name, dtype, data_shape)

    with akg.build_config(add_lower_pass=debug_mode(0), dump_pass_ir=True):
        mod = akg.build(forward_s, op_vars, "cce", name="test", attrs=attrs, polyhedral=True)
        return mod


def elemwise_chain(case_no, data_shape, dtype, kernel_name, attrs=None, target=utils.CCE):
    """elemwise chain"""
    if case_no == 1:
        return case_1(data_shape, dtype, kernel_name, attrs)
    raise RuntimeError("No support case %d" % case_no)