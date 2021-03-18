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

"""operator dsl function: div_mod_issue"""
import akg.tvm
import akg
import akg.lang.cce
from akg.utils import kernel_exec as utils



def div_mod_issue(data_shape, weight_shape, case_number):

    if (case_number == 0):
        A = akg.tvm.placeholder(data_shape, dtype='float16', name='input0')
        divisor = 2
        stage1 = akg.tvm.compute(data_shape, lambda n, c, h, w: A[n, c / divisor, h, w] + 1, name="stage1")
        op_vars = [A, stage1]
        s = akg.tvm.create_schedule([stage1.op])
        akg.lower(s, op_vars, simple_mode=True, polyhedral=True)
        with akg.build_config(add_lower_pass=utils.debug_mode(0), dump_pass_ir=True):
            mod = akg.build(s, op_vars, "cce", name="test1", polyhedral=True)
        return mod
    else:
        A = akg.tvm.placeholder(data_shape, dtype='float16', name='input0')
        B = akg.tvm.placeholder(weight_shape, dtype='float16', name='input1')

        divisor = 3
        stage1 = akg.tvm.compute(data_shape, lambda n, c, h, w: A[n, c / divisor, h, w] + 1, name="stage1")
        stage2 = akg.tvm.compute(weight_shape, lambda n, c, h, w: stage1[0, c, 0, 0] + B[n, c, h, w], name="stage2")
        op_vars = [A, B, stage2]

        s = akg.tvm.create_schedule([stage2.op])
        akg.lower(s, op_vars, simple_mode=True, polyhedral=True)

        with akg.build_config(add_lower_pass=utils.debug_mode(0), dump_pass_ir=True):
            mod_stage2 = akg.build(s, op_vars, "cce", name="test2", polyhedral=True)
        return mod_stage2
