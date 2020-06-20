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

import numpy as np
from tensorio import compare_tensor
import akg
import akg.tvm
from akg import backend as cce
from akg.utils import kernel_exec as utils
from test_op import invert_permutation
from akg.utils import validation_check as vc_util
from akg.utils import validation_check as vc_util


def invert_permutation_run(shape, dtype, attrs):
    # check shapes
    vc_util.check_shape(shape)

    if not (dtype.lower() in "int32"):
        raise RuntimeError("indices_dtype only support int32 while dtype is %s" % dtype)

    A = akg.tvm.placeholder(shape, dtype, name="A")
    op = invert_permutation.invert_permutation(A)
    s = akg.tvm.create_schedule(op.op)

    kernel_name = utils.gen_name_kernel("invert_permutation", dtype, shape)
    with akg.build_config(add_lower_pass=cce.debug_mode(0), dump_pass_ir=True):
        mod = akg.build(s, [A, op], "cce", name=kernel_name, attrs=attrs, polyhedral=True)

    input_data = np.random.permutation(np.arange(shape[0])).astype(np.int32)
    expect = np.full([shape[0]], 0, np.int32)
    for i, e in enumerate(input_data):
        expect[e] = i

    output = np.full([shape[0]], 0, np.int32)
    output = utils.mod_launch(mod, (input_data, output), expect=expect)

    return (input_data, ), output, expect, compare_tensor(output, expect, rtol=5e-03, equal_nan=True)
