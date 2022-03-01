# Copyright 2022 Huawei Technologies Co., Ltd
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
# limitations under the License
import numpy as np

import akg
from akg import tvm
from akg import composite
from akg.utils import CUDA
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array

def csr2coo(indptr, nnz, target=CUDA):
    return composite.csr2coo((indptr,), {"nnz": nnz})

def gen_data(shape, nnz, dtype):
    indptr_choice = np.arange(1, nnz, dtype=dtype)
    indptr = np.sort(np.random.choice(indptr_choice, shape[0] - 2, replace=True))
    indptr = np.concatenate((np.array([0], dtype=dtype), indptr, np.array([nnz], dtype=dtype)))
    expect = np.zeros(nnz, dtype=dtype)
    for i in range(shape[0] - 1):
        row_start = indptr[i]
        row_end = indptr[i + 1]
        expect[row_start : row_end] = i
    return indptr, expect

def csr2coo_run(shape, nnz, dtype, poly_sch=True, attrs=None):
    if not attrs:
        attrs = {"target": "cuda"}
    # gen data
    op_attrs = [nnz]
    indptr, expect = gen_data(shape, nnz, dtype)
    output_shape = expect.shape
    attrs["csr_avg_row"] = nnz // (shape[0] - 1)
    attrs["is_csr"] = True

    mod = utils.op_build_test(csr2coo, [shape], [dtype], op_attrs=op_attrs, polyhedral=poly_sch,
                              attrs=attrs, kernel_name="csr2coo")

    if len(expect.shape) == 0:
        output_shape = (1, )
    output = np.zeros(output_shape, expect.dtype)
    output = utils.mod_launch(mod, (indptr, output), expect=expect)
    atol, rtol = get_rtol_atol("csr2coo", dtype)
    res = compare_tensor(output, expect, rtol=rtol, atol=atol)
    print("Test {}".format("Pass" if res else "Failed"))
    target_name = attrs["target"].split()[0]
    if not res:
        mod_source = mod
        if target_name != "llvm":
            mod_source = mod.imported_modules[0]
        print("Error {}:========================".format(target_name))
        print(mod_source.get_source())
        raise AssertionError("Test fail")
    if attrs["profiling"]:
        args_list = to_tvm_nd_array(
            [indptr, output, expect], akg.tvm.context(target_name, 0))
        target_profiling(mod, *args_list, target=target_name,  repeat_time=attrs["repeat_time"])
    return (indptr,), output, expect, res
