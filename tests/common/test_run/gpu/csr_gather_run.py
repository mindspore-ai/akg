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
import scipy.sparse

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

def csr_gather(dense, col_idx, row_idx, shape, target=CUDA):
    assert target == CUDA, "only supports GPU"
    return composite.csr_gather((row_idx, col_idx, dense), {"dense_shape": shape})

def gen_data(shape, dtype1, dtype2):
    dense = random_gaussian(shape).astype(dtype1)
    sparse_data = scipy.sparse.rand(shape[0], shape[1], density=0.2, format='csr', dtype=dtype1)
    coo = sparse_data.tocoo()
    coo_idx = np.stack((coo.row, coo.col))
    expect = dense[coo_idx.tolist()]
    return dense, sparse_data.indices.astype(dtype2), sparse_data.indptr.astype(dtype2), np.asarray(expect)

def csr_gather_run(shape, dtype1, dtype2, poly_sch=True, attrs=None):
    if not attrs:
        attrs = {"target": "cuda"}
    # gen data
    op_attrs = [shape]
    dense, col_idx, row_idx, expect = gen_data(shape, dtype1, dtype2)
    output_shape = expect.shape
    attrs["csr_avg_row"] = col_idx.shape[0] // shape[0]
    attrs["is_csr"] = True

    mod = utils.op_build_test(csr_gather, [shape, col_idx.shape, row_idx.shape], 
                              [dtype1, dtype2, dtype2], op_attrs=op_attrs, polyhedral=poly_sch,
                              attrs=attrs, kernel_name="csr_gather")

    if len(expect.shape) == 0:
        output_shape = (1, )
    output = np.zeros(output_shape, expect.dtype)
    output = utils.mod_launch(mod, (dense, col_idx, row_idx, output), expect=expect)
    atol, rtol = get_rtol_atol("csr_gather", dtype1)
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
            [dense, col_idx, row_idx, output, expect], akg.tvm.context(target_name, 0))
        target_profiling(mod, *args_list, target=target_name,  repeat_time=attrs["repeat_time"])
    return (dense, col_idx, row_idx), output, expect, res
