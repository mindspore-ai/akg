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
import scipy as sp

import akg
from akg import tvm
from akg import composite
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array

def csr_mm(indptr, indices, data, dense):
    return composite.csr_mm((indptr, indices, data, dense), {})

def gen_data(shape1, dtype1, dtype2, dtype3, shape2, dtype4):
    csr_matrix = sp.sparse.rand(shape1[0], shape1[1], density=0.2, format='csr', dtype=dtype3)
    dense = np.random.random(shape2).astype(dtype4)
    expect = np.array(csr_matrix * dense)
    return csr_matrix.indptr.astype(dtype1), csr_matrix.indices.astype(dtype2), csr_matrix.data, dense, expect

def csr_mm_run(shape1, dtype1, dtype2, dtype3, shape2, dtype4, poly_sch=True, attrs=None):
    if not attrs:
        attrs = {"target": "cuda"}
    if attrs["target"] == "cuda":
        attrs["enable_akg_reduce_lib"] = True
        attrs["enable_atomic_add"] = True
    indptr, indices, data, dense, expect = gen_data(shape1, dtype1, dtype2, dtype3, shape2, dtype4)

    attrs["csr_avg_row"] = data.shape[0] // shape1[0]
    attrs["is_csr"] = True
    
    mod = utils.op_build_test(csr_mm, [indptr.shape, indices.shape, data.shape, dense.shape],
                              [indptr.dtype.name, indices.dtype.name, data.dtype.name, dense.dtype.name],
                              polyhedral=poly_sch, attrs=attrs, kernel_name='csr_mm')
    
    output_shape = expect.shape
    output = np.zeros(output_shape, dtype="float32")
    output = utils.mod_launch(mod, (indptr, indices, data, dense, output), expect=expect)
    res = compare_tensor(output, expect, rtol=5e-3, atol=1e-8)
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
        args_list = to_tvm_nd_array([indptr, indices, data, dense, output], akg.tvm.context(target_name, 0))
        target_profiling(mod, *args_list, target=target_name,  repeat_time=attrs["repeat_times"])
    return (indptr, indices, data, dense), output, expect, res
