# Copyright 2021 Huawei Technologies Co., Ltd
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
from tests.common.gen_random import random_gaussian
from akg.utils import kernel_exec as utils
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array
from tests.common.tensorio import compare_tensor

def csrmv(data, indices, indptr, weight):
    assert len(data.shape) == 1 and len(weight.shape) == 2, "only support 2-dim csrmv"
    assert data.dtype == weight.dtype, ("data and weight must have same dtype.")
    
    num_rows = indptr.shape[0] - 1

    def csrmv_ir(data, indices, indptr, weight, out):
        ib = tvm.ir_builder.create()
        with ib.for_range(0, num_rows, name="row") as row:
            ib.store(out, [row, 0], tvm.const(0, data.dtype))
            row_start = ib.load(indptr, row)
            row_end = ib.load(indptr, row + 1)
            row_elems = row_end - row_start
            with ib.for_range(0, row_elems, name="idx") as idx:
                elem = row_start + idx
                val = tvm.expr.Select(elem < row_end,
                                      ib.load(data, elem) * ib.load(weight, [ib.load(indices, elem), 0]),
                                      tvm.const(0, data.dtype))
                ib.scope_attr([tvm.api._IterVar((0, weight.shape[0]), "idx", 2)], "reduce_update", "")
                temp = val + ib.load(out, [row, 0])
                ib.store(out, [row, 0], temp)
        return ib.get()
    output_shape = [num_rows, 1]
    output_name = "T_csrmv_"
    out_buf = tvm.decl_buffer(output_shape, data.dtype, output_name)
    return tvm.extern([output_shape], [data, indices, indptr, weight],
                        lambda ins, outs: csrmv_ir(ins[0], ins[1], ins[2], ins[3], outs[0]),
                        dtype=data.dtype, out_buffers=[out_buf], name=output_name)

def gen_data(shape1, dtype1, shape2, dtype2):
    csr_matrix = sp.sparse.rand(shape1[0], shape1[1], density=0.2, format='csr', dtype=dtype1)
    weight = np.random.random(shape2).astype(dtype2)
    expect = csr_matrix * weight
    return csr_matrix.data, csr_matrix.indices, csr_matrix.indptr, weight, expect

def test_ms_csrmv(shape1, dtype1, shape2, dtype2, poly_sch=False, attrs=None):
    if not attrs:
        attrs = {"target": "cuda"}
    if attrs["target"] == "cuda":
        attrs["enable_akg_reduce_lib"] = True
        attrs["enable_atomic_add"] = True
    data, indices, indptr, weight, expect = gen_data(shape1, dtype1, shape2, dtype2)

    mod = utils.op_build_test(csrmv, [data.shape, indices.shape, indptr.shape, weight.shape],
                              ["float32", "int32", "int32", "float32"], polyhedral=poly_sch,
                              attrs=attrs, kernel_name='csrmv')
    
    output_shape = expect.shape
    output = np.zeros(output_shape, dtype="float32")
    output = utils.mod_launch(mod, (data, indices, indptr, weight, output), expect=expect)
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
        args_list = to_tvm_nd_array([data, indices, indptr, weight, output, expect], akg.tvm.context(target_name, 0))
        target_profiling(mod, *args_list, target=target_name,  repeat_time=attrs["repeat_time"])
