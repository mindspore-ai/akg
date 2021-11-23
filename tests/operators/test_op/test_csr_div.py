import numpy as np
import scipy.sparse

import akg
from akg import tvm
from akg import composite
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array

def csr_div(dense, sparse_data, col_idx, row_idx, shape):
    return composite.csr_div((row_idx, col_idx, sparse_data, dense), {"dense_shape": shape})

def gen_data(shape1, shape2, dtype1, dtype2):
    dense = random_gaussian(shape1).astype(dtype1)
    sparse_data = scipy.sparse.rand(shape2[0], shape2[1], density=0.2, format='csr', dtype=dtype1)
    expect = sparse_data.multiply(np.divide(1, np.broadcast_to(dense, shape2)))
    return dense, sparse_data.data, sparse_data.indices.astype(dtype2), sparse_data.indptr.astype(dtype2), expect.data
    
def test_csr_div(shape1, shape2, dtype1, dtype2, poly_sch=False, attrs=None):
    if not attrs:
        attrs = {"target": "cuda"}
    # gen data
    op_attrs = [shape2]
    dense, sparse_data, col_idx, row_idx, expect = gen_data(shape1, shape2, dtype1, dtype2)
    output_shape = expect.shape
    attrs["csr_avg_row"] = sparse_data.shape[0] // shape1[0]

    mod = utils.op_build_test(csr_div, [shape1, sparse_data.shape, col_idx.shape, row_idx.shape], 
                              [dtype1, dtype1, dtype2, dtype2], op_attrs=op_attrs, polyhedral=poly_sch,
                              attrs=attrs, kernel_name="csr_div")

    if len(expect.shape) == 0:
        output_shape = (1, )
    output = np.zeros(output_shape, expect.dtype)
    output = utils.mod_launch(mod, (dense, sparse_data, col_idx, row_idx, output), expect=expect)
    atol, rtol = get_rtol_atol("csr_div", dtype1)
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
            [dense, sparse_data, col_idx, row_idx, output, expect], akg.tvm.context(target_name, 0))
        target_profiling(mod, *args_list, target=target_name,  repeat_time=attrs["repeat_time"])
