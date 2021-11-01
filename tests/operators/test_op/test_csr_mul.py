import numpy as np
import scipy.sparse

import akg
from akg import tvm
from akg import topi
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array, get_shape
from akg.utils.dsl_create import get_broadcast_shape

def csr_mul(dense, sparse_data, col_idx, row_idx, shape):
    assert len(shape) == 2, "only supports 2-dim sparse tensor"
    assert len(dense.shape) <= 2
    assert dense.dtype == sparse_data.dtype, "data and weight must have the same dtype"

    num_rows = row_idx.shape[0] - 1
    dense_shape = get_shape(dense.shape)
    sparse_shape = get_shape(shape)
    broadcast_shape = get_broadcast_shape(dense_shape, sparse_shape)
    need_expand = tvm.const(len(dense_shape) < len(broadcast_shape))
    need_broadcast_first_dim = tvm.const(
        len(dense_shape) == len(broadcast_shape) and dense_shape[0] < broadcast_shape[0])
    need_broadcast_last_dim = tvm.const(
        len(dense_shape) == len(broadcast_shape) and dense_shape[1] < broadcast_shape[1])

    def gen_ir(dense, sparse_data, col_idx, row_idx, output):
        ib = tvm.ir_builder.create()
        with ib.for_range(0, num_rows, name='i') as i:
            start = ib.load(row_idx, i)
            end = ib.load(row_idx, i + 1)
            with ib.for_range(0, end - start, name='j') as j:
                pos = start + j
                with ib.if_scope(pos < end):
                    val = ib.load(sparse_data, pos)
                    col = ib.load(col_idx, pos)
                    with ib.if_scope(need_expand):
                        ib.store(output, pos, val * ib.load(dense, [col]))
                    with ib.else_scope():
                        with ib.if_scope(need_broadcast_first_dim):
                            ib.store(output, pos, val * ib.load(dense, [0, col]))
                        with ib.else_scope():
                            with ib.if_scope(need_broadcast_last_dim):
                                ib.store(output, pos, val * ib.load(dense, [i, 0]))
                            with ib.else_scope():
                                ib.store(output, pos, val * ib.load(dense, [i, col]))
        return ib.get()

    output_name = "T_csr_mul_" + dense.op.name + "_" + sparse_data.op.name
    out_buf = tvm.decl_buffer(sparse_data.shape, sparse_data.dtype, output_name)
    return tvm.extern([shape],
                      [dense, sparse_data, col_idx, row_idx],
                      lambda ins, outs: gen_ir(ins[0], ins[1], ins[2], ins[3], outs[0]),
                      dtype=sparse_data.dtype, out_buffers=[out_buf], name=output_name)
    
def gen_data(shape1, shape2, dtype1, dtype2):
    dense = random_gaussian(shape1).astype(dtype1)
    sparse_data = scipy.sparse.rand(shape2[0], shape2[1], density=0.2, format='csr', dtype=dtype1)
    expect = sparse_data.multiply(np.broadcast_to(dense, shape2))
    return dense, sparse_data.data, sparse_data.indices.astype(dtype2), sparse_data.indptr.astype(dtype2), expect.data

def test_csr_mul(shape1, shape2, dtype1, dtype2, poly_sch=False, attrs=None):
    if not attrs:
        attrs = {"target": "cuda"}
    # gen data
    op_attrs = [shape2]
    dense, sparse_data, col_idx, row_idx, expect = gen_data(shape1, shape2, dtype1, dtype2)
    output_shape = expect.shape

    mod = utils.op_build_test(csr_mul, [shape1, sparse_data.shape, col_idx.shape, row_idx.shape], 
                              [dtype1, dtype1, dtype2, dtype2], op_attrs=op_attrs, polyhedral=poly_sch,
                              attrs=attrs, kernel_name="csr_mul")

    if len(expect.shape) == 0:
        output_shape = (1, )
    output = np.zeros(output_shape, expect.dtype)
    output = utils.mod_launch(mod, (dense, sparse_data, col_idx, row_idx, output), expect=expect)
    atol, rtol = get_rtol_atol("csr_mul", dtype1)
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
