import numpy as np
import scipy.sparse

import akg
from akg import tvm
from tests.common.base import get_rtol_atol
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array

def csr_reduce_sum(data, col_idx, row_idx, axis, shape):
    assert len(shape) == 2, "only supports 2-dim sparse tensor"
    if axis < 0:
        axis += len(shape)
    reduce_first_axis = tvm.const(axis == 0)

    num_rows = row_idx.shape[0] - 1
    def gen_ir(data, col_idx, row_idx, output):
        ib = tvm.ir_builder.create()
        with ib.for_range(0, num_rows, name='i') as i:
            ib.store(output, i, tvm.const(0, data.dtype))
            start = ib.load(row_idx, i)
            end = ib.load(row_idx, i + 1)
            with ib.for_range(0, end - start, name='j') as j:
                pos = start + j
                val = tvm.expr.Select(pos < end, ib.load(data, pos), tvm.const(0, data.dtype))
                with ib.if_scope(reduce_first_axis):
                    col = ib.load(col_idx, pos)
                    ib.store(output, col, val + ib.load(output, col))
                with ib.else_scope():
                    ib.scope_attr([tvm.api._IterVar((0, shape[1]), "j", 2)], "reduce_update", "")
                    ib.store(output, i, val + ib.load(output, i))
        return ib.get()

    output_shape = [shape[1 - axis]]
    output_name = "T_csr_reduce_sum_" + data.op.name + "_" + str(axis)
    out_buf = tvm.decl_buffer(output_shape, data.dtype, output_name)
    return tvm.extern([output_shape],
                      [data, col_idx, row_idx],
                      lambda ins, outs: gen_ir(ins[0], ins[1], ins[2], outs[0]),
                      dtype=data.dtype, out_buffers=[out_buf], name=output_name)
    
def gen_data(shape, dtype1, dtype2, axis):
    data = scipy.sparse.rand(shape[0], shape[1], density=0.2, format='csr', dtype=dtype1)
    expect = data.sum(axis)
    return data.data, data.indices.astype(dtype2), data.indptr.astype(dtype2), expect

def test_csr_reduce_sum(shape, dtype1, dtype2, axis, poly_sch=False, attrs=None):
    if not attrs:
        attrs = {"target": "cuda"}
    if attrs["target"] == "cuda":
        attrs["enable_akg_reduce_lib"] = True
        attrs["enable_atomic_add"] = True
    op_attrs = [axis, shape]

    # gen data
    data, col_idx, row_idx, expect = gen_data(shape, dtype1, dtype2, axis)
    output_shape = expect.shape
    attrs["csr_avg_row"] = data.shape[0] // shape[0]

    mod = utils.op_build_test(csr_reduce_sum, [data.shape, col_idx.shape, row_idx.shape], 
                              [dtype1, dtype2, dtype2], op_attrs=op_attrs, polyhedral=poly_sch,
                              attrs=attrs, kernel_name="csr_reduce_sum")

    if len(expect.shape) == 0:
        output_shape = (1, )
    output = np.zeros(output_shape, expect.dtype)
    output = utils.mod_launch(mod, (data, col_idx, row_idx, output), expect=expect)
    atol, rtol = get_rtol_atol("csr_reduce_sum", dtype1)
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
            [data, col_idx, row_idx, output, expect], akg.tvm.context(target_name, 0))
        target_profiling(mod, *args_list, target=target_name,  repeat_time=attrs["repeat_time"])
