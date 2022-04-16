import functools
import operator
import scipy.sparse
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

def csr_reduce_sum(data, col_idx, row_idx, axis, shape):
    attrs = {"axis": [axis], "dense_shape": shape}
    return composite.csr_reduce_sum((row_idx, col_idx, data), attrs)

def gen_data(shape, dtype1, dtype2, axis, nnz=-1):
    axis = axis % len(shape)
    if nnz > 0:
        indptr_choice = np.arange(0, nnz, dtype=dtype2)
        indptr = np.sort(np.random.choice(indptr_choice, shape[0] - 1, replace=True))
        indptr = np.concatenate(
            (np.array([0], dtype=dtype2), indptr, np.array([nnz], dtype=dtype2)))
        indices_choice = np.arange(shape[1], dtype=dtype2)
        indices = np.zeros(nnz, dtype=dtype2)
        for i in range(0, shape[0]):
            row_start = indptr[i]
            row_end = indptr[i + 1]
            indices[row_start : row_end] = np.sort(np.random.choice(indices_choice, row_end - row_start, replace=False))
        sparse_data = random_gaussian((nnz,) + shape[2:]).astype(dtype1)
        x = sparse_data.reshape(nnz, -1)
        for i in range(functools.reduce(operator.mul, shape[2:], 1)):
            sparse = scipy.sparse.csr_matrix((x[..., i], indices, indptr), shape=shape[:2])
            out = np.array(sparse.sum(axis))
            if i == 0:
                expect = [out.data]
            else:
                expect.append(out.data)
        expect = np.moveaxis(np.stack(expect, 0).reshape(shape[2:] + (shape[1 - axis],)), -1, 0)
        return sparse_data, indices.astype(dtype2), indptr.astype(dtype2), expect
    data = scipy.sparse.rand(shape[0], shape[1], density=0.2, format='csr', dtype=dtype1)
    expect = np.array(data.sum(axis))
    return data.data, data.indices.astype(dtype2), data.indptr.astype(dtype2), expect

def csr_reduce_sum_run(shape, dtype1, dtype2, axis, nnz=-1, poly_sch=True, attrs=None):
    if not attrs:
        attrs = {"target": "cuda"}
    if attrs["target"] == "cuda":
        attrs["enable_akg_reduce_lib"] = True
        attrs["enable_atomic_add"] = True
    op_attrs = [axis, shape]

    # gen data
    data, col_idx, row_idx, expect = gen_data(shape, dtype1, dtype2, axis, nnz=nnz)
    output_shape = expect.shape
    attrs["is_csr"] = True

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
            [data, col_idx, row_idx, output], akg.tvm.context(target_name, 0))
        target_profiling(mod, *args_list, target=target_name,  repeat_time=attrs["repeat_times"])
    return (data, col_idx, row_idx), output, expect, res