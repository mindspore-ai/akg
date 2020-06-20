import akg
import akg.topi as topi
import akg.tvm as tvm
from gen_random import random_gaussian
from akg.utils import kernel_exec as utils
from akg import platform as cce
import numpy as np
import pdb

dtype = "float16"
mapKey = {"add":"binary", "sub":"binary","div":"binary","mul":"binary","min":"binary","max":"binary",
          "abs": "single", "exp": "single", "log": "single", "sqrt": "single",
          "adds": "single", "muls": "single"}
insn = "adds"
insnType = mapKey[insn]

def gen_data(shape, dtype):
    support_list = {"float16": np.float16, "float32": np.float32}
    ma = random_gaussian(shape, miu=1, sigma=0.1)
    mb = random_gaussian(shape, miu=1, sigma=0.1)
    ma = ma.astype(support_list[dtype])
    mb = mb.astype(support_list[dtype])
    expect = ma

    if insn == "add":
        expect = ma + mb
    elif insn == "sub":
        expect = ma - mb
    if insn == "mul":
        expect = ma * mb
    elif insn == "div":
        expect = ma / mb
    elif insn == "max":
        expect = np.max(ma, mb)
    elif insn == "min":
        expect = np.min(ma, mb)

    elif insn == "abs":
        expect = np.abs(ma)
    elif insn == "exp":
        expect = np.exp(ma)
    elif insn == "log":
        expect = np.log(ma)
    elif insn == "sqrt":
        expect = np.sqrt(ma)

    elif insn == "adds":
        expect = ma + 2
    elif insn == "muls":
        expect = ma * 2

    return ma, mb, expect

def gen_kernel():
    kernel_name = "dynamic_1d_" + insn + "_" + dtype
    attrs = {}
    attrs['enable_multicore'] = False
    attrs['enable_post_poly_loop_partition'] = False
    attrs['enable_unroll_loop'] = False
    attrs['enable_fix_loop_extent'] = False
    attrs['enable_double_buffer'] = False
    attrs['enable_dynamic'] = True
    attrs['dim'] = "0 0 1024 1"
    mod = my_dsl(dtype, kernel_name, attrs)
    source_code = mod.imported_modules[0].get_source()
    print(source_code)
    save_cce(source_code)
    return mod

def my_dsl(dtype, kernel_name, attrs):
    m = tvm.var("M")
    n = tvm.var("N")
    A = tvm.placeholder((m,), name="A", dtype=dtype)
    B = tvm.placeholder((m,), name="B", dtype=dtype)

    if insn == "add":
        C = topi.add(A, B)
    elif insn == "sub":
        C = topi.subtract(A, B)
    if insn == "mul":
        C = topi.multiply(A, B)
    elif insn == "div":
        C = topi.divide(A, B)
    elif insn == "max":
        C = topi.maximum(A, B)
    elif insn == "min":
        C = topi.minimum(A, B)

    elif insn == "abs":
        C = tvm.compute(A.shape, lambda *index: tvm.abs(A(*index)), name='C')
    elif insn == "exp":
        C = topi.exp(A)
    elif insn == "log":
        C = topi.log(A)
    elif insn == "sqrt":
        C = topi.sqrt(A)
        C = topi.log(A)
    elif insn == "sqrt":
        C = topi.sqrt(A)

    elif insn == "adds":
        C = A + tvm.const(2, dtype)
    elif insn == "muls":
        C = A * tvm.const(2, dtype)

    # C = tvm.compute((m, ), lambda i: A[i] + B[i], name="C")
    s = tvm.create_schedule([C.op])
    with akg.build_config(add_lower_pass=cce.debug_mode(0), dump_pass_ir=True):
        if insnType == "binary":
            mod = akg.build(s, [A, B, C], "cce", name=kernel_name, attrs = attrs, polyhedral=True)
        else:
            mod = akg.build(s, [A, C], "cce", name=kernel_name, attrs = attrs, polyhedral=True)
    return mod

def save_cce(code):
    with open("aaaa_code.cce", "w") as f:
        f.write(code)

def test_dsl(shape):
    print("\n\n\nshape:", shape, "\n\n")
    mod = gen_kernel()
    ma, mb, expect = gen_data(shape, dtype)
    output = np.full(expect.shape, 0, dtype=dtype)
    if insnType == "binary":
        output = utils.mod_launch(mod, (ma, mb, output))
    else:
        output = utils.mod_launch(mod, (ma, output))

    rtol = atol = 1e-04
    cpr_res_is = np.isclose(output, expect, rtol, atol, equal_nan=False)
    cpr_res_all = np.allclose(output, expect, rtol, atol, equal_nan=False)
    print("\noutput:", output)
    print("\nexpect:", expect)

if __name__ == "__main__":
    test_dsl((30000,))
    #test_dsl((1999,))
    # test_dsl((2001,))
