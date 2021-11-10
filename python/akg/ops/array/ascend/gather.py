# Copyright 2019-2021 Huawei Technologies Co., Ltd
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

"""operator dsl function: gather"""
import akg.tvm
import akg
import akg.utils as utils
from akg.utils.kernel_exec import create_code, debug_mode

def new_alloc(ib, dtype, shape, name, scope):
    """allocate buffer"""
    buf_var = ib.allocate(dtype, shape, name=name, scope=scope)
    new_buffer = akg.tvm.decl_buffer(shape, buf_var.dtype, name=name, scope=scope, data=buf_var)
    return new_buffer


def kernel_ir(dst, data, indices):
    """build ir"""
    ib = akg.tvm.ir_builder.create()

    # copy indices to UB
    indices_ptr = ib.buffer_ptr(indices)
    batch_size = 1024
    batch_num = indices.shape[0] // batch_size
    last_size = indices.shape[0] % batch_size
    burst_len_of_batch_size = (batch_size + 7) // 8
    burst_len_of_last_size = (last_size + 7) // 8

    data_ub = new_alloc(ib, data.dtype, (data.shape[1]), "X_UB", scope="local.UB")
    indices_ub = new_alloc(ib, indices_ptr.dtype, (batch_size,), "Y_UB", scope="local.UB")
    row_len = data.shape[1]
    burst_len = (row_len + 15) // 16

    with ib.if_scope(batch_num > 0):
        with ib.for_range(0, batch_num, name='num') as num:
            ib.emit(akg.tvm.call_extern(indices.dtype, "copy_gm_to_ubuf",
                                    indices_ub.access_ptr("w"),
                                    indices.access_ptr('r', offset=num * batch_size),
                                    0, 1, burst_len_of_batch_size, 0, 0))

            with ib.for_range(0, batch_size, name='row') as row:
                reg = ib.allocate("int32", (1,), name='reg', scope="local.REG")
                ib.emit(akg.tvm.call_extern(
                    indices.dtype, "reg_mov",
                    akg.tvm.call_extern(reg.dtype, "reg", reg[0]),
                    indices_ub.access_ptr('r', offset=row)
                ))
                gm_offset = reg[0] * row_len
                ib.emit(akg.tvm.call_extern(data.dtype, "copy_gm_to_ubuf",
                                        data_ub.access_ptr("w"),
                                        data.access_ptr('r', offset=gm_offset),
                                        0, 1, burst_len, 0, 0))
                ib.emit(akg.tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                        dst.access_ptr('w', offset=(num * batch_size + row) * row_len),
                                        data_ub.access_ptr("r"),
                                        0, 1, burst_len, 0, 0))

    with ib.if_scope(last_size > 0):
        ib.emit(akg.tvm.call_extern(indices.dtype, "copy_gm_to_ubuf",
                                indices_ub.access_ptr("w"),
                                indices.access_ptr('r', offset=batch_num * batch_size),
                                0, 1, burst_len_of_last_size, 0, 0))

        with ib.for_range(0, last_size, name='row') as row:
            reg = ib.allocate("int32", (1,), name='reg', scope="local.REG")
            ib.emit(akg.tvm.call_extern(
                indices.dtype, "reg_mov",
                akg.tvm.call_extern(reg.dtype, "reg", reg[0]),
                indices_ub.access_ptr('r', offset=row)
            ))
            gm_offset = reg[0] * row_len
            ib.emit(akg.tvm.call_extern(data.dtype, "copy_gm_to_ubuf",
                                    data_ub.access_ptr("w"),
                                    data.access_ptr('r', offset=gm_offset),
                                    0, 1, burst_len, 0, 0))
            ib.emit(akg.tvm.call_extern(dst.dtype, "copy_ubuf_to_gm",
                                    dst.access_ptr('w', offset=(batch_num * batch_size + row) * row_len),
                                    data_ub.access_ptr("r"),
                                    0, 1, burst_len, 0, 0))

    return ib.get()

@utils.check_input_type((list, tuple), (list, tuple), str, str, int, str, (str, type(None)))
def Gather(params_shape, indices_shape, params_dtype, indices_dtype, axis, kernel_name, cce_path="./", target=utils.CCE):
    """Gather data by indices"""
    utils.check_shape(params_shape, length=2)
    utils.check_shape(indices_shape, length=1)
    utils.ops_dtype_check(params_dtype, utils.DtypeForDavinci.ALL_TYPES)
    utils.ops_dtype_check(indices_dtype, utils.DtypeForDavinci.INT32)
    utils.check_equal("axis", "zero", axis, 0)

    # construct compute
    o_shape = (indices_shape[0], params_shape[1])
    xx = akg.tvm.placeholder(params_shape, dtype=params_dtype, name="X")
    yy = akg.tvm.placeholder(indices_shape, dtype=indices_dtype, name="Y")
    res = akg.tvm.extern(o_shape, [xx, yy], lambda ins, outs: kernel_ir(outs[0], ins[0], ins[1]),
        name="res", dtype=params_dtype)
    s = akg.tvm.create_schedule(res.op)

    # create cce
    attrs = {"enable_multicore": False}
    with akg.build_config(add_lower_pass=debug_mode(0), dump_pass_ir=True):
        mod = akg.build(s, [xx, yy, res], "cce", name=kernel_name, attrs=attrs)

    source_code = mod.imported_modules[0].get_source()
    create_code(kernel_name, cce_path, source_code)

    return mod
