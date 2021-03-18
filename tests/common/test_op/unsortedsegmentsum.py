# Copyright 2019 Huawei Technologies Co., Ltd
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

"""operator dsl function: unsortedsegmentsum"""
import akg.tvm

from akg.utils import kernel_exec as utils
from akg.tvm.hybrid import script
from akg.utils import custom_tiling as ct_util
from akg.utils import validation_check as vc_util

unsortedsegmentsum_set_dim_map = {
    str(((38714, 1024, 38714, 30522), "float32")): ((8, 1), (1024 * 4, 1)),
    str(((128, 128, 64, 128, 128, 34), "float16")): ((1, 1), (16, 1), (8, 1), (8, 1)),
    str(((128, 128, 64, 128, 128, 33), "float16")): ((1, 1), (16, 1), (8, 1), (8, 1), ),
    str(((128, 128, 64, 128, 128, 33), "float32")): ((1, 1), (16, 1), (8, 1), (8, 1), ),

    # issue 818
    str(((128, 768, 128, 21128), "float32")): ((1, 1), (8, 1), (128, 1)),
    str(((160, 1024, 160, 1024), "float32")): ((1, 1), (8, 1), (160, 1)),
    str(((160, 768, 160, 1024), "float32")): ((1, 1), (8, 1), (160, 1)),
    str(((320, 1024, 320, 2048), "float32")): ((1, 1), (8, 1), (320, 1)),
    str(((320, 768, 320, 2048), "float32")): ((1, 1), (8, 1), (320, 1)),
    str(((4096, 1024, 4096, 2), "float32")): ((2, 1), (1024, 1), (1, 1)),
    str(((1024, 1024, 1024, 21128), "float32")): ((1, 1), (32, 1), (1024, 1)),
    str(((128, 1024, 128, 21128), "float32")): ((1, 1), (64, 1), (128, 1)),
    str(((256, 1024, 256, 21128), "float32")): ((1, 1), (32, 1), (256, 1)),
    str(((512, 1024, 512, 21128), "float32")): ((1, 1), (8, 1), (512, 1)),
    str(((1024, 768, 1024, 21128), "float32")): ((1, 1), (8, 1), (1024, 1)),
    str(((128, 768, 128, 21128), "float32")): ((1, 1), (8, 1), (128, 1)),
    str(((256, 768, 256, 21128), "float32")): ((1, 1), (32, 1), (256, 1)),
    str(((512, 768, 512, 21128), "float32")): ((1, 1), (8, 1), (512, 1)),
    str(((4096, 768, 4096, 2), "float32")): ((2, 1), (768, 1), (1, 1)),
    str(((640, 1024, 640, 4096), "float32")): ((1, 1), (8, 1), (640, 1)),
    str(((640, 768, 640, 4096), "float32")): ((1, 1), (8, 1), (640, 1)),
    str(((80, 1024, 80, 512), "float32")): ((1, 1), (8, 1), (80, 1)),
    str(((1280, 768, 1280, 8192), "float32")): ((1, 1), (8, 1), (1280, 1)),
}


def unsortedsegmentsum_set_dim_func(input_data, ids_tensor, num_segments):
    key = []
    for x in input_data.shape:
        key.append(x)
    for x in ids_tensor.shape:
        key.append(x)
    key.append(num_segments)
    hash_key = str((tuple(key), input_data.dtype))

    return ct_util.set_dims_by_key(hash_key, unsortedsegmentsum_set_dim_map), hash_key


@ct_util.reg_set_dim_func(unsortedsegmentsum_set_dim_func)
def unsortedsegmentsum(input_data, ids_tensor, num_segments):
    """
    Computes the sum value  along ids_tensor of a akg.tvm.Tensor

    Args:
        input_data (tvm.tensor.Tensor): Tensor of type float16, float32, int32
        ids_tensor (tvm.tensor.Tensor): Tensor of type int32, shape is a prefix of input_data.shape.
        num_segments (int): the number of classes in ids_tensor

    Returns:
        tvm.tensor.Tensor of same type as input_data,
   
    Raises:
        RuntimeError: If the type of input_data is invalid.

    """
    dtype = input_data.dtype
    check_list = ["float16", "float32", "int32"]
    if not (dtype in check_list):
        raise RuntimeError("unsortedsegmentsum only support %s while dtype is %s" % (",".join(check_list), dtype))

    shape = [x.value for x in input_data.shape]
    vc_util.check_shape(shape)

    id_shape = [x.value for x in ids_tensor.shape]
    vc_util.check_shape(id_shape)

    @script
    def hy_func_2d_1d(input_data, ids_tensor, zero):
        nd0 = num_segments
        d0, d1 = input_data.shape
        out = output_tensor((nd0, d1), input_data.dtype)
        for i in range(nd0):
            for j in range(d1):
                out[i, j] = zero
                for ii in range(d0):
                    if ids_tensor[ii] == i:
                        out[i, j] = out[i, j] + input_data[ii, j]
        return out

    @script
    def hy_func_3d_1d(input_data, ids_tensor, zero):
        nd0 = num_segments
        d0, d1, d2 = input_data.shape
        out = output_tensor((nd0, d1, d2), input_data.dtype)
        for i in range(nd0):
            for j in range(d1):
                for k in range(d2):
                    out[i, j, k] = zero
                    for ii in range(d0):
                        if ids_tensor[ii] == i:
                            out[i, j, k] = out[i, j, k] + input_data[ii, j, k]
        return out

    @script
    def hy_func_3d_2d(input_data, ids_tensor, zero):
        nd0 = num_segments
        d0, d1, d2 = input_data.shape
        out = output_tensor((nd0, d2), input_data.dtype)
        for i in range(nd0):
            for j in range(d2):
                out[i, j] = zero
                for ii in range(d0):
                    for jj in range(d1):
                        if ids_tensor[ii, jj] == i:
                            out[i, j] = out[i, j] + input_data[ii, jj, j]
        return out

    @script
    def hy_func_4d_1d(input_data, ids_tensor, zero):
        nd0 = num_segments
        d0, d1, d2, d3 = input_data.shape
        out = output_tensor((nd0, d1, d2, d3), input_data.dtype)
        for i in range(nd0):
            for j in range(d1):
                for k in range(d2):
                    for l in range(d3):
                        out[i, j, k, l] = zero
                        for ii in range(d0):
                            if ids_tensor[ii] == i:
                                out[i, j, k, l] = out[i, j, k, l] + input_data[ii, j, k, l]
        return out

    @script
    def hy_func_5d_1d(input_data, ids_tensor, zero):
        nd0 = num_segments
        d0, d1, d2, d3, d4 = input_data.shape
        out = output_tensor((nd0, d1, d2, d3, d4), input_data.dtype)
        for i in range(nd0):
            for j in range(d1):
                for k in range(d2):
                    for l in range(d3):
                        for m in range(d4):
                            out[i, j, k, l, m] = zero
                            for ii in range(d0):
                                if ids_tensor[ii] == i:
                                    out[i, j, k, l, m] = out[i, j, k, l, m] + input_data[ii, j, k, l, m]
        return out

    ZERO = akg.tvm.const(0.0, dtype)

    if len(shape) == 2 and len(id_shape) == 1:
        output = hy_func_2d_1d(input_data, ids_tensor, ZERO)
    elif len(shape) == 3 and len(id_shape) == 1:
        output = hy_func_3d_1d(input_data, ids_tensor, ZERO)
    elif len(shape) == 3 and len(id_shape) == 2:
        output = hy_func_3d_2d(input_data, ids_tensor, ZERO)
    elif len(shape) == 4 and len(id_shape) == 1:
        output = hy_func_4d_1d(input_data, ids_tensor, ZERO)
    elif len(shape) == 5 and len(id_shape) == 1:
        output = hy_func_5d_1d(input_data, ids_tensor, ZERO)
    else:
        raise RuntimeError("unsortedsegmentsum not supported now")
    attr_map = {"enable_dma_sink": True}
    return output, attr_map
