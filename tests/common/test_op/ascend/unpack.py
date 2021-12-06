# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

"""operator dsl function: unpack"""
import akg
from akg import tvm
from akg.utils.format_transform import get_shape
import akg.utils as utils


def _check_params(shape, num, axis, tensor_format, dtype):
    """
    Check the parameters including shape, num, axis, format, dtype.

    Args:
        shape (tuple): the shape of tensor
        num (int): the length of the dim axis
        axis (int): axis of unapck
        tensor_format (str): the data format of input
        dtype (str): the data type

    Returns:
        None
    """
    utils.check_shape(shape)

    # check format
    format_list = ("ND", "NHWC", "NCHW", "HWCN")
    if tensor_format == "NC1HWC0":
        if len(shape) != 5:
            raise RuntimeError("NC1HWC0 shape length should be equal to 5")
        # index list of H W axis.
        suporrted_list = (-5, -3, -2, 0, 2, 3)
        if axis not in suporrted_list:
            raise RuntimeError("NC1HWC0 supports unpack of N H W axis")
    else:
        if tensor_format not in format_list:
            raise RuntimeError("Format supports ND,NCHW,NHWC,HWCN and NC1HWC0")

    # check axis value
    if axis < -len(shape) or axis >= len(shape):
        raise RuntimeError("axis= %d not in the union of [%d, %d) " %
                           (axis, -len(shape), len(shape)))

    # check num value
    if num is None:
        num = shape[axis]
    if num is None:
        raise RuntimeError("Cannot infer num value from shape %s" % shape)
    if num != shape[axis]:
        raise RuntimeError("Num shuld be equal to length of dim axis %d" %
                           shape[axis])

    # The maximum size of functional arguments is 1024B.
    # 1 param takes 8 bytes, needs Multiple output param and 1 input param,
    # so the maximum size of output is 127 = 1024 // 8 - 1.
    if num > 127:
        raise RuntimeError("Exceeds stack holding the parameters, "
                           "maximum is 127 but get %d." % num)

    utils.ops_dtype_check(
        dtype, (
            utils.DtypeForDavinci.INT8,
            utils.DtypeForDavinci.INT16,
            utils.DtypeForDavinci.INT32,
            utils.DtypeForDavinci.INT64,
            utils.DtypeForDavinci.UINT8,
            utils.DtypeForDavinci.UINT16,
            utils.DtypeForDavinci.UINT32,
            utils.DtypeForDavinci.UINT64,
            utils.DtypeForDavinci.FLOAT16,
            utils.DtypeForDavinci.FLOAT32))

def _index_offset(shape, axis, offset, *index):
    """Compute the offset of index along one dimension."""
    input_index = list(index)
    output_index = ()
    for i, _ in enumerate(shape):
        if i == axis:
            input_index[i] = input_index[i] + offset
        output_index += (input_index[i],)

    return output_index

def _unpack_compute(input_place, num, axis):
    """Unpack a tensor into `num` tensors along axis dimension."""

    input_shape = get_shape(input_place)
    for index, _ in enumerate(input_shape):
        input_shape[index] = input_shape[index] if index != axis else 1

    output_shape_list = [input_shape for i in range(num)]

    offset = 0
    out_tensor_list = []
    for i, t_shape in enumerate(output_shape_list):
        out_tensor = tvm.compute(
            t_shape,
            lambda *index, t_shape=t_shape:
            input_place(*_index_offset(t_shape, axis, offset, *index)),
            name='tensor' + str(i))
        out_tensor_list.append(out_tensor)
        offset = offset + 1

    return tuple(out_tensor_list)


@utils.check_input_type(akg.tvm.tensor.Tensor, str,
                          (int, type(None)), (int, type(None)), (str, type(None)))
def unpack(value, tensor_format, num=None, axis=0, target=utils.CCE):
    """
    Unpacks the given dimension of a rank R tensor into rank (R-1) tensors.
    It will produce `num` tensors from value, and `axis` dim of each tensor is 1.

    Args:
        value (tvm.tensor.Tensor): tensor to be unpacked
        tensor_format (str): data format, support "ND", "NHWC", "NCHW",
                             "HWCN", "NC1HWC0".
        num (int): length of the dim axis, automatically inferred
                   if None(default).
        axis (int): axis to unpack along

    Returns:
        The tuple of `tvm.tensor.Tensor` objects unpacked from value.
    """
    shape = get_shape(value)
    dtype = value.dtype
    _check_params(shape, num, axis, tensor_format, dtype)

    # infer the value of num
    real_axis = axis + len(shape) if axis < 0 else axis
    num = shape[real_axis]

    return  _unpack_compute(value, num, real_axis)
