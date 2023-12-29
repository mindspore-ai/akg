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

"""operator dsl function: scatter_"""
import akg.tvm
import akg.utils as utils



def scatter_(index, src, dim, shape, value=None, target="cce"):
    '''
    Based on tensor 'index', write values in tensor src into a 'shape' zero-tensor.

    Args:
        index: Tensor, the indices of elements to scatter
        src: Tensor, the source elements to scatter, incase 'value' is not specified, should be the same size of index
        dim: Integer, the axis along which to index
        shape: Tensor, given shape
        value: Float, the source element to scatter, incase 'src' is not specified

    Returns:
        Tensor, scattered result, should be the same size of 'shape'
    '''

    # check shapes dtype
    index_shape = [x.value for x in index.shape]
    if src is None and value is None:
        raise RuntimeError("src and value cannot be None at the same time.")

    if src is not None and value is not None:
        raise RuntimeError("src and value cannot be not None at the same time.")

    # when src is not None, check shape and type
    if src is not None:
        src_shape = [x.value for x in src.shape]
        utils.check_shape(src_shape)
        if not index_shape <= src_shape:
            raise RuntimeError("all dimensions size of index should not be larger than src.")
        dtype = src.dtype
        support_list = {"float16", "float32", "int32"}
        if not dtype.lower() in support_list:
            raise RuntimeError("scatter only support %s while dtype is %s" % (",".join(support_list), dtype))

    # check index shape and type
    utils.check_shape(index_shape)
    index_dtype = index.dtype
    if not index_dtype.lower() in "int32":
        raise RuntimeError("index dtype only support int32 while dtype is %s" % index_dtype)

    if not index_shape[:dim - 1] + index_shape[dim:] <= shape[:dim - 1] + shape[dim:]:
        raise RuntimeError("All dimensions except dim size of index should not be larger than shape.")

    n = index.shape[dim].value

    def pick(j, *indices):
        return akg.tvm.expr.Select(indices[dim] == index[indices[:dim] + (j,) + indices[dim + 1:]],
                               akg.tvm.const(1, src.dtype),
                               akg.tvm.const(0, src.dtype)) * src[indices[:dim] + (j,) + indices[dim + 1:]]

    def pick_value(j, *indices):
        return akg.tvm.expr.Select(indices[dim] == index[indices[:dim] + (j,) + indices[dim + 1:]],
                               akg.tvm.const(1, 'float'),
                               akg.tvm.const(0, 'float')) * value

    if isinstance(value, float):
        reducible = akg.tvm.compute([n] + list(shape), lambda *i: pick_value(i[0], *i[1:]), name="reduc_value")

    else:
        reducible = akg.tvm.compute([n] + list(shape), lambda *i: pick(i[0], *i[1:]), name="reduc_src")

    k = akg.tvm.reduce_axis((0, n))
    res = akg.tvm.compute(shape, lambda *i: akg.tvm.sum(reducible[(k,) + i], axis=k))
    return res
