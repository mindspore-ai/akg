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

"""operator dsl function: fused_batch_norm_grad"""
from __future__ import absolute_import
import akg
import akg.tvm
import akg.topi
import akg.utils as utils
from akg.utils.dsl_create import mul_axis_sum
from akg.utils.format_transform import get_shape
from akg.utils.kernel_exec import product_is_mini
from akg.utils import custom_tiling as ct_util
from akg.utils.validation_check import is_all_1_but_axis_equal

def copy_attrs(attrs):
    new_attrs = dict()
    for key in attrs:
        new_attrs[key] = attrs[key]
    return new_attrs

def bng1_tiling_strategy(tensor):
    """Custom tiling strategy for first part of splited fused_batch_norm_grad op"""

    # bn1 input [N, C1, C0, H, W]
    batch, _, in_h, in_w, _ = get_shape(tensor)
    batch_pos = 0
    c0_pos = 4
    c1_pos = 1
    strategy = list()
    core_num = 32

    if batch > 1:
        strategy += ct_util.create_constraint_on_tensor(
            tensor=tensor,
            values=1,
            constraints=ct_util.TileConstraint.CANDIDATE,
            tensor_pos=batch_pos)

    if in_h != 1 or in_w != 1:
        strategy += ct_util.create_constraint_on_tensor(
            tensor=tensor,
            values="FULL",
            constraints=ct_util.TileConstraint.MAX,
            tensor_pos=c0_pos)
        strategy += ct_util.create_constraint_on_tensor(
            tensor=tensor,
            values=1,
            constraints=ct_util.TileConstraint.CANDIDATE,
            tensor_pos=c1_pos)
        strategy += ct_util.create_constraint_on_tensor(
            tensor=tensor,
            values=core_num,
            constraints=ct_util.TileConstraint.CANDIDATE,
            tensor_pos=c1_pos)

    return strategy


def bng3_tiling_strategy(tensor):
    """Custom tiling strategy for 3rd part of splited fused_batch_norm_grad op"""

    # bn3 input [C1, C0, N, H, W]
    strategy_nc0 = list()
    n_pos = 0
    c0_pos = 4
    c1_pos = 1
    core_num = 32

    strategy_nc0 += ct_util.create_constraint_on_tensor(
        tensor=tensor,
        values=1,
        constraints=ct_util.TileConstraint.CANDIDATE,
        tensor_pos=n_pos)

    strategy_nc0 += ct_util.create_constraint_on_tensor(
        tensor=tensor,
        values="FULL",
        constraints=ct_util.TileConstraint.MAX,
        tensor_pos=c0_pos)

    strategy_c1 = list()
    strategy_c1 += ct_util.create_constraint_on_tensor(
        tensor=tensor,
        values=1,
        constraints=ct_util.TileConstraint.CANDIDATE,
        tensor_pos=c1_pos)

    strategy_c1 += ct_util.create_constraint_on_tensor(
        tensor=tensor,
        values=core_num,
        constraints=ct_util.TileConstraint.CANDIDATE,
        tensor_pos=c1_pos)

    strategy = strategy_nc0 + strategy_c1

    return strategy


def check_inputs(op_id, *args):
    """check inputs"""
    def check_shape(tensor_format, tensor, shape_nc1hwc0, name):
        shape = get_shape(tensor)
        if tensor_format == "C1C0":
            if not is_all_1_but_axis_equal(shape, shape_nc1hwc0, (1, 4)):
                raise AssertionError("{} shape {} did not match data_shape {}"
                                     "".format(name, shape, shape_nc1hwc0))
        elif tensor_format == "NC1C0":
            if not is_all_1_but_axis_equal(shape, shape_nc1hwc0, (0, 1, 4)):
                raise AssertionError("{} shape {} did not match data_shape {}"
                                     "".format(name, shape, shape_nc1hwc0))
        elif tensor_format == "NC1HWC0":
            if not is_all_1_but_axis_equal(shape, shape_nc1hwc0, (0, 1, 2, 3, 4)):
                raise AssertionError("{} shape {} did not match data_shape {}"
                                     "".format(name, shape, shape_nc1hwc0))
    if op_id not in range(1, 4):
        raise AssertionError("Not support {} part!".format(op_id))

    def check_op1(*args):
        dy, data, mean = args
        utils.ops_dtype_check([dy.dtype, data.dtype],
                              utils.DtypeForDavinci.ALL_FLOAT)
        utils.ops_dtype_check(mean.dtype, utils.DtypeForDavinci.FLOAT32)
        shape_nc1hwc0 = get_shape(dy)
        check_shape("NC1HWC0", data, shape_nc1hwc0, "data")
        check_shape("C1C0", mean, shape_nc1hwc0, "mean")
        return shape_nc1hwc0   
    
    def check_op2(*args):
        dgamma_red_hw, dbeta_red_hw, var, gamma, _, shape_nc1hwc0 = args
        utils.ops_dtype_check([dgamma_red_hw.dtype, dbeta_red_hw.dtype,
                               var.dtype, gamma.dtype],
                              utils.DtypeForDavinci.FLOAT32)
        if not isinstance(shape_nc1hwc0, (list, tuple)):
            raise AssertionError("data_shape must be a list or tuple!")
        check_shape("NC1C0", dgamma_red_hw, shape_nc1hwc0, "dgamma_red_hw")
        check_shape("NC1C0", dbeta_red_hw, shape_nc1hwc0, "dbeta_red_hw")
        check_shape("C1C0", var, shape_nc1hwc0, "var")
        check_shape("C1C0", gamma, shape_nc1hwc0, "gamma")
        return shape_nc1hwc0
    
    def check_op3(*args):
        dy, rs, dgamma_dx, dbeta_dx, data_minus_mean = args
        utils.ops_dtype_check(dy.dtype, utils.DtypeForDavinci.ALL_FLOAT)
        utils.ops_dtype_check([rs.dtype, dgamma_dx.dtype, dbeta_dx.dtype,
                               data_minus_mean.dtype],
                              utils.DtypeForDavinci.FLOAT32)
        shape_nc1hwc0 = get_shape(dy)
        check_shape("C1C0", rs, shape_nc1hwc0, "rs")
        check_shape("C1C0", dgamma_dx, shape_nc1hwc0, "dgamma_dx")
        check_shape("C1C0", dbeta_dx, shape_nc1hwc0, "dbeta_dx")
        check_shape("NC1HWC0", data_minus_mean,
                    shape_nc1hwc0, "data_minus_mean")
        return shape_nc1hwc0

    shape_nc1hwc0 = None
    if op_id == 1:
        shape_nc1hwc0 = check_op1(*args)
    elif op_id == 2:
        shape_nc1hwc0 = check_op2(*args)
    else:
        shape_nc1hwc0 = check_op3(*args)

    if len(shape_nc1hwc0) != 5:
        raise AssertionError(
            "fused_batch_norm_grad_split only support special5D shape!")


def sum_data(data, axes, keepdims, single_sum=False):
    """different solutions for sum multi axes"""
    if single_sum:
        data = akg.topi.sum(data, axis=axes, keepdims=keepdims)
    else:
        data = mul_axis_sum(data, axes, keepdims)
    return data


attrs_bng1_ = dict()
set_attr_map_bng1_ = {
    "float32": [("merge_outer_loop_for_multicore", 1), ("single_sum", True)],
    "float16": [("enable_bisect_optimize", False)],

    str(((32, 64, 14, 14, 16), "float16")): [("single_sum", True)],
    str(((32, 128, 7, 7, 16), "float16")): [("single_sum", True)],
    str(((32, 32, 14, 14, 16), "float32")): [("pragma_disable_whole_component", False)],
}


def set_attr_func_bng1_(*args):
    """set attr"""

    shape = tuple(get_shape(args[0]))
    dtype = args[0].dtype
    attrs_bng1_.clear()
    attrs_bng1_["dead_code_elim"] = True
    hash_key = dtype

    if hash_key in set_attr_map_bng1_.keys():
        attrs_list = set_attr_map_bng1_.get(hash_key, {})
        for attr in attrs_list:
            attrs_bng1_[attr[0]] = attr[1]

    hash_key = str((shape, dtype))
    
    if hash_key in set_attr_map_bng1_.keys():
        attrs_list = set_attr_map_bng1_.get(hash_key, {})
        for attr in attrs_list:
            attrs_bng1_[attr[0]] = attr[1]


set_dim_map_bng1_ = {
    # resnet50 V1.0 && V1.5
    str(((32, 4, 112, 112, 16), "float32")): (
        (0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 16, 1), (0, 3, 1, 1), (0, 4, 112, 1),
        (1, 0, 1, 1), (1, 1, 1, 1), (1, 2, 112, 1), (1, 3, 1, 1), (1, 4, 16, 1)),
    str(((32, 4, 56, 56, 16), "float32")): (
        (0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 16, 1), (0, 3, 1, 1), (0, 4, 56, 1),
        (1, 0, 1, 1), (1, 1, 1, 1), (1, 2, 56, 1), (1, 3, 1, 1), (1, 4, 16, 1)),
    str(((32, 16, 56, 56, 16), "float32")): (
        (0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 16, 1), (0, 3, 1, 1), (0, 4, 56, 1),
        (1, 0, 1, 1), (1, 1, 1, 1), (1, 2, 56, 1), (1, 3, 1, 1), (1, 4, 16, 1)),
    str(((32, 8, 28, 28, 16), "float32")): (
        (0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 16, 1), (0, 3, 1, 1), (0, 4, 28, 1),
        (1, 0, 1, 1), (1, 1, 1, 1), (1, 2, 28, 1), (1, 3, 1, 1), (1, 4, 16, 1)),
    str(((32, 32, 28, 28, 16), "float32")): (
        (0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 16, 1), (0, 3, 1, 1), (0, 4, 28, 1),
        (1, 0, 1, 1), (1, 1, 1, 1), (1, 2, 28, 1), (1, 3, 1, 1), (1, 4, 16, 1)),
    str(((32, 16, 14, 14, 16), "float32")): (
        (0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 16, 1), (0, 3, 14, 1), (0, 4, 14, 1),
        (1, 0, 1, 1), (1, 1, 1, 1), (1, 2, 14, 1), (1, 3, 14, 1), (1, 4, 16, 1)),
    str(((32, 64, 14, 14, 16), "float32")): (
        (0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 16, 1), (0, 3, 14, 1), (0, 4, 14, 1),
        (1, 0, 1, 1), (1, 1, 1, 1), (1, 2, 14, 1), (1, 3, 14, 1), (1, 4, 16, 1)),
    str(((32, 32, 7, 7, 16), "float32")): (
        (0, 0, 1, 1), (0, 1, 8, 1), (0, 2, 16, 1), (0, 3, 7, 1), (0, 4, 7, 1),
        (1, 0, 1, 1), (1, 1, 8, 1), (1, 2, 7, 1), (1, 3, 7, 1), (1, 4, 16, 1)),
    str(((32, 128, 7, 7, 16), "float32")): (
        (0, 0, 1, 1), (0, 1, 8, 1), (0, 2, 16, 1), (0, 3, 7, 1), (0, 4, 7, 1),
        (1, 0, 1, 1), (1, 1, 8, 1), (1, 2, 7, 1), (1, 3, 7, 1), (1, 4, 16, 1)),

    str(((32, 4, 112, 112, 16), "float16")): (
        (1, 1), (1, 1), (16, 1), (1, 1), (112, 1)),
    str(((32, 4, 56, 56, 16), "float16")): (
        (1, 1), (4, 1), (16, 1), (1, 1), (56, 1)),
    str(((32, 16, 56, 56, 16), "float16")): (
        (1, 1), (4, 1), (16, 1), (1, 1), (56, 1)),
    str(((32, 8, 28, 28, 16), "float16")): (
        (1, 1), (4, 1), (16, 1), (4, 1), (28, 1)),
    str(((32, 32, 28, 28, 16), "float16")): (
        (1, 1), (4, 1), (16, 1), (1, 1), (28, 1)),
    str(((32, 16, 14, 14, 16), "float16")): (
        (1, 1), (16, 1), (16, 1), (1, 1), (14, 1)),
    str(((32, 64, 14, 14, 16), "float16")): (
        (1, 1), (64, 1), (16, 1), (1, 1), (1, 1)),
    str(((32, 32, 7, 7, 16), "float16")): (
        (1, 1), (16, 1), (16, 1), (1, 1), (7, 1)),
    str(((32, 128, 7, 7, 16), "float16")): (
        (1, 1), (128, 1), (16, 1), (1, 1), (1, 1)),

    # resnet50 V1.5
    str(((32, 8, 56, 56, 16), "float32")): (
        (0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 16, 1), (0, 3, 1, 1), (0, 4, 56, 1),
        (1, 0, 1, 1), (1, 1, 1, 1), (1, 2, 56, 1), (1, 3, 1, 1), (1, 4, 16, 1)),
    str(((32, 16, 28, 28, 16), "float32")): (
        (0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 16, 1), (0, 3, 1, 1), (0, 4, 28, 1),
        (1, 0, 1, 1), (1, 1, 1, 1), (1, 2, 28, 1), (1, 3, 1, 1), (1, 4, 16, 1)),
    str(((32, 32, 14, 14, 16), "float32")): (
        (0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 16, 1), (0, 3, 14, 1), (0, 4, 14, 1),
        (1, 0, 1, 1), (1, 1, 1, 1), (1, 2, 14, 1), (1, 3, 14, 1), (1, 4, 16, 1)),

    str(((32, 8, 56, 56, 16), "float16")): (
        (1, 1), (1, 1), (16, 1), (1, 1), (56, 1)),
    str(((32, 16, 28, 28, 16), "float16")): (
        (1, 1), (1, 1), (16, 1), (1, 1), (28, 1)),
    str(((32, 32, 14, 14, 16), "float16")): (
        (1, 1), (1, 1), (16, 1), (14, 1), (14, 1)),
}


def set_dim_func_bng1_(*args):
    """set dim for op bn_grad_1"""
    set_attr_func_bng1_(*args)
    shape = tuple(get_shape(args[0]))
    dtype = args[0].dtype
    hash_key = str((shape, dtype))
    dim_info = set_dim_map_bng1_.get(hash_key, "")
    return ct_util.set_dims(dim_info), hash_key


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                        akg.tvm.tensor.Tensor, (str, type(None)))
def fused_bn_grad1(dy, data, mean):
    """Gradient for fused_batch_norm, reduce axis H and W."""
    check_inputs(1, dy, data, mean)
    dim_info = set_dim_func_bng1_(dy)[0]
    attrs = copy_attrs(attrs_bng1_)
    single_sum = attrs.pop("single_sum", False)
    ori_dtype = dy.dtype
    if ori_dtype != "float32":
        dy = akg.topi.cast(dy, "float32")
        data = akg.topi.cast(data, "float32")
    axes = (2, 3)

    dbeta_red_hw = sum_data(dy, axes, keepdims=True, single_sum=single_sum)

    mean = akg.lang.ascend.broadcast(mean, data.shape)
    data_minus_mean = akg.tvm.compute(
        data.shape, lambda *i: data(*i) - mean(*i), "data_minus_mean")
    dgamma_param = akg.tvm.compute(
        data.shape, lambda *i: dy(*i) * data_minus_mean(*i), "dgamma_param")
    dgamma_red_hw = sum_data(
        dgamma_param, axes, keepdims=True, single_sum=single_sum)
    if dim_info != "":
        attrs["dim"] = dim_info
    attrs["custom_tiling"] = bng1_tiling_strategy(data)
    return dgamma_red_hw, dbeta_red_hw, data_minus_mean, attrs


set_dim_map_bng2_ = {
    # resnet50 V1.0 & V1.5
    str((32, 4, 112, 112, 16)): ((1, 1), (16, 1), (33, 1)),
    str((32, 4, 56, 56, 16)): ((1, 1), (16, 1), (33, 1)),
    str((32, 16, 56, 56, 16)): ((1, 1), (16, 1), (33, 1)),
    str((32, 8, 28, 28, 16)): ((1, 1), (16, 1), (33, 1)),
    str((32, 32, 28, 28, 16)): ((1, 1), (16, 1), (33, 1)),
    str((32, 16, 14, 14, 16)): ((1, 1), (16, 1), (33, 1)),
    str((32, 64, 14, 14, 16)): ((2, 1), (16, 1), (33, 1)),
    str((32, 32, 7, 7, 16)): ((1, 1), (16, 1), (33, 1)),
    str((32, 128, 7, 7, 16)): ((4, 1), (16, 1), (33, 1)),

    # resnet50 V1.5
    str((32, 8, 56, 56, 16)): ((4, 1), (16, 1), (33, 1)),
    str((32, 16, 28, 28, 16)): ((1, 1), (16, 1), (33, 1)),
    str((32, 32, 14, 14, 16)): ((1, 1), (16, 1), (33, 1)),
}

attrs_bng2_ = {
}


def set_dim_func_bng2_(*args):
    """set dim for op bn_grad_2"""
    shape = tuple(args[-1])
    hash_key = str(shape)
    dim_info = set_dim_map_bng2_.get(hash_key, "")
    return ct_util.set_dims(dim_info), hash_key


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                        akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                        float, (list, tuple), (str, type(None)))
def fused_bn_grad2(dgamma_red_hw, dbeta_red_hw, var, gamma, eps, data_shape):
    """Second part of fused_bn_grad, reduce axis N, calculate the result of dgamma and dbeta."""
    check_inputs(2, dgamma_red_hw, dbeta_red_hw, var, gamma, eps, data_shape)
    attrs = copy_attrs(attrs_bng2_)
    dim_info = set_dim_func_bng2_(data_shape)[0]
    m = data_shape[0] * data_shape[2] * data_shape[3]
    neg_m_rec = akg.tvm.const((-1.0 / m), dtype=var.dtype)
    eps = akg.tvm.const(eps, var.dtype)
    shape = get_shape(var)

    dbeta = akg.topi.sum(dbeta_red_hw, 0, keepdims=True)

    v = akg.tvm.compute(shape, lambda *i: var(*i) + eps, name="var_plus_eps")
    if product_is_mini():
        v = akg.topi.cast(v, "float16")
    rsqvar = akg.tvm.compute(shape,
                             lambda *i:
                             akg.tvm.exp(akg.tvm.log(v(*i)) *
                                         akg.tvm.const(-0.5, v.dtype)),
                             name="rsqvar", attrs={'no_inline': 1})
    if product_is_mini():
        rsqvar = akg.topi.cast(rsqvar, "float32")

    dgamma_red_n = akg.topi.sum(dgamma_red_hw, 0, keepdims=True)
    dgamma = akg.tvm.compute(shape,
                             lambda *i: dgamma_red_n(*i) * rsqvar(*i),
                             name="dgamma")

    rs = akg.tvm.compute(shape,
                         lambda *i: gamma(*i) * rsqvar(*i),
                         name="rs", attrs={'no_inline': 1})
    rs_div_m = akg.tvm.compute(shape,
                               lambda *i: rs(*i) * neg_m_rec,
                               name="rs_div_m", attrs={'no_inline': 1})
    dgamma_dx = akg.tvm.compute(shape,
                                lambda *i:
                                rs_div_m(*i) * rsqvar(*i) * dgamma(*i),
                                name="dgamma_dx")
    dbeta_dx = akg.tvm.compute(shape,
                               lambda *i: rs_div_m(*i) * dbeta(*i),
                               name="dbeta_dx")
    if dim_info != "":
        attrs["dim"] = dim_info
    return dgamma, dbeta, rs, dgamma_dx, dbeta_dx, attrs


attrs_bng3_ = {
}
set_dim_map_bng3_ = {
    # resnet50 V1.0 & V1.5
    str(((32, 4, 112, 112, 16), "float32")): ((1, 1), (1, 1), (2, 1), (112, 1), (16, 1)),
    str(((32, 4, 56, 56, 16), "float32")): ((1, 1), (1, 1), (4, 1), (56, 1), (16, 1)),
    str(((32, 16, 56, 56, 16), "float32")): ((1, 1), (1, 1), (4, 1), (56, 1), (16, 1)),
    str(((32, 8, 28, 28, 16), "float32")): ((1, 1), (1, 1), (7, 1), (28, 1), (16, 1)),
    str(((32, 32, 28, 28, 16), "float32")): ((1, 1), (1, 1), (7, 1), (28, 1), (16, 1)),
    str(((32, 16, 14, 14, 16), "float32")): ((1, 1), (1, 1), (14, 1), (14, 1), (16, 1)),
    str(((32, 64, 14, 14, 16), "float32")): ((1, 1), (1, 1), (14, 1), (14, 1), (16, 1)),
    str(((32, 32, 7, 7, 16), "float32")): ((1, 1), (8, 1), (7, 1), (7, 1), (16, 1)),
    str(((32, 128, 7, 7, 16), "float32")): ((1, 1), (8, 1), (7, 1), (7, 1), (16, 1)),

    str(((32, 4, 112, 112, 16), "float16")): ((1, 1), (1, 1), (1, 1), (112, 1), (16, 1)),
    str(((32, 4, 56, 56, 16), "float16")): ((1, 1), (1, 1), (4, 1), (56, 1), (16, 1)),
    str(((32, 16, 56, 56, 16), "float16")): ((1, 1), (1, 1), (4, 1), (56, 1), (16, 1)),
    str(((32, 8, 28, 28, 16), "float16")): ((1, 1), (1, 1), (7, 1), (28, 1), (16, 1)),
    str(((32, 32, 28, 28, 16), "float16")): ((1, 1), (1, 1), (7, 1), (28, 1), (16, 1)),
    str(((32, 16, 14, 14, 16), "float16")): ((1, 1), (1, 1), (14, 1), (14, 1), (16, 1)),
    str(((32, 64, 14, 14, 16), "float16")): ((1, 1), (1, 1), (14, 1), (14, 1), (16, 1)),
    str(((32, 32, 7, 7, 16), "float16")): ((1, 1), (8, 1), (7, 1), (7, 1), (16, 1)),
    str(((32, 128, 7, 7, 16), "float16")): ((1, 1), (128, 1), (1, 1), (1, 1), (16, 1)),

    # resnet50 V1.5
    str(((32, 8, 56, 56, 16), "float32")): ((1, 1), (1, 1), (4, 1), (56, 1), (16, 1)),
    str(((32, 8, 56, 56, 16), "float16")): ((1, 1), (1, 1), (4, 1), (56, 1), (16, 1)),
    str(((32, 16, 28, 28, 16), "float32")): ((1, 1), (1, 1), (7, 1), (28, 1), (16, 1)),
    str(((32, 16, 28, 28, 16), "float16")): ((1, 1), (1, 1), (7, 1), (28, 1), (16, 1)),
    str(((32, 32, 14, 14, 16), "float32")): ((1, 1), (1, 1), (14, 1), (14, 1), (16, 1)),
    str(((32, 32, 14, 14, 16), "float16")): ((1, 1), (1, 1), (14, 1), (14, 1), (16, 1)),
}


def set_dim_func_bng3_(*args):
    """set dim for op bn_grad_3"""
    shape = tuple(get_shape(args[0]))
    dtype = args[0].dtype
    hash_key = str((shape, dtype))
    dim_info = set_dim_map_bng3_.get(hash_key, "")
    return ct_util.set_dims(dim_info), hash_key


@utils.check_input_type(akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                        akg.tvm.tensor.Tensor, akg.tvm.tensor.Tensor,
                        akg.tvm.tensor.Tensor, (str, type(None)))
def fused_bn_grad3(dy, rs, dgamma_dx, dbeta_dx, data_minus_mean):
    """Gradient for fused_batch_norm, calculate dx."""
    check_inputs(3, dy, rs, dgamma_dx, dbeta_dx, data_minus_mean)
    attrs = copy_attrs(attrs_bng3_)
    dim_info = set_dim_func_bng3_(dy)[0]

    ori_dtype = dy.dtype
    if ori_dtype == "float16":
        dy = akg.topi.cast(dy, "float32")
    shape = tuple(get_shape(dy))

    def map_index(i):
        return (0, i[1], 0, 0, i[4])

    dx_dbeta = akg.tvm.compute(shape,
                               lambda *i:
                               akg.lang.ascend.vmadd(
                                   dy(*i), rs(*map_index(i)), dbeta_dx(*map_index(i))),
                               name="dx_dbeta")
    dx = akg.tvm.compute(shape,
                         lambda *i:
                         akg.lang.ascend.vmla(
                             dx_dbeta(*i), data_minus_mean(*i), dgamma_dx(*map_index(i))),
                         name="bn_grad_dx")
    if ori_dtype == "float16":
        dx = akg.topi.cast(dx, ori_dtype)
    if dim_info != "":
        attrs["dim"] = dim_info
    attrs["custom_tiling"] = bng3_tiling_strategy(dx)
    return dx, attrs
