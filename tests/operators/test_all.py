#!/usr/bin/env python3
# coding: utf-8
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
# limitations under the License

from tests.common.test_utils import gen_random_shape
from tests.operators.test_op.test_ms_add import test_ms_add
from tests.operators.test_op.test_ms_reduce_sum import test_ms_reduce_sum
from tests.operators.test_op.test_ms_exp import test_ms_exp
from tests.operators.test_op.test_ms_mul import test_ms_mul
from tests.operators.test_op.test_ms_sqrt import test_ms_sqrt
from tests.operators.test_op.test_ms_batch_matmul import test_ms_bmm
from tests.operators.test_op.test_ms_abs import test_ms_abs
from tests.operators.test_op.test_ms_conv import test_ms_conv
from tests.operators.test_op.test_fused_bn_double_follow_relu import test_fused_bn_double_follow_relu
from tests.operators.test_op.test_fused_bn_follow_relu_avgpool import test_fused_bn_follow_relu_avgpool
from tests.operators.test_op.test_fused_bn_follow_relu import test_fused_bn_follow_relu
from tests.operators.test_op.test_fused_bn_reduce_grad import test_fused_bn_reduce_grad
from tests.operators.test_op.test_fused_bn_reduce import test_fused_bn_reduce
from tests.operators.test_op.test_fused_bn_update_grad import test_fused_bn_update_grad
from tests.operators.test_op.test_fused_bn_update import test_fused_bn_update
from tests.operators.test_op.test_ms_addn import test_ms_addn
from tests.operators.test_op.test_fused_gather_gather_add_mul_max_exp_scatter_add import test_fused_gather_gather_add_mul_max_exp_scatter_add
from tests.operators.test_op.test_fused_gather_mul_scatter_add import test_fused_gather_mul_scatter_add
from tests.operators.test_op.test_fused_gather_nd_reduce_sum_mul_unsorted_segment_sum import test_fused_gather_nd_reduce_sum_mul_unsorted_segment_sum
from tests.operators.test_op.test_fused_is_finite import test_fused_is_finite
from tests.operators.test_op.test_fused_l2loss_grad import test_fused_l2loss_grad
from tests.operators.test_op.test_fused_mul_div_rsqrt_mul_isfinite_red import test_fused_mul_div_rsqrt_mul_isfinite_red
from tests.operators.test_op.test_fused_pad import test_fused_pad
from tests.operators.test_op.test_fused_relu_grad_bn_double_reduce_grad import test_fused_relu_grad_bn_double_reduce_grad
from tests.operators.test_op.test_fused_relu_grad_bn_double_update_grad import test_fused_relu_grad_bn_double_update_grad
from tests.operators.test_op.test_fused_relu_grad_bn_reduce_grad import test_fused_relu_grad_bn_reduce_grad
from tests.operators.test_op.test_fused_relu_grad_bn_update_grad import test_fused_relu_grad_bn_update_grad
from tests.operators.test_op.test_fused_relu_grad import test_fused_relu_grad
from tests.operators.test_op.test_ms_assign import test_ms_assign
from tests.operators.test_op.test_ms_cast import test_ms_cast
from tests.operators.test_op.test_ms_cumprod import test_ms_cumprod
from tests.operators.test_op.test_ms_cumsum import test_ms_cumsum
from tests.operators.test_op.test_ms_divide import test_ms_divide
from tests.operators.test_op.test_ms_equal import test_ms_equal
from tests.operators.test_op.test_ms_expand_dims import test_expand_dims
from tests.operators.test_op.test_ms_gather_nd import test_ms_gather_nd
from tests.operators.test_op.test_ms_gather import test_ms_gather
from tests.operators.test_op.test_ms_greater_equal import test_ms_greater_equal
from tests.operators.test_op.test_ms_less_equal import test_ms_less_equal
from tests.operators.test_op.test_ms_log import test_ms_log
from tests.operators.test_op.test_ms_maximum import test_ms_maximum
from tests.operators.test_op.test_ms_minimum import test_ms_minimum
from tests.operators.test_op.test_ms_neg import test_ms_neg
from tests.operators.test_op.test_ms_one_hot import test_ms_one_hot
from tests.operators.test_op.test_ms_pow import test_ms_pow
from tests.operators.test_op.test_ms_reciprocal import test_ms_reciprocal
from tests.operators.test_op.test_ms_reduce_and import test_ms_reduce_and
from tests.operators.test_op.test_ms_reduce_max import test_ms_reduce_max
from tests.operators.test_op.test_ms_reduce_min import test_ms_reduce_min
from tests.operators.test_op.test_ms_reduce_or import test_ms_reduce_or
from tests.operators.test_op.test_ms_reshape import test_ms_reshape
from tests.operators.test_op.test_ms_round import test_ms_round
from tests.operators.test_op.test_ms_rsqrt import test_ms_rsqrt
from tests.operators.test_op.test_ms_select import test_ms_select
from tests.operators.test_op.test_ms_sub import test_ms_sub
from tests.operators.test_op.test_ms_tensor_scatter_add import test_ms_tensor_scatter_add
from tests.operators.test_op.test_ms_tile import test_ms_tile
from tests.operators.test_op.test_ms_trans_data import test_ms_trans_data
from tests.operators.test_op.test_ms_unsorted_segment_sum import test_ms_unsorted_segment_sum
from tests.operators.test_op.test_ms_standard_normal import test_ms_standard_normal
from tests.operators.test_op.test_ms_reduce_prod import test_ms_reduce_prod

def add(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        test_ms_add(fuzz_shape, fuzz_shape, 'float32', poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_add((512, 1), (512, 1), 'float32', poly_sch=poly_sch, attrs=attrs)
    test_ms_add((1024, 2), (1024, 2), 'float32', poly_sch=poly_sch, attrs=attrs)
    test_ms_add((2, 1024), (2, 1024), 'float32', poly_sch=poly_sch, attrs=attrs)
    test_ms_add((1024, 1024), (1024, 1024), 'float32', poly_sch=poly_sch, attrs=attrs)
    test_ms_add((1024, 10240), (1024, 10240), 'float32', poly_sch=poly_sch, attrs=attrs)
    test_ms_add((1024, 1024, 10), (1024, 1024, 10), 'float32', poly_sch=poly_sch, attrs=attrs)

def mul(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        test_ms_mul(fuzz_shape, 'float32', poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_mul((512, 1), 'float32', poly_sch=poly_sch, attrs=attrs)
    test_ms_mul((1024, 2), 'float32', poly_sch=poly_sch, attrs=attrs)
    test_ms_mul((2, 1024), 'float32', poly_sch=poly_sch, attrs=attrs)
    test_ms_mul((1024, 1024), 'float32', poly_sch=poly_sch, attrs=attrs)
    test_ms_mul((1024, 10240), 'float32', poly_sch=poly_sch, attrs=attrs)
    test_ms_mul((1024, 1024, 10), 'float32', poly_sch=poly_sch, attrs=attrs)

def exp(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        test_ms_exp(fuzz_shape, 'float32', poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_exp((512, 1), 'float32', poly_sch=poly_sch, attrs=attrs)
    test_ms_exp((1024, 2), 'float32', poly_sch=poly_sch, attrs=attrs)
    test_ms_exp((2, 1024), 'float32', poly_sch=poly_sch, attrs=attrs)
    test_ms_exp((1024, 1024), 'float32', poly_sch=poly_sch, attrs=attrs)
    test_ms_exp((1024, 10240), 'float32', poly_sch=poly_sch, attrs=attrs)
    test_ms_exp((1024, 1024, 10), 'float32', poly_sch=poly_sch, attrs=attrs)

def sqrt(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_sqrt(input_shape, 'float32', poly_sch=poly_sch, attrs=attrs)
    test_ms_sqrt((512, 1), 'float32', poly_sch=poly_sch, attrs=attrs)
    test_ms_sqrt((1024, 2), 'float32', poly_sch=poly_sch, attrs=attrs)
    test_ms_sqrt((2, 1024), 'float32', poly_sch=poly_sch, attrs=attrs)
    test_ms_sqrt((1024, 1024), 'float32', poly_sch=poly_sch, attrs=attrs)
    test_ms_sqrt((1024, 10240), 'float32', poly_sch=poly_sch, attrs=attrs)
    test_ms_sqrt((1024, 1024, 10), 'float32', poly_sch=poly_sch, attrs=attrs)

def reduce_sum(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_reduce_sum(input_shape, 'float32', axis=1, keepdims=True,
                       poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_reduce_sum((9, 1024, 1024), 'float32', axis=1, keepdims=True,
                       poly_sch=poly_sch, attrs=attrs)
    test_ms_reduce_sum((9, 1024, 1024), 'float32', axis=2, keepdims=True,
                       poly_sch=poly_sch, attrs=attrs)
    test_ms_reduce_sum((9, 1024, 1024), 'float32', axis=None, keepdims=True,
                       poly_sch=poly_sch, attrs=attrs)
    test_ms_reduce_sum((10240,), 'float32', axis=None, keepdims=True,
                       poly_sch=poly_sch, attrs=attrs)

def bmm(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_bmm(input_shape, input_shape, 'float32', 'float32', shape_bias=(1, ),
                add_bias=False, tensor_core=False, poly_sch=poly_sch, attrs=attrs)
        return
    # Test for FP32 MatMul
    test_ms_bmm((128, 64), (128, 64), 'float32', 'float32', shape_bias=(1, ),
                add_bias=False, tensor_core=False, poly_sch=poly_sch, attrs=attrs)

    # test_ms_bmm((32, 12, 128, 128), (32, 12, 128, 64), 'float16', 'float16', layout1='NHDT', layout2='NHTD', layout_out='NHDT',
    #             shape_bias=(1, ), add_bias=False, tensor_core=False, poly_sch=poly_sch)
    # test_ms_bmm((256, 128), (64, 128), 'float16', 'float16', layout1='NHDT', layout2='NHDT', layout_out='NHDT',
    #             shape_bias=(1, ), add_bias=False, tensor_core=False, poly_sch=poly_sch)
    # test_ms_bmm((128, 32), (128, 512), 'float16', 'float16', layout1='NHTD', layout2='NHTD', layout_out='NHDT',
    #             shape_bias=(1, ), add_bias=False, tensor_core=False, poly_sch=poly_sch)
    # test_ms_bmm((128, 64), (64, 32), 'float16', 'float16', layout1='NHDT', layout2='NHTD', layout_out='NHDT',
    #             shape_bias=(1, ), add_bias=False, tensor_core=False, poly_sch=poly_sch)

def abs_op(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_abs(input_shape, 'float32', poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_abs((1024, 1024), "float32", poly_sch=poly_sch, attrs=attrs)
    test_ms_abs((1024, 1024), "float16", poly_sch=poly_sch, attrs=attrs)
    test_ms_abs((1, ), "float32", poly_sch=poly_sch, attrs=attrs)
    test_ms_abs((1, 1), "float32", poly_sch=poly_sch, attrs=attrs)
    test_ms_abs((1, ), "float16", poly_sch=poly_sch, attrs=attrs)
    test_ms_abs((1, 1), "float16", poly_sch=poly_sch, attrs=attrs)

def conv(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_abs(input_shape, input_shape, (1, 1), (0, 0, 0, 0), (1, 1), "float32", "float32",
                    tensor_core=False, poly_sch=poly_sch, attrs=attrs)
        return
    # Test for FP32 Conv2D
    test_ms_conv(shape_data=(32, 64, 56, 56), shape_weight=(64, 64, 3, 3), stride=(1, 1), padding=(1, 1, 1, 1),
            dilation=(1, 1), dtype="float32", out_dtype="float32", layout="NCHW", tensor_core=False,
            attrs=attrs)

    # Test for FP16 Conv2D with auto-tiling
    test_ms_conv((16, 4, 4, 16), (16, 3, 3, 16), (1, 1), (0, 0, 0, 0), (1, 1), "float16", "float16",
                 tensor_core=True, poly_sch=poly_sch, attrs=attrs)

    test_ms_conv((16, 16, 16, 16), (16, 3, 3, 16), (1, 1), (0, 0, 0, 0), (1, 1), "float16", "float16",
                 tensor_core=True, poly_sch=poly_sch, attrs=attrs)

    test_ms_conv((64, 6, 6, 64), (64, 3, 3, 64), (1, 1), (0, 0, 0, 0), (1, 1), "float16", "float16",
                 tensor_core=True, poly_sch=poly_sch, attrs=attrs)

    test_ms_conv((64, 6, 6, 64), (64, 3, 3, 64), (1, 1), (0, 0, 0, 0), (1, 1), "float16", "float32",
                 tensor_core=True, poly_sch=poly_sch, attrs=attrs)

def fused_bn_double_follow_relu(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_fused_bn_double_follow_relu(
            fuzz_shape, layout='NHWC', poly_sch=poly_sch, attrs=attrs)
        return
    test_fused_bn_double_follow_relu(
        (256, 7, 7, 2048), layout='NHWC', poly_sch=poly_sch, attrs=attrs)

def fused_bn_follow_relu_avgpool(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_fused_bn_follow_relu_avgpool(
            input_shape, layout='NHWC', poly_sch=poly_sch, attrs=attrs)
        return
    test_fused_bn_follow_relu_avgpool(
        (256, 7, 7, 2048), layout='NHWC', poly_sch=poly_sch, attrs=attrs)

def fused_bn_follow_relu(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_fused_bn_follow_relu(
            input_shape, layout='NHWC', poly_sch=poly_sch, attrs=attrs)
        return
    test_fused_bn_follow_relu(
        (256, 7, 7, 2048), layout='NHWC', poly_sch=poly_sch, attrs=attrs)

def fused_bn_reduce_grad(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_fused_bn_reduce_grad(
            input_shape, layout='NHWC', poly_sch=poly_sch, attrs=attrs)
        return
    test_fused_bn_reduce_grad(
        (32, 7, 7, 256), layout='NHWC', poly_sch=poly_sch, attrs=attrs)

def fused_bn_reduce(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_fused_bn_reduce(
            input_shape, layout='NHWC', poly_sch=poly_sch, attrs=attrs)
        return
    test_fused_bn_reduce((32, 7, 7, 256), layout='NHWC', poly_sch=poly_sch, attrs=attrs)

def fused_bn_update_grad(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_fused_bn_update_grad(
            input_shape, (256,), layout='NHWC', poly_sch=poly_sch, attrs=attrs)
        return
    test_fused_bn_update_grad(
        (32, 7, 7, 32), (32,), layout="NHWC", poly_sch=poly_sch, attrs=attrs)

def fused_bn_update(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_fused_bn_update(
            input_shape, poly_sch=poly_sch, attrs=attrs)
        return
    test_fused_bn_update((2048,), poly_sch=poly_sch, attrs=attrs)

def addn(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_fused_bn_update(
            input_shape, "float32", 2, poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_addn((1, 1024, 1024), "float32", 2, poly_sch=poly_sch, attrs=attrs)
    test_ms_addn((1, 1024, 1024), "float16", 2, poly_sch=poly_sch, attrs=attrs)

def fused_gather_gather_add_mul_max_exp_scatter_add(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_fused_gather_gather_add_mul_max_exp_scatter_add(input_shape, (108365, ), (1,), (108365, ),
                                            'float32', 'int32', 0, poly_sch=poly_sch, attrs=attrs)
        return
    test_fused_gather_gather_add_mul_max_exp_scatter_add((19717, 8, 1), (108365, ), (1,), (108365, ),
                                                'float32', 'int32', 0, poly_sch=poly_sch, attrs=attrs)

def fused_gather_mul_scatter_add(poly_sch, fuzz_shape, attrs):
    attrs2 = attrs
    attrs2.update({"dim": "0 0 16 16 0 1 8 8 0 2 8 8", "bind_block": "1 1 6773", "bind_thread": "8 8 16"})
    if fuzz_shape:
        input_shape = fuzz_shape
        test_fused_gather_mul_scatter_add(input_shape, (108365, ), (108365, 8, 8), (108365, 1), 'float32', 'int32', 0, poly_sch=poly_sch,
                    attrs=attrs2)
        return
    test_fused_gather_mul_scatter_add((19717, 8, 8), (108365, ), (108365, 8, 8), (108365, 1), 'float32', 'int32', 0, poly_sch=poly_sch,
        attrs=attrs2)

def fused_gather_nd_reduce_sum_mul_unsorted_segment_sum(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_fused_gather_nd_reduce_sum_mul_unsorted_segment_sum(
        input_shape, (108365, 1), (108365, 8, 1), (108365,), 'float32', 'int32', -1, True, 19717, poly_sch=poly_sch, attrs=attrs)
        return
    test_fused_gather_nd_reduce_sum_mul_unsorted_segment_sum(
        (19717, 8, 8), (108365, 1), (108365, 8, 1), (108365,), 'float32', 'int32', -1, True, 19717, poly_sch=poly_sch, attrs=attrs)

def fused_is_finite(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_fused_is_finite(input_shape, layout='NHWC', poly_sch=poly_sch, attrs=attrs)
        return
    test_fused_is_finite((1, 1, 256, 1024), layout='NHWC', poly_sch=poly_sch, attrs=attrs)

def fused_l2loss_grad(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_fused_l2loss_grad(input_shape, layout='NHWC', poly_sch=poly_sch, attrs=attrs)
        return
    test_fused_l2loss_grad((1, 1, 256, 1024), layout='NHWC', poly_sch=poly_sch, attrs=attrs)

def fused_mul_div_rsqrt_mul_isfinite_red(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_fused_mul_div_rsqrt_mul_isfinite_red(input_shape, poly_sch=poly_sch, attrs=attrs)
        return
    test_fused_mul_div_rsqrt_mul_isfinite_red((64,), poly_sch=poly_sch, attrs=attrs)

def fused_pad(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_fused_pad(input_shape, (0, 0, 0, 0), (0, 0, 1, 0),
                   layout='NHWC', pad_value=0.0, poly_sch=poly_sch, attrs=attrs)
        return
    test_fused_pad((7, 7, 3, 64), (0, 0, 0, 0), (0, 0, 1, 0),
                   layout='NHWC', pad_value=0.0, poly_sch=poly_sch, attrs=attrs)

def fused_relu_grad_bn_double_reduce_grad(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_fused_relu_grad_bn_double_reduce_grad(
        (32,), input_shape, layout="NHWC", poly_sch=poly_sch, attrs=attrs)
        return
    test_fused_relu_grad_bn_double_reduce_grad(
        (32,), (32, 7, 7, 32), layout="NHWC", poly_sch=poly_sch, attrs=attrs)

def fused_relu_grad_bn_double_update_grad(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_fused_relu_grad_bn_double_update_grad(
            input_shape, (32, ), layout='NHWC', poly_sch=poly_sch, attrs=attrs)
        return
    test_fused_relu_grad_bn_double_update_grad(
        (32, 7, 7, 32), (32, ), layout='NHWC', poly_sch=poly_sch, attrs=attrs)

def fused_relu_grad_bn_reduce_grad(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_fused_relu_grad_bn_reduce_grad(
            (64, ), input_shape,  layout='NHWC', poly_sch=poly_sch, attrs=attrs)
        return
    test_fused_relu_grad_bn_reduce_grad(
        (64, ), (256, 112, 112, 64),  layout='NHWC', poly_sch=poly_sch, attrs=attrs)

def fused_relu_grad_bn_update_grad(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_fused_relu_grad_bn_update_grad(
            input_shape, (64,), layout="NHWC", poly_sch=poly_sch, attrs=attrs)
        return
    test_fused_relu_grad_bn_update_grad(
        (32, 14, 14, 8), (8,), layout="NHWC", poly_sch=poly_sch, attrs=attrs)

def fused_relu_grad(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_fused_relu_grad(input_shape, poly_sch=poly_sch, attrs=attrs)
        return
    test_fused_relu_grad((32, 7, 7, 32), poly_sch=poly_sch, attrs=attrs)

def assign(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_assign('float32', input_shape, input_shape, poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_assign('float32', (16, 16), (16, 16), poly_sch=poly_sch, attrs=attrs)

def cast(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_cast(input_shape, "float16", "float32", poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_cast((32, 32, 14, 14, 16), "float16", "float32", poly_sch=poly_sch, attrs=attrs)
    test_ms_cast((32, 32, 14, 14, 16), "float32", "float16", poly_sch=poly_sch, attrs=attrs)


def cumprod(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_cumprod(input_shape, dtype="float32", axis=0, exclusive=False, reverse=False, poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_cumprod((16, 3, 3, 16), dtype="float32", axis=0, exclusive=False, reverse=False, poly_sch=poly_sch, attrs=attrs)

def cumsum(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_cumsum(input_shape, dtype="float32", axis=0, exclusive=False, reverse=False, poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_cumsum((16, 3, 3, 16), dtype="float32", axis=0, exclusive=False, reverse=False, poly_sch=poly_sch, attrs=attrs)

def divide(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_divide(input_shape, 'float32', poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_divide((1024, 1024), 'float32', poly_sch=poly_sch, attrs=attrs)
    test_ms_divide((1024, 1024), 'float16', poly_sch=poly_sch, attrs=attrs)

def equal(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_equal((input_shape, input_shape), 'float16', poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_equal(((1, 1024), (1, 1024)), 'float16', poly_sch=poly_sch, attrs=attrs)
    test_ms_equal(((1, 1024), (1, 1024)), 'float32', poly_sch=poly_sch, attrs=attrs)

def expand_dims(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_expand_dims(input_shape, 2, 'float16', poly_sch=poly_sch, attrs=attrs)
        return
    test_expand_dims((32, 1024, 1024), 1, 'float32', poly_sch=poly_sch, attrs=attrs)
    test_expand_dims((32, 1024, 1024), 2, 'float16', poly_sch=poly_sch, attrs=attrs)

def gather_nd(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_gather_nd(input_shape, 'float32', (108365, 1), 'int32', poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_gather_nd((19717, 1, 3), 'float32', (108365, 1), 'int32', poly_sch=poly_sch, attrs=attrs)

def gather(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_gather(input_shape, 'float32', (108365, ), 'int32', 0, poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_gather((19717, 8, 1), 'float32', (108365, ), 'int32', 0, poly_sch=poly_sch, attrs=attrs)

def greater_equal(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_greater_equal(input_shape, input_shape, 'float16', poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_greater_equal((1, 1024), (1, 1024), 'float16', poly_sch=poly_sch, attrs=attrs)
    test_ms_greater_equal((1, 1024), (1, 1024), 'float32', poly_sch=poly_sch, attrs=attrs)

def less_equal(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_less_equal(input_shape, input_shape, 'float16', poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_less_equal((1, 1024), (1, 1024), 'float16', poly_sch=poly_sch, attrs=attrs)
    test_ms_less_equal((1, 1024), (1, 1024), 'float32', poly_sch=poly_sch, attrs=attrs)

def log(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_log(input_shape, 'float16', poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_log((9, 1024, 1024), 'float16', poly_sch=poly_sch, attrs=attrs)
    test_ms_log((9, 1024, 1024), 'float32', poly_sch=poly_sch, attrs=attrs)

def maximum(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_maximum(input_shape, input_shape,
                    'float32', poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_maximum((32, 1024, 1024), (32, 1024, 1024),
                    'float32', poly_sch=poly_sch, attrs=attrs)
    test_ms_maximum((32, 1024, 1024), (1, 1024, 1024),
                    'float16', poly_sch=poly_sch, attrs=attrs)
    test_ms_maximum((32, 32, 32, 256), (32, 32, 1, 256),
                    'float16', poly_sch=poly_sch, attrs=attrs)

def minimum(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_minimum(input_shape, input_shape,
                    'float32', poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_minimum((32, 1024, 1024), (32, 1024, 1024),
                    'float32', poly_sch=poly_sch, attrs=attrs)
    test_ms_minimum((32, 1024, 1024), (1, 1024, 1024),
                    'float16', poly_sch=poly_sch, attrs=attrs)
    test_ms_minimum((32, 32, 32, 256), (32, 32, 1, 256),
                    'float16', poly_sch=poly_sch, attrs=attrs)

def neg(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_neg(input_shape, "float32", poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_neg((1024, 1024), "float32", poly_sch=poly_sch, attrs=attrs)
    test_ms_neg((1024, 1024), "float16", poly_sch=poly_sch, attrs=attrs)
    test_ms_neg((1, ), "float32", poly_sch=poly_sch, attrs=attrs)
    test_ms_neg((1, 1), "float32", poly_sch=poly_sch, attrs=attrs)
    test_ms_neg((1, ), "float16", poly_sch=poly_sch, attrs=attrs)
    test_ms_neg((1, 1), "float16", poly_sch=poly_sch, attrs=attrs)

def one_hot(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_one_hot(input_shape, 16, "int32", 1, 0, 0, poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_one_hot((1024,), 16, "int32", 1, 0, 0, poly_sch=poly_sch, attrs=attrs)
    test_ms_one_hot((1024,), 16, "float32", 1, 0, 0, poly_sch=poly_sch, attrs=attrs)
    test_ms_one_hot((32,), 16, "int32", 1, 0, 0, poly_sch=poly_sch, attrs=attrs)
    test_ms_one_hot((32,), 16, "float32", 1, 0, 0, poly_sch=poly_sch, attrs=attrs)

def pow_op(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_pow(input_shape, input_shape, 'float32', poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_pow((9, 1024, 1024), (9, 1024, 1024), 'float32', poly_sch=poly_sch, attrs=attrs)
    test_ms_pow((9, 1024, 1024), (9, 1024, 1), 'float32', poly_sch=poly_sch, attrs=attrs)
    test_ms_pow((9, 1024, 1024), (9, 1, 1), 'float32', poly_sch=poly_sch, attrs=attrs)
    test_ms_pow((9, 1024, 1024), (1, 1, 1), 'float32', poly_sch=poly_sch, attrs=attrs)
    test_ms_pow((9, 1024, 1024), (9, 1024, 1024), 'float16', poly_sch=poly_sch, attrs=attrs)
    test_ms_pow((9, 1024, 1024), (9, 1024, 1), 'float16', poly_sch=poly_sch, attrs=attrs)
    test_ms_pow((9, 1024, 1024), (9, 1, 1), 'float16', poly_sch=poly_sch, attrs=attrs)
    test_ms_pow((9, 1024, 1024), (1, 1, 1), 'float16', poly_sch=poly_sch, attrs=attrs)

def reciprocal(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_reciprocal(input_shape, 'float16', poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_reciprocal((1, 1024), 'float16', poly_sch=poly_sch, attrs=attrs)
    test_ms_reciprocal((1, 1024), 'float32', poly_sch=poly_sch, attrs=attrs)

def reduce_and(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_reduce_and(fuzz_shape, 'bool', axis=None,
                       keepdims=True, poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_reduce_and((32768,), 'bool', axis=None,
                       keepdims=True, poly_sch=poly_sch, attrs=attrs)
    test_ms_reduce_and((1024, 1024), 'bool', axis=1,
                       keepdims=True, poly_sch=poly_sch, attrs=attrs)

def reduce_max(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_reduce_max(input_shape, 'float32', axis=None,
                       keepdims=False, poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_reduce_max((9, 1024, 1024), 'float32', axis=None,
                       keepdims=False, poly_sch=poly_sch, attrs=attrs)
    test_ms_reduce_max((9, 1024, 1024), 'float16', axis=None,
                       keepdims=False, poly_sch=poly_sch, attrs=attrs)
    test_ms_reduce_max((9, 1024, 1024), 'float32', axis=2,
                       keepdims=False, poly_sch=poly_sch, attrs=attrs)
    test_ms_reduce_max((9, 1024, 1024), 'float16', axis=2,
                       keepdims=False, poly_sch=poly_sch, attrs=attrs)

def reduce_min(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_reduce_min(input_shape, 'float32', axis=None,
                       keepdims=False, poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_reduce_min((9, 1024, 1024), 'float32', axis=None,
                       keepdims=False, poly_sch=poly_sch, attrs=attrs)
    test_ms_reduce_min((9, 1024, 1024), 'float16', axis=None,
                       keepdims=False, poly_sch=poly_sch, attrs=attrs)
    test_ms_reduce_min((9, 1024, 1024), 'float32', axis=2,
                       keepdims=False, poly_sch=poly_sch, attrs=attrs)
    test_ms_reduce_min((9, 1024, 1024), 'float16', axis=2,
                       keepdims=False, poly_sch=poly_sch, attrs=attrs)

def reduce_or(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_reduce_or(input_shape, 'bool', axis=None,
                       keepdims=True, poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_reduce_or((32768,), 'bool', axis=None,
                       keepdims=True, poly_sch=poly_sch, attrs=attrs)
    test_ms_reduce_or((1024, 1024), 'bool', axis=1,
                       keepdims=True, poly_sch=poly_sch, attrs=attrs)

def reshape(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_reshape("float32", input_shape,
                    input_shape, poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_reshape("float32", (64, 128, 1024),
                    (8192, 1024), poly_sch=poly_sch, attrs=attrs)
    test_ms_reshape("float16", (64, 128, 1024),
                    (8192, 1024), poly_sch=poly_sch, attrs=attrs)

def round_op(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_round(input_shape, "float32", poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_round((1024, 1024), "float32", poly_sch=poly_sch, attrs=attrs)
    test_ms_round((1024, 1024), "float16", poly_sch=poly_sch, attrs=attrs)
    test_ms_round((1, ), "float32", poly_sch=poly_sch, attrs=attrs)
    test_ms_round((1, 1), "float32", poly_sch=poly_sch, attrs=attrs)
    test_ms_round((1, ), "float16", poly_sch=poly_sch, attrs=attrs)
    test_ms_round((1, 1), "float16", poly_sch=poly_sch, attrs=attrs)

def rsqrt(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_rsqrt(input_shape, 'float32', poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_rsqrt((32, 1024, 1024), 'float32', poly_sch=poly_sch, attrs=attrs)
    test_ms_rsqrt((32, 1024, 1024), 'float16', poly_sch=poly_sch, attrs=attrs)

def select(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_select((2, ), input_shape,  "int8", "float16", poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_select((2, ), (2, 2, 2),  "int8", "float16", poly_sch=poly_sch, attrs=attrs)

def sub(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_sub(input_shape, input_shape,
                'float32', poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_sub((32, 1024, 1024), (32, 1024, 1024),
                'float32', poly_sch=poly_sch, attrs=attrs)
    test_ms_sub((32, 1024, 1024), (32, 1024, 1024),
                'float16', poly_sch=poly_sch, attrs=attrs)
    test_ms_sub((32, 1024, 1024), (1, 1024, 1024),
                'float32', poly_sch=poly_sch, attrs=attrs)
    test_ms_sub((4, 4, 4), (1, 4, 4), 'float32', poly_sch=poly_sch, attrs=attrs)

def tensor_scatter_add(poly_sch, fuzz_shape, attrs):
    attrs.update({"dim": "0 0 8 8 0 1 128 128", "bind_block": "847 1", "bind_thread": "128 8"})
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_tensor_scatter_add(input_shape, 'float32', input_shape, 'int32', 0, poly_sch=poly_sch,
                                attrs=attrs)
        return
    test_ms_tensor_scatter_add((19717, 8, 1), 'float32', (108365, 1), 'int32', 0, poly_sch=poly_sch,
        attrs=attrs)

def tile(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_tile(input_shape, (3,), 'float32', poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_tile((1024, 4096), (3,), 'float32', poly_sch=poly_sch, attrs=attrs)
    test_ms_tile((1024, 4096), (3,), 'float16', poly_sch=poly_sch, attrs=attrs)

def trans_data(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_trans_data(input_shape, (0, 2, 1, 3),
                       'float32', poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_trans_data((8, 24, 38, 38), (0, 2, 1, 3),
                       'float32', poly_sch=poly_sch, attrs=attrs)
    test_ms_trans_data((8, 24, 38, 38), (0, 2, 1, 3),
                       'float16', poly_sch=poly_sch, attrs=attrs)

def unsorted_segment_sum(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_unsorted_segment_sum(input_shape, 'float32', (108365,), 'int32',
            19717, poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_unsorted_segment_sum((108365, 8, 1), 'float32', (108365,), 'int32',
            19717, poly_sch=poly_sch, attrs=attrs)

def standard_normal(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_standard_normal(1, input_shape, poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_standard_normal(1, (1987, 64), poly_sch=poly_sch, attrs=attrs)
    test_ms_standard_normal(2, (5025, 64, 3), poly_sch=poly_sch, attrs=attrs)

def reduce_prod(poly_sch, fuzz_shape, attrs):
    if fuzz_shape:
        input_shape = fuzz_shape
        test_ms_reduce_prod(input_shape, 'float32', axis=None,
                       keepdims=False, poly_sch=poly_sch, attrs=attrs)
        return
    test_ms_reduce_prod((32,), 'float32', axis=None,
                       keepdims=False, poly_sch=poly_sch, attrs=attrs)
    test_ms_reduce_prod((65536, 3), 'float32', axis=(1,),
                       keepdims=True, poly_sch=poly_sch, attrs=attrs)
    test_ms_reduce_prod((256, 32, 1024), 'float32', axis=(1,),
                       keepdims=False, poly_sch=poly_sch, attrs=attrs)


class Logger(object):
    def __init__(self, filename, stream):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def usage(op_maps):
    print("Usage:")
    print("1. Run all with auto schedule:")
    print("\t$python test_all.py all")
    print("2. Run one with auto schedule")
    print("\t$python test_all.py func_name")
    print("3. Run fuzz test of add op with maximal dimension of shape equals 3")
    print("\t$python test_all.py -f 3 add")
    print("Available target:")
    print("\tgpu, cpu, default is cpu")
    print("Available func:")
    print("\t", list(op_maps.keys()), "\n")


if __name__ == '__main__':
    import sys
    import getopt
    import traceback
    from datetime import datetime

    op_map = {"add": add, "mul": mul, "exp": exp, "sqrt": sqrt, "reduce_sum": reduce_sum, "bmm": bmm,
              "abs": abs_op, "conv": conv, "fused_bn_double_follow_relu": fused_bn_double_follow_relu,
              "fused_bn_follow_relu_avgpool": fused_bn_follow_relu_avgpool, "addn": addn, "assign": assign,
              "fused_bn_follow_relu": fused_bn_follow_relu, "fused_bn_reduce_grad": fused_bn_reduce_grad,
              "fused_bn_reduce": fused_bn_reduce, "fused_bn_update_grad": fused_bn_update_grad, "cast": cast,
              "fused_bn_update": fused_bn_update, "fused_gather_mul_scatter_add": fused_gather_mul_scatter_add,
              "fused_gather_gather_add_mul_max_exp_scatter_add": fused_gather_gather_add_mul_max_exp_scatter_add,
              "fused_gather_nd_reduce_sum_mul_unsorted_segment_sum": fused_gather_nd_reduce_sum_mul_unsorted_segment_sum,
              "fused_is_finite": fused_is_finite, "fused_l2loss_grad": fused_l2loss_grad, "fused_pad": fused_pad,
              "fused_mul_div_rsqrt_mul_isfinite_red": fused_mul_div_rsqrt_mul_isfinite_red,
              "fused_relu_grad_bn_double_reduce_grad": fused_relu_grad_bn_double_reduce_grad,
              "fused_relu_grad_bn_double_update_grad": fused_relu_grad_bn_double_update_grad,
              "fused_relu_grad_bn_reduce_grad": fused_relu_grad_bn_reduce_grad,
              "fused_relu_grad_bn_update_grad": fused_relu_grad_bn_update_grad,
              "fused_relu_grad": fused_relu_grad, "cumprod": cumprod,
              "cumsum": cumsum, "divide": divide, "equal": equal, "expand_dims": expand_dims,
              "gather_nd": gather_nd, "gather": gather, "greater_equal": greater_equal, "less_equal": less_equal,
              "log": log, "maximum": maximum, "minimum": minimum, "neg": neg, "one_hot": one_hot, "pow": pow_op,
              "reciprocal": reciprocal, "reduce_and": reduce_and, "reduce_max": reduce_max, "reduce_min": reduce_min,
              "reduce_or": reduce_or, "reshape": reshape, "round": round_op, "rsqrt": rsqrt, "select": select,
              "sub": sub, "tensor_scatter_add": tensor_scatter_add, "tile": tile, "trans_data": trans_data,
              "unsorted_segment_sum": unsorted_segment_sum, "standard_normal": standard_normal,
              "reduce_prod": reduce_prod}
    options, args = getopt.getopt(
        sys.argv[1:], "f:t:r:p", ["fuzz=", "target=", "mind-trick-string=", "mind-trick-file=",
        "profiling-repeat-time=", "--profiling"])
    default_attrs = {"profiling": False}
    fuzz_dim = 0
    for name, value in options:
        if name in ("-f", "--fuzz"):
            fuzz_dim = int(value)
        elif name == "--mind-trick-string":
            default_attrs['mind_trick'] = value
        elif name == "--mind-trick-file":
            with open(value, 'r') as f:
                default_attrs['mind_trick'] = f.read()
        elif name in ("-t", "--target"):
            if value == "gpu":
                default_attrs['target'] = "cuda"
            elif value == "cpu":
                default_attrs['target'] = "llvm"
            else:
                print("Invalid target name, available name:  gpu, cpu")
                sys.exit()
        elif name in ("-p", "--profiling"):
            default_attrs["profiling"] = True
        elif name in ("-r", "--profiling-repeat-time"):
            default_attrs["repeat_time"] = int(value)
    if 'target' not in default_attrs.keys():
        default_attrs['target'] = 'cuda'
    if 'repeat_time' not in default_attrs.keys():
        default_attrs['repeat_time'] = 1000

    def cpu_filter(item):
        op_filter = ["standard_normal", "conv"]
        return item[0] not in op_filter
    op_map["all"] = list((dict(filter(cpu_filter, op_map.items())) if default_attrs['target'] == "llvm" else op_map).values())
    if len(sys.argv) == 1:
        usage(op_map)
        sys.exit()

    fail_op_list = dict()
    run_op_list = list()
    for op in args:
        if op_map.get(op) is not None:
            f = op_map.get(op)
            if not isinstance(f, list):
                run_op_list.append(f)
            else:
                run_op_list += f

    now = datetime.now()
    logfile = "opstest_" + \
        '-'.join(list(map(str, [now.month, now.day,
                                now.hour, now.minute]))) + ".log"
    sys.stdout = Logger(logfile, sys.stdout)
    sys.stderr = Logger(logfile, sys.stderr)

    for op in run_op_list:
        print("Operater: ", op.__name__)
        fuzz_input_shape = gen_random_shape(fuzz_dim) if fuzz_dim > 0 else None
        if fuzz_input_shape:
            print("Fuzz shape: {}".format(fuzz_input_shape))
        try:
            print("Time of auto schedule:")
            op(poly_sch=True, fuzz_shape=fuzz_input_shape, attrs=default_attrs.copy())
        except(AssertionError, ValueError, TypeError):
            if op.__name__ in fail_op_list:
                fail_op_list[op.__name__].extend(
                    ["using auto schedule:", traceback.format_exc()])
            else:
                fail_op_list[op.__name__] = [
                    "using auto schedule:", traceback.format_exc()]

    if len(fail_op_list) == 0:
        print("All test pass!")
    else:
        for op, error_info in fail_op_list.items():
            print("Run op %s error" % op)
            for e in error_info:
                print(e)
