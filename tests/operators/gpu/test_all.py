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

from tests.operators.gpu.test_ms_add import test_ms_add
from tests.operators.gpu.test_ms_addn import test_ms_addn
from tests.operators.gpu.test_ms_batch_matmul import test_ms_bmm
from tests.operators.gpu.test_ms_exp import test_ms_exp
from tests.operators.gpu.test_ms_maximum import test_ms_maximum
from tests.operators.gpu.test_ms_minimum import test_ms_minimum
from tests.operators.gpu.test_ms_mul import test_ms_mul
from tests.operators.gpu.test_ms_divide import test_ms_divide
from tests.operators.gpu.test_ms_rsqrt import test_ms_rsqrt
from tests.operators.gpu.test_ms_sub import test_ms_sub
from tests.operators.gpu.test_ms_tile import test_ms_tile
from tests.operators.gpu.test_ms_one_hot import test_ms_one_hot
from tests.operators.gpu.test_ms_sqrt import test_ms_sqrt
from tests.operators.gpu.test_ms_cast import test_ms_cast
from tests.operators.gpu.test_ms_reshape import test_ms_reshape
from tests.operators.gpu.test_ms_expand_dims import test_expand_dims
from tests.operators.gpu.test_ms_trans_data import test_ms_trans_data
from tests.operators.gpu.test_ms_log import test_ms_log
from tests.operators.gpu.test_ms_pow import test_ms_pow

from tests.operators.gpu.test_ms_abs import test_ms_abs
from tests.operators.gpu.test_ms_neg import test_ms_neg
from tests.operators.gpu.test_ms_round import test_ms_round
from tests.operators.gpu.test_ms_select import test_ms_select
from tests.operators.gpu.test_ms_equal import test_ms_equal
from tests.operators.gpu.test_ms_less_equal import test_ms_less_equal
from tests.operators.gpu.test_ms_greater_equal import test_ms_greater_equal
from tests.operators.gpu.test_ms_reciprocal import test_ms_reciprocal
from tests.operators.gpu.test_ms_reduce_sum import test_ms_reduce_sum
from tests.operators.gpu.test_ms_reduce_prod import test_ms_reduce_prod
from tests.operators.gpu.test_ms_reduce_max import test_ms_reduce_max
from tests.operators.gpu.test_ms_reduce_min import test_ms_reduce_min
from tests.operators.gpu.test_ms_reduce_and import test_ms_reduce_and
from tests.operators.gpu.test_ms_reduce_or import test_ms_reduce_or
from tests.operators.gpu.test_ms_cumsum import test_ms_cumsum
from tests.operators.gpu.test_ms_cumprod import test_ms_cumprod
from tests.operators.gpu.test_ms_conv import test_ms_conv
from tests.operators.gpu.test_ms_gather import test_ms_gather
from tests.operators.gpu.test_ms_gather_nd import test_ms_gather_nd
from tests.operators.gpu.test_ms_tensor_scatter_add import test_ms_tensor_scatter_add
from tests.operators.gpu.test_fused_gather_mul_scatter_add import test_fused_gather_mul_scatter_add
from tests.operators.gpu.test_ms_unsorted_segment_sum import test_ms_unsorted_segment_sum
from tests.operators.gpu.test_fused_gather_nd_reduce_sum_mul_unsorted_segment_sum import test_fused_gather_nd_reduce_sum_mul_unsorted_segment_sum
from tests.operators.gpu.test_ms_standard_normal import test_ms_standard_normal

from tests.operators.gpu.test_fused_pad import test_fused_pad
from tests.operators.gpu.test_fused_bn_reduce import test_fused_bn_reduce
from tests.operators.gpu.test_fused_bn_update import test_fused_bn_update
from tests.operators.gpu.test_fused_bn_follow_relu import test_fused_bn_follow_relu
from tests.operators.gpu.test_fused_bn_follow_relu_avgpool import test_fused_bn_follow_relu_avgpool
from tests.operators.gpu.test_fused_bn_double_follow_relu import test_fused_bn_double_follow_relu
from tests.operators.gpu.test_fused_bn_reduce_grad import test_fused_bn_reduce_grad
from tests.operators.gpu.test_fused_relu_grad_bn_reduce_grad import test_fused_relu_grad_bn_reduce_grad
from tests.operators.gpu.test_fused_relu_grad_bn_double_reduce_grad import test_fused_relu_grad_bn_double_reduce_grad
from tests.operators.gpu.test_fused_l2loss_grad import test_fused_l2loss_grad
from tests.operators.gpu.test_fused_is_finite import test_fused_is_finite
from tests.operators.gpu.test_fused_relu_grad_bn_update_grad import test_fused_relu_grad_bn_update_grad
from tests.operators.gpu.test_fused_relu_grad_bn_double_update_grad import test_fused_relu_grad_bn_double_update_grad
from tests.operators.gpu.test_fused_relu_grad import test_fused_relu_grad
from tests.operators.gpu.test_fused_bn_update_grad import test_fused_bn_update_grad
from tests.operators.gpu.test_fused_mul_div_rsqrt_mul_isfinite_red import test_fused_mul_div_rsqrt_mul_isfinite_red
from tests.operators.gpu.test_fused_gather_gather_add_mul_max_exp_scatter_add import test_fused_gather_gather_add_mul_max_exp_scatter_add

def add(poly_sch, fuzz_shape=None, mind_trick_str=''):
    if fuzz_shape:
        test_ms_add(fuzz_shape, fuzz_shape, 'float32', poly_sch=poly_sch)
        return

    test_ms_add((1, 1024), (1, 1024), 'float32', poly_sch=poly_sch)
    test_ms_add((2, 32, 256, 32, 32), (2, 32, 256, 32, 32),
                'float32', poly_sch=poly_sch)


def addn(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_addn((1, 1024, 1024), "float32", 2, poly_sch=poly_sch)
    test_ms_addn((1, 1024, 1024), "float16", 2, poly_sch=poly_sch)


def bmm(poly_sch, fuzz_shape=None, mind_trick_str=''):
    # Test for FP32 MatMul (Non-TensorCore)
    test_ms_bmm((768, 768), (768, 768), 'float32', 'float32', layout1='NHDT', layout2='NHDT', layout_out='NHDT',
                shape_bias=(1, ), add_bias=False, tensor_core=False, poly_sch=poly_sch)

    # Test for FP16 MatMul (Enable TensorCore)
    test_ms_bmm((32, 12, 128, 64), (32, 12, 128, 64), 'float16', 'float16', layout1='NHDT', layout2='NHDT', layout_out='NHDT',
                shape_bias=(1, ), add_bias=False, tensor_core=True, poly_sch=poly_sch)
    test_ms_bmm((32, 12, 128, 128), (32, 12, 128, 64), 'float16', 'float16', layout1='NHDT', layout2='NHTD', layout_out='NHDT',
                shape_bias=(1, ), add_bias=False, tensor_core=True, poly_sch=poly_sch)
    test_ms_bmm((256, 128), (64, 128), 'float16', 'float16', layout1='NHDT', layout2='NHDT', layout_out='NHDT',
                shape_bias=(1, ), add_bias=False, tensor_core=True, poly_sch=poly_sch)
    test_ms_bmm((128, 32), (128, 512), 'float16', 'float16', layout1='NHTD', layout2='NHTD', layout_out='NHDT',
                shape_bias=(1, ), add_bias=False, tensor_core=True, poly_sch=poly_sch)
    test_ms_bmm((128, 64), (64, 32), 'float16', 'float16', layout1='NHDT', layout2='NHTD', layout_out='NHDT',
                shape_bias=(1, ), add_bias=False, tensor_core=True, poly_sch=poly_sch)

def cast(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_cast((32, 32, 14, 14, 16), "float16", "float32", poly_sch=poly_sch)
    test_ms_cast((32, 32, 14, 14, 16), "float32", "float16", poly_sch=poly_sch)


def exp(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_exp((1024, 4096), 'float32', poly_sch=poly_sch)
    test_ms_exp((1024, 4096), 'float16', poly_sch=poly_sch)
    test_ms_exp((1024, 4095), 'float16', poly_sch=poly_sch)
    test_ms_exp((1024, 799), 'float16', poly_sch=poly_sch)


def maximum(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_maximum((32, 1024, 1024), (32, 1024, 1024),
                    'float32', poly_sch=poly_sch)
    test_ms_maximum((32, 1024, 1024), (1, 1024, 1024),
                    'float16', poly_sch=poly_sch)
    test_ms_maximum((32, 32, 32, 256), (32, 32, 1, 256),
                    'float16', poly_sch=poly_sch)


def minimum(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_minimum((32, 1024, 1024), (32, 1024, 1024),
                    'float32', poly_sch=poly_sch)
    test_ms_minimum((32, 1024, 1024), (1, 1024, 1024),
                    'float16', poly_sch=poly_sch)
    test_ms_minimum((32, 32, 32, 256), (32, 32, 1, 256),
                    'float16', poly_sch=poly_sch)


def mul(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_mul((1024, 4096), 'float32', poly_sch=poly_sch)


def divide(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_divide((1024, 1024), 'float32', poly_sch=poly_sch)
    test_ms_divide((1024, 1024), 'float16', poly_sch=poly_sch)


def reshape(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_reshape("float32", (64, 128, 1024),
                    (8192, 1024), poly_sch=poly_sch)
    test_ms_reshape("float16", (64, 128, 1024),
                    (8192, 1024), poly_sch=poly_sch)


def rsqrt(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_rsqrt((32, 1024, 1024), 'float32', poly_sch=poly_sch)
    test_ms_rsqrt((32, 1024, 1024), 'float16', poly_sch=poly_sch)


def sqrt(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_sqrt((1024, 1024), "float32", poly_sch=poly_sch)
    test_ms_sqrt((1024, 1024), "float16", poly_sch=poly_sch)


def sub(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_sub((32, 1024, 1024), (32, 1024, 1024),
                'float32', poly_sch=poly_sch)
    test_ms_sub((32, 1024, 1024), (32, 1024, 1024),
                'float16', poly_sch=poly_sch)
    test_ms_sub((32, 1024, 1024), (1, 1024, 1024),
                'float32', poly_sch=poly_sch)
    test_ms_sub((4, 4, 4), (1, 4, 4), 'float32', poly_sch=poly_sch)


def tile(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_tile((1024, 4096), (3,), 'float32', poly_sch=poly_sch)
    test_ms_tile((1024, 4096), (3,), 'float16', poly_sch=poly_sch)


def one_hot(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_one_hot((1024,), 16, "int32", 1, 0, 0, poly_sch=poly_sch)
    test_ms_one_hot((1024,), 16, "float32", 1, 0, 0, poly_sch=poly_sch)
    test_ms_one_hot((32,), 16, "int32", 1, 0, 0, poly_sch=poly_sch)
    test_ms_one_hot((32,), 16, "float32", 1, 0, 0, poly_sch=poly_sch)


def expand_dims(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_expand_dims((32, 1024, 1024), 1, 'float32', poly_sch=poly_sch)
    test_expand_dims((32, 1024, 1024), 2, 'float16', poly_sch=poly_sch)


def trans_data(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_trans_data((8, 24, 38, 38), (0, 2, 1, 3),
                       'float32', poly_sch=poly_sch)
    test_ms_trans_data((8, 24, 38, 38), (0, 2, 1, 3),
                       'float16', poly_sch=poly_sch)


def log(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_log((9, 1024, 1024), 'float16', poly_sch=poly_sch)
    test_ms_log((9, 1024, 1024), 'float32', poly_sch=poly_sch)


def pow(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_pow((9, 1024, 1024), (9, 1024, 1024), 'float32', poly_sch=poly_sch)
    test_ms_pow((9, 1024, 1024), (9, 1024, 1), 'float32', poly_sch=poly_sch)
    test_ms_pow((9, 1024, 1024), (9, 1, 1), 'float32', poly_sch=poly_sch)
    test_ms_pow((9, 1024, 1024), (1, 1, 1), 'float32', poly_sch=poly_sch)
    test_ms_pow((9, 1024, 1024), (9, 1024, 1024), 'float16', poly_sch=poly_sch)
    test_ms_pow((9, 1024, 1024), (9, 1024, 1), 'float16', poly_sch=poly_sch)
    test_ms_pow((9, 1024, 1024), (9, 1, 1), 'float16', poly_sch=poly_sch)
    test_ms_pow((9, 1024, 1024), (1, 1, 1), 'float16', poly_sch=poly_sch)


def abs(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_abs((1024, 1024), "float32", poly_sch=poly_sch)
    test_ms_abs((1024, 1024), "float16", poly_sch=poly_sch)
    test_ms_abs((1, ), "float32", poly_sch=poly_sch)
    test_ms_abs((1, 1), "float32", poly_sch=poly_sch)
    test_ms_abs((1, ), "float16", poly_sch=poly_sch)
    test_ms_abs((1, 1), "float16", poly_sch=poly_sch)


def neg(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_neg((1024, 1024), "float32", poly_sch=poly_sch)
    test_ms_neg((1024, 1024), "float16", poly_sch=poly_sch)
    test_ms_neg((1, ), "float32", poly_sch=poly_sch)
    test_ms_neg((1, 1), "float32", poly_sch=poly_sch)
    test_ms_neg((1, ), "float16", poly_sch=poly_sch)
    test_ms_neg((1, 1), "float16", poly_sch=poly_sch)


def round(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_round((1024, 1024), "float32", poly_sch=poly_sch)
    test_ms_round((1024, 1024), "float16", poly_sch=poly_sch)
    test_ms_round((1, ), "float32", poly_sch=poly_sch)
    test_ms_round((1, 1), "float32", poly_sch=poly_sch)
    test_ms_round((1, ), "float16", poly_sch=poly_sch)
    test_ms_round((1, 1), "float16", poly_sch=poly_sch)


def conv(poly_sch, fuzz_shape=None, mind_trick_str=''):
    # Test for FP32 Conv2D (Non-TensorCore)
    test_ms_conv(shape_data=(32, 64, 56, 56), shape_weight=(64, 64, 3, 3), stride=(1, 1), padding=(1, 1, 1, 1),
            dilation=(1, 1), dtype="float32", out_dtype="float32", layout="NCHW", tensor_core=False)

    # Test for FP16 Conv2D (TensorCore) with auto-tiling
    test_ms_conv((16, 4, 4, 16), (16, 3, 3, 16), (1, 1), (0, 0, 0, 0), (1, 1), "float16", "float16")

    test_ms_conv((16, 16, 16, 16), (16, 3, 3, 16), (1, 1), (0, 0, 0, 0), (1, 1), "float16", "float16")

    test_ms_conv((64, 6, 6, 64), (64, 3, 3, 64), (1, 1), (0, 0, 0, 0), (1, 1), "float16", "float16")

    test_ms_conv((64, 6, 6, 64), (64, 3, 3, 64), (1, 1), (0, 0, 0, 0), (1, 1), "float16", "float32")

def select(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_select((2, ), (2, 2, 2),  "int8", "float16", poly_sch=poly_sch)


def equal(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_equal(((1, 1024), (1, 1024)), 'float16', poly_sch=poly_sch)
    test_ms_equal(((1, 1024), (1, 1024)), 'float32', poly_sch=poly_sch)


def less_equal(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_less_equal((1, 1024), (1, 1024), 'float16', poly_sch=poly_sch)
    test_ms_less_equal((1, 1024), (1, 1024), 'float32', poly_sch=poly_sch)


def greater_equal(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_greater_equal((1, 1024), (1, 1024), 'float16', poly_sch=poly_sch)
    test_ms_greater_equal((1, 1024), (1, 1024), 'float32', poly_sch=poly_sch)


def reciprocal(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_reciprocal((1, 1024), 'float16', poly_sch=poly_sch)
    test_ms_reciprocal((1, 1024), 'float32', poly_sch=poly_sch)

def reduce_sum(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_reduce_sum((256, 256), 'float32', axis=(1,),
                       keepdims=True, poly_sch=poly_sch)
    test_ms_reduce_sum((9, 1024, 1024), 'float32', axis=None,
                       keepdims=False, poly_sch=poly_sch)
    test_ms_reduce_sum((9, 1024, 1024), 'float32', axis=2,
                       keepdims=True, poly_sch=poly_sch)
    test_ms_reduce_sum((9, 1024), 'float16', axis=None,
                       keepdims=False, poly_sch=poly_sch)
    test_ms_reduce_sum((9, 1024), 'float16', axis=1,
                       keepdims=True, poly_sch=poly_sch)

def reduce_prod(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_reduce_prod((32,), 'float32', axis=None,
                       keepdims=False, poly_sch=poly_sch)
    test_ms_reduce_prod((65536, 3), 'float32', axis=(1,),
                       keepdims=True, poly_sch=poly_sch)
    test_ms_reduce_prod((256, 32, 1024), 'float32', axis=(1,),
                       keepdims=False, poly_sch=poly_sch)



def reduce_min(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_reduce_min((9, 1024, 1024), 'float32', axis=None,
                       keepdims=False, poly_sch=poly_sch)
    test_ms_reduce_min((9, 1024, 1024), 'float16', axis=None,
                       keepdims=False, poly_sch=poly_sch)
    test_ms_reduce_min((9, 1024, 1024), 'float32', axis=2,
                       keepdims=False, poly_sch=poly_sch)
    test_ms_reduce_min((9, 1024, 1024), 'float16', axis=2,
                       keepdims=False, poly_sch=poly_sch)


def reduce_max(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_reduce_max((9, 1024, 1024), 'float32', axis=None,
                       keepdims=False, poly_sch=poly_sch)
    test_ms_reduce_max((9, 1024, 1024), 'float16', axis=None,
                       keepdims=False, poly_sch=poly_sch)
    test_ms_reduce_max((9, 1024, 1024), 'float32', axis=2,
                       keepdims=False, poly_sch=poly_sch)
    test_ms_reduce_max((9, 1024, 1024), 'float16', axis=2,
                       keepdims=False, poly_sch=poly_sch)


def reduce_and(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_reduce_and((32768,), 'bool', axis=None,
                       keepdims=True, poly_sch=poly_sch)
    test_ms_reduce_and((1024, 1024), 'bool', axis=1,
                       keepdims=True, poly_sch=poly_sch)


def reduce_or(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_reduce_or((32768,), 'bool', axis=None,
                       keepdims=True, poly_sch=poly_sch)
    test_ms_reduce_or((1024, 1024), 'bool', axis=1,
                       keepdims=True, poly_sch=poly_sch)


def standard_normal(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_standard_normal(1, (1987, 64), poly_sch=True)
    test_ms_standard_normal(2, (5025, 64, 3), poly_sch=True)


def gather(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_gather((19717, 8, 1), 'float32', (108365, ), 'int32', 0, poly_sch=True)

def gather_nd(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_gather_nd((19717, 1, 3), 'float32', (108365, 1), 'int32', poly_sch=True)

def tensor_scatter_add(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_tensor_scatter_add((19717, 8, 1), 'float32', (108365, 1), 'int32', 0, poly_sch=True)

def unsorted_segment_sum(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_ms_unsorted_segment_sum((108365, 8, 1), 'float32', (108365,), 'int32', 19717, poly_sch=True)

def fused_gather_mul_scatter_add(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_fused_gather_mul_scatter_add((19717, 8, 8), (108365, ), (108365, 8, 8), (108365, 1), 'float32', 'int32', 0, poly_sch=True)

def fused_gather_nd_reduce_sum_mul_unsorted_segment_sum(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_fused_gather_nd_reduce_sum_mul_unsorted_segment_sum(
        (19717, 8, 8), (108365, 1), (108365, 8, 1), (108365,), 'float32', 'int32', -1, True, 19717, poly_sch=True)

def fused_gather_gather_add_mul_max_exp_scatter_add(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_fused_gather_gather_add_mul_max_exp_scatter_add((19717, 8, 1), (108365, ), (1,), (108365, ),
                                                          'float32', 'int32', 0, poly_sch=True)

def fused_pad(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_fused_pad((7, 7, 3, 64), (0, 0, 0, 0), (0, 0, 1, 0),
                   layout='NHWC', pad_value=0.0, poly_sch=poly_sch)


def fused_bn_reduce(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_fused_bn_reduce((256, 7, 7, 2048), layout='NHWC', poly_sch=poly_sch)


def fused_bn_update(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_fused_bn_update((2048,), poly_sch=poly_sch, mind_trick=mind_trick_str)


def fused_bn_follow_relu(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_fused_bn_follow_relu(
        (256, 7, 7, 2048), layout='NHWC', poly_sch=poly_sch)


def fused_bn_follow_relu_avgpool(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_fused_bn_follow_relu_avgpool(
        (256, 7, 7, 2048), layout='NHWC', poly_sch=poly_sch)


def fused_bn_double_follow_relu(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_fused_bn_double_follow_relu(
        (256, 7, 7, 2048), layout='NHWC', poly_sch=poly_sch)


def fused_bn_reduce_grad(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_fused_bn_reduce_grad(
        (256, 56, 56, 256), layout='NHWC', poly_sch=poly_sch)


def fused_relu_grad_bn_reduce_grad(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_fused_relu_grad_bn_reduce_grad(
        (64, ), (256, 112, 112, 64),  layout='NHWC', poly_sch=poly_sch)


def fused_relu_grad_bn_double_reduce_grad(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_fused_relu_grad_bn_double_reduce_grad(
        (256,), (256, 56, 56, 256), layout="NHWC", poly_sch=poly_sch)


def fused_l2loss_grad(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_fused_l2loss_grad((1, 1, 256, 1024), layout='NHWC', poly_sch=poly_sch, mind_trick=mind_trick_str)


def fused_is_finite(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_fused_is_finite((1, 1, 256, 1024), layout='NHWC', poly_sch=poly_sch)


def fused_relu_grad_bn_update_grad(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_fused_relu_grad_bn_update_grad(
        (256, 112, 112, 64), (64,), layout="NHWC", poly_sch=poly_sch)


def fused_relu_grad_bn_double_update_grad(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_fused_relu_grad_bn_double_update_grad(
        (256, 56, 56, 256), (256, ), layout='NHWC', poly_sch=poly_sch)


def fused_relu_grad(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_fused_relu_grad((256, 56, 56, 256), poly_sch=poly_sch, mind_trick=mind_trick_str)


def fused_bn_update_grad(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_fused_bn_update_grad(
        (256, 56, 56, 256), (256,), layout="NHWC", poly_sch=poly_sch)


def fused_mul_div_rsqrt_mul_isfinite_red(poly_sch, fuzz_shape=None, mind_trick_str=''):
    test_fused_mul_div_rsqrt_mul_isfinite_red((64,), poly_sch=poly_sch)


class Logger(object):
    def __init__(self, filename, stream):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def usage(op_map):
    print("Usage:")
    print("1. Run func1 and func2 with manual schedule:")
    print("\t$python test_all.py -m func_name1 func_name2")
    print("\t$python test_all.py --manual func_name1 func_name2")
    print("2. Run all with auto schedule:")
    print("\t$python test_all.py -a all\n")
    print("\t$python test_all.py --auto all\n")
    print("3. Both schedule methods will be tested if no option is specified")
    print("\t$python test_all.py func_name")
    print("4. Run fuzz test of add op with maximal dimension of shape equals 3")
    print("\t$python test_all.py -f 3 add")
    print("Available func:")
    print("\t", list(op_map.keys()), "\n")


if __name__ == '__main__':
    import sys
    import getopt
    import traceback
    from datetime import datetime

    op_map = {"abs": abs, "add": add, "addn": addn, "bmm": bmm, "cast": cast, "divide": divide,
              "equal": equal, "exp": exp, "greater_equal": greater_equal, "less_equal": less_equal,
              "log": log, "max": maximum, "min": minimum, "mul": mul, "neg": neg, "pow": pow,
              "reciprocal": reciprocal, "round": round, "rsqrt": rsqrt, "select": select, "sqrt": sqrt,
              "sub": sub, "reduce_max": reduce_max, "reduce_min": reduce_min, "reduce_and":reduce_and,
              "reduce_or":reduce_or, "reduce_sum": reduce_sum, "reduce_prod":reduce_prod, "expand_dims": expand_dims, "one_hot": one_hot,
              "reshape": reshape, "tile": tile, "trans_data": trans_data,
              "conv": conv, "gather":gather, "gather_nd":gather_nd,
              "tensor_scatter_add":tensor_scatter_add,
              "unsorted_segment_sum": unsorted_segment_sum,
              "fused_gather_mul_scatter_add":fused_gather_mul_scatter_add,
              "fused_gather_nd_reduce_sum_mul_unsorted_segment_sum": fused_gather_nd_reduce_sum_mul_unsorted_segment_sum,
              "fused_pad": fused_pad,
              "fused_bn_reduce": fused_bn_reduce,
              "fused_bn_update": fused_bn_update,
              "fused_bn_follow_relu": fused_bn_follow_relu,
              "fused_bn_follow_relu_avgpool": fused_bn_follow_relu_avgpool,
              "fused_bn_double_follow_relu": fused_bn_double_follow_relu,
              "fused_bn_reduce_grad": fused_bn_reduce_grad,
              "fused_relu_grad_bn_reduce_grad": fused_relu_grad_bn_reduce_grad,
              "fused_relu_grad_bn_double_reduce_grad": fused_relu_grad_bn_double_reduce_grad,
              "fused_l2loss_grad": fused_l2loss_grad,
              "fused_is_finite": fused_is_finite,
              "fused_relu_grad_bn_update_grad": fused_relu_grad_bn_update_grad,
              "fused_relu_grad_bn_double_update_grad": fused_relu_grad_bn_double_update_grad,
              "fused_relu_grad": fused_relu_grad,
              "fused_bn_update_grad": fused_bn_update_grad,
              "fused_mul_div_rsqrt_mul_isfinite_red": fused_mul_div_rsqrt_mul_isfinite_red,
              "fused_gather_mul_scatter_add": fused_gather_mul_scatter_add,
              "fused_gather_gather_add_mul_max_exp_scatter_add": fused_gather_gather_add_mul_max_exp_scatter_add,
              "standard_normal": standard_normal,
              }
    all_f = list(op_map.values())
    op_map["all"] = all_f
    if len(sys.argv) == 1:
        usage(op_map)
        sys.exit()

    options, args = getopt.getopt(
        sys.argv[1:], "f:", ["fuzz=", "mind-trick-string=", "mind-trick-file="])
    mind_trick_str = ''
    fuzz_dim = 0
    for name, value in options:
        if name in ("-f", "--fuzz"):
            fuzz_dim = int(value)
        if name == "--mind-trick-string":
            mind_trick_str = value
        if name == "--mind-trick-file":
            with open(value, 'r') as f:
                mind_trick_str = f.read()

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
    filename = "opstest_" + \
        '-'.join(list(map(str, [now.month, now.day,
                                now.hour, now.minute]))) + ".log"
    sys.stdout = Logger(filename, sys.stdout)
    sys.stderr = Logger(filename, sys.stderr)

    for op in run_op_list:
        print("Operater: ", op.__name__)
        fuzz_shape = gen_random_shape(fuzz_dim) if fuzz_dim > 0 else None
        if fuzz_shape:
            print("Fuzz shape: {}".format(fuzz_shape))
        try:
            print("Time of auto schedule:")
            op(poly_sch=True, fuzz_shape=fuzz_shape, mind_trick_str=mind_trick_str)
        except:
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
