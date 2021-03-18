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

import pytest
from tests.st.ops.gpu.test_ms_add import test_ms_add
from tests.st.ops.gpu.test_ms_addn import test_ms_addn
from tests.st.ops.gpu.test_ms_batch_matmul import test_ms_bmm
from tests.st.ops.gpu.test_ms_exp import test_ms_exp
from tests.st.ops.gpu.test_ms_maximum import test_ms_maximum
from tests.st.ops.gpu.test_ms_minimum import test_ms_minimum
from tests.st.ops.gpu.test_ms_mul import test_ms_mul
from tests.st.ops.gpu.test_ms_divide import test_ms_divide
from tests.st.ops.gpu.test_ms_rsqrt import test_ms_rsqrt
from tests.st.ops.gpu.test_ms_sub import test_ms_sub
from tests.st.ops.gpu.test_ms_tile import test_ms_tile
from tests.st.ops.gpu.test_ms_one_hot import test_ms_one_hot
from tests.st.ops.gpu.test_ms_sqrt import test_ms_sqrt
from tests.st.ops.gpu.test_ms_cast import test_ms_cast
from tests.st.ops.gpu.test_ms_reshape import test_ms_reshape
from tests.st.ops.gpu.test_ms_expand_dims import test_expand_dims
from tests.st.ops.gpu.test_ms_trans_data import test_ms_trans_data
from tests.st.ops.gpu.test_ms_log import test_ms_log
from tests.st.ops.gpu.test_ms_pow import test_ms_pow
from tests.st.ops.gpu.test_ms_reduce_sum import test_ms_reduce_sum
from tests.st.ops.gpu.test_ms_abs import test_ms_abs
from tests.st.ops.gpu.test_ms_neg import test_ms_neg
from tests.st.ops.gpu.test_ms_round import test_ms_round
from tests.st.ops.gpu.test_ms_select import test_ms_select
from tests.st.ops.gpu.test_ms_equal import test_ms_equal
from tests.st.ops.gpu.test_ms_less_equal import test_ms_less_equal
from tests.st.ops.gpu.test_ms_greater_equal import test_ms_greater_equal
from tests.st.ops.gpu.test_ms_reciprocal import test_ms_reciprocal
from tests.st.ops.gpu.test_ms_reduce_max import test_ms_reduce_max
from tests.st.ops.gpu.test_ms_reduce_min import test_ms_reduce_min
from tests.st.ops.gpu.test_fused_pad import test_fused_pad
from tests.st.ops.gpu.test_fused_bn_reduce import test_fused_bn_reduce
from tests.st.ops.gpu.test_fused_bn_update import test_fused_bn_update
from tests.st.ops.gpu.test_fused_bn_follow_relu import test_fused_bn_follow_relu
from tests.st.ops.gpu.test_fused_bn_follow_relu_avgpool import test_fused_bn_follow_relu_avgpool
from tests.st.ops.gpu.test_fused_bn_double_follow_relu import test_fused_bn_double_follow_relu
from tests.st.ops.gpu.test_fused_bn_reduce_grad import test_fused_bn_reduce_grad
from tests.st.ops.gpu.test_fused_relu_grad_bn_reduce_grad import test_fused_relu_grad_bn_reduce_grad
from tests.st.ops.gpu.test_fused_relu_grad_bn_double_reduce_grad import test_fused_relu_grad_bn_double_reduce_grad
from tests.st.ops.gpu.test_fused_l2loss_grad import test_fused_l2loss_grad
from tests.st.ops.gpu.test_fused_is_finite import test_fused_is_finite
from tests.st.ops.gpu.test_fused_relu_grad_bn_update_grad import test_fused_relu_grad_bn_update_grad
from tests.st.ops.gpu.test_fused_relu_grad_bn_double_update_grad import test_fused_relu_grad_bn_double_update_grad
from tests.st.ops.gpu.test_fused_relu_grad import test_fused_relu_grad
from tests.st.ops.gpu.test_fused_bn_update_grad import test_fused_bn_update_grad
from tests.st.ops.gpu.test_fused_mul_div_rsqrt_mul_isfinite_red import test_fused_mul_div_rsqrt_mul_isfinite_red
from tests.st.ops.gpu.test_ms_composite_stitch import test_composite_stitch


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_add():
    test_ms_add((1, 1024), (1, 1024), 'float32', poly_sch=True)
    test_ms_add((2, 32, 256, 32, 32), (2, 32, 256, 32, 32),
                'float32', poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_addn():
    test_ms_addn((1, 1024, 1024), "float32", 2, poly_sch=True)
    test_ms_addn((1, 1024, 1024), "float16", 2, poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bmm():
    test_ms_bmm((8, 16, 32), (8, 64, 32), 'float32', poly_sch=True)  # matmul with batch
    test_ms_bmm((1, 2, 32), (1, 1, 32), 'float32', poly_sch=True)  # matmul with some axis equals to 1
    test_ms_bmm((1, 1024, 1024), (1, 1024, 1024), 'float32', poly_sch=True)
    # test_ms_bmm((1, 1024, 1024), (1, 1024, 1024), 'float32', (1, 1024, 1024))  # cannot store type float32
    test_ms_bmm((1, 1024, 512), (1, 256, 512), 'float32', poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cast():
    test_ms_cast((32, 32, 14, 14, 16), "float16", "float32", poly_sch=True)
    test_ms_cast((32, 32, 14, 14, 16), "float32", "float16", poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_exp():
    test_ms_exp((1024, 4096), 'float32', poly_sch=True)
    test_ms_exp((1024, 4096), 'float16', poly_sch=True)
    test_ms_exp((1024, 4095), 'float16', poly_sch=True)
    test_ms_exp((1024, 799), 'float16', poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_maximum():
    test_ms_maximum((32, 1024, 1024), (32, 1024, 1024), 'float32', poly_sch=True)
    test_ms_maximum((32, 1024, 1024), (1, 1024, 1024), 'float16', poly_sch=True)
    test_ms_maximum((32, 32, 32, 256), (32, 32, 1, 256), 'float16', poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_minimum():
    test_ms_minimum((32, 1024, 1024), (32, 1024, 1024), 'float32', poly_sch=True)
    test_ms_minimum((32, 1024, 1024), (1, 1024, 1024), 'float16', poly_sch=True)
    test_ms_minimum((32, 32, 32, 256), (32, 32, 1, 256), 'float16', poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_mul():
    test_ms_mul((1024, 4096), 'float32', poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_divide():
    test_ms_divide((1024, 1024), 'float32', poly_sch=True)
    test_ms_divide((1024, 1024), 'float16', poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_reshape():
    test_ms_reshape("float32", (64, 128, 1024), (8192, 1024), poly_sch=True)
    test_ms_reshape("float16", (64, 128, 1024), (8192, 1024), poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_rsqrt():
    test_ms_rsqrt((32, 1024, 1024), 'float32', poly_sch=True)
    test_ms_rsqrt((32, 1024, 1024), 'float16', poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sqrt():
    test_ms_sqrt((1024, 1024), "float32", poly_sch=True)
    test_ms_sqrt((1024, 1024), "float16", poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sub():
    test_ms_sub((32, 1024, 1024), (32, 1024, 1024), 'float32', poly_sch=True)
    test_ms_sub((32, 1024, 1024), (32, 1024, 1024), 'float16', poly_sch=True)
    test_ms_sub((32, 1024, 1024), (1, 1024, 1024), 'float32', poly_sch=True)
    test_ms_sub((4, 4, 4), (1, 4, 4), 'float32', poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tile():
    test_ms_tile((1024, 4096), (3,), 'float32', poly_sch=True)
    test_ms_tile((1024, 4096), (3,), 'float16', poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_one_hot():
    test_ms_one_hot((1024,), 16, "int32", 1, 0, 0, poly_sch=True)
    test_ms_one_hot((1024,), 16, "float32", 1, 0, 0, poly_sch=True)
    test_ms_one_hot((32,), 16, "int32", 1, 0, 0, poly_sch=True)
    test_ms_one_hot((32,), 16, "float32", 1, 0, 0, poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ms_expand_dims():
    test_expand_dims((32, 1024, 1024), 1, 'float32', poly_sch=True)
    test_expand_dims((32, 1024, 1024), 2, 'float16', poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_trans_data():
    test_ms_trans_data((8, 24, 38, 38), (0, 2, 1, 3), 'float32', poly_sch=True)
    test_ms_trans_data((8, 24, 38, 38), (0, 2, 1, 3), 'float16', poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_log():
    test_ms_log((9, 1024, 1024), 'float16', poly_sch=True)
    test_ms_log((9, 1024, 1024), 'float32', poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pow():
    test_ms_pow((9, 1024, 1024), (9, 1024, 1024), 'float32', poly_sch=True)
    test_ms_pow((9, 1024, 1024), (9, 1024, 1), 'float32', poly_sch=True)
    test_ms_pow((9, 1024, 1024), (9, 1, 1), 'float32', poly_sch=True)
    test_ms_pow((9, 1024, 1024), (1, 1, 1), 'float32', poly_sch=True)
    test_ms_pow((9, 1024, 1024), (9, 1024, 1024), 'float16', poly_sch=True)
    test_ms_pow((9, 1024, 1024), (9, 1024, 1), 'float16', poly_sch=True)
    test_ms_pow((9, 1024, 1024), (9, 1, 1), 'float16', poly_sch=True)
    test_ms_pow((9, 1024, 1024), (1, 1, 1), 'float16', poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_abs():
    test_ms_abs((1024, 1024), "float32", poly_sch=True)
    test_ms_abs((1024, 1024), "float16", poly_sch=True)
    test_ms_abs((1,), "float32", poly_sch=True)
    test_ms_abs((1, 1), "float32", poly_sch=True)
    test_ms_abs((1,), "float16", poly_sch=True)
    test_ms_abs((1, 1), "float16", poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_neg():
    test_ms_neg((1024, 1024), "float32", poly_sch=True)
    test_ms_neg((1024, 1024), "float16", poly_sch=True)
    test_ms_neg((1,), "float32", poly_sch=True)
    test_ms_neg((1, 1), "float32", poly_sch=True)
    test_ms_neg((1,), "float16", poly_sch=True)
    test_ms_neg((1, 1), "float16", poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_round():
    test_ms_round((1024, 1024), "float32", poly_sch=True)
    test_ms_round((1024, 1024), "float16", poly_sch=True)
    test_ms_round((1,), "float32", poly_sch=True)
    test_ms_round((1, 1), "float32", poly_sch=True)
    test_ms_round((1,), "float16", poly_sch=True)
    test_ms_round((1, 1), "float16", poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_reduce_sum():
    test_ms_reduce_sum((9, 1024, 1024), 'float32', axis=None, keepdims=False, poly_sch=True)
    test_ms_reduce_sum((9, 1024, 1024), 'float32', axis=2, keepdims=True, poly_sch=True)
    test_ms_reduce_sum((9, 1024), 'float16', axis=None, keepdims=False, poly_sch=True)
    test_ms_reduce_sum((9, 1024), 'float16', axis=1, keepdims=True, poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_select():
    test_ms_select((2,), (2, 2, 2), "int8", "float16", poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_equal():
    test_ms_equal(((1, 1024), (1, 1024)), 'float16', poly_sch=True)
    test_ms_equal(((1, 1024), (1, 1024)), 'float32', poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_less_equal():
    test_ms_less_equal((1, 1024), (1, 1024), 'float16', poly_sch=True)
    test_ms_less_equal((1, 1024), (1, 1024), 'float32', poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_greater_equal():
    test_ms_greater_equal((1, 1024), (1, 1024), 'float16', poly_sch=True)
    test_ms_greater_equal((1, 1024), (1, 1024), 'float32', poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_reciprocal():
    test_ms_reciprocal((1, 1024), 'float16', poly_sch=True)
    test_ms_reciprocal((1, 1024), 'float32', poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_reduce_min():
    test_ms_reduce_min((9, 1024, 1024), 'float32', axis=None, keepdims=False, poly_sch=True)
    test_ms_reduce_min((9, 1024, 1024), 'float16', axis=None, keepdims=False, poly_sch=True)
    test_ms_reduce_min((9, 1024, 1024), 'float32', axis=2, keepdims=False, poly_sch=True)
    test_ms_reduce_min((9, 1024, 1024), 'float16', axis=2, keepdims=False, poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_reduce_max():
    test_ms_reduce_max((9, 1024, 1024), 'float32', axis=None, keepdims=False, poly_sch=True)
    test_ms_reduce_max((9, 1024, 1024), 'float16', axis=None, keepdims=False, poly_sch=True)
    test_ms_reduce_max((9, 1024, 1024), 'float32', axis=2, keepdims=False, poly_sch=True)
    test_ms_reduce_max((9, 1024, 1024), 'float16', axis=2, keepdims=False, poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ms_fused_pad():
    test_fused_pad((7, 7, 3, 64), (0, 0, 0, 0), (0, 0, 1, 0), layout='NHWC', pad_value=0.0, poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ms_fused_bn_reduce():
    test_fused_bn_reduce((256, 7, 7, 2048), layout='NHWC', poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ms_fused_bn_update():
    test_fused_bn_update((2048,), poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ms_fused_bn_follow_relu():
    test_fused_bn_follow_relu((256, 7, 7, 2048), layout='NHWC', poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ms_fused_bn_follow_relu_avgpool():
    test_fused_bn_follow_relu_avgpool((256, 7, 7, 2048), layout='NHWC', poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ms_fused_bn_double_follow_relu():
    test_fused_bn_double_follow_relu((256, 7, 7, 2048), layout='NHWC', poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ms_fused_bn_reduce_grad():
    test_fused_bn_reduce_grad((256, 56, 56, 256), layout='NHWC', poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ms_fused_relu_grad_bn_reduce_grad():
    test_fused_relu_grad_bn_reduce_grad((64,), (256, 112, 112, 64), layout='NHWC', poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ms_fused_relu_grad_bn_double_reduce_grad():
    test_fused_relu_grad_bn_double_reduce_grad((256,), (256, 56, 56, 256), layout="NHWC", poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ms_fused_l2loss_grad():
    test_fused_l2loss_grad((1, 1, 256, 1024), layout='NHWC', poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ms_fused_is_finite():
    test_fused_is_finite((1, 1, 256, 1024), layout='NHWC', poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ms_fused_relu_grad_bn_update_grad():
    test_fused_relu_grad_bn_update_grad((256, 112, 112, 64), (64,), layout="NHWC", poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ms_fused_relu_grad_bn_double_update_grad():
    test_fused_relu_grad_bn_double_update_grad((256, 56, 56, 256), (256,), layout='NHWC', poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ms_fused_relu_grad():
    test_fused_relu_grad((256, 56, 56, 256), poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ms_fused_bn_update_grad():
    test_fused_bn_update_grad((256, 56, 56, 256), (256,), layout="NHWC", poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ms_fused_mul_div_rsqrt_mul_isfinite_red():
    test_fused_mul_div_rsqrt_mul_isfinite_red((64,), poly_sch=True)
    return True


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ms_composite_buffer_stitch():
    ci_path = "./stitch_cases/"
    test_composite_stitch(ci_path)
    return True


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
