#!/usr/bin/env python3
# coding: utf-8
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

"""__init__"""
from .op_build import op_build, op_build_to_func
from .message import compilewithjson
from .message import compilewithjsonname
from .ops_general import (tensor_add, add, add_n, assign, cast, equal, less_equal, mul, sub, div, divide, tile,
                          logical_or, logical_and, logical_not, not_equal, greater_equal, tensor_max, neg, log, less,
                          exp, tensor_sum, reshape, reciprocal, sqrt)
from .ops_ascend import (real_div, floor_div, argmax, simple_mean, relu, zeros_like, strided_slice,
                         sparse_softmax_cross_entropy_with_logits, softmax, relu_grad, reduce_mean, prod_force_sea,
                         prod_force_sea_grad, one_hot, simple_mean_grad, max_pool_with_argmax, mat_mul, conv_2d,
                         load_im2col, four2five, five2four, conv_bn1, gather_v2, lamb_apply_optimizer_assign, fused_bn1,
                         fused_bn2, fused_bn3, clear_zero, bias_add, bias_add_grad, batch_matmul, assign_add,
                         apply_momentum, equal_count, bn_grad1, bn_grad2, bn_grad3, fused_batch_norm,
                         fused_batch_norm_grad, fused_batch_norm_infer, conv_2d_backprop_input, conv_2d_backprop_filter)
from .ops_gpu import (squeeze, squeeze_grad, relu6, relu6_grad, h_swish, h_swish_grad, h_sigmoid, h_sigmoid_grad)
