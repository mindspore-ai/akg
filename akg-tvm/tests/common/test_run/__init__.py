# Copyright 2021 Huawei Technologies Co., Ltd
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
from .abs_run import abs_run
from .add_run import add_run
from .addn_run import addn_run
from .assign_run import assign_run
from .batch_matmul_run import batch_matmul_run
from .cast_run import cast_run
from .conv_fusion_run import conv_fusion_run
from .conv_run import conv_run
from .coo2csr_run import coo2csr_run
from .csr_gather_run import csr_gather_run
from .csr2coo_run import csr2coo_run
from .csrmv_run import csrmv_run
from .csr_div_run import csr_div_run
from .csr_mul_run import csr_mul_run
from .csr_reduce_sum_run import csr_reduce_sum_run
from .cumprod_run import cumprod_run
from .csr_mm_run import csr_mm_run
from .cumsum_run import cumsum_run
from .div_run import div_run
from .equal_run import equal_run
from .exp_run import exp_run
from .expand_dims_run import expand_dims_run
from .fused_bn_double_follow_relu_run import fused_bn_double_follow_relu_run
from .fused_bn_follow_relu_avgpool_run import fused_bn_follow_relu_avgpool_run
from .fused_bn_follow_relu_run import fused_bn_follow_relu_run
from .fused_bn_reduce_grad_run import fused_bn_reduce_grad_run
from .fused_bn_reduce_run import fused_bn_reduce_run
from .fused_bn_update_grad_run import fused_bn_update_grad_run
from .fused_bn_update_run import fused_bn_update_run
from .fused_gather_gather_add_mul_max_exp_scatter_add_run import fused_gather_gather_add_mul_max_exp_scatter_add_run
from .fused_gather_mul_scatter_add_run import fused_gather_mul_scatter_add_run
from .fused_gather_nd_reduce_sum_mul_unsorted_segment_sum_run import fused_gather_nd_reduce_sum_mul_unsorted_segment_sum_run
from .fused_is_finite_run import fused_is_finite_run
from .fused_l2loss_grad_run import fused_l2loss_grad_run
from .fused_mul_div_rsqrt_mul_isfinite_red_run import fused_mul_div_rsqrt_mul_isfinite_red_run
from .fused_pad_run import fused_pad_run
from .fused_relu_grad_bn_double_reduce_grad_run import fused_relu_grad_bn_double_reduce_grad_run
from .fused_relu_grad_bn_double_update_grad_run import fused_relu_grad_bn_double_update_grad_run
from .fused_relu_grad_bn_reduce_grad_run import fused_relu_grad_bn_reduce_grad_run
from .fused_relu_grad_bn_update_grad_run import fused_relu_grad_bn_update_grad_run
from .fused_relu_grad_run import fused_relu_grad_run
from .gather_nd_run import gather_nd_run
from .gather_run import gather_run
from .greater_equal_run import greater_equal_run
from .less_equal_run import less_equal_run
from .log_run import log_run
from .maximum_run import maximum_run
from .minimum_run import minimum_run
from .mul_run import mul_run
from .neg_run import neg_run
from .one_hot_run import one_hot_run
from .pow_run import pow_run
from .reciprocal_run import reciprocal_run
from .reduce_all_run import reduce_all_run
from .reduce_and_run import reduce_and_run
from .reduce_max_run import reduce_max_run
from .reduce_min_run import reduce_min_run
from .reduce_or_run import reduce_or_run
from .reduce_prod_run import reduce_prod_run
from .reduce_sum_run import reduce_sum_run
from .reshape_run import reshape_run
from .round_run import round_run
from .rsqrt_run import rsqrt_run
from .select_run import select_run
from .sqrt_run import sqrt_run
from .standard_normal_run import standard_normal_run
from .sub_run import sub_run
from .tensor_scatter_add_run import tensor_scatter_add_run
from .tile_run import tile_run
from .transpose_run import transpose_run
from .unsorted_segment_max_run import unsorted_segment_max_run
from .unsorted_segment_sum_run import unsorted_segment_sum_run