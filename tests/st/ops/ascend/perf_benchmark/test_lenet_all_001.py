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

import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from base_all_run import BaseCaseRun
from tests.common.test_run.conv_run import conv_run
from tests.common.test_run.conv_backprop_input_run import conv_backprop_input_run
from tests.common.test_run.conv_backprop_filter_run import conv_backprop_filter_run
from tests.common.test_run.fused_batch_norm_run import fused_batch_norm_run
from tests.common.test_run.fused_batch_norm_grad_run import fused_batch_norm_grad_run
from tests.common.test_run.batch_norm_ad_run import batch_norm_ad_run
from tests.common.test_run.batchmatmul_run import batchmatmul_execute
from tests.common.test_run.maxpool_with_argmax_run import maxpool_with_argmax_run
from tests.common.test_run.mean_run import mean_execute
from tests.common.test_run.mean_ad_run import mean_ad_run
from tests.common.test_run.relu_run import relu_run
from tests.common.test_run.relu_grad_run import relu_grad_run
from tests.common.test_run.relu_ad_run import relu_ad_run
from tests.common.test_run.add_run import add_run
from tests.common.test_run.addn_run import addn_execute
from tests.common.test_run.sparse_softmax_cross_entropy_with_logits_run import sparse_softmax_cross_entropy_with_logits_run
from tests.common.test_run.sparse_softmax_cross_entropy_with_logits_ad_run import sparse_softmax_cross_entropy_with_logits_ad_run
from tests.common.test_run.bias_add_ad_run import bias_add_ad_run
from tests.common.test_run.reshape_run import reshape_execute
from tests.common.test_run.apply_momentum_run import apply_momentum_run
from tests.common.test_run.cast_run import cast_run
from tests.common.test_run.conv_bn1_run import conv_bn1_run
from tests.common.test_run.relu_ad_run import relu_ad_run
from tests.common.test_run.conv_input_ad_run import conv_input_ad_run
from tests.common.test_run.conv_filter_ad_run import conv_filter_ad_run


class TestLenet(BaseCaseRun):
    def setup(self):
        """
        testcase preparcondition
        :return:
        """
        case_name = "test_lenet_all_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        if not super(TestLenet, self).setup():
            return False

        self.test_args = [
            ("lenet_ApplyMomentum_10_84_f32", apply_momentum_run, ((10, 84), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("lenet_ApplyMomentum_10_f32", apply_momentum_run, ((10,), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("lenet_ApplyMomentum_120_400_f32", apply_momentum_run, ((120, 400), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("lenet_ApplyMomentum_120_f32", apply_momentum_run, ((120,), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("lenet_ApplyMomentum_25_1_16_16_f32", apply_momentum_run, ((25, 1, 16, 16), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("lenet_ApplyMomentum_84_120_f32", apply_momentum_run, ((84, 120), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("lenet_ApplyMomentum_84_f32", apply_momentum_run, ((84,), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),

            ("001_cast_test_case_32_dim_2", cast_run, ((10, 84), "float32", "float16"),
             ["level0", "rpc", "rpc_cloud"]),
            ("002_cast_test_case_32_dim_1", cast_run, ((10,), "float32", "float16"), ["level0", "rpc", "rpc_cloud"]),
            ("003_cast_test_case_32_dim_2", cast_run, ((120, 400), "float32", "float16"),
             ["level0", "rpc", "rpc_cloud"]),
            (
                "004_cast_test_case_32_dim_1", cast_run, ((120,), "float32", "float16"),
                ["level0", "rpc", "rpc_cloud"]),
            ("005_cast_test_case_32_dim_4", cast_run, ((25, 1, 16, 16), "float32", "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("006_cast_test_case_16_dim_2", cast_run, ((32, 10), "float16", "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("007_cast_test_case_16_dim_2", cast_run, ((32, 120), "float16", "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("008_cast_test_case_16_dim_2", cast_run, ((32, 400), "float16", "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("09_cast_test_case_16_dim_2", cast_run, ((32, 84), "float16", "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("010_cast_test_case_32_dim_2", cast_run, ((84, 120), "float32", "float16"),
             ["level0", "rpc", "rpc_cloud"]),
            ("011_cast_test_case_32_dim_1", cast_run, ((84,), "float32", "float16"), ["level0", "rpc", "rpc_cloud"]),
            ("test_lenet_001_cast_test_case_16_dim_2", cast_run, ((10, 84), "float16", "float32"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_002_cast_test_case_16_dim_1", cast_run, ((10,), "float16", "float32"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_003_cast_test_case_16_dim_2", cast_run, ((120, 400), "float16", "float32"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_004_cast_test_case_16_dim_1", cast_run, ((120,), "float16", "float32"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_005_cast_test_case_16_dim_4", cast_run, ((25, 1, 16, 16), "float16", "float32"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_006_cast_test_case_32_dim_2", cast_run, ((32, 10), "float32", "float16"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_007_cast_test_case_32_dim_2", cast_run, ((32, 120), "float32", "float16"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_008_cast_test_case_32_dim_2", cast_run, ((32, 400), "float32", "float16"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_010_cast_test_case_16_dim_2", cast_run, ((84, 120), "float16", "float32"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_011_cast_test_case_16_dim_1", cast_run, ((84,), "float16", "float32"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_09_cast_test_case_32_dim_2", cast_run, ((32, 84), "float32", "float16"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),

            ("lenet_Conv2D_32_1_14_14_16", conv_run,
             ((32, 6, 14, 14), (16, 6, 5, 5), (0, 0, 0, 0), (1, 1), (1, 1), False), ["level0", "rpc", "rpc_cloud"]),
            ("lenet_Conv2D_32_1_32_32_16", conv_run,
             ((32, 1, 32, 32), (6, 1, 5, 5), (0, 0, 0, 0), (1, 1), (1, 1), False), ["level0", "rpc", "rpc_cloud"]),

            ("lenet_Conv2DBackpropInput_32_1_10_10_16", conv_backprop_input_run,
             ((32, 6, 14, 14), (16, 6, 5, 5), (0, 0, 0, 0), (1, 1), (1, 1)), ["level0", "rpc", "rpc_cloud"]),

            ("lenet_conv_backprop_filter_run_001", conv_backprop_filter_run,
             ((32, 6, 14, 14), (16, 6, 5, 5), (0, 0, 0, 0), (1, 1), (1, 1)), ["level0", "rpc", "rpc_cloud"]),
            ("lenet_conv_backprop_filter_run_002", conv_backprop_filter_run,
             ((32, 1, 32, 32), (6, 1, 5, 5), (0, 0, 0, 0), (1, 1), (1, 1)), ["level0", "rpc", "rpc_cloud"]),

            ("five2four_001", "five2four_run", ([32, 16, 5, 5], "float16", 'NCHW', "float16"),
             ["level0", "rpc", "rpc_cloud"]),
            ("five2four_002", "five2four_run", ([32, 16, 5, 5], "float32", 'NCHW', "float16"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),

            ("four2five_001", "four2five_run", ([32, 1, 32, 32], "float32", 'NCHW', "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("four2five_002", "four2five_run", ([32, 16, 5, 5], "float32", 'NCHW', "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("four2five_003", "four2five_run", ([32, 1, 32, 32], "float16", 'NCHW', "float32"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("four2five_004", "four2five_run", ([32, 16, 5, 5], "float16", 'NCHW', "float32"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),

            # FullConnection
            ("lenet_FullConnection_001", batchmatmul_execute,
             ((), 32, 84, 120, (84,), 'float16', False, True, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("lenet_FullConnection_002", batchmatmul_execute,
             ((), 32, 84, 120, (84,), "float32", False, True, "batchmatmul_output"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("lenet_FullConnection_003", batchmatmul_execute,
             ((), 32, 120, 400, (120,), 'float16', False, True, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("lenet_FullConnection_004", batchmatmul_execute,
             ((), 32, 120, 400, (120,), "float32", False, True, "batchmatmul_output"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("lenet_FullConnection_005", batchmatmul_execute,
             ((), 32, 10, 84, (10,), 'float16', False, True, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("lenet_FullConnection_006", batchmatmul_execute,
             ((), 32, 10, 84, (10,), "float32", False, True, "batchmatmul_output"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("lenet_FullConnection_007", batchmatmul_execute,
             ((), 32, 100, 84, (100,), 'float16', False, True, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("lenet_FullConnection_008", batchmatmul_execute,
             ((), 32, 100, 84, (100,), "float32", False, True, "batchmatmul_output"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),

            # MatMul
            ("lenet_MatMul_001", batchmatmul_execute,
             ((), 32, 84, 10, (), "float16", False, False, "batchmatmul_output"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("lenet_MatMul_002", batchmatmul_execute,
             ((), 32, 84, 10, (), 'float32', False, False, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("lenet_MatMul_003", batchmatmul_execute,
             ((), 32, 400, 120, (), "float16", False, False, "batchmatmul_output"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ('lenet_MatMul_004', batchmatmul_execute,
             ((), 32, 400, 120, (), 'float32', False, False, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("lenet_MatMul_005", batchmatmul_execute,
             ((), 32, 120, 84, (), "float16", False, False, "batchmatmul_output"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ('lenet_MatMul_006', batchmatmul_execute,
             ((), 32, 120, 84, (), 'float32', False, False, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("lenet_MatMul_007", batchmatmul_execute,
             ((), 32, 84, 100, (), "float16", False, False, "batchmatmul_output"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("lenet_MatMul_008", batchmatmul_execute,
             ((), 32, 84, 100, (), 'float32', False, False, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),

            # MatMulGe
            ("lenet_MatMulTaGe_001", batchmatmul_execute,
             ((), 120, 84, 32, (), "float16", True, False, "batchmatmul_output"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ('lenet_MatMulTaGe_002', batchmatmul_execute,
             ((), 120, 84, 32, (), 'float32', True, False, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("lenet_MatMulTaGe_003", batchmatmul_execute,
             ((), 400, 120, 32, (), "float16", True, False, "batchmatmul_output"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ('lenet_MatMulTaGe_004', batchmatmul_execute,
             ((), 400, 120, 32, (), 'float32', True, False, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ('lenet_MatMulTaGe_005', batchmatmul_execute,
             ((), 84, 32, 10, (), 'float16', True, False, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("lenet_MatMulTaGe_006", batchmatmul_execute,
             ((), 84, 32, 10, (), "float32", True, False, "batchmatmul_output"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ('lenet_MatMulTaGe_007', batchmatmul_execute,
             ((), 84, 32, 100, (), 'float16', True, False, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("lenet_MatMulTaGe_008", batchmatmul_execute,
             ((), 84, 32, 100, (), "float32", True, False, "batchmatmul_output"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),

            ("lenet_MaxPoolV2_32_1_10_10_16_f16_valid_2_0_2", maxpool_with_argmax_run,
             ((32, 1, 10, 10, 16), [2, 2], [2, 2], "VALID", True, "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("lenet_MaxPoolV2_32_1_28_28_16_f16_valid_2_0_2", maxpool_with_argmax_run,
             ((32, 1, 28, 28, 16), [2, 2], [2, 2], "VALID", True, "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),

            ("relu_001", relu_run, ((32, 1, 10, 10, 16), "float16", 1e-5), ["level0", "rpc", "rpc_cloud"]),
            ("relu_002", relu_run, ((32, 1, 28, 28, 16), "float16", 1e-5), ["level0", "rpc", "rpc_cloud"]),
            ("relu_003", relu_run, ((32, 1, 10, 10, 16), "float32", 1e-5),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("relu_004", relu_run, ((32, 1, 28, 28, 16), "float32", 1e-5),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),

            ("reshape_001", reshape_execute, [(32, 16, 5, 5), (32, -1), "float32"], ["level0", "rpc", "rpc_cloud"]),
            ("reshape_002", reshape_execute, [(32, 400), (32, 16, 5, 5), "float32"], ["level0", "rpc", "rpc_cloud"]),
            ("reshape_003", reshape_execute, [(32, 16, 5, 5), (32, -1), "float16"],
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("reshape_004", reshape_execute, [(32, 400), (32, 16, 5, 5), "float16"],
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),

            # maxpool_grad_with_argmax
            # ("test_lenet_maxpool_grad_with_argmax_001", maxpool_grad_with_argmax_run,
            #  ((32, 1, 10, 10, 16), (2, 2), (2, 2), "VALID", "float16", False, True),
            #  ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            # ("test_lenet_maxpool_grad_with_argmax_002", maxpool_grad_with_argmax_run,
            #  ((32, 1, 10, 10, 16), (2, 2), (2, 2), "VALID", "float32", False, True),
            #  ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            # ("test_lenet_maxpool_grad_with_argmax_003", maxpool_grad_with_argmax_run,
            #  ((32, 1, 28, 28, 16), (2, 2), (2, 2), "VALID", "float16", False, True),
            #  ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            # ("test_lenet_maxpool_grad_with_argmax_004", maxpool_grad_with_argmax_run,
            #  ((32, 1, 28, 28, 16), (2, 2), (2, 2), "VALID", "float32", False, True),
            #  ["level0", "rpc", "rpc_cloud", "Unavailable"]),

            # conv_bn1
            ("test_lenet_conv_bn1_32_1_14_14_16", conv_bn1_run,
             ((32, 6, 14, 14), (16, 6, 5, 5), (0, 0, 0, 0), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_conv_bn1_32_1_32_32_16", conv_bn1_run,
             ((32, 1, 32, 32), (6, 1, 5, 5), (0, 0, 0, 0), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),

            # relu_ad
            ("test_lenet_relu_ad_001", relu_ad_run, ((32, 1, 10, 10, 16), "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_relu_ad_002", relu_ad_run, ((32, 1, 28, 28, 16), "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_relu_ad_003", relu_ad_run, ((32, 1, 10, 10, 16), "float32"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_relu_ad_004", relu_ad_run, ((32, 1, 28, 28, 16), "float32"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),

            # bias_add_ad
            ("test_lenet_bias_add_ad_fp16_001", bias_add_ad_run, ([32, 10], "DefaultFormat", "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_bias_add_ad_fp16_002", bias_add_ad_run, ([32, 120], "DefaultFormat", "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_bias_add_ad_fp16_003", bias_add_ad_run, ([32, 84], "DefaultFormat", "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_bias_add_ad_fp32_001", bias_add_ad_run, ([32, 10], "DefaultFormat", "float32"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_bias_add_ad_fp32_002", bias_add_ad_run, ([32, 120], "DefaultFormat", "float32"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_bias_add_ad_fp32_003", bias_add_ad_run, ([32, 84], "DefaultFormat", "float32"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),

            # sparse_softmax_cross_entropy_with_logits_ad
            ("test_lenet_lenet_sparse_softmax_cross_entropy_with_logits_ad_32_10_f32_32_10_32_i32_32_true",
             sparse_softmax_cross_entropy_with_logits_ad_run,
             [(32,), "int32", (32, 10), "float32", "mean", "sparse_softmax_cross_entropy_with_logits_ad_fp32"],
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            #  conv_input_ad
            ("test_lenet_conv_input_ad_32_1_10_10_16", conv_input_ad_run,
             ((32, 6, 14, 14), (16, 6, 5, 5), (0, 0, 0, 0), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),

            # conv_filter_ad
            ("test_lenet_conv_filter_ad_run_001", conv_filter_ad_run,
             ((32, 6, 14, 14), (16, 6, 5, 5), (0, 0, 0, 0), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_conv_filter_ad_run_002", conv_filter_ad_run,
             ((32, 1, 32, 32), (6, 1, 5, 5), (0, 0, 0, 0), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),

            ("lenet_ApplyMomentum_1_84_f32", apply_momentum_run, ((1, 84), "float32", False),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("lenet_ApplyMomentum_1_f32", apply_momentum_run, ((1,), "float32", False),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("lenet_ApplyMomentum_1_400_f32", apply_momentum_run, ((1, 400), "float32", False),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("lenet_ApplyMomentum_1_1_16_16_f32", apply_momentum_run, ((1, 1, 16, 16), "float32", False),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("lenet_ApplyMomentum_1_120_f32", apply_momentum_run, ((1, 120), "float32", False),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),

            ("bias_add_fp32_001", bias_add_ad_run, ([1, 10], "DefaultFormat", "float32"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("bias_add_fp32_002", bias_add_ad_run, ([1, 120], "DefaultFormat", "float32"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("bias_add_fp32_003", bias_add_ad_run, ([1, 84], "DefaultFormat", "float32"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("bias_add_fp16_004", bias_add_ad_run, ([1, 10], "DefaultFormat", "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("bias_add_fp16_005", bias_add_ad_run, ([1, 120], "DefaultFormat", "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("bias_add_fp16_006", bias_add_ad_run, ([1, 84], "DefaultFormat", "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),

            ("001_cast_test_case_32_dim_2", cast_run, ((1, 84), "float32", "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("002_cast_test_case_32_dim_1", cast_run, ((1,), "float32", "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("003_cast_test_case_32_dim_2", cast_run, ((1, 400), "float32", "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("004_cast_test_case_32_dim_4", cast_run, ((1, 1, 16, 16), "float32", "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("005_cast_test_case_16_dim_2", cast_run, ((1, 10), "float16", "float32"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("006_cast_test_case_16_dim_2", cast_run, ((1, 120), "float16", "float32"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("007_cast_test_case_16_dim_2", cast_run, ((1, 400), "float16", "float32"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("08_cast_test_case_16_dim_2", cast_run, ((1, 84), "float16", "float32"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("009_cast_test_case_32_dim_2", cast_run, ((1, 120), "float32", "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_001_cast_test_case_16_dim_1", cast_run, ((1,), "float16", "float32"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_002_cast_test_case_16_dim_4", cast_run, ((1, 1, 16, 16), "float16", "float32"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_003_cast_test_case_32_dim_2", cast_run, ((1, 10), "float32", "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),

            ("five2four_001", "five2four_run", ([1, 16, 5, 5], "float16", 'NCHW', "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("five2four_002", "five2four_run", ([1, 16, 5, 5], "float32", 'NCHW', "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("four2five_001", "four2five_run", ([1, 1, 32, 32], "float32", 'NCHW', "float32"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("four2five_002", "four2five_run", ([1, 16, 5, 5], "float32", 'NCHW', "float32"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("four2five_003", "four2five_run", ([1, 1, 32, 32], "float16", 'NCHW', "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("four2five_004", "four2five_run", ([1, 16, 5, 5], "float16", 'NCHW', "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),

            ("test_lenet_lenet_sparse_softmax_cross_entropy_with_logits_ad_32_10_f32_32_10_32_i32_32_true",
             sparse_softmax_cross_entropy_with_logits_ad_run,
             [(1,), "int32", (1, 10), "float32", "mean", "sparse_softmax_cross_entropy_with_logits_ad_fp32"],
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),

            ("relu_001", relu_run, ((1, 1, 10, 10, 16), "float16", 1e-5),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("relu_002", relu_run, ((1, 1, 28, 28, 16), "float16", 1e-5),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("relu_003", relu_run, ((1, 1, 10, 10, 16), "float32", 1e-5),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("relu_004", relu_run, ((1, 1, 28, 28, 16), "float32", 1e-5),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),

            ("relu_grad_001", relu_ad_run, ((1, 1, 10, 10, 16), "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("relu_grad_002", relu_ad_run, ((1, 1, 28, 28, 16), "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("relu_grad_003", relu_ad_run, ((1, 1, 10, 10, 16), "float32"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("relu_grad_004", relu_ad_run, ((1, 1, 28, 28, 16), "float32"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),

            ("reshape_001", reshape_execute, [(1, 16, 5, 5), (1, -1), "float32"],
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("reshape_002", reshape_execute, [(1, 400), (1, 16, 5, 5), "float32"],
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("reshape_003", reshape_execute, [(1, 16, 5, 5), (1, -1), "float16"],
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("reshape_004", reshape_execute, [(1, 400), (1, 16, 5, 5), "float16"],
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),

            ("lenet_Conv2D_32_1_14_14_16", conv_run,
             ((1, 6, 14, 14), (16, 6, 5, 5), (0, 0, 0, 0), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("lenet_Conv2D_32_1_32_32_16", conv_run,
             ((1, 1, 32, 32), (6, 1, 5, 5), (0, 0, 0, 0), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("lenet_Conv2DBackpropInput_32_1_10_10_16", conv_backprop_input_run,
             ((1, 6, 14, 14), (16, 6, 5, 5), (0, 0, 0, 0), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("lenet_conv_backprop_filter_run_001", conv_backprop_filter_run,
             ((1, 6, 14, 14), (16, 6, 5, 5), (0, 0, 0, 0), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("lenet_conv_backprop_filter_run_002", conv_backprop_filter_run,
             ((1, 1, 32, 32), (6, 1, 5, 5), (0, 0, 0, 0), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),

            ("test_lenet_conv_input_ad_32_1_10_10_16", conv_input_ad_run,
             ((1, 6, 14, 14), (16, 6, 5, 5), (0, 0, 0, 0), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_conv_filter_ad_run_001", conv_filter_ad_run,
             ((1, 6, 14, 14), (16, 6, 5, 5), (0, 0, 0, 0), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_conv_filter_ad_run_002", conv_filter_ad_run,
             ((1, 1, 32, 32), (6, 1, 5, 5), (0, 0, 0, 0), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_conv_bn1_32_1_14_14_16", conv_bn1_run,
             ((1, 6, 14, 14), (16, 6, 5, 5), (0, 0, 0, 0), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_conv_bn1_32_1_32_32_16", conv_bn1_run,
             ((1, 1, 32, 32), (6, 1, 5, 5), (0, 0, 0, 0), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),

            # FullConnection
            ("lenet_FullConnection_1_001", batchmatmul_execute,
             ((1,), 32, 84, 120, (84,), 'float16', False, True, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("lenet_FullConnection_1_002", batchmatmul_execute,
             ((1,), 32, 120, 400, (120,), 'float16', False, True, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("lenet_FullConnection_1_003", batchmatmul_execute,
             ((1,), 32, 10, 84, (10,), 'float16', False, True, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("lenet_FullConnection_1_004", batchmatmul_execute,
             ((1,), 32, 100, 84, (100,), 'float16', False, True, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),

            # MatMul
            ("lenet_MatMul_1_001", batchmatmul_execute,
             ((1,), 32, 84, 10, (), 'float32', False, False, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ('lenet_MatMul_1_002', batchmatmul_execute,
             ((1,), 32, 400, 120, (), 'float32', False, False, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ('lenet_MatMul_1_003', batchmatmul_execute,
             ((1,), 32, 120, 84, (), 'float32', False, False, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("lenet_MatMul_1_004", batchmatmul_execute,
             ((1,), 32, 84, 100, (), 'float32', False, False, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),

            # MatMulGe
            ('lenet_MatMulTaGe_1_001', batchmatmul_execute,
             ((1,), 120, 84, 32, (), 'float32', True, False, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ('lenet_MatMulTaGe_1_002', batchmatmul_execute,
             ((1,), 400, 120, 32, (), 'float32', True, False, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ('lenet_MatMulTaGe_1_003', batchmatmul_execute,
             ((1,), 84, 32, 10, (), 'float32', True, False, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ('lenet_MatMulTaGe_1_004', batchmatmul_execute,
             ((1,), 84, 32, 100, (), 'float32', True, False, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
        ]


def print_args():
    cls = TestLenet()
    cls.setup()
    cls.print_args()


if __name__ == "__main__":
    print_args()
