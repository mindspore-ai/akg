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
from tests.common.test_run.conv_input_ad_run import conv_input_ad_run
from tests.common.test_run.conv_filter_ad_run import conv_filter_ad_run


class TestAlexnet(BaseCaseRun):
    def setup(self):
        """
        testcase preparcondition
        :return:
        """
        case_name = "test_alexnet_all_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        if not super(TestAlexnet, self).setup():
            return False

        self.test_args = [
            # Applymomentum
            ("test_alexnet_v1_apply_momentum_001", apply_momentum_run, ((10, 4096), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_alexnet_v1_apply_momentum_002", apply_momentum_run, ((10,), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_alexnet_v1_apply_momentum_003", apply_momentum_run, ((121, 6, 16, 16), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_alexnet_v1_apply_momentum_004", apply_momentum_run, ((144, 24, 16, 16), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_alexnet_v1_apply_momentum_005", apply_momentum_run, ((150, 16, 16, 16), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_alexnet_v1_apply_momentum_006", apply_momentum_run, ((216, 16, 16, 16), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_alexnet_v1_apply_momentum_007", apply_momentum_run, ((216, 24, 16, 16), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_alexnet_v1_apply_momentum_008", apply_momentum_run, ((4096, 4096), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_alexnet_v1_apply_momentum_009", apply_momentum_run, ((4096, 9216), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_alexnet_v1_apply_momentum_010", apply_momentum_run, ((4096,), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),

            # Cast
            ("test_alexnet_cast_000", cast_run, ([10], "float32", "float16"), ["level0", "rpc", "rpc_cloud"]),
            ("test_alexnet_cast_001", cast_run, ([10, 4096], "float32", "float16"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_alexnet_cast_002", cast_run, ([4096], "float32", "float16"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_alexnet_cast_003", cast_run, ([4096, 4096], "float32", "float16"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_alexnet_cast_004", cast_run, ([4096, 9216], "float32", "float16"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_alexnet_cast_005", cast_run, ([216, 16, 16, 16], "float32", "float16"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_alexnet_cast_006", cast_run, ([216, 24, 16, 16], "float32", "float16"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_alexnet_cast_007", cast_run, ([144, 24, 16, 16], "float32", "float16"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_alexnet_cast_008", cast_run, ([150, 16, 16, 16], "float32", "float16"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_alexnet_cast_009", cast_run, ([121, 6, 16, 16], "float32", "float16"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_alexnet_cast_010", cast_run, ([32, 10], "float16", "float32"), ["level0", "rpc", "rpc_cloud"]),
            ("test_alexnet_cast_011", cast_run, ([32, 4096], "float16", "float32"), ["level0", "rpc", "rpc_cloud"]),
            ("test_alexnet_cast_012", cast_run, ([32, 9216], "float16", "float32"), ["level0", "rpc", "rpc_cloud"]),
            ("test_alexnet_cast_014", cast_run, ([216, 16, 16, 16], "float16", "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_alexnet_cast_015", cast_run, ([216, 24, 16, 16], "float16", "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_alexnet_cast_016", cast_run, ([144, 24, 16, 16], "float16", "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_alexnet_cast_017", cast_run, ([150, 16, 16, 16], "float16", "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_alexnet_cast_018", cast_run, ([121, 6, 16, 16], "float16", "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_alexnet_cast_019", cast_run, ([10], "float16", "float32"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_cast_020", cast_run, ([10, 4096], "float16", "float32"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_cast_021", cast_run, ([4096], "float16", "float32"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_cast_022", cast_run, ([4096, 4096], "float16", "float32"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_cast_023", cast_run, ([4096, 9216], "float16", "float32"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_cast_024", cast_run, ([216, 16, 16, 16], "float16", "float32"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_cast_025", cast_run, ([216, 24, 16, 16], "float16", "float32"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_cast_026", cast_run, ([144, 24, 16, 16], "float16", "float32"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_cast_027", cast_run, ([150, 16, 16, 16], "float16", "float32"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_cast_028", cast_run, ([121, 6, 16, 16], "float16", "float32"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_cast_029", cast_run, ([32, 10], "float32", "float16"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_cast_030", cast_run, ([32, 4096], "float32", "float16"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_cast_031", cast_run, ([32, 9216], "float32", "float16"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_cast_032", cast_run, ([216, 16, 16, 16], "float32", "float16"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_cast_033", cast_run, ([216, 24, 16, 16], "float32", "float16"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_cast_034", cast_run, ([144, 24, 16, 16], "float32", "float16"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_cast_035", cast_run, ([150, 16, 16, 16], "float32", "float16"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_cast_036", cast_run, ([121, 6, 16, 16], "float32", "float16"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),

            # Five2Four
            ("five2four_001", "five2four_run", ([32, 256, 6, 6], "float16", 'NCHW', "float16"),
             ["level0", "rpc", "rpc_cloud"]),
            ("five2four_002", "five2four_run", ([32, 256, 6, 6], "float32", 'NCHW', "float32"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),

            # four2five
            ("test_alexnet_four2five_000", "four2five_run", ([32, 3, 227, 227], "float32", "NCHW", "float32"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_four2five_001", "four2five_run", ([32, 256, 6, 6], "float32", "NCHW", "float32"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_four2five_002", "four2five_run", ([32, 3, 227, 227], "float16", "NCHW", "float16"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_four2five_003", "four2five_run", ([32, 256, 6, 6], "float16", "NCHW", "float16"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),

            # FullConnection
            ('FullConnection_alexnet_001', batchmatmul_execute,
             ((), 32, 10, 4096, (10,), 'float16', False, True, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ('FullConnection_alexnet_002', batchmatmul_execute,
             ((), 32, 4096, 4096, (4096,), 'float16', False, True, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ('FullConnection_alexnet_003', batchmatmul_execute,
             ((), 32, 4096, 9216, (4096,), 'float16', False, True, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ('FullConnection_alexnet_004', batchmatmul_execute,
             ((), 32, 1001, 4096, (1001,), 'float16', False, True, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ('FullConnection_alexnet_005', batchmatmul_execute,
             ((), 32, 1000, 4096, (1000,), 'float16', False, True, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ('FullConnection_alexnet_006', batchmatmul_execute,
             ((), 32, 10, 4096, (10,), 'float32', False, True, 'batchmatmul_output'),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ('FullConnection_alexnet_007', batchmatmul_execute,
             ((), 32, 4096, 4096, (4096,), 'float32', False, True, 'batchmatmul_output'),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ('FullConnection_alexnet_008', batchmatmul_execute,
             ((), 32, 4096, 9216, (4096,), 'float32', False, True, 'batchmatmul_output'),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ('FullConnection_alexnet_009', batchmatmul_execute,
             ((), 32, 1001, 4096, (1001,), 'float32', False, True, 'batchmatmul_output'),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ('FullConnection_alexnet_010', batchmatmul_execute,
             ((), 32, 1000, 4096, (1000,), 'float32', False, True, 'batchmatmul_output'),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),

            ('FullConnection_alexnet_011', batchmatmul_execute,
             ((), 32, 100, 4096, (100,), 'float16', False, True, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ('FullConnection_alexnet_012', batchmatmul_execute,
             ((), 32, 100, 4096, (100,), 'float32', False, True, 'batchmatmul_output'),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),

            # MatMul
            ('MatMul_alexnet_001', batchmatmul_execute,
             ((), 32, 4096, 4096, (), 'float32', False, False, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ('MatMul_alexnet_002', batchmatmul_execute,
             ((), 32, 9216, 4096, (), 'float32', False, False, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ('MatMul_alexnet_003', batchmatmul_execute,
             ((), 32, 4096, 10, (), 'float32', False, False, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ('MatMul_alexnet_004', batchmatmul_execute,
             ((), 32, 4096, 1001, (), 'float32', False, False, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ('MatMul_alexnet_005', batchmatmul_execute,
             ((), 32, 4096, 1000, (), 'float32', False, False, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ('MatMul_alexnet_006', batchmatmul_execute,
             ((), 32, 4096, 4096, (), 'float16', False, False, 'batchmatmul_output'),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ('MatMul_alexnet_007', batchmatmul_execute,
             ((), 32, 9216, 4096, (), 'float16', False, False, 'batchmatmul_output'),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ('MatMul_alexnet_008', batchmatmul_execute,
             ((), 32, 4096, 10, (), 'float16', False, False, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ('MatMul_alexnet_009', batchmatmul_execute,
             ((), 32, 4096, 1001, (), 'float16', False, False, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ('MatMul_alexnet_010', batchmatmul_execute,
             ((), 32, 4096, 1000, (), 'float16', False, False, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),

            ('MatMul_alexnet_011', batchmatmul_execute,
             ((), 32, 4096, 100, (), 'float32', False, False, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ('MatMul_alexnet_012', batchmatmul_execute,
             ((), 32, 4096, 100, (), 'float16', False, False, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),

            # MatMulGe
            ('MatMulGe_alexnet_001', batchmatmul_execute,
             ((), 4096, 10, 32, (), 'float32', True, False, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ('MatMulGe_alexnet_002', batchmatmul_execute,
             ((), 4096, 4096, 32, (), 'float32', True, False, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ('MatMulGe_alexnet_003', batchmatmul_execute,
             ((), 9216, 4096, 32, (), 'float32', True, False, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ('MatMulGe_alexnet_004', batchmatmul_execute,
             ((), 4096, 1001, 32, (), 'float32', True, False, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ('MatMulGe_alexnet_005', batchmatmul_execute,
             ((), 4096, 1000, 32, (), 'float32', True, False, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ('MatMulGe_alexnet_006', batchmatmul_execute,
             ((), 4096, 10, 32, (), 'float16', True, False, 'batchmatmul_output'),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ('MatMulGe_alexnet_007', batchmatmul_execute,
             ((), 4096, 4096, 32, (), 'float16', True, False, 'batchmatmul_output'),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ('MatMulGe_alexnet_008', batchmatmul_execute,
             ((), 9216, 4096, 32, (), 'float16', True, False, 'batchmatmul_output'),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ('MatMulGe_alexnet_009', batchmatmul_execute,
             ((), 4096, 1001, 32, (), 'float16', True, False, 'batchmatmul_output'),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ('MatMulGe_alexnet_010', batchmatmul_execute,
             ((), 4096, 1000, 32, (), 'float16', True, False, 'batchmatmul_output'),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),

            ('MatMulGe_alexnet_011', batchmatmul_execute,
             ((), 4096, 100, 32, (), 'float32', True, False, 'batchmatmul_output'),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ('MatMulGe_alexnet_012', batchmatmul_execute,
             ((), 4096, 100, 32, (), 'float16', True, False, 'batchmatmul_output'),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),

            # maxpool_with_argmax_
            ("alexnet_maxpool_with_argmax_fp16_001", maxpool_with_argmax_run,
             ((32, 16, 13, 13, 16), (3, 3), (2, 2), "VALID", True, "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("alexnet_maxpool_with_argmax_fp16_002", maxpool_with_argmax_run,
             ((32, 16, 27, 27, 16), (3, 3), (2, 2), "VALID", True, "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("alexnet_maxpool_with_argmax_fp16_003", maxpool_with_argmax_run,
             ((32, 6, 55, 55, 16), (3, 3), (2, 2), "VALID", True, "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),

            # relu
            ("test_alexnet_relu_000", relu_run,
             ([32, 6, 55, 55, 16], "float16", 1e-5), ["level0", "rpc", "rpc_cloud"]),
            ("test_alexnet_relu_001", relu_run,
             ([32, 16, 27, 27, 16], "float16", 1e-5), ["level0", "rpc", "rpc_cloud"]),
            ("test_alexnet_relu_003", relu_run,
             ([32, 24, 13, 13, 16], "float16", 1e-5), ["level0", "rpc", "rpc_cloud"]),
            ("test_alexnet_relu_004", relu_run,
             ([32, 16, 13, 13, 16], "float16", 1e-5), ["level0", "rpc", "rpc_cloud"]),
            ("test_alexnet_relu_005", relu_run,
             ([32, 4096], "float16", 1e-5), ["level0", "rpc", "rpc_cloud"]),
            ("test_alexnet_relu_006", relu_run, ([32, 6, 55, 55, 16], "float32", 1e-5),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_relu_007", relu_run, ([32, 16, 27, 27, 16], "float32", 1e-5),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_relu_008", relu_run, ([32, 24, 13, 13, 16], "float32", 1e-5),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_relu_009", relu_run, ([32, 16, 13, 13, 16], "float32", 1e-5),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_relu_010", relu_run, ([32, 4096], "float32", 1e-5),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),

            # Reshape
            ("reshape_001", reshape_execute, [(32, 256, 6, 6), (32, -1), "float16"],
             ["level0", "rpc", "rpc_cloud"]),
            ("reshape_002", reshape_execute, [(32, 9216), (32, 256, 6, 6), "float32"],
             ["level0", "rpc", "rpc_cloud"]),
            ("test_alexnet_reshape_003", reshape_execute, [(32, 256, 6, 6), (32, -1), "float32"],
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_reshape_004", reshape_execute, [(32, 9216), (32, 256, 6, 6), "float16"],
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),

            # Conv
            ("Alexnet_Conv2D_32_1_227_227_16", conv_run,
             ((32, 3, 227, 227), (96, 3, 11, 11), (0, 0, 0, 0), (4, 4), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("Alexnet_Conv2D_32_16_13_13_16", conv_run,
             ((32, 256, 13, 13), (384, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("Alexnet_Conv2D_32_24_13_13_16", conv_run,
             ((32, 384, 13, 13), (256, 384, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("Alexnet_Conv2D_32_24_13_13_16_v2", conv_run,
             ((32, 384, 13, 13), (384, 384, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("Alexnet_Conv2D_32_6_27_27_16", conv_run,
             ((32, 96, 27, 27), (256, 96, 5, 5), (2, 2, 2, 2), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),

            # ConvBackward
            ("Alexnet_Conv2DBackpropInput_32_24_13_13_16", conv_backprop_input_run,
             ((32, 384, 13, 13), (256, 384, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud"]),
            ("Alexnet_Conv2DBackpropInput_32_6_27_27_16", conv_backprop_input_run,
             ((32, 96, 27, 27), (256, 96, 5, 5), (2, 2, 2, 2), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud"]),
            ("Alexnet_Conv2DBackpropInput_32_16_13_13_16", conv_backprop_input_run,
             ((32, 256, 13, 13), (384, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud"]),
            ("Alexnet_Conv2DBackpropInput_32_24_13_13_16_v2", conv_backprop_input_run,
             ((32, 384, 13, 13), (384, 384, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud"]),

            # ConvBackwardFilter
            ("test_alexnet_conv_backprop_filter_000", conv_backprop_filter_run,
             ([32, 3, 227, 227], [96, 3, 11, 11], (0, 0, 0, 0), (4, 4), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("Alexnet_conv_backprop_filter_run_001", conv_backprop_filter_run,
             ((32, 384, 13, 13), (256, 384, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1)), ["level0", "rpc", "rpc_cloud"]),
            ("Alexnet_conv_backprop_filter_run_002", conv_backprop_filter_run,
             ((32, 96, 27, 27), (256, 96, 5, 5), (2, 2, 2, 2), (1, 1), (1, 1)), ["level0", "rpc", "rpc_cloud"]),
            ("Alexnet_conv_backprop_filter_run_003", conv_backprop_filter_run,
             ((32, 256, 13, 13), (384, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1)), ["level0", "rpc", "rpc_cloud"]),
            ("Alexnet_conv_backprop_filter_run_004", conv_backprop_filter_run,
             ((32, 384, 13, 13), (384, 384, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1)), ["level0", "rpc", "rpc_cloud"]),

            # maxpool_grad_with_argmax
            # ("test_alexnet_maxpool_grad_with_argmax_001", maxpool_grad_with_argmax_run,
            #  ([32, 16, 13, 13, 16], (3, 3), (2, 2), "VALID", "float16", False, True),
            #  ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            # ("test_alexnet_maxpool_grad_with_argmax_002", maxpool_grad_with_argmax_run,
            #  ([32, 16, 27, 27, 16], (3, 3), (2, 2), "VALID", "float16", False, True),
            #  ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            # ("test_alexnet_maxpool_grad_with_argmax_003", maxpool_grad_with_argmax_run,
            #  ([32, 6, 55, 55, 16], (3, 3), (2, 2), "VALID", "float16", False, True),
            #  ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            # ("test_alexnet_maxpool_grad_with_argmax_004", maxpool_grad_with_argmax_run,
            #  ((32, 16, 13, 13, 16), (3, 3), (2, 2), "VALID", "float32", False, True),
            #  ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            # ("test_alexnet_maxpool_grad_with_argmax_005", maxpool_grad_with_argmax_run,
            #  ((32, 16, 27, 27, 16), (3, 3), (2, 2), "VALID", "float32", False, True),
            #  ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            # ("test_alexnet_maxpool_grad_with_argmax_006", maxpool_grad_with_argmax_run,
            #  ((32, 6, 55, 55, 16), (3, 3), (2, 2), "VALID", "float32", False, True),
            #  ["level0", "rpc", "rpc_cloud", "Unavailable"]),

            # conv_bn1
            ("Alexnet_conv_bn1_32_1_227_227_16", conv_bn1_run,
             ((32, 3, 227, 227), (96, 3, 11, 11), (0, 0, 0, 0), (4, 4), (1, 1), False),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("Alexnet_conv_bn1_32_16_13_13_16", conv_bn1_run,
             ((32, 256, 13, 13), (384, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("Alexnet_conv_bn1_32_24_13_13_16", conv_bn1_run,
             ((32, 384, 13, 13), (256, 384, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("Alexnet_conv_bn1_32_24_13_13_16_v2", conv_bn1_run,
             ((32, 384, 13, 13), (384, 384, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("Alexnet_conv_bn1_32_6_27_27_16", conv_bn1_run,
             ((32, 96, 27, 27), (256, 96, 5, 5), (2, 2, 2, 2), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),

            # relu_ad
            ("test_alexnet_relu_ad_001", relu_ad_run, ([32, 4096], "float32"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_relu_ad_002", relu_ad_run, ([32, 16, 13, 13, 16], "float32"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_relu_ad_003", relu_ad_run, ([32, 24, 13, 13, 16], "float32"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_relu_ad_004", relu_ad_run, ([32, 24, 13, 13, 16], "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_relu_ad_005", relu_ad_run, ([32, 16, 27, 27, 16], "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_relu_ad_006", relu_ad_run, ([32, 6, 55, 55, 16], "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_relu_ad_007", relu_ad_run, ((32, 16, 13, 13, 16), "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_relu_ad_008", relu_ad_run, ((32, 16, 28, 28, 16), "float32"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_relu_ad_009", relu_ad_run, ((32, 16, 28, 28, 16), "float16"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_relu_ad_010", relu_ad_run, ((32, 4096), "float16"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_relu_ad_011", relu_ad_run, ((32, 6, 55, 55, 16), "float32"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_alexnet_relu_ad_012", relu_ad_run, ([32, 16, 27, 27, 16], "float32"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),

            # conv_input_ad
            ("Alexnet_conv_input_ad_32_24_13_13_16", conv_input_ad_run,
             ((32, 384, 13, 13), (256, 384, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("Alexnet_conv_input_ad_32_6_27_27_16", conv_input_ad_run,
             ((32, 96, 27, 27), (256, 96, 5, 5), (2, 2, 2, 2), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("Alexnet_conv_input_ad_32_16_13_13_16", conv_input_ad_run,
             ((32, 256, 13, 13), (384, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("Alexnet_conv_input_ad_32_24_13_13_16_v2", conv_input_ad_run,
             ((32, 384, 13, 13), (384, 384, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),

            # conv_filter_ad
            ("test_alexnet_conv_filter_ad_000", conv_filter_ad_run,
             ([32, 3, 227, 227], [96, 3, 11, 11], (0, 0, 0, 0), (4, 4), (1, 1)),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("Alexnet_conv_filter_ad_001", conv_filter_ad_run,
             ((32, 384, 13, 13), (256, 384, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1)),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("Alexnet_conv_filter_ad_002", conv_filter_ad_run,
             ((32, 96, 27, 27), (256, 96, 5, 5), (2, 2, 2, 2), (1, 1), (1, 1)),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("Alexnet_conv_filter_ad_003", conv_filter_ad_run,
             ((32, 256, 13, 13), (384, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1)),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("Alexnet_conv_filter_ad_004", conv_filter_ad_run,
             ((32, 384, 13, 13), (384, 384, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1)),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),

            # sparse_softmax_cross_entropy_with_logits_ad
            ("Alexnet_sparse_softmax_cross_entropy_with_logits_ad_001",
             sparse_softmax_cross_entropy_with_logits_ad_run,
             [(32,), "int32", (32, 10), "float32", "mean", "sparse_softmax_cross_entropy_with_logits_ad_fp32"],
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("Alexnet_sparse_softmax_cross_entropy_with_logits_ad_002",
             sparse_softmax_cross_entropy_with_logits_ad_run,
             [(32,), "int32", (32, 1001), "float32", "mean", "sparse_softmax_cross_entropy_with_logits_ad_fp32"],
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("Alexnet_sparse_softmax_cross_entropy_with_logits_ad_003",
             sparse_softmax_cross_entropy_with_logits_ad_run,
             [(32,), "int32", (32, 1000), "float32", "mean", "sparse_softmax_cross_entropy_with_logits_ad_fp32"],
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),

            # bias_add_ad
            ("test_lenet_bias_add_ad_fp16_001", bias_add_ad_run, ([32, 10], "DefaultFormat", "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_bias_add_ad_fp16_002", bias_add_ad_run, ([32, 120], "DefaultFormat", "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_bias_add_ad_fp16_003", bias_add_ad_run, ([32, 84], "DefaultFormat", "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_bias_add_ad_fp32_004", bias_add_ad_run, ([32, 10], "DefaultFormat", "float32"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_bias_add_ad_fp32_005", bias_add_ad_run, ([32, 120], "DefaultFormat", "float32"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_bias_add_ad_fp32_006", bias_add_ad_run, ([32, 84], "DefaultFormat", "float32"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_bias_add_ad_fp32_007", bias_add_ad_run, ([32, 1001], "DefaultFormat", "float32"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_bias_add_ad_fp16_008", bias_add_ad_run, ([32, 1001], "DefaultFormat", "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_bias_add_ad_fp32_009", bias_add_ad_run, ([32, 1000], "DefaultFormat", "float32"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_lenet_bias_add_ad_fp16_010", bias_add_ad_run, ([32, 1000], "DefaultFormat", "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),

        ]


def print_args():
    cls = TestAlexnet()
    cls.setup()
    cls.print_args()


if __name__ == "__main__":
    print_args()
