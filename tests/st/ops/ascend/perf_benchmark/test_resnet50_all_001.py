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

"""
################################################

Testcase_PrepareCondition:

Testcase_TestSteps:

Testcase_ExpectedResult:

"""
import os
import pytest
from tests.common.base import TestBase, get_splitted_cases
from tests.common.test_run.conv_run import conv_run
from tests.common.test_run.conv_input_ad_run import conv_input_ad_run
from tests.common.test_run.conv_filter_ad_run import conv_filter_ad_run
from tests.common.test_run.batchmatmul_run import batchmatmul_execute
from tests.common.test_run.mean_run import mean_execute
from tests.common.test_run.mean_ad_run import mean_ad_run
from tests.common.test_run.relu_run import relu_run
from tests.common.test_run.relu_ad_run import relu_ad_run
from tests.common.test_run.add_run import add_run
from tests.common.test_run.addn_run import addn_execute
from tests.common.test_run.sparse_softmax_cross_entropy_with_logits_run import \
    sparse_softmax_cross_entropy_with_logits_run
from tests.common.test_run.sparse_softmax_cross_entropy_with_logits_ad_run import \
    sparse_softmax_cross_entropy_with_logits_ad_run
from tests.common.test_run.bias_add_ad_run import bias_add_ad_run
from tests.common.test_run.bias_add_run import bias_add_run
from tests.common.test_run.reshape_run import reshape_execute
from tests.common.test_run.apply_momentum_run import apply_momentum_run
from tests.common.test_run.bn_split_run import bn_split_run
from tests.common.test_run.fused_batch_norm_grad_run import fused_bn_grad_5D_all_run
from tests.common.test_run.conv_bn1_run import conv_bn1_run
from tests.common.test_run.softmax_run import softmax_execute
from tests.common.test_run.argmax_run import argmax_run
from tests.common.test_run.equal_count_run import equal_count_run
from tests.common.test_run.clear_zero_run import clear_zero_run
from tests.common.test_run.maxpool_with_argmax_run import maxpool_with_argmax_run
from tests.common.test_run.maxpool_grad_with_argmax_run import maxpool_grad_with_argmax_run


class TestResnet50_001(TestBase):
    def setup(self):
        """
        testcase preparcondition
        :return:
        """
        case_name = "test_resnet50_all_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        if not super(TestResnet50_001, self).setup():
            return False

        self.test_args = [
            # maxpool_with_argmax
            ("test_resnet50_maxpool_with_argmax_000", maxpool_with_argmax_run,
             ((32, 4, 112, 112, 16), (3, 3), (2, 2), 'SAME', True, "float16"), ["level0", "rpc", "rpc_cloud"]),

            # batch_matmul
            ("test_resnet50_matmul_000", batchmatmul_execute,
             ((), 32, 10, 2048, (10,), "float32", False, True, "batchmatmul_output"), ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_matmul_001", batchmatmul_execute,
             ((), 2048, 10, 32, (), "float32", True, False, "batchmatmul_output"), ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_matmul_002", batchmatmul_execute,
             ((), 32, 2048, 10, (), "float32", False, False, "batchmatmul_output"), ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_matmul_003", batchmatmul_execute,
             ((), 2048, 1001, 32, (), "float32", True, False, "batchmatmul_output"), ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_matmul_004", batchmatmul_execute,
             ((), 32, 2048, 1001, (), "float32", False, False, "batchmatmul_output"), ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_matmul_005", batchmatmul_execute,
             ((), 32, 1001, 2048, (1001,), "float32", False, True, "batchmatmul_output"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_matmul_006", batchmatmul_execute,
             ((), 32, 10, 2048, (10,), "float16", False, True, "batchmatmul_output"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_matmul_007", batchmatmul_execute,
             ((), 2048, 10, 32, (), "float16", True, False, "batchmatmul_output"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_matmul_008", batchmatmul_execute,
             ((), 32, 2048, 10, (), "float16", False, False, "batchmatmul_output"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_matmul_009", batchmatmul_execute,
             ((), 2048, 1001, 32, (), "float16", True, False, "batchmatmul_output"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_matmul_010", batchmatmul_execute,
             ((), 32, 2048, 1001, (), "float16", False, False, "batchmatmul_output"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_matmul_011", batchmatmul_execute,
             ((), 32, 1001, 2048, (1001,), "float16", False, True, "batchmatmul_output"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_matmul_012", batchmatmul_execute,
             ((), 1001, 2048, 32, (), "float32", True, False, "batchmatmul_output"), ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_matmul_013", batchmatmul_execute,
             ((), 32, 1001, 2048, (), "float16", False, True, "batchmatmul_output"), ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_matmul_014", batchmatmul_execute,
             ((), 1001, 2048, 32, (), "float16", True, False, "batchmatmul_output"), ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_matmul_015", batchmatmul_execute,
             ((), 32, 1001, 2048, (), "float32", False, True, "batchmatmul_output"), ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_matmul_016", batchmatmul_execute,
             ((), 10, 2048, 32, (), "float32", True, False, "batchmatmul_output"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_matmul_017", batchmatmul_execute,
             ((), 32, 10, 2048, (), "float16", False, True, "batchmatmul_output"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_matmul_018", batchmatmul_execute,
             ((), 10, 2048, 32, (), "float16", True, False, "batchmatmul_output"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_matmul_019", batchmatmul_execute,
             ((), 32, 10, 2048, (), "float32", False, True, "batchmatmul_output"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_matmul_020", batchmatmul_execute,
             ((), 100, 2048, 32, (), "float32", True, False, "batchmatmul_output"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_matmul_021", batchmatmul_execute,
             ((), 32, 100, 2048, (), "float16", False, True, "batchmatmul_output"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_matmul_022", batchmatmul_execute,
             ((), 100, 2048, 32, (), "float16", True, False, "batchmatmul_output"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_matmul_023", batchmatmul_execute,
             ((), 32, 100, 2048, (), "float32", False, True, "batchmatmul_output"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_matmul_024", batchmatmul_execute,
             ((), 32, 100, 2048, (100,), "float32", False, True, "batchmatmul_output"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_matmul_025", batchmatmul_execute,
             ((), 2048, 100, 32, (), "float32", True, False, "batchmatmul_output"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_matmul_026", batchmatmul_execute,
             ((), 32, 2048, 100, (), "float32", False, False, "batchmatmul_output"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_matmul_027", batchmatmul_execute,
             ((), 32, 100, 2048, (100,), "float16", False, True, "batchmatmul_output"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_matmul_028", batchmatmul_execute,
             ((), 2048, 100, 32, (), "float16", True, False, "batchmatmul_output"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_matmul_029", batchmatmul_execute,
             ((), 32, 2048, 100, (), "float16", False, False, "batchmatmul_output"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),

            # mean
            ("test_resnet50_mean_000", mean_execute, ((32, 128, 7, 7, 16), "float32", (2, 3), True, "cce_mean"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_mean_001", mean_execute, ((32, 128, 7, 7, 16), "float16", (2, 3), True, "cce_mean"),
             ["level0", "rpc", "rpc_cloud"]),

            # meanad
            ("test_resnet50_mean_ad_000", mean_ad_run, ((32, 2048, 7, 7), "float32", (2, 3), True),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_mean_ad_001", mean_ad_run, ((32, 2048, 7, 7), "float16", (2, 3), True),
             ["level0", "rpc", "rpc_cloud"]),

            # Add
            ("test_resnet50_add_000", add_run, ([32, 128, 7, 7, 16], [32, 128, 7, 7, 16], "float32", "cce_add_fp32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_add_001", add_run,
             ([32, 16, 56, 56, 16], [32, 16, 56, 56, 16], "float32", "cce_add_fp32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_add_002", add_run,
             ([32, 32, 28, 28, 16], [32, 32, 28, 28, 16], "float32", "cce_add_fp32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_add_003", add_run,
             ([32, 64, 14, 14, 16], [32, 64, 14, 14, 16], "float32", "cce_add_fp32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_add_004", add_run, ([32, 128, 7, 7, 16], [32, 128, 7, 7, 16], "float16", "cce_add_fp16"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_add_005", add_run,
             ([32, 16, 56, 56, 16], [32, 16, 56, 56, 16], "float16", "cce_add_fp16"), ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_add_006", add_run,
             ([32, 32, 28, 28, 16], [32, 32, 28, 28, 16], "float16", "cce_add_fp16"), ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_add_007", add_run,
             ([32, 64, 14, 14, 16], [32, 64, 14, 14, 16], "float16", "cce_add_fp16"), ["level0", "rpc", "rpc_cloud"]),

            # AddN
            ("test_resnet50_addn_000", addn_execute, ((32, 128, 7, 7, 16), "float16", 2),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_addn_001", addn_execute, ((32, 16, 56, 56, 16), "float16", 2),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_addn_002", addn_execute, ((32, 32, 28, 28, 16), "float16", 2),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_addn_003", addn_execute, ((32, 4, 56, 56, 16), "float16", 2),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_addn_004", addn_execute, ((32, 64, 14, 14, 16), "float16", 2),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_addn_005", addn_execute, ((32, 128, 7, 7, 16), "float32", 2),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_addn_006", addn_execute, ((32, 16, 56, 56, 16), "float32", 2),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_addn_007", addn_execute, ((32, 32, 28, 28, 16), "float32", 2),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_addn_008", addn_execute, ((32, 64, 14, 14, 16), "float32", 2),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_addn_009", addn_execute, ((32, 4, 56, 56, 16), "float32", 2),
             ["level0", "rpc", "rpc_cloud"]),

            # sparse_softmax_cross_entropy_with_logits
            ("test_resnet50_sparse_softmax_cross_entropy_with_logits_000",
             sparse_softmax_cross_entropy_with_logits_run,
             [(32,), "int32", (32, 10), "float32", "mean", "sparse_softmax_cross_entropy_with_logits_fp32"],
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_sparse_softmax_cross_entropy_with_logits_001",
             sparse_softmax_cross_entropy_with_logits_run,
             [(32,), "int32", (32, 1001), "float32", "mean", "sparse_softmax_cross_entropy_with_logits_fp32"],
             ["level0", "rpc", "rpc_cloud"]),

            # sparse_softmax_cross_entropy_with_logits_ad
            ("test_resnet50_sparse_softmax_cross_entropy_with_logits_ad_000",
             sparse_softmax_cross_entropy_with_logits_ad_run,
             [(32,), "int32", (32, 10), "float32", "mean", "sparse_softmax_cross_entropy_with_logits_fp32"],
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_sparse_softmax_cross_entropy_with_logits_ad_001",
             sparse_softmax_cross_entropy_with_logits_ad_run,
             [(32,), "int32", (32, 1001), "float32", "mean", "sparse_softmax_cross_entropy_with_logits_fp32"],
             ["level0", "rpc", "rpc_cloud"]),

            # apply_momentum
            ("test_resnet50_apply_momentum_000", apply_momentum_run, ((10, 2048), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_apply_momentum_001", apply_momentum_run, ((10,), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_apply_momentum_002", apply_momentum_run, ((128, 32, 16, 16), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_apply_momentum_003", apply_momentum_run, ((144, 16, 16, 16), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_apply_momentum_004", apply_momentum_run, ((16, 32, 16, 16), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_apply_momentum_005", apply_momentum_run, ((16, 4, 16, 16), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_apply_momentum_006", apply_momentum_run, ((16, 64, 16, 16), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_apply_momentum_007", apply_momentum_run, ((16, 8, 16, 16), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_apply_momentum_008", apply_momentum_run, ((1, 128, 1, 1, 16), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_apply_momentum_009", apply_momentum_run, ((1, 16, 1, 1, 16), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_apply_momentum_010", apply_momentum_run, ((1, 32, 1, 1, 16), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_apply_momentum_011", apply_momentum_run, ((1, 4, 1, 1, 16), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_apply_momentum_012", apply_momentum_run, ((1, 64, 1, 1, 16), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_apply_momentum_013", apply_momentum_run, ((1, 8, 1, 1, 16), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_apply_momentum_014", apply_momentum_run, ((288, 32, 16, 16), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_apply_momentum_015", apply_momentum_run, ((32, 128, 16, 16), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_apply_momentum_016", apply_momentum_run, ((32, 16, 16, 16), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_apply_momentum_017", apply_momentum_run, ((32, 64, 16, 16), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_apply_momentum_018", apply_momentum_run, ((32, 8, 16, 16), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_apply_momentum_019", apply_momentum_run, ((36, 4, 16, 16), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_apply_momentum_020", apply_momentum_run, ((49, 4, 16, 16), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_apply_momentum_021", apply_momentum_run, ((4, 16, 16, 16), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_apply_momentum_022", apply_momentum_run, ((4, 4, 16, 16), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_apply_momentum_023", apply_momentum_run, ((64, 128, 16, 16), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_apply_momentum_024", apply_momentum_run, ((64, 16, 16, 16), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_apply_momentum_025", apply_momentum_run, ((64, 32, 16, 16), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_apply_momentum_026", apply_momentum_run, ((72, 8, 16, 16), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_apply_momentum_027", apply_momentum_run, ((8, 32, 16, 16), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_apply_momentum_028", apply_momentum_run, ((1001, 2048), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_apply_momentum_029", apply_momentum_run, ((1001,), "float32", False),
             ["level0", "rpc", "rpc_cloud"]),

            # bias_add
            ("test_resnet50_bias_add_000", bias_add_run, ([32, 10], "DefaultFormat", "float32"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_bias_add_001", bias_add_run, ([32, 1001], "DefaultFormat", "float32"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_bias_add_002", bias_add_run, ([32, 10], "DefaultFormat", "float16"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_bias_add_003", bias_add_run, ([32, 1001], "DefaultFormat", "float16"),
             ["level0", "rpc", "rpc_cloud"]),

            # BiasAddAd
            ("test_resnet50_bias_add_ad_000", bias_add_ad_run, ([32, 10], "DefaultFormat", "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_bias_add_ad_001", bias_add_ad_run, ([32, 1001], "DefaultFormat", "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_bias_add_ad_002", bias_add_ad_run, ([32, 10], "DefaultFormat", "float16"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_bias_add_ad_003", bias_add_ad_run, ([32, 1001], "DefaultFormat", "float16"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),

            # reshape
            ("test_resnet50_reshape_000", reshape_execute, [(32, 2048, 1, 1), (32, 2048), "float32"],
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_reshape_001", reshape_execute, [(32, 2048), (32, 2048, 1, 1), "float32"],
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_reshape_002", reshape_execute, [(32, 2048, 1, 1), (32, 2048), "float16"],
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_reshape_003", reshape_execute, [(32, 2048), (32, 2048, 1, 1), "float16"],
             ["level0", "rpc", "rpc_cloud"]),

            # four2five
            ("test_resnet50_four2five_000", "four2five_run", ([32, 3, 224, 224], "float32", "NCHW", "float16"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_four2five_001", "four2five_run", ([32, 2048, 7, 7], "float32", "NCHW", "float16"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_four2five_002", "four2five_run", ([32, 224, 224, 3], "float32", 'NHWC', "float16"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_four2five_003", "four2five_run", ([32, 3, 224, 224], "float16", "NCHW", "float16"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_four2five_004", "four2five_run", ([32, 2048, 7, 7], "float16", "NCHW", "float16"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_four2five_005", "four2five_run", ([32, 224, 224, 3], "float16", 'NHWC', "float16"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),

            # five2four
            ("test_resnet50_five2four_000", "five2four_run", ([32, 2048, 1, 1], "float16", "NCHW", "float16"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_five2four_001", "five2four_run", ([32, 2048, 1, 1], "float32", "NCHW", "float16"),
             ["level1", "rpc", "rpc_cloud", "Unavailable"]),

            # maxpool_grad_with_argmax
            ("test_resnet50_maxpool_grad_with_argmax_000", maxpool_grad_with_argmax_run,
             ((32, 4, 112, 112, 16), (3, 3), (2, 2), "SAME", "float32", False),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_maxpool_grad_with_argmax_001", maxpool_grad_with_argmax_run,
             ((32, 4, 112, 112, 16), (3, 3), (2, 2), "SAME", "float16", False),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),

            # softmax
            ("test_resnet50_softmax_001", softmax_execute, ((32, 10), "float16", -1, "softmax_16"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_softmax_002", softmax_execute, ((32, 10), "float32", -1, "softmax_32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_softmax_003", softmax_execute, ((32, 1001), "float16", -1, "softmax_16"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_softmax_004", softmax_execute, ((32, 1001), "float32", -1, "softmax_32"),
             ["level0", "rpc", "rpc_cloud"]),

            # argmax
            ("test_resnet50_argmax_001", argmax_run, ((32, 10), "float16", -1),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_argmax_002", argmax_run, ((32, 10), "float32", -1),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_argmax_003", argmax_run, ((32, 1001), "float16", -1),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_argmax_004", argmax_run, ((32, 1001), "float32", -1),
             ["level0", "rpc", "rpc_cloud"]),

            # EqualCount
            ("test_resnet50_equal_count_001", equal_count_run, (((32,), (32,)), "int32", "equal_count"),
             ["level0", "rpc", "rpc_cloud"]),

            # Clear_zero
            ("test_resnet50_equal_count_001", clear_zero_run, ((32, 4, 112, 112, 16), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_equal_count_002", clear_zero_run, ((32, 4, 112, 112, 16), "float16"),
             ["level0", "rpc", "rpc_cloud"]),
        ]

        self.test_args_conv = [
            # conv
            ("test_resnet50_conv_000", conv_run,
             ((32, 2048, 7, 7), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_001", conv_run,
             ((32, 256, 14, 14), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_002", conv_run,
             ((32, 256, 14, 14), (1024, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_003", conv_run,
             ((32, 256, 56, 56), (512, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_004", conv_run,
             ((32, 256, 56, 56), (64, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_005", conv_run,
             ((32, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_007", conv_run,
             ((32, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_008", conv_run,
             ((32, 512, 28, 28), (1024, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_009", conv_run,
             ((32, 512, 28, 28), (128, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_010", conv_run,
             ((32, 512, 7, 7), (512, 512, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_011", conv_run,
             ((32, 512, 7, 7), (2048, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_012", conv_run,
             ((32, 64, 56, 56), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_013", conv_run,
             ((32, 64, 56, 56), (256, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_014", conv_run,
             ((32, 64, 56, 56), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_015", conv_run,
             ((32, 1024, 14, 14), (2048, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_016", conv_run,
             ((32, 1024, 14, 14), (256, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_017", conv_run,
             ((32, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_018", conv_run,
             ((32, 128, 28, 28), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_019", conv_run,
             ((32, 128, 28, 28), (512, 128, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_020", conv_run,
             ((32, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_021", conv_run,
             ((32, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_022", conv_run,
             ((32, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_025", conv_run,
             ((32, 512, 14, 14), (512, 512, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),

            # conv_input_ad
            ("test_resnet50_conv_input_ad_000", conv_input_ad_run,
             ((32, 512, 7, 7), (2048, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_input_ad_002", conv_input_ad_run,
             ((32, 256, 14, 14), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_conv_input_ad_007", conv_input_ad_run,
             ((32, 128, 28, 28), (512, 128, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_conv_input_ad_008", conv_input_ad_run,
             ((32, 2048, 7, 7), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_conv_input_ad_009", conv_input_ad_run,
             ((32, 512, 7, 7), (512, 512, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_conv_input_ad_013", conv_input_ad_run,
             ((32, 64, 56, 56), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_conv_input_ad_014", conv_input_ad_run,
             ((32, 256, 14, 14), (1024, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_conv_input_ad_018", conv_input_ad_run,
             ((32, 128, 28, 28), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),

            # conv_filter_ad
            ("test_resnet50_conv_filter_ad_000", conv_filter_ad_run,
             ((32, 256, 14, 14), (1024, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_filter_ad_002", conv_filter_ad_run,
             ((32, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_filter_ad_003", conv_filter_ad_run,
             ((32, 512, 7, 7), (2048, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_filter_ad_004", conv_filter_ad_run,
             ((32, 1024, 14, 14), (256, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_filter_ad_005", conv_filter_ad_run,
             ((32, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_filter_ad_006", conv_filter_ad_run,
             ((32, 64, 56, 56), (256, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_filter_ad_007", conv_filter_ad_run,
             ((32, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_filter_ad_008", conv_filter_ad_run,
             ((32, 128, 28, 28), (512, 128, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_filter_ad_009", conv_filter_ad_run,
             ((32, 2048, 7, 7), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_filter_ad_010", conv_filter_ad_run,
             ((32, 256, 56, 56), (64, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_filter_ad_011", conv_filter_ad_run,
             ((32, 64, 56, 56), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_filter_ad_012", conv_filter_ad_run,
             ((32, 128, 28, 28), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_filter_ad_013", conv_filter_ad_run,
             ((32, 256, 14, 14), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_filter_ad_014", conv_filter_ad_run,
             ((32, 512, 7, 7), (512, 512, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_filter_ad_015", conv_filter_ad_run,
             ((32, 64, 56, 56), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_filter_ad_016", conv_filter_ad_run,
             ((32, 512, 28, 28), (1024, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1)),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_filter_ad_017", conv_filter_ad_run,
             ((32, 1024, 14, 14), (2048, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1)),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_filter_ad_018", conv_filter_ad_run,
             ((32, 256, 56, 56), (512, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1)),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_filter_ad_020", conv_filter_ad_run,
             ((32, 256, 28, 28), (256, 256, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1)),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_filter_ad_021", conv_filter_ad_run,
             ((32, 512, 14, 14), (512, 512, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1)),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_filter_ad_023", conv_filter_ad_run,
             ((32, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1)),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_filter_ad_024", conv_filter_ad_run,
             ((32, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1)),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_filter_ad_025", conv_filter_ad_run,
             ((32, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1)),
             ["level0", "rpc", "rpc_cloud"]),

            # conv_bn1
            ("test_resnet50_conv_bn1_000", conv_bn1_run,
             ((32, 2048, 7, 7), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_bn1_001", conv_bn1_run,
             ((32, 256, 14, 14), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_bn1_002", conv_bn1_run,
             ((32, 256, 14, 14), (1024, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_bn1_003", conv_bn1_run,
             ((32, 256, 56, 56), (512, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_bn1_004", conv_bn1_run,
             ((32, 256, 56, 56), (64, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_bn1_005", conv_bn1_run,
             ((32, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_bn1_007", conv_bn1_run,
             ((32, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_bn1_008", conv_bn1_run,
             ((32, 512, 28, 28), (1024, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_bn1_009", conv_bn1_run,
             ((32, 512, 28, 28), (128, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_bn1_010", conv_bn1_run,
             ((32, 512, 7, 7), (512, 512, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_bn1_011", conv_bn1_run,
             ((32, 512, 7, 7), (2048, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_bn1_012", conv_bn1_run,
             ((32, 64, 56, 56), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_bn1_013", conv_bn1_run,
             ((32, 64, 56, 56), (256, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_bn1_014", conv_bn1_run,
             ((32, 64, 56, 56), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_bn1_015", conv_bn1_run,
             ((32, 1024, 14, 14), (2048, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_bn1_016", conv_bn1_run,
             ((32, 1024, 14, 14), (256, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_bn1_017", conv_bn1_run,
             ((32, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_bn1_018", conv_bn1_run,
             ((32, 128, 28, 28), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_bn1_019", conv_bn1_run,
             ((32, 128, 28, 28), (512, 128, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_bn1_020", conv_bn1_run,
             ((32, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_bn1_021", conv_bn1_run,
             ((32, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_bn1_022", conv_bn1_run,
             ((32, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_bn1_024", conv_bn1_run,
             ((32, 256, 28, 28), (256, 256, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_bn1_025", conv_bn1_run,
             ((32, 512, 14, 14), (512, 512, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
        ]

        self.test_args_relu = [
            # relu
            ("test_resnet50_relu_000", relu_run, ((32, 128, 7, 7, 16), "float32", 1e-5),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_001", relu_run, ((32, 16, 14, 14, 16), "float32", 1e-5),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_002", relu_run, ((32, 16, 56, 56, 16), "float32", 1e-5),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_003", relu_run, ((32, 32, 28, 28, 16), "float32", 1e-5),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_004", relu_run, ((32, 32, 7, 7, 16), "float32", 1e-5),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_005", relu_run, ((32, 4, 112, 112, 16), "float32", 1e-5),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_006", relu_run, ((32, 4, 56, 56, 16), "float32", 1e-5),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_007", relu_run, ((32, 64, 14, 14, 16), "float32", 1e-5),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_008", relu_run, ((32, 8, 28, 28, 16), "float32", 1e-5),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_009", relu_run, ((32, 8, 56, 56, 16), "float32", 1e-5),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_010", relu_run, ((32, 16, 28, 28, 16), "float32", 1e-5),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_011", relu_run, ((32, 32, 14, 14, 16), "float32", 1e-5),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_012", relu_run, ((32, 128, 7, 7, 16), "float16", 1e-5),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_013", relu_run, ((32, 16, 14, 14, 16), "float16", 1e-5),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_014", relu_run, ((32, 16, 56, 56, 16), "float16", 1e-5),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_015", relu_run, ((32, 32, 28, 28, 16), "float16", 1e-5),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_016", relu_run, ((32, 32, 7, 7, 16), "float16", 1e-5),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_017", relu_run, ((32, 4, 112, 112, 16), "float16", 1e-5),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_018", relu_run, ((32, 4, 56, 56, 16), "float16", 1e-5),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_019", relu_run, ((32, 64, 14, 14, 16), "float16", 1e-5),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_020", relu_run, ((32, 8, 28, 28, 16), "float16", 1e-5),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_021", relu_run, ((32, 8, 56, 56, 16), "float16", 1e-5),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_022", relu_run, ((32, 16, 28, 28, 16), "float16", 1e-5),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_023", relu_run, ((32, 32, 14, 14, 16), "float16", 1e-5),
             ["level0", "rpc", "rpc_cloud"]),

            # relu_ad
            ("test_resnet50_relu_ad_000", relu_ad_run, ((32, 128, 7, 7, 16), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_ad_001", relu_ad_run, ((32, 16, 14, 14, 16), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_ad_002", relu_ad_run, ((32, 16, 56, 56, 16), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_ad_003", relu_ad_run, ((32, 32, 28, 28, 16), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_ad_004", relu_ad_run, ((32, 32, 7, 7, 16), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_ad_005", relu_ad_run, ((32, 4, 112, 112, 16), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_ad_006", relu_ad_run, ((32, 4, 56, 56, 16), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_ad_007", relu_ad_run, ((32, 64, 14, 14, 16), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_ad_008", relu_ad_run, ((32, 8, 28, 28, 16), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_ad_009", relu_ad_run, ((32, 8, 56, 56, 16), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_ad_010", relu_ad_run, ((32, 16, 28, 28, 16), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_ad_011", relu_ad_run, ((32, 32, 14, 14, 16), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_relu_ad_012", relu_ad_run, ((32, 128, 7, 7, 16), "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_relu_ad_013", relu_ad_run, ((32, 16, 14, 14, 16), "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_relu_ad_014", relu_ad_run, ((32, 16, 56, 56, 16), "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_relu_ad_015", relu_ad_run, ((32, 32, 28, 28, 16), "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_relu_ad_016", relu_ad_run, ((32, 32, 7, 7, 16), "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_relu_ad_017", relu_ad_run, ((32, 4, 112, 112, 16), "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_relu_ad_018", relu_ad_run, ((32, 4, 56, 56, 16), "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_relu_ad_019", relu_ad_run, ((32, 64, 14, 14, 16), "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_relu_ad_020", relu_ad_run, ((32, 8, 28, 28, 16), "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_relu_ad_021", relu_ad_run, ((32, 8, 56, 56, 16), "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_relu_ad_022", relu_ad_run, ((32, 16, 28, 28, 16), "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_relu_ad_023", relu_ad_run, ((32, 32, 14, 14, 16), "float16"),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
        ]

        self.test_args_bn = [
            # bn_split
            ("test_resnet50_bn_split_000", bn_split_run,
             ((32, 128, 7, 7, 16), "float32", 0.1, 1e-4, "resnet50_bn_split_000"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_bn_split_001", bn_split_run,
             ((32, 16, 14, 14, 16), "float32", 0.1, 1e-4, "resnet50_bn_split_001"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_bn_split_002", bn_split_run,
             ((32, 16, 56, 56, 16), "float32", 0.1, 1e-4, "resnet50_bn_split_002"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_bn_split_003", bn_split_run,
             ((32, 32, 28, 28, 16), "float32", 0.1, 1e-4, "resnet50_bn_split_003"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_bn_split_004", bn_split_run,
             ((32, 32, 7, 7, 16), "float32", 0.1, 1e-4, "resnet50_bn_split_004"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_bn_split_005", bn_split_run,
             ((32, 4, 112, 112, 16), "float32", 0.1, 1e-4, "resnet50_bn_split_005"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_bn_split_006", bn_split_run,
             ((32, 4, 56, 56, 16), "float32", 0.1, 1e-4, "resnet50_bn_split_006"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_bn_split_007", bn_split_run,
             ((32, 64, 14, 14, 16), "float32", 0.1, 1e-4, "resnet50_bn_split_007"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_bn_split_008", bn_split_run,
             ((32, 8, 28, 28, 16), "float32", 0.1, 1e-4, "resnet50_bn_split_008"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_bn_split_009", bn_split_run,
             ((32, 8, 56, 56, 16), "float32", 0.1, 1e-4, "resnet50_bn_split_009"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_bn_split_010", bn_split_run,
             ((32, 16, 28, 28, 16), "float32", 0.1, 1e-4, "resnet50_bn_split_010"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_bn_split_011", bn_split_run,
             ((32, 32, 14, 14, 16), "float32", 0.1, 1e-4, "resnet50_bn_split_011"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_bn_split_012", bn_split_run,
             ((32, 128, 7, 7, 16), "float16", 0.1, 1e-4, "resnet50_bn_split_012"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_bn_split_013", bn_split_run,
             ((32, 16, 14, 14, 16), "float16", 0.1, 1e-4, "resnet50_bn_split_013"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_bn_split_014", bn_split_run,
             ((32, 16, 56, 56, 16), "float16", 0.1, 1e-4, "resnet50_bn_split_014"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_bn_split_015", bn_split_run,
             ((32, 32, 28, 28, 16), "float16", 0.1, 1e-4, "resnet50_bn_split_015"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_bn_split_016", bn_split_run,
             ((32, 32, 7, 7, 16), "float16", 0.1, 1e-4, "resnet50_bn_split_016"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_bn_split_017", bn_split_run,
             ((32, 4, 112, 112, 16), "float16", 0.1, 1e-4, "resnet50_bn_split_017"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_bn_split_018", bn_split_run,
             ((32, 4, 56, 56, 16), "float16", 0.1, 1e-4, "resnet50_bn_split_018"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_bn_split_019", bn_split_run,
             ((32, 64, 14, 14, 16), "float16", 0.1, 1e-4, "resnet50_bn_split_019"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_bn_split_020", bn_split_run,
             ((32, 8, 28, 28, 16), "float16", 0.1, 1e-4, "resnet50_bn_split_020"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_bn_split_021", bn_split_run,
             ((32, 8, 56, 56, 16), "float16", 0.1, 1e-4, "resnet50_bn_split_021"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_bn_split_022", bn_split_run,
             ((32, 16, 28, 28, 16), "float16", 0.1, 1e-4, "resnet50_bn_split_022"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_bn_split_023", bn_split_run,
             ((32, 32, 14, 14, 16), "float16", 0.1, 1e-4, "resnet50_bn_split_023"),
             ["level0", "rpc", "rpc_cloud"]),

            # fused_bn_grad_5D_all
            ("test_resnet50_fused_bn_grad_5D_all_000", fused_bn_grad_5D_all_run,
             ((32, 128, 7, 7, 16), "float32", 1e-4, "resnet50_fused_bn_grad_5D_all_000"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_fused_bn_grad_5D_all_001", fused_bn_grad_5D_all_run,
             ((32, 16, 14, 14, 16), "float32", 1e-4, "resnet50_fused_bn_grad_5D_all_001"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_fused_bn_grad_5D_all_002", fused_bn_grad_5D_all_run,
             ((32, 16, 56, 56, 16), "float32", 1e-4, "resnet50_fused_bn_grad_5D_all_002"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_fused_bn_grad_5D_all_003", fused_bn_grad_5D_all_run,
             ((32, 32, 28, 28, 16), "float32", 1e-4, "resnet50_fused_bn_grad_5D_all_003"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_fused_bn_grad_5D_all_004", fused_bn_grad_5D_all_run,
             ((32, 32, 7, 7, 16), "float32", 1e-4, "resnet50_fused_bn_grad_5D_all_004"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_fused_bn_grad_5D_all_005", fused_bn_grad_5D_all_run,
             ((32, 4, 112, 112, 16), "float32", 1e-4, "resnet50_fused_bn_grad_5D_all_005"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_fused_bn_grad_5D_all_006", fused_bn_grad_5D_all_run,
             ((32, 4, 56, 56, 16), "float32", 1e-4, "resnet50_fused_bn_grad_5D_all_006"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_fused_bn_grad_5D_all_007", fused_bn_grad_5D_all_run,
             ((32, 64, 14, 14, 16), "float32", 1e-4, "resnet50_fused_bn_grad_5D_all_007"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_fused_bn_grad_5D_all_008", fused_bn_grad_5D_all_run,
             ((32, 8, 28, 28, 16), "float32", 1e-4, "resnet50_fused_bn_grad_5D_all_008"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_fused_bn_grad_5D_all_009", fused_bn_grad_5D_all_run,
             ((32, 8, 56, 56, 16), "float32", 1e-4, "resnet50_fused_bn_grad_5D_all_009"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_fused_bn_grad_5D_all_010", fused_bn_grad_5D_all_run,
             ((32, 16, 28, 28, 16), "float32", 1e-4, "resnet50_fused_bn_grad_5D_all_010"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_fused_bn_grad_5D_all_011", fused_bn_grad_5D_all_run,
             ((32, 32, 14, 14, 16), "float32", 1e-4, "resnet50_fused_bn_grad_5D_all_011"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_fused_bn_grad_5D_all_012", fused_bn_grad_5D_all_run,
             ((32, 128, 7, 7, 16), "float16", 1e-4, "resnet50_fused_bn_grad_5D_all_012"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_fused_bn_grad_5D_all_013", fused_bn_grad_5D_all_run,
             ((32, 16, 14, 14, 16), "float16", 1e-4, "resnet50_fused_bn_grad_5D_all_013"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_fused_bn_grad_5D_all_014", fused_bn_grad_5D_all_run,
             ((32, 16, 56, 56, 16), "float16", 1e-4, "resnet50_fused_bn_grad_5D_all_014"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_fused_bn_grad_5D_all_015", fused_bn_grad_5D_all_run,
             ((32, 32, 28, 28, 16), "float16", 1e-4, "resnet50_fused_bn_grad_5D_all_015"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_fused_bn_grad_5D_all_016", fused_bn_grad_5D_all_run,
             ((32, 32, 7, 7, 16), "float16", 1e-4, "resnet50_fused_bn_grad_5D_all_016"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_fused_bn_grad_5D_all_017", fused_bn_grad_5D_all_run,
             ((32, 4, 112, 112, 16), "float16", 1e-4, "resnet50_fused_bn_grad_5D_all_017"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_fused_bn_grad_5D_all_018", fused_bn_grad_5D_all_run,
             ((32, 4, 56, 56, 16), "float16", 1e-4, "resnet50_fused_bn_grad_5D_all_018"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_fused_bn_grad_5D_all_019", fused_bn_grad_5D_all_run,
             ((32, 64, 14, 14, 16), "float16", 1e-4, "resnet50_fused_bn_grad_5D_all_019"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_fused_bn_grad_5D_all_020", fused_bn_grad_5D_all_run,
             ((32, 8, 28, 28, 16), "float16", 1e-4, "resnet50_fused_bn_grad_5D_all_020"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_fused_bn_grad_5D_all_021", fused_bn_grad_5D_all_run,
             ((32, 8, 56, 56, 16), "float16", 1e-4, "resnet50_fused_bn_grad_5D_all_021"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_fused_bn_grad_5D_all_022", fused_bn_grad_5D_all_run,
             ((32, 16, 28, 28, 16), "float16", 1e-4, "resnet50_fused_bn_grad_5D_all_022"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_fused_bn_grad_5D_all_023", fused_bn_grad_5D_all_run,
             ((32, 32, 14, 14, 16), "float16", 1e-4, "resnet50_fused_bn_grad_5D_all_023"),
             ["level0", "rpc", "rpc_cloud"]),
        ]

        self.test_args_single = [
            ("test_resnet50_conv_006", conv_run,
             ((32, 3, 224, 224), (64, 3, 7, 7), (2, 3, 2, 3), (2, 2), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_023", conv_run,
             ((32, 128, 56, 56), (128, 128, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_024", conv_run,
             ((32, 256, 28, 28), (256, 256, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_input_ad_001", conv_input_ad_run,
             ((32, 1024, 14, 14), (2048, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_conv_input_ad_003", conv_input_ad_run,
             ((32, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_conv_input_ad_004", conv_input_ad_run,
             ((32, 1024, 14, 14), (256, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_conv_input_ad_005", conv_input_ad_run,
             ((32, 64, 56, 56), (256, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_conv_input_ad_006", conv_input_ad_run,
             ((32, 256, 56, 56), (512, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_conv_input_ad_010", conv_input_ad_run,
             ((32, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_conv_input_ad_011", conv_input_ad_run,
             ((32, 256, 56, 56), (64, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_conv_input_ad_012", conv_input_ad_run,
             ((32, 64, 56, 56), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_conv_input_ad_015", conv_input_ad_run,
             ((32, 512, 28, 28), (1024, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_conv_input_ad_016", conv_input_ad_run,
             ((32, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_conv_input_ad_017", conv_input_ad_run,
             ((32, 512, 28, 28), (128, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_conv_input_ad_019", conv_input_ad_run,
             ((32, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_conv_input_ad_020", conv_input_ad_run,
             ((32, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_conv_input_ad_021", conv_input_ad_run,
             ((32, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_conv_input_ad_022", conv_input_ad_run,
             ((32, 128, 56, 56), (128, 128, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_conv_input_ad_023", conv_input_ad_run,
             ((32, 256, 28, 28), (256, 256, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_conv_input_ad_024", conv_input_ad_run,
             ((32, 512, 14, 14), (512, 512, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_conv_filter_ad_019", conv_filter_ad_run,
             ((32, 128, 56, 56), (128, 128, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1)),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_filter_ad_022", conv_filter_ad_run,
             ((32, 3, 224, 224), (64, 3, 7, 7), (2, 3, 2, 3), (2, 2), (1, 1)),
             ["level0", "rpc", "rpc_cloud", "Unavailable"]),
            ("test_resnet50_conv_bn1_006", conv_bn1_run,
             ((32, 3, 224, 224), (64, 3, 7, 7), (2, 3, 2, 3), (2, 2), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_resnet50_conv_bn1_023", conv_bn1_run,
             ((32, 128, 56, 56), (128, 128, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1), False),
             ["level0", "rpc", "rpc_cloud"]),
        ]

    def get_cases(self, split_nums, split_idx):
        cases = []
        cases.extend(get_splitted_cases(self.test_args, split_nums, split_idx))
        cases.extend(get_splitted_cases(self.test_args_conv, split_nums, split_idx))
        cases.extend(get_splitted_cases(self.test_args_relu, split_nums, split_idx))
        cases.extend(get_splitted_cases(self.test_args_bn, split_nums, split_idx))

        return cases

    def test(self, split_nums, split_idx):
        return self.run_test_arg_func(self.get_cases(split_nums, split_idx), "level0")

    def test_single(self, split_nums, split_idx):
        return self.run_test_arg_func(get_splitted_cases(self.test_args_single, split_nums, split_idx), "level0")

    def teardown(self):
        self._log.info("{0} Teardown".format(self.casename))
        super(TestResnet50_001, self).teardown()
        return


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test0():
    a = TestResnet50_001()
    a.setup()
    a.test(20, 0)
    a.teardown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test1():
    a = TestResnet50_001()
    a.setup()
    a.test(20, 1)
    a.teardown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test2():
    a = TestResnet50_001()
    a.setup()
    a.test(20, 2)
    a.teardown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test3():
    a = TestResnet50_001()
    a.setup()
    a.test(20, 3)
    a.teardown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test4():
    a = TestResnet50_001()
    a.setup()
    a.test(20, 4)
    a.teardown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test5():
    a = TestResnet50_001()
    a.setup()
    a.test(20, 5)
    a.teardown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test6():
    a = TestResnet50_001()
    a.setup()
    a.test(20, 6)
    a.teardown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test7():
    a = TestResnet50_001()
    a.setup()
    a.test(20, 7)
    a.teardown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test8():
    a = TestResnet50_001()
    a.setup()
    a.test(20, 8)
    a.teardown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test9():
    a = TestResnet50_001()
    a.setup()
    a.test(20, 9)
    a.teardown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test10():
    a = TestResnet50_001()
    a.setup()
    a.test(20, 10)
    a.teardown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test11():
    a = TestResnet50_001()
    a.setup()
    a.test(20, 11)
    a.teardown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test12():
    a = TestResnet50_001()
    a.setup()
    a.test(20, 12)
    a.teardown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test13():
    a = TestResnet50_001()
    a.setup()
    a.test(20, 13)
    a.teardown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test14():
    a = TestResnet50_001()
    a.setup()
    a.test(20, 14)
    a.teardown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test15():
    a = TestResnet50_001()
    a.setup()
    a.test(20, 15)
    a.teardown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test16():
    a = TestResnet50_001()
    a.setup()
    a.test(20, 16)
    a.teardown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test17():
    a = TestResnet50_001()
    a.setup()
    a.test(20, 17)
    a.teardown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test18():
    a = TestResnet50_001()
    a.setup()
    a.test(20, 18)
    a.teardown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test19():
    a = TestResnet50_001()
    a.setup()
    a.test(20, 19)
    a.teardown()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single0():
    a = TestResnet50_001()
    a.setup()
    a.test_single(24, 0)
    a.teardown()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single1():
    a = TestResnet50_001()
    a.setup()
    a.test_single(24, 1)
    a.teardown()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single2():
    a = TestResnet50_001()
    a.setup()
    a.test_single(24, 2)
    a.teardown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single3():
    a = TestResnet50_001()
    a.setup()
    a.test_single(24, 3)
    a.teardown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single4():
    a = TestResnet50_001()
    a.setup()
    a.test_single(24, 4)
    a.teardown()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single5():
    a = TestResnet50_001()
    a.setup()
    a.test_single(24, 5)
    a.teardown()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single6():
    a = TestResnet50_001()
    a.setup()
    a.test_single(24, 6)
    a.teardown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single7():
    a = TestResnet50_001()
    a.setup()
    a.test_single(24, 7)
    a.teardown()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single8():
    a = TestResnet50_001()
    a.setup()
    a.test_single(24, 8)
    a.teardown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single9():
    a = TestResnet50_001()
    a.setup()
    a.test_single(24, 9)
    a.teardown()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single10():
    a = TestResnet50_001()
    a.setup()
    a.test_single(24, 10)
    a.teardown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single11():
    a = TestResnet50_001()
    a.setup()
    a.test_single(24, 11)
    a.teardown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single12():
    a = TestResnet50_001()
    a.setup()
    a.test_single(24, 12)
    a.teardown()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single13():
    a = TestResnet50_001()
    a.setup()
    a.test_single(24, 13)
    a.teardown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single14():
    a = TestResnet50_001()
    a.setup()
    a.test_single(24, 14)
    a.teardown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single15():
    a = TestResnet50_001()
    a.setup()
    a.test_single(24, 15)
    a.teardown()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single16():
    a = TestResnet50_001()
    a.setup()
    a.test_single(24, 16)
    a.teardown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single17():
    a = TestResnet50_001()
    a.setup()
    a.test_single(24, 17)
    a.teardown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single18():
    a = TestResnet50_001()
    a.setup()
    a.test_single(24, 18)
    a.teardown()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single19():
    a = TestResnet50_001()
    a.setup()
    a.test_single(24, 19)
    a.teardown()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single20():
    a = TestResnet50_001()
    a.setup()
    a.test_single(24, 20)
    a.teardown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single21():
    a = TestResnet50_001()
    a.setup()
    a.test_single(24, 21)
    a.teardown()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single22():
    a = TestResnet50_001()
    a.setup()
    a.test_single(24, 22)
    a.teardown()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single23():
    a = TestResnet50_001()
    a.setup()
    a.test_single(24, 23)
    a.teardown()
