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

"""unsortedsegmentsum test cast"""
import os
import pytest
from tests.common.base import TestBase


class TestCase(TestBase):

    def setup(self):
        case_name = "test_conv_bn_fusion_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {} Setup case============".format(self.casename))
        self.testarg = [
            #(testflag, opfuncname, case_num, ((in_n, in_c, in_h, in_w),
            #    (cout, in_c, w_h, w_w), dtype, (p_left, p_right, p_top, p_bottom),
            #    (s_h, s_w), (d_h, d_w), bias, momentem, eps,
            #    bypass_l1 , dump_data, [cutH, cutCo, cutM, cutK, cutN]))
        ]
        self.testarg_rpc_cloud= [
            #  ("conv_bn_fusion_01", conv_bn_fusion_run, (
            #      (32, 1024, 14, 14), (2048, 1024, 1, 1), "float16",
            #      (0, 0, 0, 0), (2, 2), (1, 1), False, 0.3, 1e-3)),
            #  ("conv_bn_fusion_02", conv_bn_fusion_run, (
            #      (32, 1024, 14, 14), (256, 1024, 1, 1), "float16",
            #      (0, 0, 0, 0), (1, 1), (1, 1), False, 0.3, 1e-3)),
            #  ("conv_bn_fusion_03", conv_bn_fusion_run, (
            #      (32, 1024, 14, 14), (512, 1024, 1, 1), "float16",
            #      (0, 0, 0, 0), (1, 1), (1, 1), False, 0.3, 1e-3)),
            #  ("conv_bn_fusion_04", conv_bn_fusion_run, (
            #      (32, 1024, 14, 14), (512, 1024, 1, 1), "float16",
            #      (0, 0, 0, 0), (2, 2), (1, 1), False, 0.3, 1e-3)),
            #  ("conv_bn_fusion_05", conv_bn_fusion_run, (
            #      (32, 128, 28, 28), (128, 128, 3, 3), "float16",
            #      (1, 1, 1, 1), (1, 1), (1, 1), False, 0.3, 1e-3)),
            #  ("conv_bn_fusion_06", conv_bn_fusion_run, (
            #      (32, 128, 28, 28), (512, 128, 1, 1), "float16",
            #      (0, 0, 0, 0), (1, 1), (1, 1), False, 0.3, 1e-3)),
            #  ("conv_bn_fusion_07", conv_bn_fusion_run, (
            #      (32, 128, 56, 56), (128, 128, 3, 3), "float16",
            #      (0, 1, 0, 1), (2, 2), (1, 1), False, 0.3, 1e-3)),
            #  ("conv_bn_fusion_08", conv_bn_fusion_run, (
            #      (32, 2048, 7, 7), (512, 2048, 1, 1), "float16",
            #      (0, 0, 0, 0), (1, 1), (1, 1), False, 0.3, 1e-3)),
            #  ("conv_bn_fusion_09", conv_bn_fusion_run, (
            #      (32, 256, 14, 14), (1024, 256, 1, 1), "float16",
            #      (0, 0, 0, 0), (1, 1), (1, 1), False, 0.3, 1e-3)),
            #  ("conv_bn_fusion_10", conv_bn_fusion_run, (
            #      (32, 256, 14, 14), (256, 256, 3, 3), "float16",
            #      (1, 1, 1, 1), (1, 1), (1, 1), False, 0.3, 1e-3)),
            #  ("conv_bn_fusion_11", conv_bn_fusion_run, (
            #      (32, 256, 28, 28), (256, 256, 3, 3), "float16",
            #      (0, 1, 0, 1), (2, 2), (1, 1), False, 0.3, 1e-3)),
            #  ("conv_bn_fusion_12", conv_bn_fusion_run, (
            #      (32, 256, 56, 56), (128, 256, 1, 1), "float16",
            #      (0, 0, 0, 0), (1, 1), (1, 1), False, 0.3, 1e-3)),
            #  ("conv_bn_fusion_13", conv_bn_fusion_run, (
            #      (32, 256, 56, 56), (128, 256, 1, 1), "float16",
            #      (0, 0, 0, 0), (2, 2), (1, 1), False, 0.3, 1e-3)),
            #  ("conv_bn_fusion_14", conv_bn_fusion_run, (
            #      (32, 256, 56, 56), (512, 256, 1, 1), "float16",
            #      (0, 0, 0, 0), (2, 2), (1, 1), False, 0.3, 1e-3)),
            #  ("conv_bn_fusion_15", conv_bn_fusion_run, (
            #      (32, 256, 56, 56), (64, 256, 1, 1), "float16",
            #      (0, 0, 0, 0), (1, 1), (1, 1), False, 0.3, 1e-3)),
            #  ("conv_bn_fusion_16", conv_bn_fusion_run, (
            #      (32, 3, 224, 224), (64, 3, 7, 7), "float16",
            #      (2, 3, 2, 3), (2, 2), (1, 1), False, 0.3, 1e-3)),
            #  ("conv_bn_fusion_17", conv_bn_fusion_run, (
            #      (32, 512, 14, 14), (512, 512, 3, 3), "float16",
            #      (0, 1, 0, 1), (2, 2), (1, 1), False, 0.3, 1e-3)),
            #  ("conv_bn_fusion_18", conv_bn_fusion_run, (
            #      (32, 512, 28, 28), (1024, 512, 1, 1), "float16",
            #      (0, 0, 0, 0), (2, 2), (1, 1), False, 0.3, 1e-3)),
            #  ("conv_bn_fusion_19", conv_bn_fusion_run, (
            #      (32, 512, 28, 28), (128, 512, 1, 1), "float16",
            #      (0, 0, 0, 0), (1, 1), (1, 1), False, 0.3, 1e-3)),
            #  ("conv_bn_fusion_20", conv_bn_fusion_run, (
            #      (32, 512, 28, 28), (256, 512, 1, 1), "float16",
            #      (0, 0, 0, 0), (1, 1), (1, 1), False, 0.3, 1e-3)),
            #  ("conv_bn_fusion_21", conv_bn_fusion_run, (
            #      (32, 512, 28, 28), (256, 512, 1, 1), "float16",
            #      (0, 0, 0, 0), (2, 2), (1, 1), False, 0.3, 1e-3)),
            #  ("conv_bn_fusion_22", conv_bn_fusion_run, (
            #      (32, 512, 7, 7), (2048, 512, 1, 1), "float16",
            #      (0, 0, 0, 0), (1, 1), (1, 1), False, 0.3, 1e-3)),
            #  ("conv_bn_fusion_23", conv_bn_fusion_run, (
            #      (32, 512, 7, 7), (512, 512, 3, 3), "float16",
            #      (1, 1, 1, 1), (1, 1), (1, 1), False, 0.3, 1e-3)),
            #  ("conv_bn_fusion_24", conv_bn_fusion_run, (
            #      (32, 64, 56, 56), (256, 64, 1, 1), "float16",
            #      (0, 0, 0, 0), (1, 1), (1, 1), False, 0.3, 1e-3)),
            #  ("conv_bn_fusion_25", conv_bn_fusion_run, (
            #      (32, 64, 56, 56), (64, 64, 1, 1), "float16",
            #      (0, 0, 0, 0), (1, 1), (1, 1), False, 0.3, 1e-3)),
            #  ("conv_bn_fusion_26", conv_bn_fusion_run, (
            #      (32, 64, 56, 56), (64, 64, 3, 3), "float16",
            #      (1, 1, 1, 1), (1, 1), (1, 1), False, 0.3, 1e-3)),
        ]
        self.testarg_level1 = [
        ]

        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg, is_conv=True)

    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    def test_run_level1(self):
        self.common_run(self.testarg_level1, is_conv=True)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {} Teardown============".format(self.casename))
