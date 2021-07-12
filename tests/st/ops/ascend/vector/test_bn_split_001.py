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

""" test splited BN """
import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.bn_split_run import bn_split_run
from tests.common.test_run.bn_split_run import bn_1_run, bn_2_run, bn_3_run


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_bn1_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============"
                       "".format(self.casename))
        self.testarg = [
            # testflag, opfuncname, testRunArgs:
            # shape, dtype, momentum, eps, kernel_name, dimArgs
        ]

        self.testarg_l1 = [
        ]

        self.testarg_aic = [
        ]

        self.testarg_cloud = [
            # resnet50 V1.0 shapes
            # float16
            ("r50_v1.0_splited_BN_fp16_01", bn_split_run,
             ((32, 4, 112, 112, 16), "float16", 0.1, 1e-4, "r50_sBN_fp16_01")),
            ("r50_v1.0_splited_BN_fp16_02", bn_split_run,
             ((32, 4, 56, 56, 16), "float16", 0.1, 1e-4, "r50_sBN_fp16_02")),
            ("r50_v1.0_splited_BN_fp16_03", bn_split_run,
             ((32, 16, 56, 56, 16), "float16", 0.1, 1e-4, "r50_sBN_fp16_03")),
            ("r50_v1.0_splited_BN_fp16_04", bn_split_run,
             ((32, 8, 28, 28, 16), "float16", 0.1, 1e-4, "r50_sBN_fp16_04")),
            ("r50_v1.0_splited_BN_fp16_05", bn_split_run,
             ((32, 32, 28, 28, 16), "float16", 0.1, 1e-4, "r50_sBN_fp16_05")),
            ("r50_v1.0_splited_BN_fp16_06", bn_split_run,
             ((32, 16, 14, 14, 16), "float16", 0.1, 1e-4, "r50_sBN_fp16_06")),
            ("r50_v1.0_splited_BN_fp16_07", bn_split_run,
             ((32, 64, 14, 14, 16), "float16", 0.1, 1e-4, "r50_sBN_fp16_07")),
            ("r50_v1.0_splited_BN_fp16_08", bn_split_run,
             ((32, 32, 7, 7, 16), "float16", 0.1, 1e-4, "r50_sBN_fp16_08")),
            ("r50_v1.0_splited_BN_fp16_09", bn_split_run,
             ((32, 128, 7, 7, 16), "float16", 0.1, 1e-4, "r50_sBN_fp16_09")),

            ("r50_v1.5_splited_BN_fp16_01", bn_split_run,
             ((32, 8, 56, 56, 16), "float16", 0.1, 1e-4, "r50_sBN_fp32_09")),
            ("r50_v1.5_splited_BN_fp16_02", bn_split_run,
             ((32, 16, 28, 28, 16), "float16", 0.1, 1e-4, "r50_sBN_fp32_09")),
            ("r50_v1.5_splited_BN_fp16_03", bn_split_run,
             ((32, 32, 14, 14, 16), "float16", 0.1, 1e-4, "r50_sBN_fp32_09")),

            # float32
            ("r50_v1.0_splited_BN_fp32_01", bn_split_run,
             ((32, 4, 112, 112, 16), "float32", 0.1, 1e-4, "r50_sBN_fp32_01")),
            ("r50_v1.0_splited_BN_fp32_02", bn_split_run,
             ((32, 4, 56, 56, 16), "float32", 0.1, 1e-4, "r50_sBN_fp32_02")),
            ("r50_v1.0_splited_BN_fp32_03", bn_split_run,
             ((32, 16, 56, 56, 16), "float32", 0.1, 1e-4, "r50_sBN_fp32_03")),
            ("r50_v1.0_splited_BN_fp32_04", bn_split_run,
             ((32, 8, 28, 28, 16), "float32", 0.1, 1e-4, "r50_sBN_fp32_04")),
            ("r50_v1.0_splited_BN_fp32_05", bn_split_run,
             ((32, 32, 28, 28, 16), "float32", 0.1, 1e-4, "r50_sBN_fp32_05")),
            ("r50_v1.0_splited_BN_fp32_06", bn_split_run,
             ((32, 16, 14, 14, 16), "float32", 0.1, 1e-4, "r50_sBN_fp32_06")),
            ("r50_v1.0_splited_BN_fp32_07", bn_split_run,
             ((32, 64, 14, 14, 16), "float32", 0.1, 1e-4, "r50_sBN_fp32_07")),
            ("r50_v1.0_splited_BN_fp32_08", bn_split_run,
             ((32, 32, 7, 7, 16), "float32", 0.1, 1e-4, "r50_sBN_fp32_08")),
            ("r50_v1.0_splited_BN_fp32_09", bn_split_run,
             ((32, 128, 7, 7, 16), "float32", 0.1, 1e-4, "r50_sBN_fp32_09")),

            ("r50_v1.5_splited_BN_fp32_01", bn_split_run,
             ((32, 8, 56, 56, 16), "float32", 0.1, 1e-4, "r50_sBN_fp32_09")),
            ("r50_v1.5_splited_BN_fp32_02", bn_split_run,
             ((32, 16, 28, 28, 16), "float32", 0.1, 1e-4, "r50_sBN_fp32_09")),
            ("r50_v1.5_splited_BN_fp32_03", bn_split_run,
             ((32, 32, 14, 14, 16), "float32", 0.1, 1e-4, "r50_sBN_fp32_09")),
        ]

        self.testarg_level2 = [
        ]
        return

    @pytest.mark.level2
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_run_l1(self):
        self.common_run(self.testarg_l1)

    def test_run_aic(self):
        self.common_run(self.testarg_aic)

    def test_run_cloud(self):
        self.common_run(self.testarg_cloud)

    def test_run_level2(self):
        self.common_run(self.testarg_level2)

    def teardown(self):
        self._log.info("============= {0} Teardown============"
                       "".format(self.casename))
        return
