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
from tests.common.base import TestBase
from tests.common.test_run.apply_momentum_run import apply_momentum_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_applymomentum"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))

        self.testarg = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("apply_momentum_01", apply_momentum_run, ((16,), "float16", False)),
            ("apply_momentum_02", apply_momentum_run, ((16, 16), "float16", True)),
            ("test_resnet50_v1_apply_momentum_001", apply_momentum_run, ((10, 2048), "float32", False, 1 / 1024.0)),
            ("test_resnet50_v1_apply_momentum_002", apply_momentum_run, ((10,), "float32", False, 1 / 1024.0)),
            ("test_resnet50_v1_apply_momentum_003", apply_momentum_run, ((128, 32, 16, 16), "float32", False, 1 / 1024.0)),
            ("test_resnet50_v1_apply_momentum_004", apply_momentum_run, ((144, 16, 16, 16), "float32", False, 1 / 1024.0)),
            ("test_resnet50_v1_apply_momentum_005", apply_momentum_run, ((16, 32, 16, 16), "float32", False, 1 / 1024.0)),
            ("test_resnet50_v1_apply_momentum_006", apply_momentum_run, ((16, 4, 16, 16), "float32", False, 1 / 1024.0)),
            ("test_resnet50_v1_apply_momentum_007", apply_momentum_run, ((16, 64, 16, 16), "float32", False, 1 / 1024.0)),
            ("test_resnet50_v1_apply_momentum_008", apply_momentum_run, ((16, 8, 16, 16), "float32", False, 1 / 1024.0)),
            ("test_resnet50_v1_apply_momentum_009", apply_momentum_run, ((1, 128, 1, 1, 16), "float32", False, 1 / 1024.0)),
            ("test_resnet50_v1_apply_momentum_010", apply_momentum_run, ((1, 16, 1, 1, 16), "float32", False, 1 / 1024.0)),
            ("test_resnet50_v1_apply_momentum_011", apply_momentum_run, ((1, 32, 1, 1, 16), "float32", False, 1 / 1024.0)),
            ("test_resnet50_v1_apply_momentum_012", apply_momentum_run, ((1, 4, 1, 1, 16), "float32", False, 1 / 1024.0)),
            ("test_resnet50_v1_apply_momentum_013", apply_momentum_run, ((1, 64, 1, 1, 16), "float32", False, 1 / 1024.0)),
            ("test_resnet50_v1_apply_momentum_014", apply_momentum_run, ((1, 8, 1, 1, 16), "float32", False, 1 / 1024.0)),
            ("test_resnet50_v1_apply_momentum_015", apply_momentum_run, ((288, 32, 16, 16), "float32", False, 1 / 1024.0)),
            ("test_resnet50_v1_apply_momentum_016", apply_momentum_run, ((32, 128, 16, 16), "float32", False, 1 / 1024.0)),
            ("test_resnet50_v1_apply_momentum_017", apply_momentum_run, ((32, 16, 16, 16), "float32", False, 1 / 1024.0)),
            ("test_resnet50_v1_apply_momentum_018", apply_momentum_run, ((32, 64, 16, 16), "float32", False, 1 / 1024.0)),
            ("test_resnet50_v1_apply_momentum_019", apply_momentum_run, ((32, 8, 16, 16), "float32", False, 1 / 1024.0)),
            ("test_resnet50_v1_apply_momentum_020", apply_momentum_run, ((36, 4, 16, 16), "float32", False, 1 / 1024.0)),
            ("test_resnet50_v1_apply_momentum_021", apply_momentum_run, ((49, 4, 16, 16), "float32", False, 1 / 1024.0)),
            ("test_resnet50_v1_apply_momentum_022", apply_momentum_run, ((4, 16, 16, 16), "float32", False, 1 / 1024.0)),
            ("test_resnet50_v1_apply_momentum_023", apply_momentum_run, ((4, 4, 16, 16), "float32", False, 1 / 1024.0)),
            ("test_resnet50_v1_apply_momentum_024", apply_momentum_run, ((64, 128, 16, 16), "float32", False, 1 / 1024.0)),
            ("test_resnet50_v1_apply_momentum_025", apply_momentum_run, ((64, 16, 16, 16), "float32", False, 1 / 1024.0)),
            ("test_resnet50_v1_apply_momentum_026", apply_momentum_run, ((64, 32, 16, 16), "float32", False, 1 / 1024.0)),
            ("test_resnet50_v1_apply_momentum_027", apply_momentum_run, ((72, 8, 16, 16), "float32", False, 1 / 1024.0)),
            ("test_resnet50_v1_apply_momentum_028", apply_momentum_run, ((8, 32, 16, 16), "float32", False, 1 / 1024.0)),
        ]

        self.testarg_level1 = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("apply_momentum_03_True", apply_momentum_run, ((304,), "float16", True)),
            ("apply_momentum_03_False", apply_momentum_run, ((304,), "float16", False)),
            ("apply_momentum_04_True", apply_momentum_run, ((3, 3, 728, 1), "float16", True)),
            ("apply_momentum_04_False", apply_momentum_run, ((3, 3, 728, 1), "float16", False),),
            ("apply_momentum_05_True", apply_momentum_run, ((256,), "float16", True)),
            ("apply_momentum_05_False", apply_momentum_run, ((256,), "float16", False)),
            ("apply_momentum_06_True", apply_momentum_run, ((3, 3, 128, 1), "float16", True)),
            ("apply_momentum_06_False", apply_momentum_run, ((3, 3, 128, 1), "float16", False)),
            ("apply_momentum_07_True", apply_momentum_run, ((1, 1, 256, 256), "float16", True)),
            ("apply_momentum_07_False", apply_momentum_run, ((1, 1, 256, 256), "float16", False)),
            ("apply_momentum_08_True", apply_momentum_run, ((1536,), "float16", True)),
            ("apply_momentum_08_False", apply_momentum_run, ((1536,), "float16", False)),
            ("apply_momentum_09_True", apply_momentum_run, ((1, 1, 256, 728), "float16", True)),
            ("apply_momentum_09_False", apply_momentum_run, ((1, 1, 256, 728), "float16", False)),
            ("apply_momentum_10_True", apply_momentum_run, ((1, 1, 128, 256), "float16", True)),
            ("apply_momentum_10_False", apply_momentum_run, ((1, 1, 128, 256), "float16", False)),
            ("apply_momentum_11_True", apply_momentum_run, ((1024,), "float16", True)),
            ("apply_momentum_11_False", apply_momentum_run, ((1024,), "float16", False)),
            ("apply_momentum_12_True", apply_momentum_run, ((48,), "float16", True)),
            ("apply_momentum_12_False", apply_momentum_run, ((48,), "float16", False)),
            ("apply_momentum_05_True", apply_momentum_run, ((3, 3, 1536, 1), "float16", True)),
            ("apply_momentum_05_False", apply_momentum_run, ((3, 3, 1536, 1), "float16", False)),
            ("apply_momentum_07_True", apply_momentum_run, ((3, 3, 2048, 1), "float16", True)),
            ("apply_momentum_15_True", apply_momentum_run, ((3, 3, 304, 1), "float16", True)),
            ("apply_momentum_15_False", apply_momentum_run, ((3, 3, 304, 1), "float16", False)),
            ("apply_momentum_16_True", apply_momentum_run, ((1, 1, 256, 48), "float16", True)),
            ("apply_momentum_16_False", apply_momentum_run, ((1, 1, 256, 48), "float16", False)),
            ("apply_momentum_17_True", apply_momentum_run, ((3, 3, 1024, 1), "float16", True)),
            ("apply_momentum_17_False", apply_momentum_run, ((3, 3, 1024, 1), "float16", False)),
            ("apply_momentum_18_True", apply_momentum_run, ((1, 1, 728, 728), "float16", True)),
            ("apply_momentum_18_False", apply_momentum_run, ((1, 1, 728, 728), "float16", False)),
            ("apply_momentum_20_True", apply_momentum_run, ((2048,), "float16", True)),
            ("apply_momentum_20_False", apply_momentum_run, ((2048,), "float16", False)),
            ("apply_momentum_21_True", apply_momentum_run, ((128,), "float16", True)),
            ("apply_momentum_21_False", apply_momentum_run, ((128,), "float16", False)),
            ("apply_momentum_25_True", apply_momentum_run, ((1, 1, 256, 21), "float16", True)),
            ("apply_momentum_25_False", apply_momentum_run, ((1, 1, 256, 21), "float16", False)),
            ("apply_momentum_26_True", apply_momentum_run, ((728,), "float16", True)),
            ("apply_momentum_26_False", apply_momentum_run, ((728,), "float16", False)),
        ]
        self.testarg_5d_rpc_cloud = [
            ("apply_momentum_1_fp16_False", apply_momentum_run, ((32, 63, 1, 1, 16), "float16", False)),
            ("apply_momentum_2_fp32_False", apply_momentum_run, ((32, 63, 1, 1, 16), "float32", False)),
            ("apply_momentum_3_fp16_False", apply_momentum_run, ((32, 1, 1, 1, 16), "float16", False)),
            ("apply_momentum_4_fp32_False", apply_momentum_run, ((32, 1, 1, 1, 16), "float32", False)),
            ("apply_momentum_5_fp16_False", apply_momentum_run, ((32, 128, 1, 1, 16), "float16", False)),
            ("apply_momentum_6_fp32_False", apply_momentum_run, ((32, 128, 1, 1, 16), "float32", False)),
            ("apply_momentum_7_fp16_False", apply_momentum_run, ((512, 32, 1, 1, 16), "float16", False)),
            ("apply_momentum_8_fp32_False", apply_momentum_run, ((512, 32, 1, 1, 16), "float32", False)),
        ]
        self.testlenet_rpc_cloud = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("apply_momentum_001", apply_momentum_run, ((10,), "float32", False)),
            ("apply_momentum_002", apply_momentum_run, ((10, 84), "float32", False)),
            ("apply_momentum_003", apply_momentum_run, ((84, 120), "float32", False)),
            ("apply_momentum_004", apply_momentum_run, ((120,), "float32", False)),
            ("apply_momentum_005", apply_momentum_run, ((120, 784), "float32", False)),
            ("apply_momentum_006", apply_momentum_run, ((16, 16, 3, 3), "float32", False)),
            ("apply_momentum_007", apply_momentum_run, ((6, 3, 3, 3), "float32", False)),
        ]
        return

    @pytest.mark.level2
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run_level1(self):
        self.common_run(self.testarg_level1)

    def test_run_rpc_cloud(self):
        """
        run case
        :return:
        """
        # for arg in self.testarg_5d_rpc_cloud:
        #     self.print_debug(arg)
        # assert self.caseresult
        # self.common_run(self.testarg_5d_rpc_cloud)
        self.common_run(self.testlenet_rpc_cloud)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return
