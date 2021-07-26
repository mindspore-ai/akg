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
from tests.common.test_run.relu6_run import relu6_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_relu6_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("001_relu6", relu6_run, ((128, 128), "float16"), ((1, 1), (128, 128))),
            ("relu6_002", relu6_run, ((8, 28, 28, 4), "float16")),
            ("relu6_003", relu6_run, ((8, 14, 14, 6), "float16")),
            ("relu6_004", relu6_run, ((8, 7, 7, 6), "float16")),
            ("relu6_005", relu6_run, ((8, 4, 4, 6), "float16")),
            ("relu6_006", relu6_run, ((8, 2, 2, 4), "float16")),
        ]
        self.testarg_cloud = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("001_relu6", relu6_run, ((128, 128), "float32"), ((1, 1), (128, 128))),
        ]

        self.testarg_rpc_cloud = [
            ("relu6_fp32_1", relu6_run, ((8, 28, 28, 4), "float32")),
            ("relu6_fp32_2", relu6_run, ((8, 14, 14, 6), "float32")),
            ("relu6_fp32_3", relu6_run, ((8, 7, 7, 6), "float32")),
            ("relu6_fp32_4", relu6_run, ((8, 4, 4, 6), "float32")),
            ("relu6_fp32_5", relu6_run, ((8, 2, 2, 4), "float32")),
        ]

        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_run_cloud(self):
        self.common_run(self.testarg_cloud)

    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return
