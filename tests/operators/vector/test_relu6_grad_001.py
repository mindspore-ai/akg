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
import datetime
import os

from base import TestBase
import pytest
from test_run.relu6_grad_run import relu6_grad_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_relu6_grad_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("relu6_grad_001", relu6_grad_run, ((1, 128), "float16")),
            ("relu6_grad_002", relu6_grad_run, ((8, 28, 28, 4), "float16")),
            ("relu6_grad_003", relu6_grad_run, ((8, 14, 14, 6), "float16")),
            ("relu6_grad_004", relu6_grad_run, ((8, 7, 7, 6), "float16")),
            ("relu6_grad_005", relu6_grad_run, ((8, 4, 4, 6), "float16")),
            ("relu6_grad_006", relu6_grad_run, ((8, 2, 2, 4), "float16")),
        ]

        self.testarg_cloud = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("relu6_grad_001", relu6_grad_run, ((1, 128), "float32")),
        ]

        self.testarg_rpc_cloud = [
            ("relu6_grad_fp32_001", relu6_grad_run, ((8, 28, 28, 4), "float32")),
            ("relu6_grad_fp32_002", relu6_grad_run, ((8, 14, 14, 6), "float32")),
            ("relu6_grad_fp32_003", relu6_grad_run, ((8, 7, 7, 6), "float32")),
            ("relu6_grad_fp32_004", relu6_grad_run, ((8, 4, 4, 6), "float32")),
            ("relu6_grad_fp32_005", relu6_grad_run, ((8, 2, 2, 4), "float32")),
        ]
        return

    @pytest.mark.rpc_mini
    @pytest.mark.level0
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run(self):
        self.common_run(self.testarg)

    @pytest.mark.aicmodel
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_cloud(self):
        self.common_run(self.testarg_cloud)

    @pytest.mark.rpc_cloud
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return

# a=TestCase()
# a.setup()
# a.test_run()
