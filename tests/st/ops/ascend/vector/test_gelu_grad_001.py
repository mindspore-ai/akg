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


############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_gelu_grad_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # caseflag,opfuncname,testRunArgs, dimArgs
            # shape, dtype
            # ("gelu_grad_run1", "gelu_grad_run", ((64 * 20, 1024), "float16")),
            # ("gelu_grad_run3", "gelu_grad_run", ((8, 16), "float16"))
        ]
        self.testlevel1 = [
            # caseflag,opfuncname,testRunArgs, dimArgs
            # shape, dtype
            # precision error
            # ("gelu_grad_run2", "gelu_grad_run", ((64 * 128, 4096), "float16")),
        ]

        self.testarg_rpc_cloud = [
            # caseflag,opfuncname,testRunArgs, dimArgs
            # shape, dtype
            # float16:[64 * 128, 4096] = float16:[64 * 128, 4096]
            ("gelu_grad_001_input_8192_4096", "gelu_grad_run", ((64 * 128, 4096), "float32")),
            # float16:[64 * 20, 1024] = float:[64 * 20, 1024]
            ("gelu_grad_002_input_1280_1024", "gelu_grad_run", ((64 * 20, 1024), "float32")),
        ]

        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run_1(self):
        self.common_run(self.testlevel1)

    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return
