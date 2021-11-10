# Copyright 2020 Huawei Technologies Co., Ltd
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
from tests.common.test_run.ascend.lin_space_run import lin_space_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):
    def setup(self):
        """set test case """
        case_name = "test_lin_space_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("lin_space_004", lin_space_run, ((32,), (1,), "float32", "int32")),
            ("lin_space_004", lin_space_run, ((128,), (1,), "float32", "int32")),
            ("lin_space_004", lin_space_run, ((4,), (1,), "float32", "int32")),
            ("lin_space_004", lin_space_run, ((233,), (1,), "float32", "int32")),

        ]
        self.testarg_rpc_cloud = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("lin_space_004", lin_space_run, ((32,), (1,), "float32", "int32")),
            ("lin_space_004", lin_space_run, ((128,), (1,), "float32", "int32")),
            ("lin_space_004", lin_space_run, ((4,), (1,), "float32", "int32")),
            ("lin_space_004", lin_space_run, ((233,), (1,), "float32", "int32")),

        ]

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
