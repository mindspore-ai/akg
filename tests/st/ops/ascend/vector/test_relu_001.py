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
Testcase_Name:
Testcase_Number:
Testcase_Stage:
Testcase_Level:
Testcase_TestType: Function Test
Testcase_Scenario:
################################################
Testcase_PrepareCondition:

Testcase_TestSteps:

Testcase_ExpectedResult:

"""
import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.relu_run import relu_run


class TestCase(TestBase):
    def setup(self):
        case_name = "test_relu_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        # Set some small shape in level 0
        self.testarg = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            # This two cases is created by gao xiong
            ("relu_001_gx", relu_run, ((1, 128), "float16", 1e-5), ((16, 0), (1, 0))),
            ("relu_002_gx", relu_run, ((4, 16, 16, 5), "float16", 1e-5), ((4, 0), (2, 0), (2, 0), (5, 0))),



        ]
        self.testlenet_rpc_cloud = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("relu_001", relu_run, ((1, 16, 7, 7), "float16", 1e-5)),
            ("relu_002", relu_run, ((1, 6, 15, 15), "float16", 1e-5)),

        ]
        # Set all shape in cloud
        self.testarg_rpc_cloud = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("relu_001", relu_run, ((32, 4, 112, 112, 16), "float16", 1e-5)),
            ("relu_002", relu_run, ((32, 4, 56, 56, 16), "float16", 1e-5)),
            ("relu_003", relu_run, ((32, 16, 56, 56, 16), "float16", 1e-5)),
            ("relu_004", relu_run, ((32, 8, 28, 28, 16), "float16", 1e-5)),
            ("relu_005", relu_run, ((32, 32, 28, 28, 16), "float16", 1e-5)),
            ("relu_006", relu_run, ((32, 16, 14, 14, 16), "float16", 1e-5)),
            ("relu_007", relu_run, ((32, 64, 14, 14, 16), "float16", 1e-5)),
            ("relu_008", relu_run, ((32, 32, 7, 7, 16), "float16", 1e-5)),
            ("relu_009", relu_run, ((32, 128, 7, 7, 16), "float16", 1e-5)),
            ("relu_010", relu_run, ((32, 4, 112, 112, 16), "float32", 1e-5)),
            ("relu_011", relu_run, ((32, 4, 56, 56, 16), "float32", 1e-5)),
            ("relu_012", relu_run, ((32, 16, 56, 56, 16), "float32", 1e-5)),
            ("relu_013", relu_run, ((32, 8, 28, 28, 16), "float32", 1e-5)),
            ("relu_014", relu_run, ((32, 32, 28, 28, 16), "float32", 1e-5)),
            ("relu_015", relu_run, ((32, 16, 14, 14, 16), "float32", 1e-5)),
            ("relu_016", relu_run, ((32, 64, 14, 14, 16), "float32", 1e-5)),
            ("relu_017", relu_run, ((32, 32, 7, 7, 16), "float32", 1e-5)),
            ("relu_018", relu_run, ((32, 128, 7, 7, 16), "float32", 1e-5)),
        ]
        return

    @pytest.mark.level2
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run_level1(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

    def test_run_rpc_cloud(self):
        """
        run case.#
        :return:
        """
        # self.common_run(self.testarg_rpc_cloud)
        self.common_run(self.testlenet_rpc_cloud)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
