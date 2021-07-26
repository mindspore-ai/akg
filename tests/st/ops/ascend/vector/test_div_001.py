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
        case_name = "test_div_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("div_002", "div_run", ((3,), (2, 3), "float16")),
            ("div_011", "div_run", ((2, 1024), (1,), "float32")),
            ("div_012", "div_run", ((1024, ), (1,), "float32")),
            ("div_013", "div_run", ((33, 64), (1,), "float32")),
            ("div_014", "div_run", ((4096, ), (1,), "float32")),
            ("div_015", "div_run", ((2, ), (1,), "float32")),
            ("div_004", "div_run", ((1, 3), (2, 3), "float32")),

        ]
        self.testarg_rpc_cloud = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("div_001", "div_run", ((3, 4, 5), (3, 1, 5), "int32")),
            ("div_003", "div_run", ((2, 3), (2, 3), "int8")),
            ("div_005", "div_run", ((8, 24, 42), (8, 24, 42), "uint8")),
            ("div_016", "div_run", ((2, 1024), (1,), "int32")),
            ("div_017", "div_run", ((1024, ), (1,), "int32")),
            ("div_018", "div_run", ((33, 64), (1,), "int32")),
            ("div_019", "div_run", ((4096, ), (1,), "int32")),
            ("div_020", "div_run", ((2, ), (1,), "int32")),
            ("test_bert_div_002", "div_run", ((2, 1024), (1,), "float32")),
            ("test_bert_div_010", "div_run", ((2,), (1,), "float32")),
            ("div_006", "div_run", ((3, 4, 5), (3, 1, 5), "int32")),
            ("div_007", "div_run", ((2, 3), (3,), "float16")),
            ("div_008", "div_run", ((2, 3), (2, 3), "int8")),
            ("div_009", "div_run", ((1, 3), (2, 3), "float32")),
            ("div_010", "div_run", ((8, 24, 42), (8, 24, 42), "uint8")),

            # bert cases
            ("div_021", "div_run", ((21128, 1024), (1,), "float32")),
            ("div_022", "div_run", ((1024, 1024), (1,), "float32")),
            ("div_023", "div_run", ((1024, 4096), (1,), "float32")),
            ("div_024", "div_run", ((4096, 1024), (1,), "float32")),
            ("div_025", "div_run", ((21128, ), (1,), "float32")),
            ("div_026", "div_run", ((21128, 1024), (1,), "int32")),
            ("div_027", "div_run", ((1024, 1024), (1,), "int32")),
            ("div_028", "div_run", ((1024, 4096), (1,), "int32")),
            ("div_029", "div_run", ((4096, 1024), (1,), "int32")),
            ("div_030", "div_run", ((21128, ), (1,), "int32")),
        ]
        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
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
        self.common_run(self.testarg_rpc_cloud)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
