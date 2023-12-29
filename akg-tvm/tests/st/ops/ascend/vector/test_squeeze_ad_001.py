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
from tests.common.test_run.ascend.squeeze_ad_run import squeeze_ad_run


class TestCase(TestBase):
    def setup(self):
        """
        testcase preparcondition
        :return:
        """
        case_name = "test_akg_squeeze_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            ("test_squeeze_16_1__1", squeeze_ad_run, [(16, 1), 1, "int32"]),
            ("test_squeeze_8_16_1__2", squeeze_ad_run, [(8, 16, 1), 2, "int32"]),
            ("test_squeeze_8_1_16_1__none", squeeze_ad_run, [(8, 1, 16, 1), None, "int32"]),
            ("test_squeeze_1_1_8_16__0", squeeze_ad_run, [(1, 1, 8, 16), 0, "float16"]),
            ("test_squeeze_8_1_16_16__1", squeeze_ad_run, [(8, 1, 16, 16), 1, "float16"]),
            ("test_squeeze_1_3_1_4_1__0_2", squeeze_ad_run, [(1, 3, 1, 4, 1), (0, 2), "int32"]),
        ]
        self.testarg_cloud = [
            #("test_squeeze_1_1_8_16__0", squeeze_run, [(1,1,8,16), 0, "float32", "squeeze"], [(1,1),(1,1),(8,8),(16,16)]),
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

    def test_run_cloud(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_cloud)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
