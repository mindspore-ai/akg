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
from tests.common.test_run.ascend.ceil_run import ceil_run


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_ceil_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("========================{0}  Setup case=================".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("ceil_8", ceil_run, [(8,), "float16"]),
            ("ceil_8_16", ceil_run, [(8, 16), "float16"]),
            ("ceil_8_16_16", ceil_run, [(8, 16, 16), "float16"]),
        ]
        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def teardown(self):
        """
          clean environment
          :return:
          """
        self._log.info("========================{0} Teardown case=================".format(self.casename))


if __name__ == "__main__":
    t = TestCase()
    t.setup()
    t.test_run()
    t.teardown()
