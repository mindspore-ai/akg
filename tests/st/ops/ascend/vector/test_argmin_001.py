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
from tests.common.test_run.argmin_run import argmin_run


############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_argmin_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # to be fixed
            ("argmin_001", argmin_run, ((8, 10), "int8", 1),),
            ("argmin_002", argmin_run, ((3, 2, 9019), "int8", -3),),
            ("argmin_003", argmin_run, ((3, 1020), "int8", 1),),
            ("argmin_004", argmin_run, ((3, 1020), "float32", 1),),
            ("argmin_005", argmin_run, ((8, 10), "float16", 1),),
            ("argmin_006", argmin_run, ((75,), "int32", -1),),
            ("argmin_007", argmin_run, ((75,), "float32", -1),),
            ("argmin_008", argmin_run, ((75,), "int8", -1),),
            ("argmin_009", argmin_run, ((75,), "float16", -1),),
            ("argmin_010", argmin_run, ((4, 9, 5, 2), "float16", -1),),
            ("argmin_011", argmin_run, ((3, 1020), "float16", 1),),
            ("argmin_012", argmin_run, ((3, 896), "float16", -1),),
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

    def test_run_level2(self):
        self.common_run(self.testarg_level2)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return


if __name__ == "__main__":
    a = TestCase()
    a.setup()
    a.test_run()
