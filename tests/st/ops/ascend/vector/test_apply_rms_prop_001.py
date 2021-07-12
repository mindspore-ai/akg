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
from tests.common.test_run.apply_rms_prop_run import apply_rms_prop_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_apply_rms_prop"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag, opfuncname, testRunArgs, dimArgs
            # testRunArgs: (shape, dtype, lr, momentum, rho, epsilon, attrs)
            ("apply_rms_prop_1", apply_rms_prop_run, ((1024,), "float16", 0.5, 0.9, 0.6, 1e-4)),
            ("apply_rms_prop_2", apply_rms_prop_run, ((16, 16), "float32", 0.5, 0.9, 0.6, 1e-6)),
        ]
        self.testarg_cloud = [
            ("apply_rms_prop_1", apply_rms_prop_run, ((1, 1, 64, 128), "float32", 0.1, 0.5, 0.8, 1e-6)),
        ]

    @pytest.mark.level2
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_run_cloud(self):
        self.common_run(self.testarg_cloud)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
