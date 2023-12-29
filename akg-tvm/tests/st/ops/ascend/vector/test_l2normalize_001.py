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
from tests.common.test_run.ascend.l2normalize_run import l2normalize_run


############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_l2normalize_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # caseflag,testfuncname,testRunArgs, dimArgs
            ("001_l2normalize", l2normalize_run, ((512,), "float16", "l2normalize")),
            ("001_l2normalize", l2normalize_run, ((32, 32), "float16", "l2normalize")),
        ]

        self.testarg_cloud = [
            # caseflag,testfuncname,testRunArgs, dimArgs
            ("001_l2normalize", l2normalize_run, ((512,), "float16", "l2normalize")),
            ("001_l2normalize", l2normalize_run, ((32, 16), "float16", "l2normalize")),
        ]
        self.testarg_level1 = [
            # caseflag,testfuncname,testRunArgs, dimArgs
            ("001_l2normalize", l2normalize_run, ((1, 512), "float16", "l2normalize")),
            ("001_l2normalize", l2normalize_run, ((96, 512), "float16", "l2normalize")),
            ("001_l2normalize", l2normalize_run, ((128, 512), "float16", "l2normalize")),
        ]
        self.testarg_level2 = [
            ("003_l2normalize", l2normalize_run, ((512000,), "float16", "l2normalize")),
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

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run_level1(self):
        self.common_run(self.testarg_level1)

    def test_run_level2(self):
        self.common_run(self.testarg_level2)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return
