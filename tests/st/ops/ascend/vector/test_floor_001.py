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
from tests.common.test_run.ascend.floor_run import floor_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_floor_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg_ci = [
            #caseflag,opfuncname,testRunArgs, dimArgs
            # Deeplab v3
        #    ("004_floor_4_33_33_256", floor_run, ((4, 33, 33, 256), "float16", "cce_floor_fp16")),
            ("005_floor", floor_run, ((128, 1280), "float32", "cce_floor_fp32")),

        ]
        self.testarg = [
            #caseflag,opfuncname,testRunArgs, dimArgs
            ("001_floor_8192_1024", floor_run, ((8192, 1024), "float16", "cce_floor_fp16"), ((8, 8), (1024, 1024))),
            ("002_floor_64_16_128_128", floor_run, ((64, 16, 128, 128), "float16", "cce_floor_fp16"), ((1, 1), (1, 1), (64, 64), (128, 128))),
            ("003_floor_64_128_1024", floor_run, ((64, 128, 1024), "float16", "cce_floor_fp16"), ((1, 1), (8, 8), (1024, 1024))),
        ]
        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run_ci(self):
        self.common_run(self.testarg_ci)

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return
