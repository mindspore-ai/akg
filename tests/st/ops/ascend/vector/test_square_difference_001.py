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
from tests.common.test_run.square_difference_run import square_difference_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_square_difference_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            #caseflag,testfuncname,testRunArgs, dimArgs
            ("001_square_difference_160_1024_160_1", square_difference_run, ((160, 1024), (160, 1), "float16", "cce_squaredifference_fp16"), ((1, 1), (1024, 1024))),
            ("002_square_difference_1024_1024_1024_1", square_difference_run, ((1024, 1024), (1024, 1), "float16", "cce_squaredifference_fp16"), ((1, 1), (1024, 1024))),
            ("003_square_difference_1280_1024_1024_1", square_difference_run, ((1280, 1024), (1280, 1), "float16", "cce_squaredifference_fp16"), ((1, 1), (1024, 1024))),
            # ("004_square_difference_8192_1024_8192_1", square_difference_run,((8192, 1024), (8192, 1), "float16", "cce_squaredifference_fp16"), ((1, 1), (1024, 1024))),
            # ("005_square_difference_8_128_1024_8_128_1", square_difference_run,((8,128, 1024), (8,128, 1), "float16", "cce_squaredifference_fp16"), ((1,1), (1, 1), (1024, 1024))),
            # ("006_square_difference_64_128_1024_64_128_1", square_difference_run,((64,128, 1024), (64,128, 1), "float16", "cce_squaredifference_fp16"), ((1,1),(1, 1), (1024, 1024))),
        ]
        self.testarg_cloud = [
            #caseflag,testfuncname,testRunArgs, dimArgs
            ("001_square_difference_160_1024_160_1", square_difference_run, ((160, 1024), (160, 1), "float32", "cce_squaredifference_fp16"), ((1, 1), (1024, 1024))),
        ]
        self.testarg_level1 = [
            #caseflag,testfuncname,testRunArgs, dimArgs
            # ("001_square_difference_160_1024_160_1", square_difference_run,((160,1024),(160,1),"float16","cce_squaredifference_fp16"), ((1, 1), (1024, 1024))),
            # ("002_square_difference_1024_1024_1024_1", square_difference_run,((1024,1024),(1024,1),"float16","cce_squaredifference_fp16"), ((1, 1), (1024, 1024))),
            # ("003_square_difference_1280_1024_1024_1", square_difference_run,((1280, 1024), (1280, 1), "float16", "cce_squaredifference_fp16"), ((1, 1), (1024, 1024))),
            ("004_square_difference_8192_1024_8192_1", square_difference_run, ((8192, 1024), (8192, 1), "float16", "cce_squaredifference_fp16"), ((1, 1), (1024, 1024))),
            ("005_square_difference_8_128_1024_8_128_1", square_difference_run, ((8, 128, 1024), (8, 128, 1), "float16", "cce_squaredifference_fp16"), ((1, 1), (1, 1), (1024, 1024))),
            ("006_square_difference_64_128_1024_64_128_1", square_difference_run, ((64, 128, 1024), (64, 128, 1), "float16", "cce_squaredifference_fp16"), ((1, 1), (1, 1), (1024, 1024))),
        ]
        return

    @pytest.mark.level2
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

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return
