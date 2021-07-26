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
from tests.common.test_run.avgpool_run import avgpool_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):

    def setup(self):
        case_name = "test_avgpool_v2_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg_ci = [
            # caseflag,func, shape(NC1HWC0),kernel,stride,pad,dtype, dimAttr
            ("001_avgpool_v2_input_128_128", avgpool_run, [(1, 1, 16, 16, 16), (2, 2), (4, 4), (1, 1, 1, 1), "float16"]),
            ("002_avgpool_v2_input_128_128", avgpool_run, [(1, 1, 16, 16, 16), (2, 2), (4, 4), (0, 0, 0, 0), "float16"]),
            ("002_avgpool_v2_input_128_128", avgpool_run, [(1, 1, 16, 16, 16), (2, 2), (4, 4), (0, 0, 0, 0), "float16"]),
            ("004_avgpool_v2_input_128_128", avgpool_run, [(10, 3, 16, 16, 16), (4, 4), (3, 3), (0, 0, 0, 0), "float16"], [(2, 2), (3, 3), (16, 16), (5, 5), (5, 5)]),
            ("006_avgpool_v2_input_128_128", avgpool_run, [(1, 2, 16, 16, 16), (4, 4), (3, 3), (1, 1, 1, 1), "float16"], [(1, 1), (16, 16), (19, 19)]),

            ("003_avgpool_v2_input_128_128", avgpool_run, [(1, 1, 16, 16, 16), (4, 4), (3, 3), 'VALID', "float16"]),
            ("003_avgpool_v2_input_128_128", avgpool_run, [(1, 1, 16, 16, 16), (4, 4), (3, 3), 'SAME', "float16"]),
        ]

        self.testarg_nightly = [
            # caseflag,func, shape(NC1HWC0),kernel,stride,pad,dtype, dimAttr
            #("005_avgpool_v2_input_128_128",avgpool_run,[(1,1,128,16,16),(4, 4),(3, 3), (0, 0), "float16"],[(16,16),(42,42),(5,5),(168,168)]),
            # ("006_avgpool_v2_input_128_128",avgpool_run,[(1,2,16,16,16),(4, 4),(3, 3), (1, 1), "float16"]),
            ("007_avgpool_v2_input_128_128", avgpool_run, [(10, 3, 16, 16, 16), (4, 4), (3, 3), (0, 0, 0, 0), "float16"]),
            ("008_avgpool_v2_input_128_128", avgpool_run, [(1, 1, 64, 64, 16), (4, 4), (3, 3), (0, 0, 0, 0), "float16"]),
            ## ("008_avgpool_v2_input_128_128",avgpool_run,[(1,3,64,64,16),(4, 4),(3, 3), (0, 0, 0, 0), "float16"],[(1,1),(16,16),(3,3),(3,3),(84,84)]),
        ]

        self.testarg_cloud = [
            # caseflag,func, shape(NC1HWC0),kernel,stride,pad,dtype, dimAttr
            ("001_avgpool_v2_input_128_128", avgpool_run, [(1, 1, 16, 16, 16), (2, 2), (4, 4), (1, 1, 1, 1), "float32"]),
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
    def test_run_nightly(self):
        self.common_run(self.testarg_nightly)

    def test_run_cloud(self):
        self.common_run(self.testarg_cloud)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return

#a = TestCase()
# a.setup()
# a.test_run_ci()
