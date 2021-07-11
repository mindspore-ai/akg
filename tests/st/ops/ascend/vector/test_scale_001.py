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
from tests.common.test_run.scale_run import scale_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_scale"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag, opfuncname, testRunArgs, dimArgs
            # input_shape, scale_shape, bias_shape, dtype, kernel_name, attrs
            ("scale_1", scale_run, ((1, 4, 128, 128), (1, 4, 1, 1), (), "float16", "cce_scale_fp16")),
            ("scale_2", scale_run, ((1, 1, 128, 128, 16), (1, 1, 1, 1, 16), (1, 1, 1, 1, 16), "float32", "cce_scale_fp32")),
            ("scale_3", scale_run, ((2, 3, 64, 128), (1, 3, 1, 1), (1, 3, 1, 1), "int8", "cce_scale_int8")),
            ("scale_4", scale_run, ((4, 4, 64, 1024), (1, 4, 1, 1), (1, 4, 1, 1), "uint8", "cce_scale_uint8")),
        ]
        self.testarg_cloud = [
            ("scale_1", scale_run, ((1, 16, 256, 32), (1, 16, 1, 1), (1, 16, 1, 1), "int32", "cce_scale_int32")),
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
