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
from tests.common.test_run.reduce_max_run import reduce_max_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_reduce_max"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag, opfuncname, testRunArgs, dimArgs
            # shape, axis, keepdims, dtype, attrs
            ("reduce_max_1", reduce_max_run, ((32, 64, 32), (1,), True, "float16", "cce_reduce_max_fp16")),
            ("reduce_max_2", reduce_max_run, ((4, 128, 1024), (0, 1), True, "float32", "cce_reduce_max_fp32")),
            ("reduce_max_3", reduce_max_run, ((2, 1280), (1,), False, "int8", "cce_reduce_max_int8")),
            ("reduce_max_4", reduce_max_run, ((8, 256), (0,), False, "uint8", "cce_reduce_max_uint8")),
            ("reduce_max_5", reduce_max_run, ((1024,), (0,), True, "int32", "cce_reduce_max_int32")),
        ]

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
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
