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
from tests.common.test_run.reduce_min_run import reduce_min_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_reduce_min"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag, opfuncname, testRunArgs, dimArgs
            # shape, axis, keepdims, dtype, attrs
            ("reduce_min_1", reduce_min_run, ((32, 64, 32), (0, 1), True, "float16", "cce_reduce_min_fp16")),
            ("reduce_min_2", reduce_min_run, ((4, 128, 1024), (1, 2), False, "float32", "cce_reduce_min_fp32")),
            ("reduce_min_3", reduce_min_run, ((2, 1280), (0,), False, "int8", "cce_reduce_min_int8")),
            ("reduce_min_4", reduce_min_run, ((8, 256), (1,), False, "uint8", "cce_reduce_min_uint8")),
            ("reduce_min_5", reduce_min_run, ((1024,), (0,), True, "int32", "cce_reduce_min_int32")),
        ]

    @pytest.mark.level2
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
