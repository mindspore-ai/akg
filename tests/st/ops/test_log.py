# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
import akg.utils as utils
from tests.common.base import TestBase
from tests.common.test_run import log_run

############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_log"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info(
            "============= {0} Setup case============".format(self.casename))
        self.args_default = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("log_01", log_run, ((1, 128), "float16", "log_fp16"), ["level0"]),
            ("log_02", log_run, ((1280, 1280), "float16", "log_fp16"), ["level0"]),
            ("log_03", log_run, ((32, 128), "float16", "log_fp16"), ["level0"]),
            ("log_04", log_run, ((128, 32), "float16", "log_fp16"), ["level0"]),
            ("log_05", log_run, ((32, 32), "float16", "log_fp16"), ["level0"]),
            ("log_06", log_run, ((384, 32), "float16", "log_fp16"), ["level0"]),
        ]
        return True

    @pytest.mark.level0
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_gpu_level0(self):
        return self.run_cases(self.args_default, utils.CUDA, "level0")

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_cpu_level0(self):
        return self.run_cases(self.args_default, utils.LLVM, "level0")

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        return self.run_cases(self.args_default, utils.CCE, "level0")

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info(
            "============= {0} Teardown============".format(self.casename))
        return
