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
from tests.common.test_run.log1p_run import log1p_run


############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_log1p_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("log1p_01", log1p_run, [(1, 128), "float16", "cce_log1p_fp16"], [(1, 1), (128, 128)]),
            ("log1p_02", log1p_run, [(128, 128), "float16", "cce_log1p_fp16"], [(128, 128), (128, 128)]),
            ("log1p_03", log1p_run, [(32, 128), "float16", "cce_log1p_fp16"], [(32, 32), (128, 128)]),
            ("log1p_04", log1p_run, [(128, 32), "float16", "cce_log1p_fp16"], [(128, 128), (32, 32)]),
            ("log1p_05", log1p_run, [(32, 32), "float16", "cce_log1p_fp16"], [(32, 32), (32, 32)]),
            ("log1p_06", log1p_run, [(384, 32), "float16", "cce_log1p_fp16"], [(384, 384), (32, 32)]),
        ]
        self.testarg_cloud = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("log1p_05", log1p_run, [(32, 32), "float32", "cce_log1p_fp16"], [(32, 32), (32, 32)]),
        ]
        return

    @pytest.mark.level2
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

    def test_run_cloud(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_cloud)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
