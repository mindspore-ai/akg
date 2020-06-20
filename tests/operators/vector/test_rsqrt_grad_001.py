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

from base import TestBase
import pytest
from test_run.rsqrt_grad_run import rsqrt_grad_run


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_rsqrt_grad_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            #caseflag,testfuncname,testRunArgs, dimArgs
            ("rsqrt_grad_01", rsqrt_grad_run, ((64, 128, 1), "float16", "cce_rsqrt_grad_fp16"), ((64, 128), (128, 128))),
            ("rsqrt_grad_02", rsqrt_grad_run, ((8192, 1), "float16", "cce_rsqrt_grad_fp16"), ((128, 0), (128, 128))),
            ("rsqrt_grad_03", rsqrt_grad_run, ((1280, 1), "float16", "cce_rsqrt_grad_fp16"), ((128, 0), (128, 128))),
        ]
        self.testarg_cloud = [
            #caseflag,testfuncname,testRunArgs, dimArgs
            ("rsqrt_grad_01", rsqrt_grad_run, ((64, 128, 1), "float32", "cce_rsqrt_grad_fp16"), ((64, 128), (128, 128))),
        ]
        return

    @pytest.mark.rpc_mini
    @pytest.mark.level0
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

    @pytest.mark.aicmodel
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
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
