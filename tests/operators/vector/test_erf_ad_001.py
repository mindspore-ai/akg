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
from base import TestBase
from nose.plugins.attrib import attr
from test_run.erf_ad_run import erf_ad_run


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_erf_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            #caseflag,testfuncname,testRunArgs, dimArgs
        ]
        self.testarg_cloud = [
            #caseflag,testfuncname,testRunArgs, dimArgs
            #("erf_01",erf_adnos_run,((1,128),"float32","cce_erf_fp16"), ((128, 128), (128, 128))),
            ("erf_01", erf_ad_run, ((1, 128), "float16", "cce_erf_fp16")),
            ("erf_02", erf_ad_run, ((128, 128), "float16", "cce_erf_fp16")),
            ("erf_03", erf_ad_run, ((128, 256), "float16", "cce_erf_fp16")),
        ]
        return

    @pytest.mark.level0
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

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
