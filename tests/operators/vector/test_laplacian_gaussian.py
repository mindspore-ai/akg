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

import datetime
import os

from base import TestBase
import pytest
from test_run.laplacian_of_gaussian_run import laplacian_of_gaussian_ad_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):
    def setup(self):
        case_name = "test_log_filter"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            ("laplacian_of_gaussian_1", laplacian_of_gaussian_ad_run, ((16,), "float16")),
            ("laplacian_of_gaussian_2", laplacian_of_gaussian_ad_run, ((16, 16), "float16")),
            ("laplacian_of_gaussian_3", laplacian_of_gaussian_ad_run, ((16,), "float32")),
            ("laplacian_of_gaussian_4", laplacian_of_gaussian_ad_run, ((16, 16), "float32")),
        ]

    @pytest.mark.rpc_mini
    @pytest.mark.level0
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run(self):
        self.common_run(self.testarg)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return

# if __name__=="__main__":
#     t = TestCase()
#     t.setup()
#     t.test_run()
