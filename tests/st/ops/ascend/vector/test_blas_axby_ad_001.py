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
from tests.common.test_run.ascend.blas_axby_ad_run import blas_axby_ad_run


############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_blas_axby_ad_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # caseflag,testfuncname,testRunArgs, dimArgs
            ("001_blas_axby_ad", blas_axby_ad_run, ((512,), "float16", "blas_axby_ad")),
            ("001_blas_axby_ad", blas_axby_ad_run, ((32, 32, 1, 12), "float16", "blas_axby_ad")),
            ("001_blas_axby_ad", blas_axby_ad_run, ((32, 32, 1, 12), "float32", "blas_axby_ad")),
        ]

        self.testarg_cloud = [
            # caseflag,testfuncname,testRunArgs, dimArgs
            ("001_blas_axby_ad", blas_axby_ad_run, ((512,), "float32", "blas_axby_ad")),
            ("001_blas_axby_ad", blas_axby_ad_run, ((32, 16, 16, 8), "float16", "blas_axby_ad")),
        ]
        self.testarg_level1 = [
            # caseflag,testfuncname,testRunArgs, dimArgs
            ("001_blas_axby_ad", blas_axby_ad_run, ((1024,), "float16", "blas_axby_ad")),
            ("001_blas_axby_ad", blas_axby_ad_run, ((32, 32, 16, 16), "float16", "blas_axby_ad")),
            ("001_blas_axby_ad", blas_axby_ad_run, ((32, 32, 16, 16), "float32", "blas_axby_ad")),
        ]

        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_run_cloud(self):
        self.common_run(self.testarg_cloud)

    def test_run_level1(self):
        self.common_run(self.testarg_level1)

    def teardown(self):
        self._log.info("============= {0} Teardown============".format(self.casename))
        return


if __name__ == "__main__":
    t = TestCase()
    t.setup()
    t.test_run()
    t.teardown()
