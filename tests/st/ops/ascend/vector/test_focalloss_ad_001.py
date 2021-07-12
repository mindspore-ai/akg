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
from tests.common.test_run.focalloss_ad_run import focalloss_ad_run
from tests.common.test_run.focalloss_ad_run import focalloss_grad_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_focalloss_ad_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("focalloss_grad_001", focalloss_grad_run, ((8, 8732, 6), "float16", "float16", 2)),
        ]
        self.test_rpc_cloud = [
            ("focalloss_grad_002", focalloss_grad_run, ((8, 8732, 6), "float32", "float32", 2)),
            ("focalloss_grad_003", focalloss_grad_run, ((32, 8732, 6), "float16", "float16", 2)),
            ("focalloss_grad_004", focalloss_grad_run, ((32, 8732, 6), "float32", "float32", 2)),
        ]
        self.test_autodiff = [
            ("focalloss_ad_001", focalloss_ad_run, ((8, 8732, 6), "float16", "float16", 2.0, "focalloss_1")),
            ("focalloss_ad_001", focalloss_ad_run, ((8, 8732, 6), "float32", "float32", 2.0, "focalloss_2")),
        ]
        return

    @pytest.mark.level2
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_rpc_cloud_run(self):
        self.common_run(self.test_rpc_cloud)

    def test_auto_diff(self):
        self.common_run(self.test_autodiff)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return


# if __name__ == "__main__":
#    a = TestCase()
#    a.setup()
#    a.test_run()
