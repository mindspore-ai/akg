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
from tests.common.base import TestBase
from tests.common.test_run.ascend.roi_align_ad_run import roi_align_ad_run


############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):
    def setup(self):
        """
        testcase preparcondition
        :return:
        """
        case_name = "test_akg_roi_align_ad"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("roi_align_ad_00", roi_align_ad_run, ((1, 16, 16, 16), 2, "float32", 8, 0.5, -1), ),
            #("roi_align_ad_01", roi_align_ad_run, ((2, 256, 24, 40), 2, "float32",7,0.03125,2), ),
            #("roi_align_ad_02", roi_align_ad_run, ((2, 256, 48, 80), 2, "float32",7,0.0625,2), ),
            #("roi_align_ad_03", roi_align_ad_run, ((2, 256, 96, 160), 2, "float32",7,0.125,2), ),
            #("roi_align_ad_04", roi_align_ad_run, ((2, 256, 192, 320), 2, "float32",7,0.25,2), ),
        ]

        self.testarg_rpc_cloud = [
        ]
        self.testarg_level = [
        ]
        return

    def test_run(self):
        self.common_run(self.testarg)

    def test_run_rpc_cloud(self):
        if len(self.testarg_rpc_cloud) > 0:
            self.common_run([self.testarg_rpc_cloud[0]])

    def test_run_level1(self):
        self.common_run(self.testarg_level)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return


if __name__ == "__main__":
    a = TestCase()
    a.setup()
    a.test_run()
