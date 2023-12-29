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
################################################
"""
import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.ascend.bc_run import bc_run
from tests.common.test_run import addn_run

############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_bankconflicts_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            #testflag,opfuncname,testRunArgs, dimArgs

            # test 001 can be used to verify that the number of operations using a pair of buffers is taken into account
            # by the read-read cost (i.e., the number of RR conflicts is lower if ilp_rr_cost=True)
            ("001_bc_256_fp16", bc_run, ([256], [256], [256], "float16", "cce_bc_fp16"), ([256, 256],)),

            # test 002 can be used to verify the functionality of ilp_reuse_inplace=True. When ilp_reuse_inplace=False, the
            # StorageRewriteILP pass fails (lp_solve returns no solution), because the buffers won't fit in ubuf
            # unless inplace reuse is exploited.
            ("002_bc_addn_input_1280_1024_2_dim_2", addn_run, ((128, 1024), "float16", 3), ((32, 32), (1024, 1024)))
        ]
        return

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


if __name__ == "__main__":
    t = TestCase()
    t.setup()
    t.test_run()
    t.teardown()
