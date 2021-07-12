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
from tests.common.test_run.scatter__run import scatter__run


############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_scatter_nd_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ('scatter__001', scatter__run, ([5, 5, 7], None, 1, [5, 6, 7], 1.1, "int32", 'float32')),
            ('scatter__002', scatter__run, ([12, 15], [12, 15], 0, [13, 15], None, "int32", 'float32'),((13, 1), (15, 1), (12, 1))),
            ('scatter__003', scatter__run, ([128, 128, 128], [128, 128, 128], 2, [128, 128, 150], None, "int32", 'float32'), ((1, 1), (1, 1), (1, 1))),
        ]

        self.testarg_cloud = [
            ('scatter__004', scatter__run, ([30, 25], [30, 25], 1, [30, 25], None, "int32", 'float32')),
        ]
        return

    @pytest.mark.level2
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_run_cloud(self):
        self.common_run(self.testarg_cloud)

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
    a.teardown()
