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
import datetime
import os
import pytest
from base import TestBase
from nose.plugins.attrib import attr
from test_run.invert_permutation_run import invert_permutation_run


############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_invert_permutation_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ('invert_permutation_001', invert_permutation_run, ([8], 'int32'), ((16, 1), (16, 1))),
            ('invert_permutation_002', invert_permutation_run, ([16], 'int32'), ((16, 1), (16, 1))),
            ('invert_permutation_003', invert_permutation_run, ([32], 'int32'), ((16, 1), (16, 1))),
            ('invert_permutation_004', invert_permutation_run, ([64], 'int32'), ((16, 1), (16, 1))),
        ]
        return

    @pytest.mark.level2
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


if __name__ == "__main__":
    a = TestCase()
    a.setup()
    a.test_run()
    a.teardown()
