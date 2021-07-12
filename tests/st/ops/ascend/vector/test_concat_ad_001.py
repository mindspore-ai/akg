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
from tests.common.test_run.concat_ad_run import concat_ad_run


############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_concat_ad_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ## ('concat_ad_01', concat_ad_run, ([[1], [1]], 'int32', 0), ((1, 1),)),
            ## ('concat_ad_02', concat_ad_run, ([[2, 1], [2, 1]], 'int32', 0), ((2,2),)),
            #('concat_ad_03', concat_ad_run, ([[30522,], [8192,]], 'int32', 0), ((1024, 1024),)),
            #('concat_ad_04', concat_ad_run, ([[30522, 1024], [8192, 1024]], 'float16', 0), ((32, 32), (1024, 1024))),
            ## ('concat_ad_01', concat_ad_run, ([[1600,]], 'float16', 0), ((32, 0), (16, 0))),
            ## ('concat_ad_02', concat_ad_run, ([[1600,], [160,]], 'float16', 0), ((32, 0), (16, 0))),
            ## ('concat_ad_02', concat_ad_run, ([[40, 16], [40, 16]], 'float16', 0), ((32, 32), (16, 16))),
            ## ('concat_ad_02', concat_ad_run, ([[40, 16], [40, 16]], 'float16', 1), ((32, 0), (16, 0))),
            ## ('concat_ad_02', concat_ad_run, ([[32, 16], [32, 16]], 'float16', 0), ((32, 0), (16, 0))),
            ## ('concat_ad_02', concat_ad_run, ([[32, 16], [32, 16]], 'float16', 1), ((32, 0), (16, 0))),
            ## ('concat_ad_02', concat_ad_run, ([[16, 16], [32, 16], [64, 16]], 'float16', 0), ((32, 0), (16, 0))),
            ## ('concat_ad_02', concat_ad_run, ([[16, 16], [16, 32], [16, 64]], 'float16', 1), ((32, 0), (16, 0))),
            ## ('concat_ad_02', concat_ad_run, ([[16, 16, 16], [16, 32, 16], [16, 64, 16]], 'float16', 1), ((32, 0), (16, 0))),
            ## ('concat_ad_02', concat_ad_run, ([[2, 16, 16, 16], [2, 32, 16, 16], [2, 64, 16, 16]], 'float16', 1), ((32, 0), (16, 0))),
            ## ('concat_ad_02', concat_ad_run, ([[1, 16, 16, 16], [1, 31, 16, 16], [1, 64, 16, 16]], 'float16', 1), ((32, 0), (16, 0))),
            ## ('concat_ad_02', concat_ad_run, ([[30522, 1024], [8192, 1024]], 'float16', 0), ((32, 32), (1024, 1024))),
            ## ('concat_ad_02', concat_ad_run, ([[30522, 1024], [8192, 1024]], 'int32', 0), ((32, 32), (1024, 1024))),
        ]
        self.testarg_level2 = [
            # testflag,opfuncname,testRunArgs, dimArgs
            # ('concat_ad_001', concat_ad_run, ([[1], [1]], 'int32', 0),
            # ('concat_ad_002', concat_ad_run, ([[2, 1], [2, 1]], 'int32', 0),
            # ('concat_ad_003', concat_ad_run, ([[30522, ], [8192, ]], 'int32', 0),
            # ('concat_ad_004', concat_ad_run, ([[1600, ]], 'float16', 0)),
            # ('concat_ad_005', concat_ad_run, ([[1600, ], [160, ]], 'float16', 0)),
            ('concat_ad_006', concat_ad_run, ([[40, 16], [40, 16]], 'float16', 0)),
            # ('concat_ad_007', concat_ad_run, ([[40, 16], [40, 16]], 'float16', 1)),
            # ('concat_ad_008', concat_ad_run, ([[32, 16], [32, 16]], 'float16', 0)),
            # ('concat_ad_009', concat_ad_run, ([[32, 16], [32, 16]], 'float16', 1)),
            # ('concat_ad_010', concat_ad_run, ([[16, 16], [32, 16], [64, 16]], 'float16', 0)),
            # ('concat_ad_011', concat_ad_run, ([[16, 16], [16, 32], [16, 64]], 'float16', 1)),
            # ('concat_ad_012', concat_ad_run, ([[16, 16, 16], [16, 32, 16], [16, 64, 16]], 'float16', 1)),
            # ('concat_ad_013', concat_ad_run, ([[2, 16, 16, 16], [2, 32, 16, 16], [2, 64, 16, 16]], 'float16', 1)),
            # ('concat_ad_014', concat_ad_run, ([[1, 16, 16, 16], [1, 31, 16, 16], [1, 64, 16, 16]], 'float16', 1)),

            # ('concat_ad_101', concat_ad_run, ([[30522, 1024], [8192, 1024]], 'float16', 0)),
            # ('concat_ad_102', concat_ad_run, ([[30522, 1024], [8192, 1024]], 'float16', 0)),
            # ('concat_ad_103', concat_ad_run, ([[30522, 1024], [8192, 1024]], 'int32', 0)),
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

    def test_run_level2(self):
        self.common_run(self.testarg_level2)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return


if __name__ == "__main__":
    #a = TestCase("test_concat_ad_001", os.getcwd())
    a = TestCase()
    a.setup()
    a.test_run_level2()
    a.teardown()
