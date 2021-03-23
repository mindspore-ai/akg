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
from tests.common.test_run.proposal_sort_run import proposal_sort_run


############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_proposal_sort_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("proposal_sort_01", proposal_sort_run, ((1, 256, 8), 32, "float16", "cce_proposal_sort_fp16"),),
            ("proposal_sort_02", proposal_sort_run, ((2, 64, 8), 16, "float16", "cce_proposal_sort_fp16"),),
        ]

        self.testarg_rpc_cloud = [
            ("proposal_sort_02", proposal_sort_run, ((2, 32, 8), 32, "float16", "cce_proposal_sort_fp16"),),
        ]

        self.testarg_level1 = [
            ("proposal_sort_01", proposal_sort_run, ((2, 8192, 8), 16, "float16", "cce_proposal_sort_fp16"),),
            ("proposal_sort_02", proposal_sort_run, ((32, 848, 8), 32, "float16", "cce_proposal_sort_fp16"),),
            ("proposal_sort_03", proposal_sort_run, ((8, 1616, 8), 512, "float16", "cce_proposal_sort_fp16"),),
            ("proposal_sort_04", proposal_sort_run, ((1, 12016, 8), 16, "float16", "cce_proposal_sort_fp16"),),
        ]
        return

    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

    def test_rpc_cloud(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_rpc_cloud)

    def test_run_level1(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_level1)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
