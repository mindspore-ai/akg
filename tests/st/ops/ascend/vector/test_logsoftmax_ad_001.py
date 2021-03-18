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
from tests.common.test_run.logsoftmax_ad_run import logsoftmax_ad_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):
    def setup(self):
        case_name = "test_autodiff_logsoftmax_ad"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("logsoftmax_ad_001", logsoftmax_ad_run, ((64, 2), "float16", -1, "cce_logsoftmax_ad_fp16"), ((32, 32), (94, 94))),
            ("logsoftmax_ad_002", logsoftmax_ad_run, ((8, 2), "float16", -1, "cce_logsoftmax_ad_fp16"), ((2, 2), (94, 94))),
            ("logsoftmax_ad_003", logsoftmax_ad_run, ((160, 30522), "float16", -1, "cce_logsoftmax_ad_fp16"), ((1, 1), (61042 + 30522, 61042 + 30522))),
            ("logsoftmax_ad_004", logsoftmax_ad_run, ((1280, 30522), "float16", -1, "cce_logsoftmax_ad_fp16"), ((1, 1), (61042 + 30522, 61042 + 30522))),

        ]
        return

    def test_run(self):
        self.common_run(self.testarg)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return
