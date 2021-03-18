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
from tests.common.test_run.topk_run import topk_run


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_topk_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("========================{0}  Setup case=================".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("topk_001", topk_run, [(10,), 3, "int32"], [(1, 1)]),
            #("topk_002",topk_run,[(6,5), 3, "int32"], [(6,6),(5,5)]),
            #("topk_003",topk_run,[(10,5), 3, "float16"], [(10,10),(5,5)]),
        ]
        return

    def test_run(self):
        self.common_run(self.testarg)

    def teardown(self):
        """
          clean environment
          :return:
          """
        self._log.info("========================{0} Teardown case=================".format(self.casename))


if __name__ == "__main__":
    a = TestCase()
    a.setup()
    a.test_run()
