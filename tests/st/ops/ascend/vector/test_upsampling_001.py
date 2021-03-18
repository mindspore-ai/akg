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
from tests.common.test_run.upsampling_run import upsampling_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_upsampling_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            #("upsampling_00", upsampling_run, ([2,5,5,3], [2,10,10,3], "float16", "cce_upsampling_fp16")),
            ("upsampling_01", upsampling_run, ([8, 20, 20, 512], [8, 40, 40, 512], "float16", "cce_upsampling_fp16")),
            ("upsampling_02", upsampling_run, ([8, 40, 40, 256], [8, 80, 80, 256], "float16", "cce_upsampling_fp16")),
        ]
        return

    # these cases are covered in test_resize_nearest_001
    def test_run(self):
        self.common_run(self.testarg)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return


if __name__ == "__main__":
    a = TestCase()
    a.setup()
    a.test_run()
    a.teardown()
