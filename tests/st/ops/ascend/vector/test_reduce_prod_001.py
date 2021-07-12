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
from tests.common.test_run.reduce_prod_run import reduce_prod_run


############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_prod_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("reduce_prod_01", reduce_prod_run, ((1195,), "float16", None, True)),
            ("reduce_prod_01", reduce_prod_run, ((1195,), "float32", (0,), True)),
            ("reduce_prod_01", reduce_prod_run, ((1195,), "int8", (0,), True)),
            ("prod_01", reduce_prod_run, ((1195,), "uint8", (0,), True)),
        ]
        self.testarg_cloud = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("reduce_prod_01", reduce_prod_run, ((1195,), "float32", (0,), True)),
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

    def test_run_cloud(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_cloud)

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
