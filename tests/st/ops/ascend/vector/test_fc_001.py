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
from tests.common.test_run.fc_run import fc_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_fc_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs:(fmapshape,weightshape,fctype,block_size,random_type),dimArgs
            ("001_fc_input_23244_323244_dim_2", fc_run, ((2, 32, 4, 4), (32, 32, 4, 4), "float16", 16, False), ((32, 8), )),
            ("002_fc_input_23244_323244_dim_2", fc_run, ((2, 32, 4, 4), (32, 32, 4, 4), "float16", 16, True), ((32, 8), )),
            ("003_fc_input_23284_323284_dim_2", fc_run, ((2, 32, 8, 4), (32, 32, 8, 4), "float16", 16, True), ((32, 8), )),
            ("004_fc_input_26444_326444_dim_2", fc_run, ((2, 64, 4, 4), (32, 64, 4, 4), "float16", 16, True), ((32, 8), )),
        ]
        return

    def test_run(self):
        self.common_run(self.testarg)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return
