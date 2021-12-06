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

import os
from tests.common.base import TestBase
from tests.common.test_run.ascend.SecondOrder_diag_split_matrix_run import diag_split_matrix_run

############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):
    def __init__(self):
        """
        testcase preparcondition
        :return:
        """
        casename = "test_diag_split_matrix_001"
        casepath = os.getcwd()
        super(TestCase,self).__init__(casename,casepath)

    def setup(self):
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            # ("positive_matrix_inv_001",positive_matrix_inv_run,((64, 64), "float32")),
        ]
        self.testarg_rpc_cloud = [
            ("diag_split_matrix_001",diag_split_matrix_run,((256,256), "float32")),
            ("diag_split_matrix_002",diag_split_matrix_run,((1024,1024), "float32")),
            ("diag_split_matrix_003",diag_split_matrix_run,((1152,1152), "float32")),
            ("diag_split_matrix_004",diag_split_matrix_run,((576,576), "float32")),
            ("diag_split_matrix_005",diag_split_matrix_run,((512,512), "float32")),
            ("diag_split_matrix_006",diag_split_matrix_run,((2048,2048), "float32")),
            ("diag_split_matrix_007",diag_split_matrix_run,((2304,2304), "float32")),
            ("diag_split_matrix_008",diag_split_matrix_run,((4608,4608), "float32")),
        ]
        return

    def test_run(self):
        self.common_run(self.testarg)

    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

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
    t.test_run_rpc_cloud()
    t.teardown()
