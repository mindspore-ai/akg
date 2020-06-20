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
from base import TestBase
from nose.plugins.attrib import attr
from test_run.SecondOrder_trace_extract_run import trace_extract_run

############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):
    def __init__(self):
        """
        testcase preparcondition
        :return:
        """
        casename = "test_trace_001"
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
            ("trace_001",trace_extract_run,((1,160,160), "float32"), ((1,1),(128,1),(128,1))),
            ("trace_002",trace_extract_run,((1,2304,2304), "float32"), ((1,1),(128,1),(128,1))),
            ("trace_003",trace_extract_run,((1,2048,2048), "float32"), ((1,1),(128,1),(128,1))),
            ("trace_004",trace_extract_run,((1,1152,1152), "float32"), ((1,1),(128,1),(128,1))),
            ("trace_005",trace_extract_run,((1,576,576), "float32"), ((1,1),(128,1),(128,1))),
            ("trace_006",trace_extract_run,((1,512,512), "float32"), ((1,1),(128,1),(128,1))),
            ("trace_007",trace_extract_run,((1,256,256), "float32"), ((1,1),(128,1),(128,1))),
            ("trace_008",trace_extract_run,((1,128,128), "float32"), ((1,1),(128,1),(128,1))),
            ("trace_extract_010",trace_extract_run,((1,64,64), "float32"), ((1,1),(128,1),(128,1))),
            ("trace_extract_011",trace_extract_run,((1,1008,1008), "float32"), ((1,1),(128,1),(128,1))),
            ("trace_extract_012",trace_extract_run,((1,1024,1024), "float32"), ((1,1),(128,1),(128,1))),
        ]
        return

    @attr(type='rpc_mini')
    @attr(level=0)
    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

    @attr(type='rpc_cloud')
    def test_run_rpc_cloud(self):
        """
        run case.#
        :return:
        """
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
