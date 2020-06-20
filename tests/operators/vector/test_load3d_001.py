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

from base import TestBase
import pytest
from test_run.load3d_run import load3d_run


############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):
    def setup(self):
        """
        case feature map     kernel  stride  pad
           1 32,3,224,224    7,7     2,2     3,3,3,3
           2 32,64,56,56     1,1     1,1     0,0,0,0
           3 32,64,56,56     3,3     1,1     1,1,1,1
           4 32,256,56,56    1,1     1,1     0,0,0,0
           5 32,256,56,56    1,1     2,2     0,0,0,0
           6 32,128,56,56    3,3     2,2     1,1,1,1
           7 32,128,28,28    1,1     1,1     0,0,0,0
           8 32,512,28,28    1,1     1,1     0,0,0,0
           9 32,128,28,28    3,3     1,1     1,1,1,1
          10 32,512,28,28    1,1     2,2     0,0,0,0
          11 32,256,28,28    3,3     2,2     1,1,1,1
          12 32,256,14,14    1,1     1,1     0,0,0,0
          13 32,1024,14,14   1,1     1,1     0,0,0,0
          14 32,256 ,14,14   1,1     1,1     0,0,0,0
          15 32,1024,14,14   1,1     2,2     0,0,0,0
          16 32,512 ,14,14   3,3     2,2     1,1,1,1
          17 32,512,7,7      1,1     1,1     0,0,0,0
          18 32,2048,7,7     1,1     1,1     0,0,0,0
          19 32,512,7,7      3,3     1,1     1,1,1,1
        testflag,opfuncname,testRunArgs, setdimArgs
        """

        case_name = "test_akg_load3d_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            ("001_load3d_input", load3d_run, ((32, 16, 224, 224), (7,7), (2,2), (3,3,3,3), "float16")),
            ("002_load3d_input", load3d_run, ((32,64, 56, 56), (1,1), (1,1), (0,0,0,0), "float16")),
            # ("003_load3d_input", load3d_run, ((32,64, 56, 56), (3,3), (1,1), (1,1,1,1),"float16")),
            ("004_load3d_input", load3d_run, ((32,256, 56, 56), (1,1), (1,1), (0,0,0,0),"float16")),
            ("005_load3d_input", load3d_run, ((32,256, 56, 56), (1,1), (2,2), (0,0,0,0),"float16")),
            # ("006_load3d_input", load3d_run, ((32,128, 56, 56), (3,3), (2,2), (1,1,1,1),"float16")),
            ("007_load3d_input", load3d_run, ((32,128, 28, 28), (1,1), (1,1), (0,0,0,0),"float16")),
            ("008_load3d_input", load3d_run, ((32,512, 28, 28), (1,1), (1,1), (0,0,0,0),"float16")),
            ("009_load3d_input", load3d_run, ((32,128, 28, 28), (3,3), (1,1), (1,1,1,1),"float16")),
            ("0010_load3d_input", load3d_run, ((32,512, 28, 28), (1,1), (2,2), (0,0,0,0),"float16")),
            # ("0011_load3d_input", load3d_run, ((32,256, 28, 28), (3,3), (2,2), (1,1,1,1),"float16")),
            ("0012_load3d_input", load3d_run, ((32,256, 14, 14), (1,1), (1,1), (0,0,0,0),"float16")),
            ("0013_load3d_input", load3d_run, ((32,1024, 14, 14), (1,1), (1,1), (0,0,0,0),"float16")),
            ("0014_load3d_input", load3d_run, ((32,256, 14, 14), (1,1), (1,1), (0,0,0,0),"float16")),
            ("0015_load3d_input", load3d_run, ((32,1024, 14, 14), (1,1), (2,2), (0,0,0,0),"float16")),
            # ("0016_load3d_input", load3d_run, ((32,512, 14, 14), (3,3), (2,2), (1,1,1,1),"float16")),
            ("017_load3d_input", load3d_run, ((32,512, 7,  7), (1,1), (1,1), (0,0,0,0),"float16")),
            ("018_load3d_input", load3d_run, ((32,2048, 7, 7), (1,1), (1,1), (0,0,0,0), "float16")),
            ("019_load3d_input", load3d_run, ((32,512, 7, 7), (3,3), (1,1), (1,1,1,1), "float16")),

        ]

        self.testarg_rpc_cloud = [
        ]
        self.testarg_level1 = [
        ]

        return

    @pytest.mark.rpc_mini
    @pytest.mark.level0
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

    @pytest.mark.rpc_cloud
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_rpc_cloud(self):
        """
        run case.#
        :return:
        """
        self.common_run([self.testarg_rpc_cloud[0]])

    @pytest.mark.level1
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
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

if __name__ == "__main__":
    t = TestCase()
    t.setup()
    t.test_run()
    t.teardown()
