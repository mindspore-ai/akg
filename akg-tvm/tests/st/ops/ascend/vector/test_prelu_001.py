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
Testcase_Name:
Testcase_Number:
Testcase_Stage:
Testcase_Level:
Testcase_TestType: Function Test
Testcase_Scenario:
################################################
Testcase_PrepareCondition:

Testcase_TestSteps:

Testcase_ExpectedResult:

"""
import os
from tests.common.base import TestBase
from tests.common.test_run.ascend.prelu_run import prelu_run


class TestCase(TestBase):
    def setup(self):
        case_name = "test_prelu_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        # Set all shape in cloud
        self.testarg_rpc_cloud = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            #("prelu_001", prelu_run, ((1, 64, 16, 16), (64,), "float16", 1e-5)),
            ("prelu_001", prelu_run, ((1, 64, 112, 112), (64,), "float16", 1e-5)),
            ("prelu_002", prelu_run, ((1, 64, 56, 56), (64,), "float16", 1e-5)),
            ("prelu_003", prelu_run, ((1, 128, 56, 56), (128,), "float16", 1e-5)),
            ("prelu_004", prelu_run, ((1, 128, 28, 28), (128,), "float16", 1e-5)),
            ("prelu_005", prelu_run, ((1, 256, 28, 28), (256,), "float16", 1e-5)),
            ("prelu_006", prelu_run, ((1, 256, 14, 14), (256,), "float16", 1e-5)),
            ("prelu_007", prelu_run, ((1, 512, 14, 14), (512,), "float16", 1e-5)),
            ("prelu_008", prelu_run, ((1, 512, 7, 7), (512,), "float16", 1e-5)),
            #
            ("prelu_011", prelu_run, ((1, 64, 112, 112), (64,), "float32", 1e-5)),
            ("prelu_012", prelu_run, ((1, 64, 56, 56), (64,), "float32", 1e-5)),
            ("prelu_013", prelu_run, ((1, 128, 56, 56), (128,), "float32", 1e-5)),
            ("prelu_014", prelu_run, ((1, 128, 28, 28), (128,), "float32", 1e-5)),
            ("prelu_015", prelu_run, ((1, 256, 28, 28), (256,), "float32", 1e-5)),
            ("prelu_016", prelu_run, ((1, 256, 14, 14), (256,), "float32", 1e-5)),
            ("prelu_017", prelu_run, ((1, 512, 14, 14), (512,), "float32", 1e-5)),
            ("prelu_018", prelu_run, ((1, 512, 7, 7), (512,), "float32", 1e-5)),
            #
            ("prelu_021", prelu_run, ((1, 64, 112, 112), (1,), "float16", 1e-5)),
            ("prelu_022", prelu_run, ((1, 64, 56, 56), (1,), "float16", 1e-5)),
            ("prelu_023", prelu_run, ((1, 128, 56, 56), (1,), "float16", 1e-5)),
            ("prelu_024", prelu_run, ((1, 128, 28, 28), (1,), "float16", 1e-5)),
            ("prelu_025", prelu_run, ((1, 256, 28, 28), (1,), "float16", 1e-5)),
            ("prelu_026", prelu_run, ((1, 256, 14, 14), (1,), "float16", 1e-5)),
            ("prelu_027", prelu_run, ((1, 512, 14, 14), (1,), "float16", 1e-5)),
            ("prelu_028", prelu_run, ((1, 512, 7, 7), (1,), "float16", 1e-5)),
            #
            ("prelu_031", prelu_run, ((1, 64, 112, 112), (1,), "float32", 1e-5)),
            ("prelu_032", prelu_run, ((1, 64, 56, 56), (1,), "float32", 1e-5)),
            ("prelu_033", prelu_run, ((1, 128, 56, 56), (1,), "float32", 1e-5)),
            ("prelu_034", prelu_run, ((1, 128, 28, 28), (1,), "float32", 1e-5)),
            ("prelu_035", prelu_run, ((1, 256, 28, 28), (1,), "float32", 1e-5)),
            ("prelu_036", prelu_run, ((1, 256, 14, 14), (1,), "float32", 1e-5)),
            ("prelu_037", prelu_run, ((1, 512, 14, 14), (1,), "float32", 1e-5)),
            ("prelu_038", prelu_run, ((1, 512, 7, 7), (1,), "float32", 1e-5)),
            #
            ("prelu_041", prelu_run, ((16, 64, 112, 112), (64,), "float16", 1e-5)),
            ("prelu_042", prelu_run, ((64, 64, 56, 56), (64,), "float16", 1e-5)),
            ("prelu_043", prelu_run, ((32, 128, 56, 56), (128,), "float16", 1e-5)),
            ("prelu_044", prelu_run, ((64, 128, 28, 28), (128,), "float16", 1e-5)),
            ("prelu_045", prelu_run, ((64, 256, 28, 28), (256,), "float16", 1e-5)),
            ("prelu_046", prelu_run, ((64, 256, 14, 14), (256,), "float16", 1e-5)),
            ("prelu_047", prelu_run, ((64, 512, 14, 14), (512,), "float16", 1e-5)),
            ("prelu_048", prelu_run, ((64, 512, 7, 7), (512,), "float16", 1e-5)),
            #
            ("prelu_051", prelu_run, ((16, 64, 112, 112), (64,), "float32", 1e-5)),
            ("prelu_052", prelu_run, ((64, 64, 56, 56), (64,), "float32", 1e-5)),
            ("prelu_053", prelu_run, ((32, 128, 56, 56), (128,), "float32", 1e-5)),
            ("prelu_054", prelu_run, ((64, 128, 28, 28), (128,), "float32", 1e-5)),
            ("prelu_055", prelu_run, ((64, 256, 28, 28), (256,), "float32", 1e-5)),
            ("prelu_056", prelu_run, ((64, 256, 14, 14), (256,), "float32", 1e-5)),
            ("prelu_057", prelu_run, ((64, 512, 14, 14), (512,), "float32", 1e-5)),
            ("prelu_058", prelu_run, ((64, 512, 7, 7), (512,), "float32", 1e-5)),
            #
            ("prelu_061", prelu_run, ((16, 64, 112, 112), (1,), "float16", 1e-5)),
            ("prelu_062", prelu_run, ((64, 64, 56, 56), (1,), "float16", 1e-5)),
            ("prelu_063", prelu_run, ((32, 128, 56, 56), (1,), "float16", 1e-5)),
            ("prelu_064", prelu_run, ((64, 128, 28, 28), (1,), "float16", 1e-5)),
            ("prelu_065", prelu_run, ((64, 256, 28, 28), (1,), "float16", 1e-5)),
            ("prelu_066", prelu_run, ((64, 256, 14, 14), (1,), "float16", 1e-5)),
            ("prelu_067", prelu_run, ((64, 512, 14, 14), (1,), "float16", 1e-5)),
            ("prelu_068", prelu_run, ((64, 512, 7, 7), (1,), "float16", 1e-5)),
            #
            ("prelu_071", prelu_run, ((16, 64, 112, 112), (1,), "float32", 1e-5)),
            ("prelu_072", prelu_run, ((64, 64, 56, 56), (1,), "float32", 1e-5)),
            ("prelu_073", prelu_run, ((32, 128, 56, 56), (1,), "float32", 1e-5)),
            ("prelu_074", prelu_run, ((64, 128, 28, 28), (1,), "float32", 1e-5)),
            ("prelu_075", prelu_run, ((64, 256, 28, 28), (1,), "float32", 1e-5)),
            ("prelu_076", prelu_run, ((64, 256, 14, 14), (1,), "float32", 1e-5)),
            ("prelu_077", prelu_run, ((64, 512, 14, 14), (1,), "float32", 1e-5)),
            ("prelu_078", prelu_run, ((64, 512, 7, 7), (1,), "float32", 1e-5)),
            #
        ]
        return

    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
