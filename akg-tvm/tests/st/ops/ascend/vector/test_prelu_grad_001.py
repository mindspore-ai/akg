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
from tests.common.test_run.ascend.prelu_grad_run import prelu_grad_run


class TestCase(TestBase):
    def setup(self):
        case_name = "test_prelu_grad_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        # Set all shape in cloud
        # self.testarg_rpc_cloud = [
        #     # testflag,opfuncname,testRunArgs, setdimArgs
        #     #("prelu_grad_000", prelu_grad_run, ((1, 64, 2, 2), (64,), "float16", 1e-5)),
        #     # ("prelu_grad_001", prelu_grad_run, ((1, 64, 112, 112), (64,), "float16", 1e-3), ((1,1),(64,1),(112,1),(112,1))),
        #     # ("prelu_grad_002", prelu_grad_run, ((1, 64, 56, 56), (64,), "float16", 1e-3), ((1,1),(64,1),(56,1),(56,1))),
        #     # ("prelu_grad_003", prelu_grad_run, ((1, 128, 56, 56), (128,), "float16", 1e-3), ((1,1),(128,1),(56,1),(56,1))),
        #     # ("prelu_grad_004", prelu_grad_run, ((1, 128, 28, 28), (128,), "float16", 1e-3), ((1,1),(128,1),(28,1),(28,1))),
        #     # ("prelu_grad_005", prelu_grad_run, ((1, 256, 28, 28), (256,), "float16", 1e-3), ((1,1),(256,1),(28,1),(28,1))),
        #     # ("prelu_grad_006", prelu_grad_run, ((1, 256, 14, 14), (256,), "float16", 1e-3), ((1,1),(256,1),(14,1),(14,1))),
        #     # ("prelu_grad_007", prelu_grad_run, ((1, 512, 14, 14), (512,), "float16", 1e-3), ((1,1),(512,1),(14,1),(14,1))),
        #     # ("prelu_grad_008", prelu_grad_run, ((1, 512, 7, 7), (512,), "float16", 1e-3), ((1,1),(512,1),(7,1),(7,1))),
        #     # #
        #     # ("prelu_grad_011", prelu_grad_run, ((1, 64, 112, 112), (64,), "float32", 1e-5), ((1,1),(64,1),(112,1),(112,1))),
        #     # ("prelu_grad_012", prelu_grad_run, ((1, 64, 56, 56), (64,), "float32", 1e-5), ((1,1),(64,1),(56,1),(56,1))),
        #     # ("prelu_grad_013", prelu_grad_run, ((1, 128, 56, 56), (128,), "float32", 1e-5), ((1,1),(128,1),(56,1),(56,1))),
        #     # ("prelu_grad_014", prelu_grad_run, ((1, 128, 28, 28), (128,), "float32", 1e-5), ((1,1),(128,1),(28,1),(28,1))),
        #     # ("prelu_grad_015", prelu_grad_run, ((1, 256, 28, 28), (256,), "float32", 1e-5), ((1,1),(256,1),(28,1),(28,1))),
        #     # ("prelu_grad_016", prelu_grad_run, ((1, 256, 14, 14), (256,), "float32", 1e-5), ((1,1),(256,1),(14,1),(14,1))),
        #     # ("prelu_grad_017", prelu_grad_run, ((1, 512, 14, 14), (512,), "float32", 1e-5), ((1,1),(512,1),(14,1),(14,1))),
        #     # ("prelu_grad_018", prelu_grad_run, ((1, 512, 7, 7), (512,), "float32", 1e-5), ((1,1),(512,1),(7,1),(7,1))),
        #     # #
        #     # ("prelu_grad_021", prelu_grad_run, ((1, 64, 112, 112), (1,), "float16", 5e-3), ((1,1),(64,1),(112,1),(112,1))),
        #     # ("prelu_grad_022", prelu_grad_run, ((1, 64, 56, 56), (1,), "float16", 5e-3), ((1,1),(64,1),(56,1),(56,1))),
        #     # ("prelu_grad_023", prelu_grad_run, ((1, 128, 56, 56), (1,), "float16", 5e-3), ((1,1),(128,1),(56,1),(56,1))),
        #     # ("prelu_grad_024", prelu_grad_run, ((1, 128, 28, 28), (1,), "float16", 5e-3), ((1,1),(128,1),(28,1),(28,1))),
        #     # ("prelu_grad_025", prelu_grad_run, ((1, 256, 28, 28), (1,), "float16", 5e-3), ((1,1),(256,1),(28,1),(28,1))),
        #     # ("prelu_grad_026", prelu_grad_run, ((1, 256, 14, 14), (1,), "float16", 5e-3), ((1,1),(256,1),(14,1),(14,1))),
        #     # ("prelu_grad_027", prelu_grad_run, ((1, 512, 14, 14), (1,), "float16", 5e-3), ((1,1),(512,1),(14,1),(14,1))),
        #     # ("prelu_grad_028", prelu_grad_run, ((1, 512, 7, 7), (1,), "float16", 5e-3), ((1,1),(512,1),(7,1),(7,1))),
        #     # #
        #     # ("prelu_grad_031", prelu_grad_run, ((1, 64, 112, 112), (1,), "float32", 1e-5), ((1,1),(64,1),(112,1),(112,1))),
        #     # ("prelu_grad_032", prelu_grad_run, ((1, 64, 56, 56), (1,), "float32", 1e-5), ((1,1),(64,1),(56,1),(56,1))),
        #     # ("prelu_grad_033", prelu_grad_run, ((1, 128, 56, 56), (1,), "float32", 1e-5), ((1,1),(128,1),(56,1),(56,1))),
        #     # ("prelu_grad_034", prelu_grad_run, ((1, 128, 28, 28), (1,), "float32", 1e-5), ((1,1),(128,1),(28,1),(28,1))),
        #     # ("prelu_grad_035", prelu_grad_run, ((1, 256, 28, 28), (1,), "float32", 1e-5), ((1,1),(256,1),(28,1),(28,1))),
        #     # ("prelu_grad_036", prelu_grad_run, ((1, 256, 14, 14), (1,), "float32", 1e-5), ((1,1),(256,1),(14,1),(14,1))),
        #     # ("prelu_grad_037", prelu_grad_run, ((1, 512, 14, 14), (1,), "float32", 1e-5), ((1,1),(512,1),(14,1),(14,1))),
        #     # ("prelu_grad_038", prelu_grad_run, ((1, 512, 7, 7), (1,), "float32", 1e-5), ((1,1),(512,1),(7,1),(7,1))),
        #     #
        #     # ("prelu_grad_041", prelu_grad_run, ((128, 64, 112, 112), (64,), "float16", 5e-03), ((1,1),(1,1),(16,1),(112,1))),
        #     # ("prelu_grad_042", prelu_grad_run, ((128, 64, 56, 56), (64,), "float16", 5e-03), ((1,1),(1,1),(56,1),(56,1))),
        #     # ("prelu_grad_043", prelu_grad_run, ((128, 128, 56, 56), (128,), "float16", 5e-03), ((1,1),(1,1),(56,1),(56,1))),
        #     # ("prelu_grad_044", prelu_grad_run, ((128, 128, 28, 28), (128,), "float16", 5e-03), ((1,1),(1,1),(28,1),(28,1))),
        #     # ("prelu_grad_045", prelu_grad_run, ((128, 256, 28, 28), (256,), "float16", 5e-03), ((1,1),(1,1),(28,1),(28,1))),
        #     # ("prelu_grad_046", prelu_grad_run, ((128, 256, 14, 14), (256,), "float16", 5e-03), ((1,1),(1,1),(14,1),(14,1))),
        #     # ("prelu_grad_047", prelu_grad_run, ((128, 512, 14, 14), (512,), "float16", 5e-03), ((1,1),(1,1),(14,1),(14,1))),
        #     # ("prelu_grad_048", prelu_grad_run, ((128, 512, 7, 7), (512,), "float16", 5e-03), ((1,1),(1,1),(7,1),(7,1))),
        #     # #
        #     # ("prelu_grad_051", prelu_grad_run, ((128, 64, 112, 112), (64,), "float32", 1e-5), ((1,1),(1,1),(112,1),(112,1))),
        #     # ("prelu_grad_052", prelu_grad_run, ((128, 64, 56, 56), (64,), "float32", 1e-5), ((1,1),(1,1),(56,1),(56,1))),
        #     # ("prelu_grad_053", prelu_grad_run, ((128, 128, 56, 56), (128,), "float32", 1e-5), ((1,1),(1,1),(56,1),(56,1))),
        #     # ("prelu_grad_054", prelu_grad_run, ((128, 128, 28, 28), (128,), "float32", 1e-5), ((1,1),(1,1),(28,1),(28,1))),
        #     # ("prelu_grad_055", prelu_grad_run, ((128, 256, 28, 28), (256,), "float32", 1e-5), ((1,1),(1,1),(28,1),(28,1))),
        #     # ("prelu_grad_056", prelu_grad_run, ((128, 256, 14, 14), (256,), "float32", 1e-5), ((1,1),(1,1),(14,1),(14,1))),
        #     # ("prelu_grad_057", prelu_grad_run, ((128, 512, 14, 14), (512,), "float32", 1e-5), ((1,1),(1,1),(14,1),(14,1))),
        #     # ("prelu_grad_058", prelu_grad_run, ((128, 512, 7, 7), (512,), "float32", 1e-5), ((1,1),(1,1),(7,1),(7,1))),
        #     # #
        #     ("prelu_grad_061", prelu_grad_run, ((128, 64, 112, 112), (1,), "float16", 5e-03), ((1,1),(1,1),(112,1),(112,1))),
        #     ("prelu_grad_062", prelu_grad_run, ((128, 64, 56, 56), (1,), "float16", 5e-03), ((1,1),(1,1),(56,1),(56,1))),
        #     ("prelu_grad_063", prelu_grad_run, ((128, 128, 56, 56), (1,), "float16", 5e-03), ((1,1),(1,1),(56,1),(56,1))),
        #     ("prelu_grad_064", prelu_grad_run, ((128, 128, 28, 28), (1,), "float16", 5e-03), ((1,1),(1,1),(28,1),(28,1))),
        #     ("prelu_grad_065", prelu_grad_run, ((128, 256, 28, 28), (1,), "float16", 5e-03), ((1,1),(1,1),(28,1),(28,1))),
        #     ("prelu_grad_066", prelu_grad_run, ((128, 256, 14, 14), (1,), "float16", 5e-03), ((1,1),(1,1),(14,1),(14,1))),
        #     ("prelu_grad_067", prelu_grad_run, ((128, 512, 14, 14), (1,), "float16", 5e-03), ((1,1),(1,1),(14,1),(14,1))),
        #     ("prelu_grad_068", prelu_grad_run, ((128, 512, 7, 7), (1,), "float16", 5e-03), ((1,1),(1,1),(7,1),(7,1))),
        #     #
        #     ("prelu_grad_071", prelu_grad_run, ((128, 64, 112, 112), (1,), "float32", 1e-5), ((1,1),(1,1),(112,1),(112,1))),
        #     ("prelu_grad_072", prelu_grad_run, ((128, 64, 56, 56), (1,), "float32", 1e-5), ((1,1),(1,1),(56,1),(56,1))),
        #     ("prelu_grad_073", prelu_grad_run, ((128, 128, 56, 56), (1,), "float32", 1e-5), ((1,1),(1,1),(56,1),(56,1))),
        #     ("prelu_grad_074", prelu_grad_run, ((128, 128, 28, 28), (1,), "float32", 1e-5), ((1,1),(1,1),(28,1),(28,1))),
        #     ("prelu_grad_075", prelu_grad_run, ((128, 256, 28, 28), (1,), "float32", 1e-5), ((1,1),(1,1),(28,1),(28,1))),
        #     ("prelu_grad_076", prelu_grad_run, ((128, 256, 14, 14), (1,), "float32", 1e-5), ((1,1),(1,1),(14,1),(14,1))),
        #     ("prelu_grad_077", prelu_grad_run, ((128, 512, 14, 14), (1,), "float32", 1e-5), ((1,1),(1,1),(14,1),(14,1))),
        #     ("prelu_grad_078", prelu_grad_run, ((128, 512, 7, 7), (1,), "float32", 1e-5), ((1,1),(1,1),(7,1),(7,1))),
        #     #
        # ]

        self.testarg_rpc_cloud = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            #("prelu_grad_000", prelu_grad_run, ((1, 64, 2, 2), (64,), "float16", 1e-5)),
            ("prelu_grad_001", prelu_grad_run, ((1, 64, 112, 112), (64,), "float16", 5e-3)),
            # ("prelu_grad_002", prelu_grad_run, ((1, 64, 56, 56), (64,), "float16", 5e-3)),
            # ("prelu_grad_003", prelu_grad_run, ((1, 128, 56, 56), (128,), "float16", 5e-3)),
            # ("prelu_grad_004", prelu_grad_run, ((1, 128, 28, 28), (128,), "float16", 5e-3)),
            # ("prelu_grad_005", prelu_grad_run, ((1, 256, 28, 28), (256,), "float16", 5e-3)),
            # ("prelu_grad_006", prelu_grad_run, ((1, 256, 14, 14), (256,), "float16", 5e-3)),
            # ("prelu_grad_007", prelu_grad_run, ((1, 512, 14, 14), (512,), "float16", 5e-3)),
            # ("prelu_grad_008", prelu_grad_run, ((1, 512, 7, 7), (512,), "float16", 5e-3)),
            # #
            # ("prelu_grad_011", prelu_grad_run, ((1, 64, 112, 112), (64,), "float32", 1e-5)),
            # ("prelu_grad_012", prelu_grad_run, ((1, 64, 56, 56), (64,), "float32", 1e-5)),
            # ("prelu_grad_013", prelu_grad_run, ((1, 128, 56, 56), (128,), "float32", 1e-5)),
            # ("prelu_grad_014", prelu_grad_run, ((1, 128, 28, 28), (128,), "float32", 1e-5)),
            # ("prelu_grad_015", prelu_grad_run, ((1, 256, 28, 28), (256,), "float32", 1e-5)),
            # ("prelu_grad_016", prelu_grad_run, ((1, 256, 14, 14), (256,), "float32", 1e-5)),
            # ("prelu_grad_017", prelu_grad_run, ((1, 512, 14, 14), (512,), "float32", 1e-5)),
            # ("prelu_grad_018", prelu_grad_run, ((1, 512, 7, 7), (512,), "float32", 1e-5)),
            # #
            # ("prelu_grad_021", prelu_grad_run, ((1, 64, 112, 112), (1,), "float16", 5e-3)),
            # ("prelu_grad_022", prelu_grad_run, ((1, 64, 56, 56), (1,), "float16", 5e-3)),
            # ("prelu_grad_023", prelu_grad_run, ((1, 128, 56, 56), (1,), "float16", 5e-3)),
            # ("prelu_grad_024", prelu_grad_run, ((1, 128, 28, 28), (1,), "float16", 5e-3)),
            # ("prelu_grad_025", prelu_grad_run, ((1, 256, 28, 28), (1,), "float16", 5e-3)),
            # ("prelu_grad_026", prelu_grad_run, ((1, 256, 14, 14), (1,), "float16", 5e-3)),
            # ("prelu_grad_027", prelu_grad_run, ((1, 512, 14, 14), (1,), "float16", 5e-3)),
            # ("prelu_grad_028", prelu_grad_run, ((1, 512, 7, 7), (1,), "float16", 5e-3)),
            # #
            # ("prelu_grad_031", prelu_grad_run, ((1, 64, 112, 112), (1,), "float32", 1e-5)),
            # ("prelu_grad_032", prelu_grad_run, ((1, 64, 56, 56), (1,), "float32", 1e-5)),
            # ("prelu_grad_033", prelu_grad_run, ((1, 128, 56, 56), (1,), "float32", 1e-5)),
            # ("prelu_grad_034", prelu_grad_run, ((1, 128, 28, 28), (1,), "float32", 1e-5)),
            # ("prelu_grad_035", prelu_grad_run, ((1, 256, 28, 28), (1,), "float32", 1e-5)),
            # ("prelu_grad_036", prelu_grad_run, ((1, 256, 14, 14), (1,), "float32", 1e-5)),
            # ("prelu_grad_037", prelu_grad_run, ((1, 512, 14, 14), (1,), "float32", 1e-5)),
            # ("prelu_grad_038", prelu_grad_run, ((1, 512, 7, 7), (1,), "float32", 1e-5)),

            # ("prelu_grad_041", prelu_grad_run, ((128, 64, 112, 112), (64,), "float16", 5e-03)),
            # ("prelu_grad_042", prelu_grad_run, ((128, 64, 56, 56), (64,), "float16", 5e-03)),
            # ("prelu_grad_043", prelu_grad_run, ((128, 128, 56, 56), (128,), "float16", 5e-03)),
            # ("prelu_grad_044", prelu_grad_run, ((128, 128, 28, 28), (128,), "float16", 5e-03)),
            # ("prelu_grad_045", prelu_grad_run, ((128, 256, 28, 28), (256,), "float16", 5e-03)),
            # ("prelu_grad_046", prelu_grad_run, ((128, 256, 14, 14), (256,), "float16", 5e-03)),
            # ("prelu_grad_047", prelu_grad_run, ((128, 512, 14, 14), (512,), "float16", 5e-03)),
            # ("prelu_grad_048", prelu_grad_run, ((128, 512, 7, 7), (512,), "float16", 5e-03)),
            # #
            # ("prelu_grad_051", prelu_grad_run, ((128, 64, 112, 112), (64,), "float32", 1e-4)),
            # ("prelu_grad_052", prelu_grad_run, ((128, 64, 56, 56), (64,), "float32", 1e-5)),
            # ("prelu_grad_053", prelu_grad_run, ((128, 128, 56, 56), (128,), "float32", 1e-5)),
            # ("prelu_grad_054", prelu_grad_run, ((128, 128, 28, 28), (128,), "float32", 1e-5)),
            # ("prelu_grad_055", prelu_grad_run, ((128, 256, 28, 28), (256,), "float32", 1e-5)),
            # ("prelu_grad_056", prelu_grad_run, ((128, 256, 14, 14), (256,), "float32", 1e-5)),
            # ("prelu_grad_057", prelu_grad_run, ((128, 512, 14, 14), (512,), "float32", 1e-5)),
            # ("prelu_grad_058", prelu_grad_run, ((128, 512, 7, 7), (512,), "float32", 1e-5)),
            # #
            # ("prelu_grad_061", prelu_grad_run, ((128, 64, 112, 112), (1,), "float16", 5e-03)),
            # ("prelu_grad_062", prelu_grad_run, ((128, 64, 56, 56), (1,), "float16", 5e-03)),
            # ("prelu_grad_063", prelu_grad_run, ((128, 128, 56, 56), (1,), "float16", 5e-03)),
            # ("prelu_grad_064", prelu_grad_run, ((128, 128, 28, 28), (1,), "float16", 5e-03)),
            # ("prelu_grad_065", prelu_grad_run, ((128, 256, 28, 28), (1,), "float16", 5e-03)),
            # ("prelu_grad_066", prelu_grad_run, ((128, 256, 14, 14), (1,), "float16", 5e-03)),
            # ("prelu_grad_067", prelu_grad_run, ((128, 512, 14, 14), (1,), "float16", 5e-03)),
            # ("prelu_grad_068", prelu_grad_run, ((128, 512, 7, 7), (1,), "float16", 5e-03)),
            # #
            # ("prelu_grad_071", prelu_grad_run, ((128, 64, 112, 112), (1,), "float32", 1e-5)),
            # ("prelu_grad_072", prelu_grad_run, ((128, 64, 56, 56), (1,), "float32", 1e-5)),
            # ("prelu_grad_073", prelu_grad_run, ((128, 128, 56, 56), (1,), "float32", 1e-5)),
            # ("prelu_grad_074", prelu_grad_run, ((128, 128, 28, 28), (1,), "float32", 1e-5)),
            # ("prelu_grad_075", prelu_grad_run, ((128, 256, 28, 28), (1,), "float32", 1e-5)),
            # ("prelu_grad_076", prelu_grad_run, ((128, 256, 14, 14), (1,), "float32", 1e-5)),
            # ("prelu_grad_077", prelu_grad_run, ((128, 512, 14, 14), (1,), "float32", 1e-5)),
            # ("prelu_grad_078", prelu_grad_run, ((128, 512, 7, 7), (1,), "float32", 1e-5)),
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
