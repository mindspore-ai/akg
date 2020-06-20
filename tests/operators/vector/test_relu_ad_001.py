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
import datetime
import os

from base import TestBase
import pytest
from test_run.relu_ad_run import relu_ad_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_relu_ad_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, dimArgs
            #("relu_ad_001",relu_ad_run,((1, 128), "float16"), ((16, 0), (1, 0))),
            ("relu_ad_001", relu_ad_run, ((1, 128), "float16")),
            ("relu_ad_002", relu_ad_run, ((4, 129, 129, 256), "float16")),
            ("relu_ad_003", relu_ad_run, ((4, 129, 129, 48), "float16")),
            ("relu_ad_004", relu_ad_run, ((4, 257, 257, 128), "float16")),
            ("relu_ad_005", relu_ad_run, ((4, 257, 257, 64), "float16")),
            ("relu_ad_006", relu_ad_run, ((4, 65, 65, 728), "float16"), ((1, 1), (1, 1), (1, 1), (728, 1))),
            ("relu_ad_007", relu_ad_run, ((4, 33, 33, 256), "float16")),
            ("relu_ad_008", relu_ad_run, ((4, 33, 33, 2048), "float16")),
            ("relu_ad_009", relu_ad_run, ((4, 33, 33, 1024), "float16")),
            ("relu_ad_010", relu_ad_run, ((4, 33, 33, 728), "float16"), ((1, 1), (1, 1), (1, 1), (768, 1))),
            ("relu_ad_011", relu_ad_run, ((4, 1, 1, 256), "float16")),
            ("relu_ad_012", relu_ad_run, ((4, 129, 129, 128), "float16")),
            ("relu_ad_013", relu_ad_run, ((4, 257, 257, 32), "float16")),
            ("relu_ad_014", relu_ad_run, ((4, 129, 129, 304), "float16")),
            ("relu_ad_015", relu_ad_run, ((4, 33, 33, 1536), "float16")),
            ("relu_ad_016", relu_ad_run, ((4, 65, 65, 256), "float16")),

            # resnet50
            ("relu_ad_017", relu_ad_run, ((1, 64, 112, 112), "float16")),
            ("relu_ad_018", relu_ad_run, ((1, 64, 56, 56), "float16")),
            ("relu_ad_019", relu_ad_run, ((1, 256, 56, 56), "float16")),
            ("relu_ad_020", relu_ad_run, ((1, 64, 28, 28), "float16")),
            ("relu_ad_021", relu_ad_run, ((1, 256, 28, 28), "float16")),
            ("relu_ad_022", relu_ad_run, ((1, 128, 28, 28), "float16")),
            ("relu_ad_023", relu_ad_run, ((1, 512, 28, 28), "float16")),
            ("relu_ad_024", relu_ad_run, ((1, 128, 14, 14), "float16")),
            ("relu_ad_025", relu_ad_run, ((1, 512, 14, 14), "float16")),
            ("relu_ad_026", relu_ad_run, ((1, 256, 14, 14), "float16")),
            ("relu_ad_027", relu_ad_run, ((1, 1024, 14, 14), "float16")),
            ("relu_ad_028", relu_ad_run, ((1, 256, 7, 7), "float16")),
            ("relu_ad_029", relu_ad_run, ((1, 1024, 7, 7), "float16")),
            ("relu_ad_030", relu_ad_run, ((1, 512, 7, 7), "float16")),
            ("relu_ad_031", relu_ad_run, ((1, 2048, 7, 7), "float16")),
        ]
        # Set all shape in cloud
        self.testarg_rpc_cloud = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("relu_ad_001", relu_ad_run, ((32, 4, 112, 112, 16), "float16")),
            ("relu_ad_002", relu_ad_run, ((32, 4, 56, 56, 16), "float16")),
            ("relu_ad_003", relu_ad_run, ((32, 16, 56, 56, 16), "float16")),
            ("relu_ad_004", relu_ad_run, ((32, 8, 28, 28, 16), "float16")),
            ("relu_ad_005", relu_ad_run, ((32, 32, 28, 28, 16), "float16")),
            ("relu_ad_006", relu_ad_run, ((32, 16, 14, 14, 16), "float16")),
            ("relu_ad_007", relu_ad_run, ((32, 64, 14, 14, 16), "float16")),
            ("relu_ad_008", relu_ad_run, ((32, 32, 7, 7, 16), "float16")),
            ("relu_ad_009", relu_ad_run, ((32, 128, 7, 7, 16), "float16")),
            ("relu_ad_010", relu_ad_run, ((32, 4, 112, 112, 16), "float32")),
            ("relu_ad_011", relu_ad_run, ((32, 4, 56, 56, 16), "float32")),
            ("relu_ad_012", relu_ad_run, ((32, 16, 56, 56, 16), "float32")),
            ("relu_ad_013", relu_ad_run, ((32, 8, 28, 28, 16), "float32")),
            ("relu_ad_014", relu_ad_run, ((32, 32, 28, 28, 16), "float32")),
            ("relu_ad_015", relu_ad_run, ((32, 16, 14, 14, 16), "float32")),
            ("relu_ad_016", relu_ad_run, ((32, 64, 14, 14, 16), "float32")),
            ("relu_ad_017", relu_ad_run, ((32, 32, 7, 7, 16), "float32")),
            ("relu_ad_018", relu_ad_run, ((32, 128, 7, 7, 16), "float32")),

        ]

        self.testarg_resnet50 = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("test_resnet50_relu_ad_001", relu_ad_run, ((32, 128, 7, 7, 16), "float32")),
            ("test_resnet50_relu_ad_002", relu_ad_run, ((32, 64, 14, 14, 16), "float32")),
            ("test_resnet50_relu_ad_003", relu_ad_run, ((32, 8, 28, 28, 16), "float32")),
            ("test_resnet50_relu_ad_004", relu_ad_run, ((32, 16, 14, 14, 16), "float32")),
            ("test_resnet50_relu_ad_005", relu_ad_run, ((32, 16, 56, 56, 16), "float32")),
            ("test_resnet50_relu_ad_006", relu_ad_run, ((32, 32, 28, 28, 16), "float32")),
            ("test_resnet50_relu_ad_007", relu_ad_run, ((32, 32, 7, 7, 16), "float32")),
            ("test_resnet50_relu_ad_008", relu_ad_run, ((32, 4, 112, 112, 16), "float32")),
            ("test_resnet50_relu_ad_009", relu_ad_run, ((32, 4, 56, 56, 16), "float32")),
            ("test_resnet50_relu_ad_010", relu_ad_run, ((32, 8, 56, 56, 16), "float32")),
            ("test_resnet50_relu_ad_011", relu_ad_run, ((32, 16, 28, 28, 16), "float32")),
            ("test_resnet50_relu_ad_012", relu_ad_run, ((32, 32, 14, 14, 16), "float32")),

        ]

        return

    @pytest.mark.rpc_mini
    @pytest.mark.level0
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run(self):
        self.common_run(self.testarg)

    @pytest.mark.rpc_cloud
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_rpc_cloud(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_rpc_cloud)

    @pytest.mark.rpc_cloud
    @pytest.mark.level1
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_restnet(self):
        self.common_run(self.testarg_resnet50)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return
