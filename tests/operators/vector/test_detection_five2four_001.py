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
import pytest
from base import TestBase
from nose.plugins.attrib import attr
from test_run.detection_five2four_run import detection_five2four_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_five2four"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, dimArgs
            # the shape is a tuple of (batch_size, box_number, width and lenth
            ("detection_five2four_001", detection_five2four_run, ((16, 4, 1), "float16")),
            ("detection_five2four_002", detection_five2four_run, ((16, 4, 3), "float16")),
            ("detection_five2four_003", detection_five2four_run, ((16, 6, 5), "float16")),
            #("detection_five2four_004", detection_five2four_run, ((16, 6, 10), "float16")),
            #("detection_five2four_005", detection_five2four_run, ((16, 6, 19), "float16")),
            ("detection_five2four_006", detection_five2four_run, ((16, 4, 38), "float16")),
        ]
        self.testarg_cloud = [
            # testflag,opfuncname,testRunArgs, dimArgs
            # ("five2four_001",five2four_run,((2, 6, 3), "float16")),
        ]

        self.testarg_rpc_cloud = [
            # ("five2four_001",five2four_run,((2, 6, 3), "float16")),
        ]

        return

    @pytest.mark.rpc_mini
    @pytest.mark.level0
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run(self):
        self.common_run(self.testarg)

    @pytest.mark.aicmodel
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_cloud(self):
        self.common_run(self.testarg_cloud)

    @pytest.mark.rpc_cloud
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return
