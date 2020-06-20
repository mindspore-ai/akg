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
from test_run.assign_run import assign_run


############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_assgin_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # caseflag,testfuncname,testRunArgs, dimArgs

            ("test_assgin_001", assign_run, ((30522,), (30522,), "float16", "cce_assgin_fp16")),
            ("test_assgin_002", assign_run, ((2,), (2,), "float32", "cce_assgin_fp32")),
            ("test_assgin_003", assign_run, ((4096,), (4096,), "int32", "cce_assgin_int32")),
            ("test_assgin_004", assign_run, ((1,), (1,), "int32", "cce_assgin_int32")),

            # run 98 s("test_assgin_003", assign_run, ((30522,1024), "float16", "cce_assgin_fp16"), ((32, 32),(1024, 1024))),
            # ci random fail: ("test_assgin_006", assign_run, ((1024,1024), "float16", "cce_assgin_fp16"), ((32, 32),(1024,1024))),
            # run fail ("test_assgin_010", assign_run, ((1, ), "int32", "cce_assgin_int32"), ((1, 1),)),

        ]
        self.testarg_rpc_cloud = [
            ("test_assgin_001", assign_run, ((1024, 4096), (1024, 4096), "float16", "cce_assgin_fp16")),
            ("test_assgin_002", assign_run, ((2, 1024), (2, 1024), "int32", "cce_assgin_int32")),
            ("test_assgin_003", assign_run, ((512, 1024), (512, 1024), "float16", "cce_assgin_fp16")),
            ("test_assgin_004", assign_run, ((1024,), (1024,), "float32", "cce_assgin_fp32")),
        ]
        self.testarg_cloud = [
            # ci random fail("test_assgin_002", assign_run, ((2,), "float32", "cce_assgin_fp16"), ((2, 2),)),
        ]
        self.testarg_level1 = [
            # caseflag,testfuncname,testRunArgs, dimArgs

            ("test_assgin_003", assign_run, ((30522, 1024), (30522, 1024), "float16", "cce_assgin_fp16")),

        ]
        self.testarg_level2 = [
            # caseflag,testfuncname,testRunArgs, dimArgs

            ("test_assgin_010", assign_run, ((1,), (1,), "int32", "cce_assgin_int32")),

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

    @pytest.mark.aicmodel
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_cloud(self):
        self.common_run(self.testarg_cloud)

    @pytest.mark.level1
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_level1(self):
        self.common_run(self.testarg_level1)

    @pytest.mark.level2
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_level2(self):
        self.common_run(self.testarg_level2)

    def teardown(self):
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
