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
from test_run.argmax_run import argmax_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_argmax_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("argmax_001", argmax_run, ((3, 1020), "float32", 1),),
            ("argmax_002", argmax_run, ((3, 2, 9019), "int8", -3),),
            ("argmax_003", argmax_run, ((75,), "int32", -1),),
            ("argmax_004", argmax_run, ((75,), "float32", -1),),
            ("argmax_005", argmax_run, ((75,), "int8", -1),),
            ("argmax_006", argmax_run, ((75,), "float16", -1),),
            ("argmax_007", argmax_run, ((4, 9, 5, 2), "float16", -1),),
            ("argmax_008", argmax_run, ((3, 1020), "float16", -1),),
            ("argmax_009", argmax_run, ((3, 896), "float16", -1), ),
            ("argmax_012", argmax_run, ((6, 1024), "float16", -1), ),

            # deeplabv3
            ("argmax_1_513_513_21_dim_4", argmax_run, ((1, 513, 513, 21), "float32", -1)),
        ]
        self.testarg_rpc_cloud = [

            ("argmax_001", argmax_run, ((32, 10), "float32", -1, )),
            ("argmax_002", argmax_run, ((32, 10), "float16", -1, )),

        ]
        self.testarg_level2 = [
        ]
        return

    @pytest.mark.rpc_mini
    @pytest.mark.level0
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run(self):
        self.common_run(self.testarg)

    @pytest.mark.level2
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_level2(self):
        self.common_run(self.testarg_level2)

    @pytest.mark.rpc_cloud
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_rpc_cloud(self):
        self.common_run([self.testarg_rpc_cloud[0]])

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return
