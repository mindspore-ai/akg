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
import pytest
from nose.plugins.attrib import attr
from base import TestBase
from test_run.axpy_run import axpy_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):

    def setup(self):
        """set test case """
        case_name = "test_axpy_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("axpy_001", axpy_run, ((16,), (16,), -1, "float16")),
            ("axpy_002", axpy_run, ((16, 16), (16, 16), 2, "float16")),
            ("axpy_003", axpy_run, ((16, 16, 16), (16, 16, 16), 2.5, "float32")),
            ("axpy_004", axpy_run, ((6,), (6,), 2.5, "float16")),
            ("axpy_005", axpy_run, ((6, 7), (6, 7), 2.5, "float16")),
            ("axpy_006", axpy_run, ((6, 6, 6), (6, 6, 6), 2.5, "float32")),

        ]
        self.testarg_rpc_cloud = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("axpy_007", axpy_run, ((16,), (16,), 2.5, "float16")),
            ("axpy_008", axpy_run, ((16, 16), (16, 16), 2.5, "float16")),
            ("axpy_009", axpy_run, ((16, 16, 16), (16, 16, 16), 2.5, "float32")),
            ("axpy_010", axpy_run, ((6,), (6,), 2.5, "float16")),
            ("axpy_011", axpy_run, ((6, 6), (6, 6), -2.321321, "float16")),
            ("axpy_012", axpy_run, ((6, 6, 6), (6, 6, 6), 34324, "float32")),

        ]

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
