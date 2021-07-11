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
from tests.common.base import TestBase
from tests.common.test_run.minimum_run import minimum_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):
    def setup(self):
        case_name = "test_minimum_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("minimum_001", minimum_run, ([4, 3], [4, 3], 'float16')),
            ("minimum_002", minimum_run, ([4, 3], [4, 3], 'float32')),
            ("minimum_003", minimum_run, ([4, 3], [4, 3], 'int32')),
            ("minimum_004", minimum_run, ([4, 3], [4, 3], 'int8')),
            ("minimum_005", minimum_run, ([4, 3], [4, 3], 'uint8')),
        ]
        self.testarg_rpc_cloud = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("minimum_006", minimum_run, ([4, 3], [4, 3], 'float32')),
            ("minimum_007", minimum_run, ([4, 3], [4, 3], 'int8')),
            ("minimum_008", minimum_run, ([4, 3], [4, 3], 'int32')),
            ("minimum_009", minimum_run, ([4, 3], [4, 3], 'uint8')),
            ("minimum_010", minimum_run, ([4, 3], [4, 3], 'float16')),
        ]
        return

    @pytest.mark.level2
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

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
