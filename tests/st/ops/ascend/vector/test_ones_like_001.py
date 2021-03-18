# Copyright 2020 Huawei Technologies Co., Ltd
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
from tests.common.test_run.ones_like_run import ones_like_run


class TestOnesLike(TestBase):
    def setup(self):
        case_name = "test_akg_ones_like_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag, opfuncname, testRunArgs, dimArgs
            ("ones_like_01", ones_like_run, ((17, 19), "float16")),
            ("ones_like_02", ones_like_run, ((5, 9, 15), "float16")),
            ("ones_like_03", ones_like_run, ((2, 4, 13, 29), "float16")),
            ("ones_like_04", ones_like_run, ((2, 1024), "float32")),
            ("ones_like_05", ones_like_run, ((32, 4, 30), "float32")),
            ("ones_like_06", ones_like_run, ((32, 4, 30), "int32")),
            ("ones_like_06", ones_like_run, ((32, 4, 30), "int8")),
            ("ones_like_06", ones_like_run, ((32, 4, 30), "uint8")),
        ]
        self.testarg_rpc_cloud = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("ones_like_01", ones_like_run, ((17, 19), "float16")),
            ("ones_like_02", ones_like_run, ((5, 9, 15), "float16")),
            ("ones_like_03", ones_like_run, ((2, 4, 13, 29), "float16")),
            ("ones_like_04", ones_like_run, ((2, 1024), "float32")),
            ("ones_like_05", ones_like_run, ((32, 4, 30), "float32")),
            ("ones_like_06", ones_like_run, ((32, 4, 30), "int32")),
            ("ones_like_06", ones_like_run, ((32, 4, 30), "int8")),
            ("ones_like_06", ones_like_run, ((32, 4, 30), "uint8")),

        ]
        return

    @pytest.mark.level0
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
