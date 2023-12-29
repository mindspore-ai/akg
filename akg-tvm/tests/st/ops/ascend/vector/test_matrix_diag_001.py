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

"""testcase for matrix_diag op"""

import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.ascend.matrix_diag_run import matrix_diag_run


class TestCase(TestBase):
    def setup(self):
        """
        testcase preparcondition
        :return:
        """
        case_name = "test_akg_matrix_diag_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # caseflag,testfuncname,testRunArgs, dimArgs
            ("matrix_diag_01", matrix_diag_run, ((10,), (10, 10), "float16")),
            ("matrix_diag_01", matrix_diag_run, ((10,), (20, 20), "float16")),
            ("matrix_diag_01", matrix_diag_run, ((20, 20), (20, 10, 30), "float16")),
            ("matrix_diag_01", matrix_diag_run, ((10, 10), (10, 10, 10), "float32")),
            ("matrix_diag_01", matrix_diag_run, ((10, 10), (10, 10, 10), "int32")),
        ]
        self.testarg_cloud = [
            # caseflag,testfuncname,testRunArgs, dimArgs
            ("matrix_diag_01", matrix_diag_run, ((5, 10, 20), (5, 10, 20, 20), "float32")),
        ]
        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_run_cloud(self):
        self.common_run(self.testarg_cloud)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
