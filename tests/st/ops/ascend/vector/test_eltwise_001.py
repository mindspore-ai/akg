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

"""eltwise testcases"""

import os
import pytest
from tests.common.base import TestBase


class TestCase(TestBase):

    def setup(self):
        case_name = "test_eltwise_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            # shape, dtype, n, mod, coeff
        ]
        self.testarg_cloud = [
            # testflag, opfuncname, testRunArgs, setdimArgs
            # shape, dtype, n, mod, coeff
        ]

        self.testarg_rpc_cloud = [
            # testflag, opfuncname, testRunArgs, setdimArgs
            # shape, dtype, n, mod, coeff
        ]
        self.testarg_level1 = [
            # testflag, opfuncname, testRunArgs, setdimArgs
            # shape, dtype, n, mod, coeff
            ("eltwise_001", "eltwise_run", ((1, 16), "float16", 2, 0, ()),),
            ("eltwise_002", "eltwise_run", ((1, 16), "float32", 2, 0, ()),),
            ("eltwise_003", "eltwise_run", (( 1, 16), "float16", 2, 1, (2, 1)),),
            ("eltwise_004", "eltwise_run", (( 1, 16), "float32", 2, 1, (1, 1)),),
            ("eltwise_005", "eltwise_run", (( 1, 16), "float32", 2, 1, (2, 1)),),
            ("eltwise_006", "eltwise_run", (( 1, 16), "float32", 2, 1, (1, 2)),),
            ("eltwise_007", "eltwise_run", (( 1, 16), "float16", 2, 2, ()),),
            ("eltwise_008", "eltwise_run", (( 1, 16), "float32", 2, 2, ()),),
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

    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run_level1(self):
        self.common_run(self.testarg_level1)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return
