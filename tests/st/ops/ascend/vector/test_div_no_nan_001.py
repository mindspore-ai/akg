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

"""div_no_nan testcases"""

import os
import pytest
from tests.common.base import TestBase


class TestCase(TestBase):

    def setup(self):
        case_name = "test_div_no_nan_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            # shape, dtype,
            ("div_no_nan_001", "div_no_nan_run", (((16,),(16,)), "float32")),
        ]
        self.testarg_cloud = [
            # testflag, opfuncname, testRunArgs, setdimArgs
            # shape, dtype,
            ("001_div_no_nan", "div_no_nan_run", (((1,16),(1,)), "float32")),
        ]

        self.testarg_rpc_cloud = [
            # testflag, opfuncname, testRunArgs, setdimArgs
            # shape, dtype,
            ("div_no_nan", "div_no_nan_run", (((1,16),(1,)), "float16")),
            ("div_no_nan_013", "div_no_nan_run", (((1,16),(1,)), "float16")),
        ]
        self.testarg_level1 = [
            # testflag, opfuncname, testRunArgs, setdimArgs
            # shape, dtype,
            ("div_no_nan_001", "div_no_nan_run", (((16, 128), (16,128)), "float16")),
            ("div_no_nan_002", "div_no_nan_run", (((16,16),(16,16)), "int32")),
            ("div_no_nan_003", "div_no_nan_run", (((16,16),(16,16)), "float32")),
            ("div_no_nan_004", "div_no_nan_run", (((16, 16), (16,16)), "uint8")),
            ("div_no_nan_005", "div_no_nan_run", (((16,16),(16,16)), "int8")),
            ("div_no_nan_006", "div_no_nan_run", (((1,),(1024,)), "float32")),
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
