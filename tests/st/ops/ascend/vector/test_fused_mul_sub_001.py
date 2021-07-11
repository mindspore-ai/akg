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

import os
import pytest
from tests.common.base import TestBase


############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_fused_mul_sub_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("001_fused_mul_sub", "fused_mul_sub_run", ([2], [2], [2], "float16")),
            ("002_fused_mul_sub", "fused_mul_sub_run", ([1024], [1024], [1024], "float16")),
            ("003_fused_mul_sub", "fused_mul_sub_run", ([1025], [1025], [1025], "float16")),
            ("004_fused_mul_sub", "fused_mul_sub_run", ([2, 1024], [2, 1024], [2, 1024], "float16")),
        ]

        self.testarg_rpc_cloud = [
        ]

        self.testarg_cloud_level0 = [
        ]
        self.testarg_level2 = [
        ]
        return

    @pytest.mark.level2
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_run_rpc_cloud(self):
        self.common_run([self.testarg_rpc_cloud[15], self.testarg_rpc_cloud[-1]])

    def test_run_cloud_level0(self):
        self.common_run(self.testarg_cloud_level0)

    def test_run_level2(self):
        self.common_run(self.testarg_level2)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return


if __name__ == "__main__":
    t = TestCase()
    t.setup()
    t.test_run()
    t.teardown()
