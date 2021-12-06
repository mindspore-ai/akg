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

"""testcase for logical_not op"""

import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.ascend.logical_not_run import logical_not_run


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_logical_not_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            #caseflag,testfuncname,testRunArgs, dimArgs
            ("001_logical_not", logical_not_run, (((128,)), "bool", "logical_not")),
            ("002_logical_not", logical_not_run, (((128, 128)), "bool", "logical_not")),
            ("003_logical_not", logical_not_run, (((1,)), "bool", "logical_not")),
            # Bert shapes from TF(64)
            ("004_logical_not", logical_not_run, (((399,)), "bool", "logical_not")),
            ("005_logical_not", logical_not_run, (((410,)), "bool", "logical_not")),
            ("006_logical_not", logical_not_run, (((1195,)), "bool", "logical_not")),
            # DeepLabV3 shapes
            ("007_logical_not", logical_not_run, (((733,)), "bool", "logical_not")),
            ("008_logical_not", logical_not_run, (((735,)), "bool", "logical_not")),
            ("009_logical_not", logical_not_run, (((1173,)), "bool", "logical_not")),
        ]
        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return

if __name__ == "__main__":
    t = TestCase()
    t.setup()
    t.test_run()
