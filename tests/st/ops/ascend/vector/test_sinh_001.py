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

"""sinh test case"""

import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.ascend.sinh_run import sinh_run


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_sinh_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("======================== %s Setup case=================", self.casename)
        self.testarg = [
            ("sinh_001", sinh_run, ((8, 16), "float16")),
            ("sinh_002", sinh_run, ((8, 16), "float32")),
        ]
        return

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def teardown(self):
        self._log.info("============= %s Teardown============", self.casename)
        return
