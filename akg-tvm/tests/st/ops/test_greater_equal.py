# Copyright 2021 Huawei Technologies Co., Ltd
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

"""testcase for greater_equal op"""

import os
import pytest
import akg.utils as utils
from tests.common.base import TestBase
from tests.common.test_run import greater_equal_run

class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_greater_equal_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.args_default = [
            #caseflag,testfuncname,testRunArgs, dimArgs
            ("001_greater_equal", greater_equal_run, (((128,), (128,)), "float16"), ["level0"]),
            ("002_greater_equal", greater_equal_run, (((128, 128), (128, 128)), "float16"), ["level0"]),
            ("003_greater_equal", greater_equal_run, (((1,), (1,)), "float16"), ["level0"]),
        ]
        return True

    @pytest.mark.level0
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_gpu_level0(self):
        return self.run_cases(self.args_default, utils.CUDA, "level0")
    
    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_cpu_level0(self):
        return self.run_cases(self.args_default, utils.LLVM, "level0")

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        return self.run_cases(self.args_default, utils.CCE, "level0")

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return
