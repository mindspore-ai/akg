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

import os
import pytest
import akg.utils as utils
from tests.common.base import TestBase
from tests.common.test_run import reduce_min_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):
    def setup(self):
        case_name = "reduce_min"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        
        self.testarg = [
            # testflag, opfuncname, testRunArgs, dimArgs
            # shape, axis, keepdims, dtype, attrs
            ("000_case", reduce_min_run, ((32, 64, 32), "float16", (1,), True), ["level0"]),
            ("001_case", reduce_min_run, ((4, 128, 1024), "float32", (0, 1), True), ["level0"]),
            ("002_case", reduce_min_run, ((8, 256), "uint8", (0,), False), ["level0"]),
            ("003_case", reduce_min_run, ((1024,), "int32", (0,), True), ["level0"]),
        ]
        self.testarg_ascend = [
            ("004_case", reduce_min_run, ((2, 1280), "int8", (1,), False), ["level0"]),
        ]

        return True

    @pytest.mark.level0
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_gpu_level0(self):
        return self.run_cases(self.testarg, utils.CUDA, "level0")
    
    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_cpu_level0(self):
        return self.run_cases(self.testarg, utils.LLVM, "level0")

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        return self.run_cases(self.testarg + self.testarg_ascend, utils.CCE, "level0")

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
