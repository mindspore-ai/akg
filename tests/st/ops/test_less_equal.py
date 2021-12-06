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

"""
################################################

Testcase_PrepareCondition:

Testcase_TestSteps:

Testcase_ExpectedResult:

"""
import os
import pytest
import akg.utils as utils
from tests.common.base import TestBase
from tests.common.test_run import less_equal_run

############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):
    def setup(self):
        case_name = "test_auto_less_equal_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.args_default = [
            ("003_less_equal", less_equal_run, (((1,), (1,)), "float32"), ["level0"]),
            ("003_less_equal", less_equal_run, (((1,), (1,)), "int32"), ["level0"]),
            ("001_less_equal", less_equal_run, (((128,), (128,)), "float16"), ["level0"]),
            ("002_less_equal", less_equal_run, (((128, 128), (128, 128)), "float32"), ["level0"]),
            ("003_less_equal", less_equal_run, (((1,), (1,)), "float16"), ["level0"]),
            ("005_less_equal", less_equal_run, (((64, 128, 768), (1,)), "float16"), ["level0"]),
            ("006_less_equal", less_equal_run, (((128, 128, 64), (1,)), "float16"), ["level0"]),
            ("007_less_equal", less_equal_run, (((2, 1), (3, 3, 1, 3)), "float16"), ["level0"]),
        ]

        self.testarg_rpc_cloud = [
            ("003_less_equal", less_equal_run, (((1,), (1,)), "int32")),
            ("002_less_equal", less_equal_run, (((1,), (1,)), "float32")),
            ("003_less_equal", less_equal_run, (((128, 128, 64), (128, 128, 64)), "float32")),
            ("004_less_equal", less_equal_run, (((64, 128, 768), (1,)), "float32")),
            ("005_less_equal", less_equal_run, (((128, 128, 64), (1,)), "float32")),
            ("less_equal_001", less_equal_run, (((64, 128, 768), (64, 128, 768)), "float32")),
            ("less_equal_002", less_equal_run, (((128, 128, 64), (128, 128, 64)), "float32")),

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

    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return
