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
from tests.common.test_run import assign_run

############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_assgin_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.args_default = [
            # caseflag,testfuncname,testRunArgs, dimArgs
            ("test_assgin_001", assign_run, ((30522,), (30522,), "float16"), ["level0"]),
            ("test_assgin_002", assign_run, ((2,), (2,), "float32"), ["level0"]),
            ("test_assgin_003", assign_run, ((4096,), (4096,), "int32"), ["level0"]),
            ("test_assgin_004", assign_run, ((1,), (1,), "int32"), ["level0"]),
        ]
        self.testarg_rpc_cloud = [
            ("test_assgin_001", assign_run, ((1024, 4096), (1024, 4096), "float16", "cce_assgin_fp16")),
            ("test_assgin_002", assign_run, ((2, 1024), (2, 1024), "int32", "cce_assgin_int32")),
            ("test_assgin_003", assign_run, ((512, 1024), (512, 1024), "float16", "cce_assgin_fp16")),
            ("test_assgin_004", assign_run, ((1024,), (1024,), "float32", "cce_assgin_fp32")),
        ]
        self.testarg_cloud = [
            # ci random fail("test_assgin_002", assign_run, ((2,), "float32", "cce_assgin_fp16"), ((2, 2),)),
        ]
        self.testarg_level1 = [
            # caseflag,testfuncname,testRunArgs, dimArgs

            ("test_assgin_003", assign_run, ((30522, 1024), (30522, 1024), "float16", "cce_assgin_fp16")),

        ]
        self.testarg_level2 = [
            # caseflag,testfuncname,testRunArgs, dimArgs

            ("test_assgin_010", assign_run, ((1,), (1,), "int32", "cce_assgin_int32")),

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

    def test_run_cloud(self):
        self.common_run(self.testarg_cloud)

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run_level1(self):
        self.common_run(self.testarg_level1)

    def test_run_level2(self):
        self.common_run(self.testarg_level2)

    def teardown(self):
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
