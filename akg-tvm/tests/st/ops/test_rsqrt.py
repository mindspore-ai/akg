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
from tests.common.test_run import rsqrt_run

class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_rsqrt_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            #caseflag,testfuncname,testRunArgs, dimArgs
            ("rsqrt_01", rsqrt_run, ((1, 128), "float16", "cce_rsqrt_fp16"), ((1, 1), (128, 128))),
            ("rsqrt_02", rsqrt_run, ((128, 128), "float16", "cce_rsqrt_fp16"), ((64, 64), (128, 128))),
            ("rsqrt_03", rsqrt_run, ((128, 256), "float16", "cce_rsqrt_fp16"), ((32, 32), (256, 256))),
            ("rsqrt_04", rsqrt_run, ((160, 1), "float16", "cce_rsqrt_fp16"), ((160, 160), (1, 1))),
            ("rsqrt_05", rsqrt_run, ((1280, 1), "float16", "cce_rsqrt_fp16"), ((1280, 1280), (1, 1))),
            ("rsqrt_06", rsqrt_run, ((8192, 1), "float16", "cce_rsqrt_fp16"), ((8192, 8192), (1, 1))),
            ("rsqrt_07", rsqrt_run, ((8, 128, 1), "float16", "cce_rsqrt_fp16"), ((8, 8), (128, 128), (1, 1))),
            ("rsqrt_08", rsqrt_run, ((1024, 1), "float16", "cce_rsqrt_fp16"), ((1024, 1024), (1, 1))),
            ("rsqrt_09", rsqrt_run, ((64, 128, 1), "float16", "cce_rsqrt_fp16"), ((64, 64), (128, 128), (1, 1))),
        ]
        self.testarg_cloud = [
            #caseflag,testfuncname,testRunArgs, dimArgs
            ("rsqrt_01", rsqrt_run, ((1, 128), "float32", "cce_rsqrt_fp16"), ((1, 1), (128, 128))),
        ]

        self.args_default = [
            ("000_case", rsqrt_run, ((32, 1024), 'float32'), ["level0"]),
            ("000_case", rsqrt_run, ((32, 1024), 'float32'), ["level0"]),
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
