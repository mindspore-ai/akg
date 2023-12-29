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

# pylint: disable=invalid-name, unused-variable
"""
test_concat
"""

import os
import pytest
import akg.utils as utils
from tests.common.base import TestBase
from tests.common.test_run import expand_dims_run

class TestCase(TestBase):

    def setup(self):
        """
        testcase preparcondition
        :return:
        """
        case_name = "test_akg_expand_dims_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            #testRunArgs, dimArgs
            ("test_expand_dims_1", expand_dims_run, ((8, 128), 2, "int32"), ((8, 8), (128, 128))),
            ("test_expand_dims_2", expand_dims_run, ((64, 128), 2, "int32"), ((64, 64), (128, 128))),
            ("test_expand_dims_3", expand_dims_run, ((64, 128, 128), 1, "float16"), ((1, 1), (128, 128), (128, 128))),
            # test_expand_dims_4, shape: int32-int32:[]-[]=int32:[1]
            ("test_expand_dims_5", expand_dims_run, ((8, 128, 128), 1, "float16"), ((1, 1), (128, 128), (128, 128))),
        ]
        self.testarg_cloud = [
            #testRunArgs, dimArgs
            ("test_expand_dims_5", expand_dims_run, ((8, 128, 128), 1, "float32"), ((1, 1), (128, 128), (128, 128))),
        ]
        self.args_default = [
            ("000_case", expand_dims_run, ((32, 128, 128), 1, 'float32'), ["level0"]),
            ("000_case", expand_dims_run, ((32, 128, 128), 1, 'float16'), ["level0"]),
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
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

    def test_run_cloud(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_cloud)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
