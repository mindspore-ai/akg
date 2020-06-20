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

"""xlogy_grad test case"""

import os

from base import TestBase
import pytest
from test_run.xlogy_grad_run import xlogy_grad_run


class TestCos(TestBase):
    def setup(self):
        case_name = "test_akg_xlogy_grad_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("========================{0}  Setup case=================".format(self.casename))
        self.testarg = [
            # testflag, opfuncname, (shape1, shape2, dtype)
            # if there is two different shape, the different part should not
            # be too big for mini device because the bad precision.
            ("xlogy_grad_f16_01", xlogy_grad_run, ((32, 16), (32, 16), "float16")),
            ("xlogy_grad_f16_02", xlogy_grad_run, ((16,), (5, 16), "float16")),
            ("xlogy_grad_f16_03", xlogy_grad_run, ((32, 16), (16,), "float16")),
            ("xlogy_grad_f32_04", xlogy_grad_run, ((32, 16), (32, 16), "float32")),
            ("xlogy_grad_f32_05", xlogy_grad_run, ((16,), (5, 16), "float32")),
            ("xlogy_grad_f32_06", xlogy_grad_run, ((32, 16), (16,), "float32")),
        ]
        self.testarg_cloud = [
            # testflag, opfuncname, (shape1, shape2, dtype) 
            ("xlogy_grad_f16_01", xlogy_grad_run, ((32, 16), (32, 16), "float16")),
            ("xlogy_grad_f16_02", xlogy_grad_run, ((16,), (256, 16), "float16")),
            ("xlogy_grad_f16_03", xlogy_grad_run, ((32, 16), (16,), "float16")),
            ("xlogy_grad_f32_04", xlogy_grad_run, ((32, 16), (32, 16), "float32")),
            ("xlogy_grad_f32_05", xlogy_grad_run, ((16,), (256, 16), "float32")),
            ("xlogy_grad_f32_06", xlogy_grad_run, ((32, 16), (16,), "float32")),
        ]
        self.testarg_bug = [
            ("xlogy_grad_f16_02", xlogy_grad_run, ((16,), (32, 16), "float16")),
            ("xlogy_grad_f16_02", xlogy_grad_run, ((16,), (32, 16), "float32")),
        ]
        return

    @pytest.mark.rpc_mini
    @pytest.mark.level1
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run(self):
        """run mini case"""
        self.common_run(self.testarg)

    @pytest.mark.level2
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_bug_run(self):
        """bug case in mini: if len(shape1) < len(shape2), the precision maynot meet demand"""
        self.common_run(self.testarg_bug)

    @pytest.mark.rpc_cloud
    @pytest.mark.level1
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_cloud_run(self):
        """run cloud case"""
        self.common_run(self.testarg_cloud)

    def teardown(self):
        """clean environment"""
        self._log.info("============= {0} Teardown============".format(self.casename))
