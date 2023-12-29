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

"""xdivy_grad test case"""

import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.ascend.xdivy_grad_run import xdivy_grad_run


class TestCos(TestBase):
    def setup(self):
        case_name = "test_akg_xdivy_grad_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("========================{0}  Setup case=================".format(self.casename))
        self.testarg = [
            # testflag, opfuncname, (shape1, shape2, dtype) 
            ("xdivy_grad_f16_01", xdivy_grad_run, ((32, 16), (32, 16), "float16")),
            ("xdivy_grad_f16_02", xdivy_grad_run, ((16,), (32, 16), "float16")),
            ("xdivy_grad_f16_03", xdivy_grad_run, ((32, 16), (16,), "float16")),
            ("xdivy_grad_f32_04", xdivy_grad_run, ((32, 16), (32, 16), "float32")),
            ("xdivy_grad_f32_05", xdivy_grad_run, ((16,), (32, 16), "float32")),
            ("xdivy_grad_f32_06", xdivy_grad_run, ((32, 16), (16,), "float32")),
        ]
        return

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
