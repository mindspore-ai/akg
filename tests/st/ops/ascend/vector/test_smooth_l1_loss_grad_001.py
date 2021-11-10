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

import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.ascend.smooth_l1_loss_grad_run import smooth_l1_loss_grad_run


############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):
    def setup(self):
        case_name = "test_smooth_l1_loss_grad"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        kernel = smooth_l1_loss_grad_run
        kernel_name = "smooth_l1_loss_grad"
        self.testarg = [
            ## testflag,opfuncname,testRunArgs, dimArgs
        ]
        self.testarg_cloud = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("test_smooth_l1_loss_grad_05_fp32", kernel, ((1, 16, 4), "float16")),
        ]

        self.testarg_rpc_cloud = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("test_smooth_l1_loss_grad_01_fp16", kernel, ((8, 4718, 4), "float16")),
            ("test_smooth_l1_loss_grad_02_fp32", kernel, ((8, 4718, 4), "float32")),
            ("test_smooth_l1_loss_grad_03_fp16", kernel, ((8, 8732, 4), "float16")),
            ("test_smooth_l1_loss_grad_04_fp16", kernel, ((8, 8732, 4), "float32")),
            #  ("test_smooth_l1_loss_grad_05_fp16_pad", kernel, ((8, 8732, '4,16'), "float16")), # multicore wrong
            ("test_smooth_l1_loss_grad_06_fp16", kernel, ((32, 8732, 4), "float16")),
            ("test_smooth_l1_loss_grad_07_fp16", kernel, ((32, 8732, 4), "float32")),
        ]

        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_run_cloud(self):
        self.common_run(self.testarg_cloud)

    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
