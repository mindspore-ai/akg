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

import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.ascend.apply_proximal_adagrad_run import apply_proximal_adagrad_run


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_apply_proximal_adagrad"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("test_akg_apply_proximal_adagrad_001", apply_proximal_adagrad_run, [(16,), "float16"]),
            ("test_akg_apply_proximal_adagrad_002", apply_proximal_adagrad_run, [(16,), "float32"]),
        ]
        self.testarg_rpc_cloud = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("test_akg_apply_proximal_adagrad_001", apply_proximal_adagrad_run, [(32, 4095), "float16"]),
            ("test_akg_apply_proximal_adagrad_002", apply_proximal_adagrad_run, [(32, 4095,), "float32"]),
        ]

        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
