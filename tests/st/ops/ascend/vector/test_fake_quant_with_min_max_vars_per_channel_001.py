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
from tests.common.test_run.fake_quant_with_min_max_vars_per_channel_run import fake_quant_with_min_max_vars_per_channel_run


class TestCase(TestBase):

    def setup(self):
        """set test case """
        case_name = "test_fake_quant_with_min_max_vars_per_channel_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= %s Setup case============", self.casename)
        self.testarg = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("fake_quant_with_min_max_vars_per_channel_001", fake_quant_with_min_max_vars_per_channel_run,
             ((2, 3), (3, ), (3, ), "float32", 8, False)),
            ("fake_quant_with_min_max_vars_per_channel_001", fake_quant_with_min_max_vars_per_channel_run,
             ((4, 4, 4, 4), (4, ), (4, ), "float32", 8, True)),

        ]
        self.testarg_rpc_cloud = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("fake_quant_with_min_max_vars_per_channel_001", fake_quant_with_min_max_vars_per_channel_run,
             ((2, 3), (3, ), (3, ), "float32", 8, False)),
            ("fake_quant_with_min_max_vars_per_channel_001", fake_quant_with_min_max_vars_per_channel_run,
             ((2, 3), (3, ), (3, ), "float32", 8, True)),
            ("fake_quant_with_min_max_vars_per_channel_001", fake_quant_with_min_max_vars_per_channel_run,
             ((4, ), (4, ), (4, ), "float32", 8, True)),
            ("fake_quant_with_min_max_vars_per_channel_001", fake_quant_with_min_max_vars_per_channel_run,
             ((4, 4, 4, 4), (4, ), (4, ), "float32", 8, False)),
        ]

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

    def test_run_rpc_cloud(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_rpc_cloud)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= %s Setup case============", self.casename)
