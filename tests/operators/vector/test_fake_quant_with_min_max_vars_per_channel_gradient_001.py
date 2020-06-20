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
from nose.plugins.attrib import attr
import pytest
from base import TestBase
from test_run.fake_quant_with_min_max_vars_per_channel_gradient_run import fake_quant_with_min_max_vars_per_channel_gradient_run


class TestCase(TestBase):

    def setup(self):
        """set test case """
        case_name = "test_fake_quant_with_min_max_vars_per_channel_gradient_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= %s Setup case============", self.casename)
        self.testarg = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("fake_quant_with_min_max_vars_per_channel_gradient_001", fake_quant_with_min_max_vars_per_channel_gradient_run,
             ((2, 3), (2, 3), (3, ), (3, ), "float32", 8, False)),
            ("fake_quant_with_min_max_vars_per_channel_gradient_002", fake_quant_with_min_max_vars_per_channel_gradient_run,
             ((4, 4, 4, 4), (4, 4, 4, 4), (4, ), (4, ), "float32", 8, True)),

        ]
        self.testarg_rpc_cloud = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("fake_quant_with_min_max_vars_per_channel_gradient_003", fake_quant_with_min_max_vars_per_channel_gradient_run,
             ((2, 3), (2, 3), (3, ), (3, ), "float32", 8, False)),
            ("fake_quant_with_min_max_vars_per_channel_gradient_004", fake_quant_with_min_max_vars_per_channel_gradient_run,
             ((2, 3), (2, 3), (3, ), (3, ), "float32", 8, True)),
            ("fake_quant_with_min_max_vars_per_channel_gradient_005", fake_quant_with_min_max_vars_per_channel_gradient_run,
             ((4, ), (4, ), (4, ), (4, ), "float32", 8, True)),
            ("fake_quant_with_min_max_vars_per_channel_gradient_006", fake_quant_with_min_max_vars_per_channel_gradient_run,
             ((4, 4, 4, 4), (4, 4, 4, 4), (4, ), (4, ), "float32", 8, False)),
        ]

    @pytest.mark.rpc_mini
    @pytest.mark.level1
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

    @pytest.mark.rpc_cloud
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
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
