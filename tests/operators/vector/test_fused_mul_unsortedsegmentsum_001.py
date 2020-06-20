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
from base import TestBase
from nose.plugins.attrib import attr


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_fused_mul_unsortedsegmentsum_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.args = [
            # testflag, opfuncname, testRunArgs, dimArgs
            # need to move tiling constraint to dsl later
            ("001_fused_mul_unsortedsegmentsum", "fused_mul_unsortedsegmentsum_run", ([32, 64, 32], [32, 64, 32], [32, 64], 18, "float16"), ((32, 1), (18, 1), (16, 1))),
            # ("002_fused_mul_unsortedsegmentsum",  "fused_mul_unsortedsegmentsum_run", ([64, 128, 64], [64, 128, 64], [64, 128], 34, "float16")),
        ]
        self.args_level1 = [
        ]
        self.args_rpc_cloud = [
        ]
        return

    @pytest.mark.rpc_mini
    @pytest.mark.level0
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.args)

    @pytest.mark.level1
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_level1(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.args_level1)

    @pytest.mark.rpc_cloud
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_rpc_cloud(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.args_rpc_cloud)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
