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

"""
exp test cast
"""

import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.ascend.exp_ad_run import exp_ad_run


class TestExp(TestBase):

    def setup(self):
        case_name = "test_akg_exp_ad_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag, opfuncname, testRunArgs, dimArgs
            ("exp_ad_01", exp_ad_run, ((640, 25), "float16")),
            ("exp_ad_01", exp_ad_run, ((6400, 1234), "float16")),
        ]
        self.testarg_cloud = [
            # testflag, opfuncname, testRunArgs, dimArgs
            ####("exp_ad_01", exp_ad_run, ((64, 2), "float32")),
        ]
        self.testarg_level1 = [
            # testflag, opfuncname, testRunArgs, dimArgs
            #("exp_ad_01", exp_ad_run, ((64, 2), "float16"), ((64, 64), (2, 2))),
            ####("exp_ad_02", exp_ad_run, ((1280, 30522), "float16")),

        ]
        return

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

    def test_run_level1(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_level1)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
