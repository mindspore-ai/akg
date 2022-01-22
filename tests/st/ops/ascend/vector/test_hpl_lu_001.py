# Copyright 2022 Huawei Technologies Co., Ltd
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


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_hpl_lu_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info(
            "============= {0} Setup case============".format(self.casename))
        self.args = [
            # testflag, opfuncname, testRunArgs
            ("001_hpl_lu", "hpl_lu_run", ([16, 16], "float16")),
            ("001_hpl_lu", "hpl_lu_run", ([32, 32], "float16")),
            ("001_hpl_lu", "hpl_lu_run", ([16, 16], "float32")),
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
        self.common_run(self.args)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info(
            "============= {0} Teardown============".format(self.casename))
        return
