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
zeros_like test case
"""

import os
from base import TestBase
import pytest
from test_run.zeros_like_run import zeros_like_run


class TestZerosLike(TestBase):
    def setup(self):
        case_name = "test_akg_zeros_like_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag, opfuncname, testRunArgs, dimArgs
            ("zeros_like_01", zeros_like_run, ((17, 19), "float16")),
            ("zeros_like_02", zeros_like_run, ((5, 9, 15), "float16")),
            ("zeros_like_03", zeros_like_run, ((2, 4, 13, 29), "float16")),
            ("zeros_like_04", zeros_like_run, ((2, 1024), "float32")),
            ("zeros_like_05", zeros_like_run, ((32, 4, 30), "float32")),
            ("zeros_like_06", zeros_like_run, ((32, 4, 30), "int32")),
        ]
        return

    @pytest.mark.level0
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
