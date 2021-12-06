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
decode
"""
import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.ascend.bounding_box_encode_run import bounding_box_encode_run


class TestCase(TestBase):

    def setup(self):
        case_name = "test_bounding_box_encode_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # caseflag,opfuncname,testRunArgs, dimArgs
            # ("bouding_box_encode_run_fp16_001", bounding_box_encode_run, ((16, 8), (2, 16, 8), (2, 16), "float16", [10.0, 10.0, 5.0, 5.0], 1e-5, "cce_bounding_box_encode_fp16")),
        ]

        return

    @pytest.mark.level0
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
