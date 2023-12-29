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

"""pooling test case"""

import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.ascend.pooling_run import pooling_run


class TestPooling(TestBase):
    def setup(self):
        case_name = "test_akg_pooling_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        """setup case parameters for test"""
        self.caseresult = True
        self._log.info("=================%s Setup case=================", self.casename)
        self.testarg_mini = [
            # testflag, opfuncname,
            # testRunArgs(shape, dtype, window, stride, mode, pad_mode,
            #             (0, 0, 0, 0), False, 0), dimArgs
            ("pool_max_fp16_mini", pooling_run, (
                (1, 1, 16, 16, 16), "float16", (2, 2), (1, 1), 0, 5,
                (0, 0, 0, 0), False, 0)),
            ("pool_avg_fp16_mini", pooling_run, (
                (1, 1, 16, 16, 16), "float16", (2, 2), (4, 4), 1, 5,
                (0, 0, 0, 0), False, 0)),
        ]
        self.testarg_cloud = [
            # testflag, opfuncname, testRunArgs, dimArgs
            ("pool_max_fp16_cld", pooling_run, (
                (32, 4, 112, 112, 16), "float16", (3, 3), (2, 2), 0, 6,
                (0, 0, 0, 0), False, 0)),
            ("pool_avg_fp16_cld", pooling_run, (
                (1, 1, 64, 64, 16), "float16", (4, 4), (3, 3), 1, 5,
                (0, 0, 0, 0), False, 0)),
        ]

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_mini_run(self):
        """run case for mini"""
        self.common_run(self.testarg_mini)

    def test_cloud_run(self):
        """run case for cloud"""
        self.common_run(self.testarg_cloud)

    def teardown(self):
        """clean environment"""
        self._log.info("=============%s Teardown===========", self.casename)
