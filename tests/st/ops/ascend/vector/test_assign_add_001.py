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
assign_add test cases
"""

import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.assign_add_run import assign_add_run


class TestAssignAdd(TestBase):

    def setup(self):
        case_name = "test_assign_add_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag, opfuncname, testRunArgs
            ("test_assign_add_160_1024", assign_add_run, ([160, 1024], [1, 1024], "float16")),
            ("test_assign_add_2_2_fp16", assign_add_run, ([2], [2], "float16")),
            ("test_assign_add_1024_1024_fp16", assign_add_run, ([1024], [1024], "float16")),
            ("test_assign_add_30522_30522_fp16", assign_add_run, ([30522], [30522], "float16")),
            ("test_assign_add_2_1024_2_1024_fp16", assign_add_run, ([2, 1024], [2, 1024], "float16")),
            ("test_assign_add_160_1024_160_1024_fp16", assign_add_run, ([160, 1024], [160, 1024], "float16")),
            ("test_assign_add_2_1_fp16", assign_add_run, ([2], [1], "float16")),
            ("test_assign_add_1024_1_fp16", assign_add_run, ([1024], [1], "float16")),
            ("test_assign_add_160_1_1_fp16", assign_add_run, ([160, 1], [1], "float16")),
            ("test_assign_add_1024_1_1_fp16", assign_add_run, ([1024, 1], [1], "float16")),
            ("test_assign_add_1280_1_1_fp16", assign_add_run, ([1280, 1], [1], "float16")),
            ("test_assign_add_2_1024_1_fp16", assign_add_run, ([2, 1024], [1], "float16")),
            ("test_assign_add_512_1024_1_fp16", assign_add_run, ([512, 1024], [1], "float16")),
            ("test_assign_add_64_128_1_1_fp16", assign_add_run, ([64, 128, 1], [1], "float16")),
            ("test_assign_add_1_int32", assign_add_run, ([1], [1], "int32")),
            ("test_assign_add_1_fp16", assign_add_run, ([1], [1], "float16")),
            ("test_assign_add_1_fp32", assign_add_run, ([1], [1], "float32")),
        ]

        self.testarg_level1 = [
            ## testflag, opfuncname, testRunArgs
            ("test_assign_add_8192_1024_8192_1024_fp16", assign_add_run, ([8192, 1024], [8192, 1024], "float16")),
            ("test_assign_add_30522_1024_30522_1024_fp16", assign_add_run, ([30522, 1024], [30522, 1024], "float16")),
            ("test_assign_add_1024_4096_1024_4096_fp16", assign_add_run, ([1024, 4096], [1024, 4096], "float16")),
            ("test_assign_add_8192_4096_8192_4096_fp16", assign_add_run, ([8192, 4096], [8192, 4096], "float16")),
            ("test_assign_add_8_128_1024_8_128_1024_fp16", assign_add_run, ([8, 128, 1024], [8, 128, 1024], "float16")),
            ("test_assign_add_64_128_1024_64_128_1024_fp16", assign_add_run,
             ([64, 128, 1024], [64, 128, 1024], "float16")),
            ("test_assign_add_8_16_128_128_8_1_128_128_fp16", assign_add_run,
             ([8, 16, 128, 128], [8, 1, 128, 128], "float16")),
            ("test_assign_add_1024_1024_1_fp16", assign_add_run, ([1024, 1024], [1], "float16"), ([1, 1], [1024, 1024])),
            ("test_assign_add_30522_1024_1_fp16", assign_add_run, ([30522, 1024], [1], "float16")),

        ]
        return

    @pytest.mark.level2
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
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
