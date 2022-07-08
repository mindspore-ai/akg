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
import akg.utils as utils
from tests.common.base import TestBase
from tests.common.test_run.cpu import pooling_run
from tests.common.test_run.cpu import global_pooling_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):
    def setup(self):
        case_name = "cpu_pooling"
        case_path = os.getcwd()

        self.params_init(case_name, case_path)

        self.args_default = [
            # max/avg pooling2d
            ("000_case", pooling_run, ((1, 256, 32, 32), (2, 2), (2, 2), (0, 0,
             0, 0), "avg", "float32", False, True, "NCHW", True), ["level0"]),
            ("001_case", pooling_run, ((1, 256, 32, 32), (2, 2), (2, 2), (0, 0,
             0, 0), "max", "float32", False, True, "NCHW", True), ["level0"]),
            ("002_case", pooling_run, ((1, 256, 31, 31), (3, 3), (3, 3), (0, 0,
             0, 0), "avg", "float32", False, True, "NCHW", True), ["level0"]),
            ("003_case", pooling_run, ((1, 256, 31, 31), (3, 3), (3, 3), (0, 0,
             0, 0), "max", "float32", False, True, "NCHW", True), ["level0"]),
            ("004_case", pooling_run, ((1, 16, 31, 31, 8), (3, 3), (3, 3), (0, 0,
             0, 0), "max", "float32", False, True, "NCHW8c", True), ["level0"]),
            ("005_case", pooling_run, ((1, 16, 31, 31, 8), (3, 3), (3, 3), (0, 0,
             0, 0), "avg", "float32", False, True, "NCHW8c", True), ["level0"]),
            ("006_case", pooling_run, ((1, 16, 31, 31, 8), (3, 3), (3, 3), (0, 0,
             0, 0), "max", "float32", True, True, "NCHW8c", True), ["level0"]),

            # max/avg global pooling2d
            ("007_case", global_pooling_run, ((4, 1024, 7, 7),
             "max", "float32", "NCHW", True), ["level0"]),
            ("008_case", global_pooling_run, ((4, 1024, 7, 7),
             "avg", "float32", "NCHW", True), ["level0"]),
            ("009_case", global_pooling_run, ((4, 128, 7, 7, 8),
             "max", "float32", "NCHW8c", True), ["level0"]),
            ("0019_case", global_pooling_run, ((4, 128, 7, 7, 8),
             "avg", "float32", "NCHW8c", True), ["level0"]),
        ]

        return True

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_cpu_level0(self):
        return self.run_cases(self.args_default, utils.LLVM, "level0")

    def teardown(self):
        self._log.info("{0} Teardown".format(self.casename))
        super(TestCase, self).teardown()
        return
