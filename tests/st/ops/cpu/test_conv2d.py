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
from tests.common.test_run.cpu import conv2d_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):
    def setup(self):
        case_name = "cpu_conv2d"
        case_path = os.getcwd()

        self.params_init(case_name, case_path)

        self.args_default = [
            ("000_case", conv2d_run, ((1, 4, 28, 28, 8), (8, 4, 3, 3, 8, 8), (1, 1),
             (0, 0, 0, 0), (1, 1), "float32", "NCHWc", "NCHWc", False), ["level0"]),
            ("001_case", conv2d_run, ((1, 32, 56, 56, 8), (8, 32, 1, 1, 8, 8), (1, 1),
             (0, 0, 0, 0), (1, 1), "float32", "NCHWc", "NCHWc", True), ["level0"]),
            ("002_case", conv2d_run, ((1, 16, 28, 28, 8), (64, 16, 1, 1, 8, 8), (1, 1),
             (0, 0, 0, 0), (1, 1), "float32", "NCHWc", "NCHWc", True), ["level0"]),
            ("003_case", conv2d_run, ((1, 32, 14, 14, 8), (64, 32, 1, 1, 8, 8), (2, 2),
             (0, 0, 0, 0), (1, 1), "float32", "NCHWc", "NCHWc", True), ["level0"]),
            ("004_case", conv2d_run, ((1, 32, 7, 7, 8), (64, 32, 1, 1, 8, 8), (1, 1),
             (0, 0, 0, 0), (1, 1), "float32", "NCHWc", "NCHWc", True), ["level0"]),
            ("005_case", conv2d_run, ((1, 4, 28, 28, 8), (8, 4, 3, 3, 8, 8), (1, 1),
             (0, 0, 0, 0), (1, 1), "float32", "NCHWc", "NCHWc", True), ["level0"]),
            ("006_case", conv2d_run, ((1, 4, 28, 28, 8), (8, 4, 3, 3, 8, 8), (1, 1),
             (0, 0, 0, 0), (2, 2), "float32", "NCHWc", "NCHWc", True), ["level0"]),
            ("007_case", conv2d_run, ((1, 120, 1, 1, 8), (16, 120, 1, 1, 8, 8), (1, 1),
             (0, 0, 0, 0), (1, 1), "float32", "NCHWc", "NCHWc", True), ["level0"]),
            ("008_case", conv2d_run, ((1, 30, 7, 7, 8), (16, 30, 1, 1, 8, 8), (1, 1),
             (0, 0, 0, 0), (1, 1), "float32", "NCHWc", "NCHWc", True), ["level0"]),
            ("009_case", conv2d_run, ((1, 1, 7, 7, 1), (16, 1, 1, 1, 1, 8), (1, 1),
             (0, 0, 0, 0), (1, 1), "float32", "NCHWc", "NCHWc", True), ["level0"]),
            ("010_case", conv2d_run, ((1, 2, 7, 7, 8), (1, 2, 1, 1, 8, 1), (1, 1),
             (0, 0, 0, 0), (1, 1), "float32", "NCHWc", "NCHWc", True), ["level0"]),
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
