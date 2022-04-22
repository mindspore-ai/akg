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
from audioop import getsample
import os
import pytest
import akg.utils as utils
from tests.common.base import TestBase
from tests.common.test_run.cpu import depthwise_conv2d_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):
    def setup(self):
        case_name = "cpu_depthwise_conv2d"
        case_path = os.getcwd()

        self.params_init(case_name, case_path)

        # we should add default attrs
        self.args_default = [
            ("000_case", depthwise_conv2d_run, ((1, 4, 114, 114, 8), (4, 1, 3, 3, 1, 8), (1, 1),
                                                "VALID", (1, 1), "float32", "NCHWc", "NCHWc", True), ["level0"]),
            ("001_case", depthwise_conv2d_run, ((1, 8, 114, 114, 8), (8, 1, 3, 3, 1, 8), (2, 2),
                                                "VALID", (1, 1), "float32", "NCHWc", "NCHWc", True), ["level0"]),
            ("002_case", depthwise_conv2d_run, ((1, 16, 58, 58, 8), (16, 1, 3, 3, 1, 8), (1, 1),
                                                "VALID", (1, 1), "float32", "NCHWc", "NCHWc", True), ["level0"]),
            ("003_case", depthwise_conv2d_run, ((1, 16, 58, 58, 8), (16, 1, 3, 3, 1, 8), (2, 2),
                                                "VALID", (1, 1), "float32", "NCHWc", "NCHWc", True), ["level0"]),
            ("004_case", depthwise_conv2d_run, ((1, 32, 30, 30, 8), (32, 1, 3, 3, 1, 8), (1, 1),
                                                "VALID", (1, 1), "float32", "NCHWc", "NCHWc", True), ["level0"]),
            ("005_case", depthwise_conv2d_run, ((1, 32, 30, 30, 8), (32, 1, 3, 3, 1, 8), (2, 2),
                                                "VALID", (1, 1), "float32", "NCHWc", "NCHWc", True), ["level0"]),
            ("006_case", depthwise_conv2d_run, ((1, 64, 16, 16, 8), (64, 1, 3, 3, 1, 8), (1, 1),
                                                "VALID", (1, 1), "float32", "NCHWc", "NCHWc", True), ["level0"]),
            ("007_case", depthwise_conv2d_run, ((1, 64, 16, 16, 8), (64, 1, 3, 3, 1, 8), (2, 2),
                                                "VALID", (1, 1), "float32", "NCHWc", "NCHWc", True), ["level0"]),
            ("008_case", depthwise_conv2d_run, ((1, 128, 9, 9, 8), (128, 1, 3, 3, 1, 8), (1, 1),
                                                "VALID", (1, 1), "float32", "NCHWc", "NCHWc", True), ["level0"]),
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
