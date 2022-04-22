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
from tests.common.test_run.cpu import layout_transform_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):
    def setup(self):
        case_name = "layout_transform"
        case_path = os.getcwd()

        self.params_init(case_name, case_path)

        self.args_default = [
            ("000_case", layout_transform_run, ((4, 32, 56, 56, 8), "float32",
                                                "NCHW8c", "NCHW", True), ["level0"]),
            ("001_case", layout_transform_run, ((8, 32, 56, 56, 8), "float32",
                                                "NCHW8c", "NCHW4c", True), ["level0"]),
            ("002_case", layout_transform_run, ((16, 32, 56, 56, 8), "float32",
                                                "NCHW8c", "NHWC", True), ["level0"]),
            ("003_case", layout_transform_run, ((128, 32, 7, 7), "float32",
                                                "OHWI", "OIHW", True), ["level0"]),
            ("004_case", layout_transform_run, ((128, 7, 7, 32), "float32", "OHWI",
                                                "OIHW8i4o", True), ["level0"]),
            ("005_case", layout_transform_run, ((16, 4, 7, 7, 8, 8), "float32",
                                                "OIHW8i8o", "OHWI", True), ["level0"]),
            ("006_case", layout_transform_run, ((12, 32, 56, 56), "float32",
                                                "NCHW", "NHWC", True), ["level0"]),
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