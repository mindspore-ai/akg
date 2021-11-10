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

"""cross test case"""

import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.ascend.cross_run import cross_run


class TestCross(TestBase):
    """test class for cross"""

    def setup(self):
        """setup for test"""
        case_name = "test_akg_cross_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("=================%s Setup case=================", self.casename)
        self.testarg_mini = [
            # testflag, opfuncname, (shape1, dtype1, shape2, dtype2), dimArgs
            # the first dim of shape1 must be 3,
            # and must meet shape1 == shape2, dtype1 == dtype2
            # compile error
            # ("cross_f16", cross_run, ((3, 3), "float16", (3, 3), "float16")),
            # ("cross_f32", cross_run, ((3, 3), "float32", (3, 3), "float32")),
        ]
        self.testarg_cloud = [
            # testflag, opfuncname, (shape1, dtype1, shape2, dtype2), dimArgs
            ("cross_i8", cross_run, ((3, 3), "int8", (3, 3), "int8")),
            ("cross_u8", cross_run, ((3, 3), "uint8", (3, 3), "uint8")),
            ("cross_i32", cross_run, ((3, 3), "int32", (3, 3), "int32")),
            ("cross_f16", cross_run, ((3, 3), "float16", (3, 3), "float16")),
            ("cross_f32", cross_run, ((3, 3), "float32", (3, 3), "float32")),
            ("cross_f16_3d", cross_run, ((3, 5, 8), "float16", (3, 5, 8), "float16")),
        ]

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_mini_run(self):
        """run case"""
        self.common_run(self.testarg_mini)

    def test_cloud_run(self):
        """run case"""
        self.common_run(self.testarg_cloud)

    def teardown(self):
        """clean environment"""
        self._log.info("=============%s Teardown============", self.casename)
