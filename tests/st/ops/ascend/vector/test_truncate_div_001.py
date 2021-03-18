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

"""truncate_div test case"""

import os
from tests.common.base import TestBase
from tests.common.test_run.truncate_div_run import truncate_div_run


class TestTD(TestBase):
    def setup(self):
        case_name = "test_akg_truncate_div_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        """setup for test"""
        self.caseresult = True
        self._log.info("=================%s Setup case=================", self.casename)
        self.testarg_mini = [
            # testflag, opfuncname, (shape1, dtype1, shape2, dtype2), dimArgs
            ("truncate_div_f16", truncate_div_run, ((8, 16), "float16", (8, 16), "float16")),
            ("truncate_div_f32", truncate_div_run, ((8, 16), "float32", (8, 16), "float32")),
        ]
        self.testarg_cloud = [
            # testflag, opfuncname, (shape1, dtype1, shape2, dtype2), dimArgs
            ("truncate_div_i8", truncate_div_run, ((8, 16), "int8", (8, 16), "int8")),
            ("truncate_div_u8", truncate_div_run, ((8, 16), "uint8", (8, 16), "uint8")),
            ("truncate_div_f16", truncate_div_run, ((8, 16), "float16", (8, 16), "float16")),
            ("truncate_div_f32", truncate_div_run, ((8, 16), "float32", (8, 16), "float32")),
            ("truncate_div_f32", truncate_div_run, ((8, 16), "float16", (16,), "float16")),
        ]

    def test_mini_run(self):
        """run case"""
        self.common_run(self.testarg_mini)

    def test_cloud_run(self):
        """run case"""
        self.common_run(self.testarg_cloud)

    def teardown(self):
        """clean environment"""
        self._log.info("=============%s Teardown============", self.casename)
