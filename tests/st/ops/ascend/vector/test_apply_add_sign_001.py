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

import os
from tests.common.base import TestBase


class TestCase(TestBase):

    def setup(self):
        case_name = "test_apply_add_sign_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= %s Setup case============", self.casename)
        self.testarg_level1 = [
            ("apply_add_sign_001", "apply_add_sign_run", ((16, 16), "float16")),
            ("apply_add_sign_002", "apply_add_sign_run", ((16, 16), "float32")),
        ]

    def test_run_level1(self):
        self.common_run(self.testarg_level1)

    def teardown(self):
        self._log.info("============= %s Teardown============", self.casename)
        return
