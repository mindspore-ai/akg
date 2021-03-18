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
        case_name = "test_apply_centered_rms_prop_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        # testflag, opfuncname, testRunArgs, dimArgs
        # testRunArgs: (shape, dtype, lr, momentum, rho, epsilon, attrs)
        self.testarg_level1 = [
            ("apply_centered_rms_prop_001", "apply_centered_rms_prop_run", ((16, 16), "float16", 0.5, 0.9, 0.6, 1e-6)),
            ("apply_centered_rms_prop_002", "apply_centered_rms_prop_run", ((16, 16), "float32", 0.5, 0.9, 0.6, 1e-6)),
        ]

    def test_run_level1(self):
        self.common_run(self.testarg_level1)

    def teardown(self):
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
