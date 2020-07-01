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
import pytest
from base import TestBase
from nose.plugins.attrib import attr


class TestCase(TestBase):

    def setup(self):
        case_name = "test_apply_adagrad_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            ("apply_adagrad_001", "apply_adagrad_run", ((16, 16), "float16", True)),
            ("apply_adagrad_002", "apply_adagrad_run", ((16, 16), "float32", True)),
            ("apply_adagrad_003", "apply_adagrad_run", ((16, 16), "float16", False)),
            ("apply_adagrad_004", "apply_adagrad_run", ((16, 16), "float32", False)),
        ]

    @pytest.mark.rpc_mini
    @pytest.mark.level0
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run(self):
        self.common_run(self.testarg)

    def teardown(self):
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
