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
from tests.common.base import TestBase
from tests.common.test_run.ascend.reduce_any_d_run import reduce_any_d_run


class TestCase(TestBase):
    """define test class"""
    def setup(self):
        case_name = "test_reduce_any_d_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        """set test case """
        self.caseresult = True
        self._log.info("============= %s Setup case============", self.casename)
        self.testarg = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("reduce_any_d_001", reduce_any_d_run, ((16, ),  "int8", (0, ), True)),
            ("reduce_any_d_002", reduce_any_d_run, ((16, 16),  "int8", (1, ), False)),
            ("reduce_any_d_003", reduce_any_d_run, ((2, 2, 2),  "int8", (0, 2), True)),

        ]
        self.testarg_rpc_cloud = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("reduce_any_d_001", reduce_any_d_run, ((16, ),  "int8", (0, ), False)),
            ("reduce_any_d_002", reduce_any_d_run, ((16, 16),  "int8", (1, ), False)),
            ("reduce_any_d_003", reduce_any_d_run, ((16, 128, 16),  "int8", (1, 2), False)),
            ("reduce_any_d_004", reduce_any_d_run, ((2, 2, 2),  "int8", None, True)),
            ("reduce_any_d_005", reduce_any_d_run, ((2, 2, 2),  "int8", (-1, -2), True)),
            ("reduce_any_d_006", reduce_any_d_run, ((2, 2, 2, 16, 16),  "int8", (0, 2, 4), False)),
        ]

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= %s Setup case============", self.casename)
