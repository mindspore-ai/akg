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
from test_run.reverse_run import reverse_run


class TestCase(TestBase):
    """define test class"""
    def setup(self):
        case_name = "test_selu_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        """set test case """
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("reverse_001", reverse_run, ((2, 3, 3, 5, 7), "int32", [0, 2])),
            ("reverse_002", reverse_run, ((5, 7), "float32", [0, ])),
            ("reverse_003", reverse_run, ((3, 5, 7), "float16", [-2, ])),
        ]
        self.testarg_rpc_cloud = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("reverse_001", reverse_run, ((11, 3, 3, 5, 7), "int32", [0, 2, 3])),
            ("reverse_002", reverse_run, ((1, 3, 3, 5, 7), "int32", [0, -2])),
            ("reverse_003", reverse_run, ((2, 3, 3, 5, 7), "float32", [0, 3])),
            ("reverse_004", reverse_run, ((6, 3, 3, 5, 7), "float16", [-3, ])),
            ("reverse_005", reverse_run, ((12, 3, 3, 5, 7), "float16", [1, 2, 3])),
        ]

    @pytest.mark.rpc_mini
    @pytest.mark.level1
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

    @pytest.mark.rpc_cloud
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_rpc_cloud(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_rpc_cloud)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
