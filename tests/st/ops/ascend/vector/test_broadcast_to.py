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
from tests.common.test_run.broadcast_to_run import broadcast_to_run


class TestCase(TestBase):

    def setup(self):
        case_name = "test_broadcast_to"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("test_akg_broadcast_to_001", broadcast_to_run, [(1, ), "float16", (16,)]),
            ("test_akg_broadcast_to_002", broadcast_to_run, [(1, 16), "float16", (16, 16)]),
            ("test_akg_broadcast_to_003", broadcast_to_run, [(16, ), "float16", (16, 16)]),
            ("test_akg_broadcast_to_004", broadcast_to_run, [(1,), "float32", (16,)]),
            ("test_akg_broadcast_to_005", broadcast_to_run, [(1, 16), "float32", (16, 16)]),
            ("test_akg_broadcast_to_006", broadcast_to_run, [(16,), "float32", (16, 16)]),
            ("test_akg_broadcast_to_007", broadcast_to_run, [(1,), "int32", (16,)]),
            ("test_akg_broadcast_to_008", broadcast_to_run, [(1, 16), "int32", (16, 16)]),
            ("test_akg_broadcast_to_009", broadcast_to_run, [(16,), "int32", (16, 16)]),
            ("test_akg_broadcast_to_010", broadcast_to_run, [(1,), "int8", (16,)]),
            ("test_akg_broadcast_to_011", broadcast_to_run, [(1, 16), "int8", (16, 16)]),
            ("test_akg_broadcast_to_012", broadcast_to_run, [(16,), "int8", (16, 16)]),
            ("test_akg_broadcast_to_013", broadcast_to_run, [(1,), "uint8", (16,)]),
            ("test_akg_broadcast_to_014", broadcast_to_run, [(1, 16), "uint8", (16, 16)]),
            ("test_akg_broadcast_to_015", broadcast_to_run, [(16,), "uint8", (16, 16)]),
            ("test_akg_broadcast_to_016", broadcast_to_run, [(16,), "float16", (16,)]),
        ]
        self.testarg_rpc_cloud = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("test_akg_broadcast_to_001", broadcast_to_run, [(1,), "float16", (4095,)]),
            ("test_akg_broadcast_to_002", broadcast_to_run, [(1, 4095), "float16", (4095, 4095)]),
            ("test_akg_broadcast_to_003", broadcast_to_run, [(4095,), "float16", (4095, 4095)]),
            ("test_akg_broadcast_to_004", broadcast_to_run, [(1,), "float32", (4095,)]),
            ("test_akg_broadcast_to_005", broadcast_to_run, [(1, 4095), "float32", (4095, 4095)]),
            ("test_akg_broadcast_to_006", broadcast_to_run, [(4095,), "float32", (4095, 4095)]),
            ("test_akg_broadcast_to_007", broadcast_to_run, [(1,), "int32", (4095,)]),
            ("test_akg_broadcast_to_008", broadcast_to_run, [(1, 4095), "int32", (4095, 4095)]),
            ("test_akg_broadcast_to_009", broadcast_to_run, [(4095,), "int32", (4095, 4095)]),
            ("test_akg_broadcast_to_010", broadcast_to_run, [(1,), "int8", (4095,)]),
            ("test_akg_broadcast_to_011", broadcast_to_run, [(1, 4095), "int8", (4095, 4095)]),
            ("test_akg_broadcast_to_012", broadcast_to_run, [(4095,), "int8", (4095, 4095)]),
            ("test_akg_broadcast_to_013", broadcast_to_run, [(1,), "uint8", (4095,)]),
            ("test_akg_broadcast_to_014", broadcast_to_run, [(1, 4095), "uint8", (4095, 4095)]),
            ("test_akg_broadcast_to_015", broadcast_to_run, [(4095,), "uint8", (4095, 4095)]),
        ]

        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

    def test_rpc_cloud(self):
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
        return
