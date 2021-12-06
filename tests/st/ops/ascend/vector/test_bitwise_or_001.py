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
from tests.common.test_run.ascend.bitwise_or_run import bitwise_or_run


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_bitwise_or"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("test_akg_bitwise_or_001", bitwise_or_run, [(1, ), "int16",  (12, ), "int16"]),
            ("test_akg_bitwise_or_002", bitwise_or_run, [(16, ), "int16",  (4, 16), "int16"]),
            ("test_akg_bitwise_or_003", bitwise_or_run, [(4, 16), "int16",  (4, 16), "int16"]),
            ("test_akg_bitwise_or_004", bitwise_or_run, [(1, ), "uint16",  (12, ), "uint16"]),
            ("test_akg_bitwise_or_005", bitwise_or_run, [(16, ), "uint16",  (4, 16), "uint16"]),
            ("test_akg_bitwise_or_006", bitwise_or_run, [(4, 16), "uint16",  (4, 16), "uint16"]),
            
        ]

        self.testarg_rpc_cloud = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("test_akg_bitwise_or_001", bitwise_or_run, [(4095,), "int16", (4095,), "int16"]),
            ("test_akg_bitwise_or_002", bitwise_or_run, [(4095,), "uint16", (4095,), "uint16"]),
        ]
        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
