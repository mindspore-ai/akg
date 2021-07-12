# Copyright 2019 Huawei Technologies Co., Ltd
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

"""
################################################

Testcase_PrepareCondition:

Testcase_TestSteps:

Testcase_ExpectedResult:

"""
import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.reduce_all_run import reduce_all_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_reduce_all_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("reduce_all_001", reduce_all_run, ((768, 768), (0,), True, "bool", "cce_all_bool")),
            ("reduce_all_002", reduce_all_run, ((16, 512), (1,), True, "bool", "cce_all_bool")),
            ("reduce_all_00x", reduce_all_run, ((16, 512 ), (1,), False, "bool", "cce_all_bool")),
            ("reduce_all_003", reduce_all_run, ((263169,), (0,), True, "bool")),
            ("reduce_all_004", reduce_all_run, ((16, 512), (0,), False, "bool")),
            ("reduce_all_005", reduce_all_run, ((16, 16), (0,), True, "bool")),
            ("reduce_all_006", reduce_all_run, ((16, 8, 4, 128), (1,), False, "bool")),
            ("reduce_all_007", reduce_all_run, ((16, 8, 64), (0,), True, "bool")),
            ("reduce_all_008", reduce_all_run, ((16, 8, 64), (1,), True, "bool")),
            ("reduce_all_009", reduce_all_run, ((32, 16, 64), (2,), True, "bool")),
            ("reduce_all_010", reduce_all_run, ((32, 16, 64), (0, 1), True, "bool")),
            ("reduce_all_011", reduce_all_run, ((32, 16, 64), (0, 2), True, "bool")),
            ("reduce_all_012", reduce_all_run, ((32, 16, 64), (1, 2), True, "bool")),
            ("reduce_all_00z", reduce_all_run, ((32, 16, 64 ), (0, 1, 2), True, "bool")),
        ]

        self.testarg_rpc_cloud = [
            # bool - int32:[768, 3072] - [2] = bool:[]
            ("reduce_all_001", reduce_all_run, ((768, 3072), (0, 1), True, "bool", "cce_all_bool")),
            # bool - int32:[229] - [1] = bool:[]
            ("reduce_all_002", reduce_all_run, ((229,), (0,), True, "bool", "cce_all_bool")),
            # bool - int32:[21128] - [1] = bool:[]
            ("reduce_all_003", reduce_all_run, ((21128,), (0,), True, "bool", "cce_all_bool")),
            # bool - int32:[2] - [1] = bool:[]
            ("reduce_all_004", reduce_all_run, ((2,), (0,), True, "bool", "cce_all_bool")),
            # bool - int32:[768] - [1] = bool:[]
            ("reduce_all_005", reduce_all_run, ((768,), (0,), True, "bool", "cce_all_bool")),
            # bool - int32:[3072, 768] - [2] = bool:[]
            ("reduce_all_006", reduce_all_run, ((3072, 768), (0, 1), True, "bool", "cce_all_bool")),
            # bool - int32:[768, 768] - [2] = bool:[]
            ("reduce_all_007", reduce_all_run, ((768, 768), (0, 1), True, "bool", "cce_all_bool")),
            # bool - int32:[2, 768] - [2] = bool:[]
            ("reduce_all_008", reduce_all_run, ((2, 768), (0, 1), True, "bool", "cce_all_bool")),
            # bool - int32:[21128, 768] - [2] = bool:[]
            ("reduce_all_009", reduce_all_run, ((21128, 768), (0, 1), True, "bool", "cce_all_bool")),
            # bool - int32:[3072] - [1] = bool:[]
            ("reduce_all_010", reduce_all_run, ((3072,), (0,), True, "bool", "cce_all_bool")),
            # bool - int32:[33, 64] - [2] = bool:[]
            ("reduce_all_011", reduce_all_run, ((33, 64), (0, 1), True, "bool", "cce_all_bool")),
        ]
        return

    @pytest.mark.level2
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_run_rpc_cloud(self):
        self.common_run([self.testarg_rpc_cloud[1]])

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return


if __name__ == "__main__":
    a = TestCase()
    a.setup()
    a.test_run()
    a.teardown()
