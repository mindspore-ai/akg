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


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_one_hot_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # caseflag,opfuncname,testRunArgs, dimArgs
            ("test_one_hot1", "one_hot_run", ((8, 8732), 6, "float16", 1, 0, -1)),
            ("test_one_hot1", "one_hot_run", ((8, 8732), 6, "float32", 1, 0, -1)),
            ("test_one_hot1", "one_hot_run", ((8,), 6, "float32", 1, 0, -1)),
            ("test_one_hot1", "one_hot_run", ((16,), 16, "float32", 1, 0, -1)),
            ("test_one_hot2", "one_hot_run", ((123,), 128, "int32", 1, 0, -1)),
            ("test_one_hot3", "one_hot_run", ((1024,), 16, "int32", 1, 0, -1)),
            ("test_one_hot4", "one_hot_run", ((160,), 160, "int32", 1, 0, -1)),
            ("test_one_hot5", "one_hot_run", ((1280,), 16, "int32", 1, 0, -1)),
            ("test_one_hot6", "one_hot_run", ((8,), 8, "int32", 1, 0, -1)),
            ("test_one_hot7", "one_hot_run", ((64,), 64, "int32", 1, 0, -1)),
            ("test_one_hot8", "one_hot_run", ((8192,), 64, "int32", 1, 0, -1)),
            ("test_one_hot9", "one_hot_run", ((1024,), 16, "int32", 1, 0, 0)),
            #("test_one_hot10", "one_hot_run", ((1052676,), 21, "int32", 1, 0, -1)),
            ("test_one_hot10", "one_hot_run", ((1,), 32000, "float32", 1, 0, -1)),
            # ("test_one_hot10", "one_hot_run",((1024, 16), 16,"int32", 1, 0, 0)),
        ]

        self.testarg_rpc_cloud = [
            # int32 - int32 - float - float:[160] - [] - [] - [] = float:[160, 30522]
            ("test_one_hot_001", "one_hot_run", ((160,), 30522, "int32", 1, 0, -1)),
            # int32 - int32 - float - float:[8192] - [] - [] - [] = float:[8192, 2]
            ("test_one_hot_002", "one_hot_run", ((8192,), 2, "int32", 1, 0, -1)),
            # int32 - int32 - float - float:[1024] - [] - [] - [] = float:[1024, 2]
            ("test_one_hot_003", "one_hot_run", ((1024,), 2, "int32", 1, 0, -1)),
            # int32 - int32 - float - float:[1280] - [] - [] - [] = float:[1280, 30522]
            ("test_one_hot_004", "one_hot_run", ((1280,), 30522, "int32", 1, 0, -1)),
            # int32 - int32 - float - float:[8] - [] - [] - [] = float:[8, 2]
            ("test_one_hot_005", "one_hot_run", ((8,), 2, "int32", 1, 0, -1)),
            # int32 - int32 - float - float:[64] - [] - [] - [] = float:[64, 2]
            ("test_one_hot_006", "one_hot_run", ((64,), 2, "int32", 1, 0, -1)),
            # int32 - int32 - float - float:[8192] - [] - [] - [] = float:[8192, 21128]
            ("test_one_hot_007", "one_hot_run", ((8192,), 21128, "int32", 1, 0, -1)),
            # int32 - int32 - float - float:[8192] - [] - [] - [] = float:[8192, 2]
            ("test_one_hot_008", "one_hot_run", ((8192,), 2, "int32", 1, 0, -1)),
            # int32 - int32 - float - float:[1280] - [] - [] - [] = float:[1280, 21128]
            ("test_one_hot_009", "one_hot_run", ((1280,), 21128, "int32", 1, 0, -1)),
            # int32 - int32 - float - float:[64] - [] - [] - [] = float:[64, 2]
            ("test_one_hot_010", "one_hot_run", ((64,), 2, "int32", 1, 0, -1)),
            ("test_one_hot_011", "one_hot_run", ((160,), 21128, "float32", 1, 0, -1)),
        ]
        return

    @pytest.mark.level2
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return


if __name__ == "__main__":
    t = TestCase()
    t.setup()
    t.test_run()
    t.teardown()
