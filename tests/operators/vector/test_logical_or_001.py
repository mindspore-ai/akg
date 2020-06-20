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

"""testcase for logical_or op"""

import datetime
import os
from base import TestBase
import pytest
from test_run.logical_or_run import logical_or_run


class TestCase(TestBase):
    def setup(self):
        """
        testcase preparcondition
        :return:
        """
        case_name = "test_akg_logical_or_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            #caseflag,testfuncname,testRunArgs, dimArgs
            # ("test_3_128_bool",                        logical_or_run, ([256], [256],                      "bool", "cce_or_bool"), ( [256, 256],)),
            ("test_3_128_bool", logical_or_run, ([3, 128], [3, 128], "bool", "cce_or_bool"), ([2, 2], [32, 32])),
            # ("test_1195_1195_bool",                        logical_or_run, ([1195], [1195],                      "bool", "cce_or_bool"), ( [1024, 1024],)),
            # ("test_410_410_bool",                        logical_or_run, ([410], [410],                      "bool", "cce_or_bool"), ( [1024, 1024],)),
            # ("test_399_399_bool",                        logical_or_run, ([399], [399],                      "bool", "cce_or_bool"), ( [1024, 1024],)),
            ("test_1_1_bool", logical_or_run, ([1], [1], "bool", "cce_or_bool"), ([1, 1],)),
            ("test_1024_1024_bool", logical_or_run, ([1024], [1024], "bool", "cce_or_bool"), ([1024, 1024],)),
            ("test_4096_4096_bool", logical_or_run, ([4096], [4096], "bool", "cce_or_bool"), ([4096, 4096],)),
            # # ("test_30522_30522_bool",                logical_or_run, ([30522], [30522],              "bool", "cce_or_bool"), ([16384, 16384],)),
            ("test_2_1024_2_1024_bool", logical_or_run, ([2, 1024], [2, 1024], "bool", "cce_or_bool"), ([2, 2], [1024, 1024])),
            ("test_160_1024_160_1024_bool", logical_or_run, ([160, 1024], [160, 1024], "bool", "cce_or_bool"), ([16, 16], [1024, 1024])),
            # ("test_512_1024_512_1024_bool",          logical_or_run, ([512,1024], [512,1024],          "bool", "cce_or_bool"), ([16, 16],[1024,1024])),
            # ("test_1024_1024_1024_1024_bool",        logical_or_run, ([1024,1024], [1024,1024],        "bool", "cce_or_bool"), ([16,16],[1024, 1024])),
            # ("test_1280_1024_1280_1024_bool",        logical_or_run, ([1280,1024], [1280,1024],        "bool", "cce_or_bool"), ([16,16],[1024, 1024])),
            # ("test_4096_1024_4096_1024_bool",        logical_or_run, ([4096,1024], [4096,1024],        "bool", "cce_or_bool"), ([16,16],[1024, 1024])),
            # ("test_8192_1024_8192_1024_bool",        logical_or_run, ([8192,1024], [8192,1024],        "bool", "cce_or_bool"), ([16,16],[1024, 1024])),
            # ("test_30522_1024_30522_1024_bool",      logical_or_run, ([30522,1024], [30522,1024],      "bool", "cce_or_bool"), ([16,16],[1024, 1024])),
            # ("test_1024_4096_1024_4096_bool",        logical_or_run, ([1024,4096], [1024,4096],        "bool", "cce_or_bool"), ([4, 4], [4096, 4096])),
            # ("test_8192_4096_8192_4096_bool",        logical_or_run, ([8192,4096], [8192,4096],        "bool", "cce_or_bool"), ([4, 4], [4096, 4096])),
            # ("test_8_128_1024_8_128_1024_bool",      logical_or_run, ([8,128,1024], [8,128,1024],      "bool", "cce_or_bool"), ([1,1],[16, 16], [1024, 1024])),
            # ("test_64_128_1024_64_128_1024_bool",    logical_or_run, ([64,128,1024], [64,128,1024],    "bool", "cce_or_bool"), ([1,1],[16, 16], [1024, 1024]))
        ]

        return

    @pytest.mark.rpc_mini
    @pytest.mark.level0
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return


if __name__ == "__main__":
    a = TestCase()
    a.setup()
    a.test_run()
    a.teardown()
