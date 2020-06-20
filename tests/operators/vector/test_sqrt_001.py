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
sqrt test cast
"""

import datetime
import os

from base import TestBase
import pytest
from test_run.sqrt_run import sqrt_run


class TestSqrt(TestBase):
    def setup(self):
        case_name = "test_akg_sqrt_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag, opfuncname, testRunArgs, dimArgs
            ("sqrt_01", sqrt_run, ((1, 128), "float16"), ((128, 128), (128, 128))),
            ("sqrt_02", sqrt_run, ((128, 128), "float16"), ((0, 0), (128, 128))),
            ("sqrt_03", sqrt_run, ((128, 256), "float16"), ((0, 0), (128, 128))),
            # ("sqrt_04", sqrt_run, ((2, 1024),     "float16"), ((2, 2), (1024, 1024))  ),
            # ("sqrt_05", sqrt_run, ((30522,),      "float16"), ((15261, 15261),)       ),
            # ("sqrt_06", sqrt_run, ((4096, 1024),  "float16"), ((16, 16), (1024, 1024))),
            ("sqrt_07", sqrt_run, ((1,), "float16"), ((1, 1),)),
            # ("sqrt_08", sqrt_run, ((1024, 4096),  "float16"), ((4, 4),(4096, 4096))   ),
            # ("sqrt_09", sqrt_run, ((4096,),       "float16"), ((4096, 4096),)         ),
            # ("sqrt_10", sqrt_run, ((30522, 1024), "float16"), ((16, 16), (1024, 1024))),
            ("sqrt_11", sqrt_run, ((1024,), "float16"), ((1024, 1024),)),
            ("sqrt_12", sqrt_run, ((2,), "float16"), ((2, 2),)),
            ("sqrt_13", sqrt_run, ((512, 1024), "float16"), ((16, 16), (1024, 1024))),
            ("sqrt_14", sqrt_run, ((1024, 1024), "float16"), ((16, 16), (1024, 1024))),
        ]
        self.testarg_cloud = [
            ("sqrt_01", sqrt_run, ((1, 128), "float32"), ((128, 128), (128, 128))),
        ]
        self.testarg_level1 = [
            # testflag, opfuncname, testRunArgs, dimArgs
            # ("sqrt_01", sqrt_run, ((1, 128),      "float16"), ((128, 128), (128, 128))),
            # ("sqrt_02", sqrt_run, ((128, 128),    "float16"), ((0, 0), (128, 128))    ),
            # ("sqrt_03", sqrt_run, ((128, 256),    "float16"), ((0, 0), (128, 128))    ),
            ("sqrt_04", sqrt_run, ((2, 1024), "float16"), ((2, 2), (1024, 1024))),
            ("sqrt_05", sqrt_run, ((30522,), "float16"), ((15261, 15261),)),
            ("sqrt_06", sqrt_run, ((4096, 1024), "float16"), ((16, 16), (1024, 1024))),
            # run fail("sqrt_07", sqrt_run, ((1,),          "float16"), ((1, 1),)               ),
            ("sqrt_08", sqrt_run, ((1024, 4096), "float16"), ((4, 4), (4096, 4096))),
            ("sqrt_09", sqrt_run, ((4096,), "float16"), ((4096, 4096),)),
            ("sqrt_10", sqrt_run, ((30522, 1024), "float16"), ((16, 16), (1024, 1024))),
            # ("sqrt_11", sqrt_run, ((1024,),       "float16"), ((1024, 1024),)         ),
            # ("sqrt_12", sqrt_run, ((2,),          "float16"), ((2, 2),)               ),
            # ("sqrt_13", sqrt_run, ((512, 1024),   "float16"), ((16, 16), (1024, 1024))),
            # ("sqrt_14", sqrt_run, ((1024, 1024),  "float16"), ((16, 16), (1024, 1024))),
            # ("sqrt_15", sqrt_run, ((128, 1024),   "float16"), ((16, 16), (1024, 1024))),

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

    @pytest.mark.aicmodel
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_cloud(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_cloud)

    @pytest.mark.level1
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_level1(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_level1)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
