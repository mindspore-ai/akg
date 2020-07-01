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

from base import TestBase
import pytest
from test_run.mean_ad_run import mean_ad_run


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_mean_ad_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            #caseflag,testfuncname,testRunArgs, dimArgs
            ("mean_ad_01", mean_ad_run, ((32, 2048, 1, 1), "float16", 0, True)),
            #("mean_ad_02", mean_ad_run, ((32, 2048, 1, 1), "float16", 0, False)),
            ("mean_ad_03", mean_ad_run, ((32, 2048, 1, 1), "float16", 1, True)),
            ("mean_ad_04", mean_ad_run, ((32, 2048, 1, 1), "float16", 1, False)),
            ("mean_ad_05", mean_ad_run, ((32, 2048, 1, 1), "float32", 2, True)),
            ("mean_ad_06", mean_ad_run, ((32, 2048, 1, 1), "float32", 2, False)),
            ("mean_ad_07", mean_ad_run, ((32, 2048, 1, 1), "float32", 3, True)),
            ("mean_ad_08", mean_ad_run, ((32, 2048, 1, 1), "float32", 3, False)),
            # resnet50 5d:
            ("mean_ad_09", mean_ad_run, ((32, 128, 7, 7, 16), "float16", (2, 3), True)),
            # corner cases
            ("mean_ad_10", mean_ad_run, ((65536, 32, 1, 1), "float16", 0, True)),
            ("mean_ad_11", mean_ad_run, ((1024, 32, 1, 1), "float16", 0, True)),
        ]
        self.testarg_cloud = [
            #caseflag,testfuncname,testRunArgs, dimArgs
            ("mean_ad_01", mean_ad_run, ((32, 2048, 1, 1), "float16", 0, True)),
            #("mean_ad_02", mean_ad_run, ((32, 2048, 1, 1), "float16", 0, False)),
            ("mean_ad_03", mean_ad_run, ((32, 2048, 1, 1), "float16", 1, True)),
            ("mean_ad_04", mean_ad_run, ((32, 2048, 1, 1), "float16", 1, False)),
            ("mean_ad_05", mean_ad_run, ((32, 2048, 1, 1), "float32", 2, True)),
            ("mean_ad_06", mean_ad_run, ((32, 2048, 1, 1), "float32", 2, False)),
            ("mean_ad_07", mean_ad_run, ((32, 2048, 1, 1), "float32", 3, True)),
            ("mean_ad_08", mean_ad_run, ((32, 2048, 1, 1), "float32", 3, False)),
            # resnet50 5d:
            ("mean_ad_09", mean_ad_run, ((32, 128, 7, 7, 16), "float32", (2, 3), True)),
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

    @pytest.mark.rpc_cloud
    @pytest.mark.level1
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_cloud(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_cloud)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
