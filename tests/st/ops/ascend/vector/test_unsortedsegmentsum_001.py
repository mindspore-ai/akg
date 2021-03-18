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

"""testcase for unsortedsegmentsum op"""

import os
import pytest
from tests.common.base import TestBase


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_unsortedsegmentsum_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.args = [
            # testflag, opfuncname, testRunArgs, dimArgs
            ("001_uss_1280_1024_8192_fp16", "unsortedsegmentsum_run", ([1280, 1024], [1280], 8192, "float16")),
            ("002_uss_1280_1024_8192_fp32", "unsortedsegmentsum_run", ([1280, 1024], [1280], 8192, "float32")),
            # ("003_uss_128_128_64_32_fp16", "unsortedsegmentsum_run", ([128, 128, 64], [128, 128], 34, "float16")),
            #("004_uss_128_128_64_32_fp32",  "unsortedsegmentsum_run", ([128, 128, 64], [128, 128], 34, "float32")),
        ]
        self.args_level1 = [
            # testflag, opfuncname, testRunArgs, dimArgs
            # ("001_uss_1280_1024_1280",  "unsortedsegmentsum_run", ([38714,1024], [38714], 30522, "float32")),
            #("001_uss_1280_1024_1280",  "unsortedsegmentsum_run", ([128, 128, 64], [128,128], 33, "float32")),
        ]
        self.args_rpc_cloud = [
            # testflag, opfuncname, testRunArgs, dimArgs
            ("bert_unsortedsegmentsum_001", "unsortedsegmentsum_run", ([1280, 768], [1280], 8192, "float32")),
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
        self.common_run(self.args)

    def test_run_level1(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.args_level1)

    def test_run_rpc_cloud(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.args_rpc_cloud)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
