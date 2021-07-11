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

import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.bias_add_ad_run import bias_add_ad_run


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_bias_add_ad_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag, opfuncname, testRunArgs, dimArgs
            ("bias_add_2d_3_16", bias_add_ad_run, ([3, 16], "DefaultFormat", "float16", False), ),
            ("bias_add_4d_23_14_7_16", bias_add_ad_run, ([23, 14, 7, 16], "NHWC", "float16"), ),
            ("bias_add_ad_64_1024", bias_add_ad_run, ([64, 1024], "DefaultFormat", "float16"), ),
            ("bias_add_ad_64_2", bias_add_ad_run, ([64, 2], "DefaultFormat", "float16"), ),
            ("bias_add_ad_64_1024", bias_add_ad_run, ([64, 1024], "DefaultFormat", "float16"), ),
            ("bias_add_ad_64_1024", bias_add_ad_run, ([64, 1024], "DefaultFormat", "float16"), ),
            ("bias_add_ad_1280_1024", bias_add_ad_run, ([1280, 1024], "DefaultFormat", "float16"), ),
            ("bias_add_2d_5_1024", bias_add_ad_run, ([8192, 1024], "DefaultFormat", "float16"), ),
            ("bias_add_ad_8192_4096", bias_add_ad_run, ([8192, 4096], "DefaultFormat", "float16"),),
        ]
        self.testarg_cloud = [
            # testflag, opfuncname, testRunArgs, dimArgs
            # resnet50
            ("test_resnet50_bias_add_grad_001", bias_add_ad_run, ([32, 1001], "DefaultFormat", "float32")),
            ("test_resnet50_bias_add_grad_002", bias_add_ad_run, ([32, 10], "DefaultFormat", "float32")),

        ]
        self.testarg_rpc_cloud = [
            ("bias_add_5d_fp32_001", bias_add_ad_run, ([32, 63, 1, 1, 16], "NC1HWC0", "float32")),
            ("bias_add_fp16_002", bias_add_ad_run, ([32, 1001], "DefaultFormat", "float16"), ),
            ("bias_add_fp16_003", bias_add_ad_run, ([32, 1, 1, 1001], "NHWC", "float16"), ),
            ("bias_add_fp32_002", bias_add_ad_run, ([32, 1001], "DefaultFormat", "float32"), ),
            ("bias_add_fp32_003", bias_add_ad_run, ([32, 1, 1, 1001], "NHWC", "float32"), ),
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

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
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


if __name__ == "__main__":
    a = TestCase()
    a.setup()
    a.test_run()
