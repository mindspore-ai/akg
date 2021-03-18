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
smooth_l1_loss
"""
import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.smooth_l1_loss_ad_run import smooth_l1_loss_ad_run


class TestCase(TestBase):
    def setup(self):
        case_name = "test_auto_smooth_l1_loss_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            #caseflag,opfuncname,testRunArgs, dimArgs
            # 1. 8,80*80*6,4: ok
            # ("smooth_l1_loss_ad_run1", smooth_l1_loss_ad_run,((8, 38400, 4), "float16", (8, 38400, 4), "float16", (8, 38400), "int32", 0, 1.0, "smooth_l1_loss_output")),
            # 2. 8,80*80*6,4: ok
            # ("smooth_l1_loss_ad_run2", smooth_l1_loss_ad_run,((8, 38400, 4), "float32", (8, 38400, 4), "float32", (8, 38400), "int32",  0, 1.0, "smooth_l1_loss_output")),
            # 3. 8,4718,4: ok
            ("smooth_l1_loss_ad_run3", smooth_l1_loss_ad_run, ((8, 4718, 4), "float16", (8, 4718, 4), "float16", (8, 4718), "int32", 0, 1.0, "smooth_l1_loss_output")),
            # 4. 8,4718,4: ok
            #("smooth_l1_loss_ad_run4", smooth_l1_loss_ad_run,((8, 4718, 4), "float32", (8, 4718, 4), "float32", (8, 4718), "int32",  0, 1.0, "smooth_l1_loss_output")),
            # 5. 8,8732,4: ok
            #("smooth_l1_loss_ad_run5", smooth_l1_loss_ad_run, ((8, 8732, 4), "float16", (8, 8732, 4), "float16", (8, 8732), "int32",  1, 1.0, "smooth_l1_loss_output")),
            # 6. 8,8732,4: ok
            #  ("smooth_l1_loss_ad_run6", smooth_l1_loss_ad_run, ((8, 8732, 4), "float32", (8, 8732, 4), "float32", (8, 8732), "int32",  0, 1.0, "smooth_l1_loss_output")),

        ]
        self.testarg_cloud = [
            #caseflag,opfuncname,testRunArgs, dimArgs
            ####("smooth_l1_loss_cloud_run0", smooth_l1_loss_ad_run, ((8, 8732, 4), "float32", (8, 8732, 4), "float32", (8, 8732), "int32",  0, 1.0, "smooth_l1_loss_output0")),
            ####("smooth_l1_loss_cloud_run1", smooth_l1_loss_ad_run, ((32, 8732, 4), "float16", (32, 8732, 4), "float16", (32, 8732), "int32",  0, 1.0, "smooth_l1_loss_output1")),
            ####("smooth_l1_loss_cloud_run2", smooth_l1_loss_ad_run, ((32, 8732, 4), "float32", (32, 8732, 4), "float32", (32, 8732), "int32",  0, 1.0, "smooth_l1_loss_output2")),
        ]
        self.testarg_aic = [
            #caseflag,opfuncname,testRunArgs, dimArgs
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

    def test_run_aicmodel(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_aic)

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

# if __name__ == "__main__":
#     a = TestCase()
#     a.setup()
#     a.test_run()
#     a.teardown()
