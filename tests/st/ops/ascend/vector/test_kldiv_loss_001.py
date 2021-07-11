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
kldiv_loss
"""
import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.kldiv_loss_run import kldiv_loss_run


class TestCase(TestBase):
    def setup(self):
        case_name = "test_auto_kldiv_loss_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            #caseflag,opfuncname,testRunArgs, dimArgs
            # 1. none
            ("kldiv_loss_none_run1", kldiv_loss_run, ((2, 2), "float16", 'none', "kldiv_loss_output")),
            ("kldiv_loss_none_run2", kldiv_loss_run, ((2, 2, 2), "float16", 'none', "kldiv_loss_output")),
            ("kldiv_loss_none_run3", kldiv_loss_run, ((2, 128, 128), "float16", 'none', "kldiv_loss_output")),
            # 2. sum
            ("kldiv_loss_sum_run1", kldiv_loss_run, ((4, 2), "float16", 'sum', "kldiv_loss_output")),
            ("kldiv_loss_sum_run2", kldiv_loss_run, ((2, 2, 2), "float16", 'sum', "kldiv_loss_output")),
            ("kldiv_loss_sum_run3", kldiv_loss_run, ((2, 128, 128), "float16", 'sum', "kldiv_loss_output")),
            # 3. mean
            ("kldiv_loss_mean_run1", kldiv_loss_run, ((4, 2), "float16", 'mean', "kldiv_loss_output")),
            ("kldiv_loss_mean_run2", kldiv_loss_run, ((2, 2, 2), "float16", 'mean', "kldiv_loss_output")),
            ("kldiv_loss_mean_run3", kldiv_loss_run, ((2, 128, 128), "float16", 'mean', "kldiv_loss_output")),
            # 4. batchmean
            ("kldiv_loss_batchmean_run1", kldiv_loss_run, ((4, 2), "float16", 'batchmean', "kldiv_loss_output")),
            ("kldiv_loss_batchmean_run2", kldiv_loss_run, ((4, 2, 2), "float16", 'batchmean', "kldiv_loss_output")),
            ("kldiv_loss_batchmean_run3", kldiv_loss_run, ((2, 128, 128), "float16", 'batchmean', "kldiv_loss_output")),

        ]
        self.testarg_cloud = [
            #caseflag,opfuncname,testRunArgs, dimArgs
            # 1. none
            ("kldiv_loss_none_cloud_run1", kldiv_loss_run, ((4, 2), "float16", 'none', "kldiv_loss_output")),
            ("kldiv_loss_none_cloud_run2", kldiv_loss_run, ((2, 2, 2), "float16", 'none', "kldiv_loss_output")),
            ("kldiv_loss_none_cloud_run3", kldiv_loss_run, ((2, 128, 128), "float16", 'none', "kldiv_loss_output")),
            # 2. sum
            ("kldiv_loss_sum_cloud_run1", kldiv_loss_run, ((2, 128, 128), "float16", 'sum', "kldiv_loss_output")),
            # 3. mean
            ("kldiv_loss_mean_cloud_run1", kldiv_loss_run, ((2, 128, 128), "float16", 'mean', "kldiv_loss_output")),
            # 4. mean
            ("kldiv_loss_batchmean_cloud_run1", kldiv_loss_run, ((2, 2, 2), "float16", 'batchmean', "kldiv_loss_output")),

        ]
        self.testarg_aic = [
            #caseflag,opfuncname,testRunArgs, dimArgs
            # 1. none
            ("kldiv_loss_none_cloud_run0", kldiv_loss_run, ((4, 2), "float16", 'none', "kldiv_loss_output")),
            # 2. mean
            ("kldiv_loss_mean_cloud_run1", kldiv_loss_run, ((4, 2), "float16", 'mean', "kldiv_loss_output")),
            # 4. batchmean
            ("kldiv_loss_batchmean_cloud_run1", kldiv_loss_run, ((4, 2), "float16", 'batchmean', "kldiv_loss_output")),
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
#    t = TestCase()
#    t.setup()
#    t.test_run()
#    t.teardown()
