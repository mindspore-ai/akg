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
mul test cast
"""

import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.elemwise_sum_run import elemwise_sum_run


class TestCase(TestBase):

    def setup(self):
        case_name = "test_autodiff_elemwise_sum_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag, opfuncname, testRunArgs, dimArgs
            ("elemwise_sum_run_3_3", elemwise_sum_run, ([3, 3], "float16")),

            # resnet50
            ("elemwise_sum_run_3_3", elemwise_sum_run, ([1, 256, 56, 56], "float16")),
            ("elemwise_sum_run_3_3", elemwise_sum_run, ([1, 256, 28, 28], "float16")),
            ("elemwise_sum_run_3_3", elemwise_sum_run, ([1, 512, 28, 28], "float16")),
            ("elemwise_sum_run_3_3", elemwise_sum_run, ([1, 512, 14, 14], "float16")),
            ("elemwise_sum_run_3_3", elemwise_sum_run, ([1, 1024, 14, 14], "float16")),
            ("elemwise_sum_run_3_3", elemwise_sum_run, ([1, 1024, 7, 7], "float16")),
            ("elemwise_sum_run_3_3", elemwise_sum_run, ([1, 2048, 7, 7], "float16")),

            #manual schedule
            ("elemwise_sum_run_3_3", elemwise_sum_run, ([1, 3, 3, 3], "float16"), (), False),
            ("elemwise_sum_run_3_3", elemwise_sum_run, ([1, 16, 16, 16], "float16"), (), False),
        ]
        self.testarg_cloud = [
            # testflag, opfuncname, testRunArgs, dimArgs
            # ("elemwise_sum_run_3_3", elemwise_sum_run, ([3,3], "float32")),
            # ("elemwise_sum_run_3_3", elemwise_sum_run, ([3, 3], "float16")),
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