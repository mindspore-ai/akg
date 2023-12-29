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
batch_norm
"""

import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.ascend.batch_norm_run import batch_norm_run


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_batch_norm_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            #caseflag,opfuncname,testRunArgs, dimArgs
            ("batch_norm_run1", batch_norm_run, ((1, 2, 4, 128), "float16", 1e-5, "batch_norm_forward_output"), ((1, 1), (4, 4), (128, 128)), True),
            # ("batch_norm_run2", batch_norm_run, ((1, 2, 4, 128), "float16", 1e-5, "batch_norm_forward_output"), ((1, 1), (4, 4), (128, 128)), False),
            # ("batch_norm_run3",batch_norm_run,((1, 2, 4, 128, 16),"float16",1e-5,"batch_norm_forward_output"), ((1,1),(128,128),(16,16)), False),
        ]
        self.testarg_cloud = [
            #caseflag,opfuncname,testRunArgs, dimArgs
            #("batch_norm_run3",batch_norm_run,((1, 2, 4, 128),"float16",1e-5,"batch_norm_forward_output"), ((1,1),(4,4),(128,128)), True),
            ("batch_norm_run4", batch_norm_run, ((1, 2, 4, 128), "float16", 1e-5, "batch_norm_forward_output"), ((1, 1), (4, 4), (128, 128)), False),
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

