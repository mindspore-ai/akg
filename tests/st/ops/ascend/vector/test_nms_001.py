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
iou_for_train test cast
"""

import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.nms_run import nms_run


class Testcase(TestBase):
    def setup(self):
        case_name = "test_akg_iou_for_train_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            ("nms_01", nms_run, ((1, 16, 8), 0.5, "float16", "cce_nms"),),
            ("nms_02", nms_run, ((1, 64, 8), 0.5, "float16", "cce_nms"),),
            ("nms_03", nms_run, ((2, 16, 8), 0.5, "float16", "cce_nms"),),
            ("nms_04", nms_run, ((2, 2048, 8), 0.5, "float16", "cce_nms"),),
        ]
        self.testarg_nightly = [
            ("nms_01", nms_run, ((1, 16, 8), 0.7, "float16", "cce_nms"),),
        ]

        self.testarg_rpc_cloud = [
            ("nms_01", nms_run, ((1, 16, 8), 0.3, "float16", "cce_nms"),),
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

    def test_run_1(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_nightly)

    def test_run_rpc_cloud(self):
        self.common_run([self.testarg_rpc_cloud[0]])

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
