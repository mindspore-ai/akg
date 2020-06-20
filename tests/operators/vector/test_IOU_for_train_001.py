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
from base import TestBase
from nose.plugins.attrib import attr
from test_run.IOU_for_train_run import iou_for_train_run


class Testcase(TestBase):

    def setup(self):
        case_name = "test_akg_iou_for_train_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            ("iou_01", iou_for_train_run, ((1, 16, 8), (1, 16, 8), "float16", "iou"),),
            ("iou_02", iou_for_train_run, ((1, 64, 8), (1, 48, 8), "float16", "iou"),),
            ("iou_03", iou_for_train_run, ((8, 64, 8), (8, 48, 8), "float16", "iou"),),
            ("iou_04", iou_for_train_run, ((2, 4704, 8), (2, 32, 8), "float16", "iou"),),
            ("iou_05", iou_for_train_run, ((2, 64, 8), (2, 2048, 8), "float16", "iou"),),
        ]

        self.testarg_nightly = [
            ("iou_01", iou_for_train_run, ((1, 16, 8), (1, 16, 8), "float16", "iou"),),
        ]

        self.testarg_rpc_cloud = [
            ("iou_01", iou_for_train_run, ((1, 16, 8), (1, 16, 8), "float16", "iou"),),
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

    @pytest.mark.level1
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_1(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_nightly)

    @pytest.mark.rpc_cloud
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_rpc_cloud(self):
        self.common_run([self.testarg_rpc_cloud[0]])

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
