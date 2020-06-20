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

import datetime
import os

from base import TestBase
from test_run.softmax_ad_run import softmax_ad_run
import pytest


############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):
    def setup(self):
        """
        testcase preparcondition
        :return:
        """
        case_name = "test_akg_softmax"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("softmax_ad_00", softmax_ad_run, ((16,), "float16", -1, "cce_softmax_ad_fp16", True), ),
            ("softmax_ad_00", softmax_ad_run, ((64,), "float16", -1, "cce_softmax_ad_fp16", True), ),

            ("softmax_ad_00", softmax_ad_run, ((1, 16), "float16", -1, "cce_softmax_ad_fp16", True), ),
            ("softmax_ad_01", softmax_ad_run, ((16, 64), "float16", -1, "cce_softmax_ad_fp16", True), ),
            ("softmax_ad_02", softmax_ad_run, ((1, 5), "float16", -1, "cce_softmax_ad_fp16", True), ),  # softmax in mobilenet
            ("softmax_ad_03", softmax_ad_run, ((1, 12), "float16", -1, "cce_softmax_ad_fp16", True), ),  # softmax in mobilenet
            ("softmax_ad_04", softmax_ad_run, ((1, 1000), "float16", -1, "cce_softmax_ad_fp16", True), ),  # softmax in resnet
            # ("softmax_ad_05", softmax_ad_run, ((4298, 30522), "float16", -1, "cce_softmax_ad_fp16", True), ), # logsoftmax in Bert # out of memory

            ("softmax_ad_010", softmax_ad_run, ((1, 16, 64), "float16", -1, "cce_softmax_ad_fp16", True)),
            ("softmax_ad_010", softmax_ad_run, ((1, 64, 128), "float16", -1, "cce_softmax_ad_fp16", True)),

            ("softmax_ad_010", softmax_ad_run, ((1, 16, 32, 64), "float16", -1, "cce_softmax_ad_fp16", True)),
        ]

        self.testarg_rpc_cloud = [
            ## testflag,opfuncname,testRunArgs, dimArgs

            # float:[64, 16, 128, 128] = float:[64, 16, 128, 128]
            ####("softmax_ad_001", softmax_ad_run, ((64, 16, 128, 128), "float32", -1, "cce_softmax_ad_fp32")),
            # float:[8, 16, 128, 128] = float:[8, 16, 128, 128]
            ####("softmax_ad_002", softmax_ad_run, ((8, 16, 128, 128), "float32", -1, "cce_softmax_ad_fp32")),

        ]
        self.testarg_level = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ####("softmax_ad_001", softmax_ad_run, ((64, 16, 128, 128), "float16", -1, "cce_softmax_ad_fp16")),
        ]
        return

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
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_rpc_cloud(self):
        if len(self.testarg_rpc_cloud) > 0:
            self.common_run([self.testarg_rpc_cloud[0]])

    @pytest.mark.level0
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_level1(self):
        self.common_run(self.testarg_level)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
