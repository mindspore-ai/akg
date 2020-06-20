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
dropout test cast
"""
import os
from base import TestBase
from nose.plugins.attrib import attr
import pytest


class Testdropout(TestBase):

    def setup(self):
        case_name = "test_akg_dropout_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag, opfuncname, testRunArgs, dimArgs
            ("dropout_01", "dropout_run", ((128, 128), 0.5, "float16", "cce_dropout_do_mask")),
            # ("dropout_02", "dropout_run", ((8, 8), 0.5, "float32", "cce_dropout_do_mask")),
            ("dropout_01", "dropout_run", ((7, 7), 0.5, "float16", "cce_dropout_do_mask")),
            ("dropout_01", "dropout_run", ((7, 7), 0.5, "float32", "cce_dropout_do_mask")),
            ("dropout_01", "dropout_run", ((512, 7, 7), 0.5, "float16", "cce_dropout_do_mask")),
            ("dropout_01", "dropout_run", ((512, 7, 7), 0.5, "float32", "cce_dropout_do_mask")),
            ("dropout_01", "dropout_run", ((1024, 7, 7), 0.5, "float16", "cce_dropout_do_mask")),
            ("dropout_01", "dropout_run", ((1024, 7, 7), 0.5, "float32", "cce_dropout_do_mask")),
            ("dropout_02", "dropout_run", ((8, 128), 0.5, "float16", "cce_dropout_do_mask")),
            ("dropout_03", "dropout_run", ((8, 128), 0.5, "float32", "cce_dropout_do_mask")),
            ("dropout_04", "dropout_run", ((64, 128, 768), 0.5, "float16", "cce_dropout_do_mask")),
            ("dropout_05", "dropout_run", ((64, 128, 768), 0.5, "float32", "cce_dropout_do_mask")),
            ("dropout_06", "dropout_run", ((64, 12, 128, 128), 0.5, "float16", "cce_dropout_do_mask")),
            ("dropout_07", "dropout_run", ((64, 12, 128, 128), 0.5, "float32", "cce_dropout_do_mask")),
            ("dropout_08", "dropout_run", ((8196, 768), 0.5, "float16", "cce_dropout_do_mask")),
            ("dropout_09", "dropout_run", ((8196, 768), 0.5, "float32", "cce_dropout_do_mask")),
            ("dropout_10", "dropout_run", ((2, 1280), 0.5, "float16", "cce_dropout_do_mask")),
            ("dropout_11", "dropout_run", ((2, 1280), 0.5, "float32", "cce_dropout_do_mask")),
            ("dropout_12", "dropout_run", ((1, 128, 1024), 0.5, "float32", "cce_dropout_do_mask")),
            ("dropout_13", "dropout_run", ((1, 16, 128, 128), 0.5, "float32", "cce_dropout_do_mask")),
            ("dropout_14", "dropout_run", ((1, 16, 128, 128), 0.5, "float16", "cce_dropout_do_mask")),
        ]

        self.testarg_nightly = [
            ("dropout_06", "dropout_run", ((64, 12, 128, 128), 0.5, "float16", "cce_dropout_do_mask")),
            ("dropout_07", "dropout_run", ((64, 12, 128, 128), 0.5, "float32", "cce_dropout_do_mask")),
        ]

        self.testarg_rpc_cloud = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            # ("dropout_04", "dropout_run", ((64, 128, 1024), "float32")),
            ("dropout_01", "dropout_run", ((84, 128, 768), 0.5, "float16", "cce_dropout_do_mask")),
            ("dropout_02", "dropout_run", ((84, 128, 768), 0.5, "float32", "cce_dropout_do_mask")),
            ("dropout_07", "dropout_run", ((1, 128, 1024), 0.5, "float32", "cce_dropout_do_mask")),
            ("dropout_08", "dropout_run", ((1, 16, 128, 128), 0.5, "float32", "cce_dropout_do_mask")),
            ("dropout_08", "dropout_run", ((1, 16, 128, 128), 0.5, "float16", "cce_dropout_do_mask")),
            # ("dropout_03", "dropout_run", ((4, 10), 0.5, "float32", "cce_dropout_do_mask")),
            # float16:[64, 128, 1024] = float:[64, 128, 1024]
            ("dropout_001", "dropout_run", ((64, 128, 1024), 0.5, "float16", "cce_dropout_do_mask")),
            # float16:[64, 16, 128, 128] = float:[64, 16, 128, 128]
            ("dropout_002", "dropout_run", ((64, 16, 128, 128), 0.5, "float16", "cce_dropout_do_mask")),
            # float16:[64 * 128, 1024] = float:[64 * 128, 1024]
            ("dropout_003", "dropout_run", ((64 * 128, 1024), 0.5, "float16", "cce_dropout_do_mask")),
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
