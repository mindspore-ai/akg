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


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_tf_four2five"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # caseflag,opfuncname,testRunArgs, dimArgs
            # NCHW
            ("four2five_001", "four2five_run", ([1, 16, 16, 16], "float16", 'NCHW', 'float16')),
            ("four2five_002", "four2five_run", ([6, 16, 2, 8], "float16", 'NCHW', 'float16')),
            ("four2five_003", "four2five_run", ([32, 32, 2, 8], "float16", 'NCHW', 'float16')),
            ("four2five_004", "four2five_run", ([1, 64, 16, 16], "float16", 'NCHW', 'float16')),
            ("four2five_005", "four2five_run", ([8, 64, 16, 16], "float16", 'NCHW', 'float16')),
            ("four2five_006", "four2five_run", ([1, 64, 15, 15], "float16", 'NCHW', 'float16')),
            ("four2five_007", "four2five_run", ([1, 24, 16, 16], "float16", 'NCHW', 'float16')),
            ("four2five_008", "four2five_run", ([1, 60, 15, 15], "float16", 'NCHW', 'float16')),
            ("four2five_009", "four2five_run", ([1, 59, 121, 15], "float16", 'NCHW', 'float16')),
            # Resnet50
            ("four2five_011", "four2five_run", ([64, 64, 56, 56], "float16", 'NCHW', 'float16')),
            ("four2five_012", "four2five_run", ([32, 1001, 1, 1], "float16", 'NCHW', 'float16')),
            ("four2five_013", "four2five_run", ([1001, 2048, 1, 1], "float16", 'NCHW', 'float16')),
            ("four2five_014", "four2five_run", ([32, 2048, 1, 1], "float16", 'NCHW', 'float16')),
            ("four2five_017", "four2five_run", ([32, 256, 14, 14], "float16", 'NCHW', 'float16')),
            # ("four2five_017", four2five_run, ([32, 2048, 7, 7], "float16", 'NCHW', 'float16')),
            # ("four2five_017", four2five_run, ([32, 1024, 14, 14], "float16", 'NCHW', 'float16')),
            # ("four2five_017", four2five_run, ([32, 256, 28, 28], "float16", 'NCHW', 'float16')),
            # ("four2five_017", "four2five_run", ([32, 512, 28, 28], "float16", 'NCHW', 'float16')),
            ("four2five_017", "four2five_run", ([32, 512, 7, 7], "float16", 'NCHW', 'float16')),

            # NHWC
            ("four2five_012", "four2five_run", ([1, 16, 16, 16], "float16", 'NHWC', 'float16')),
            ("four2five_013", "four2five_run", ([6, 2, 8, 16], "float16", 'NHWC', 'float16')),
            ("four2five_014", "four2five_run", ([1, 16, 16, 64], "float16", 'NHWC', 'float16')),
            ("four2five_015", "four2five_run", ([8, 16, 16, 64], "float16", 'NHWC', 'float16')),
            #("four2five_016", four2five_run, ([1, 16, 16, 24], "float16", 'NHWC', 'float16')),
            #("four2five_017", four2five_run, ([1, 15, 15, 60], "float16", 'NHWC', 'float16')),
            ("four2five_018", "four2five_run", ([1, 121, 15, 59], "float16", 'NHWC', 'float16')),

            # Float32 case
            ("four2five_019", "four2five_run", ([6, 16, 2, 8], "float32", 'NCHW', 'float32')),
            #("four2five_020", four2five_run, ([8, 64, 16, 16], "float32", 'NCHW', 'float16')),
            #("four2five_021", four2five_run, ([1, 64, 15, 15], "float32", 'NCHW', 'float16')),
            #("four2five_022", four2five_run, ([1, 24, 16, 16], "float32", 'NCHW', 'float16')),
            ("four2five_023", "four2five_run", ([1, 59, 121, 15], "float32", 'NCHW', 'float32')),
            ("four2five_001", "four2five_run", ([32, 1, 1, 1], "float32", 'NCHW', 'float32')),

            ("four2five_024", "four2five_run", ([6, 16, 2, 8], "float32", 'NHWC', 'float32')),
            #("four2five_025", four2five_run, ([8, 64, 16, 16], "float32", 'NHWC', 'float16')),
            #("four2five_026", four2five_run, ([1, 64, 15, 15], "float32", 'NHWC', 'float16')),
            #("four2five_027", four2five_run, ([1, 24, 16, 16], "float32", 'NHWC', 'float16')),
            ("four2five_028", "four2five_run", ([1, 59, 121, 15], "float32", 'NHWC', 'float32')),
            ("four2five_017", "four2five_run", ([32, 2048, 7, 7], "float32", 'NCHW', 'float32')),
        ]

        self.testarg_rpc_cloud = [
            # Lenet
            ("four2five_012_fp32", "four2five_run", ([1, 3, 32, 32], "float16", 'NCHW', 'float16')),
            ("four2five_012_fp32", "four2five_run", ([1, 6, 15, 15], "float16", 'NCHW', 'float16')),
            ("four2five_012_fp32", "four2five_run", ([1, 16, 7, 7], "float16", 'NCHW', 'float16')),
            # Resnet50
            ("four2five_012_fp32", "four2five_run", ([32, 1001, 1, 1], "float32", 'NCHW', 'float16')),
            ("four2five_fp32_nhwc_001", "four2five_run", ([32, 224, 224, 3], "float32", 'NHWC', 'float16')),
            ("four2five_013_fp32", "four2five_run", ([1001, 2048, 1, 1], "float32", 'NCHW', 'float16')),
            # ("four2five_014_fp32", "four2five_run", ([32, 2048, 1, 1], "float32", 'NCHW', 'float16')),
            # ("four2five_015_fp16", "four2five_run", ([32, 256, 14, 14], "float16", 'NCHW', 'float16')),
            ("four2five_016", "four2five_run", ([1, 1024, 14, 14], "float16", 'NCHW', 'float16')),
            ("four2five_017", "four2five_run", ([1, 256, 14, 14], "float16", 'NCHW', 'float16')),
            ("four2five_018", "four2five_run", ([1, 512, 14, 14], "float16", 'NCHW', 'float16')),
            ("four2five_019", "four2five_run", ([1, 2048, 14, 14], "float16", 'NCHW', 'float16')),
            ("four2five_020", "four2five_run", ([32, 128, 14, 14], "float16", 'NCHW', 'float16'))
        ]
        self.testarg_level1 = [
            ("four2five_001", "four2five_run", ([1, 64, 120, 16], "float16", 'NCHW', 'float16')),
            ("four2five_002", "four2five_run", ([1, 64, 224, 16], "float16", 'NCHW', 'float16')),
            ("four2five_003", "four2five_run", ([1, 64, 121, 15], "float16", 'NCHW', 'float16')),
            ("four2five_004", "four2five_run", ([16, 4796, 1, 1], "float16", 'NCHW', 'float16')),
            ("four2five_005", "four2five_run", ([7, 114681, 1, 1], "float16", 'NCHW', 'float16')),
            ("four2five_006", "four2five_run", ([5, 214681, 1, 1], "float16", 'NCHW', 'float16')),
            ("four2five_007", "four2five_run", ([32, 2, 8, 32], "float16", 'NHWC', 'float16')),
            ("four2five_008", "four2five_run", ([1, 224, 16, 64], "float16", 'NHWC', 'float16')),
            ("four2five_009", "four2five_run", ([1, 120, 16, 64], "float16", 'NHWC', 'float16')),
            ("four2five_010", "four2five_run", ([1, 121, 15, 64], "float16", 'NHWC', 'float16')),
            ("four2five_011", "four2five_run", ([16, 1, 1, 4796], "float16", 'NHWC', 'float16')),
            ("four2five_012", "four2five_run", ([7, 1, 1, 114681], "float16", 'NHWC', 'float16')),
            ("four2five_013", "four2five_run", ([5, 1, 1, 214681], "float16", 'NHWC', 'float16')),
            ("four2five_010", "four2five_run", ([64, 3, 224, 224], "float16", 'NCHW', 'float16')),
            ("four2five_017", "four2five_run", ([32, 256, 56, 56], "float16", 'NCHW', 'float16')),
            ("four2five_029", "four2five_run", ([32, 256, 56, 56], "float32", 'NCHW', 'float16')),

            # Resnet50
            ("four2five_014", "four2five_run", ([64, 256, 56, 56], "float16", 'NCHW', 'float16')),
            ("four2five_015", "four2five_run", ([64, 128, 28, 28], "float16", 'NCHW', 'float16')),
            ("four2five_016", "four2five_run", ([64, 512, 28, 28], "float16", 'NCHW', 'float16')),
            ("four2five_017", "four2five_run", ([64, 256, 14, 14], "float16", 'NCHW', 'float16')),
            ("four2five_018", "four2five_run", ([64, 1024, 14, 14], "float16", 'NCHW', 'float16')),
            ("four2five_019", "four2five_run", ([64, 512, 7, 7], "float16", 'NCHW', 'float16')),
            ("four2five_020", "four2five_run", ([64, 2048, 7, 7], "float16", 'NCHW', 'float16')),
            ("four2five_021", "four2five_run", ([64, 64, 112, 112], "float16", 'NCHW', 'float16')),
            ("four2five_022", "four2five_run", ([32, 2048, 1, 1], "float16", 'NCHW', 'float16')),
            ("four2five_023", "four2five_run", ([64, 64, 112, 112], "float32", 'NCHW', 'float16')),
            ("four2five_024", "four2five_run", ([32, 2048, 1, 1], "float32", 'NCHW', 'float16')),
            ("four2five_001", "four2five_run", ([32, 512, 7, 7], "float32", 'NCHW', 'float16')),
            ("four2five_001", "four2five_run", ([32, 1024, 14, 14], "float32", 'NCHW', 'float16')),
        ]
        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run_level1(self):
        self.common_run(self.testarg_level1[0:10])

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
