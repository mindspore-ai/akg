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
from tests.common.test_run.mul_run import mul_run


class TestMul(TestBase):

    def setup(self):
        case_name = "test_akg_mul_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))

        self.testarg_ci = [
            ("mul_01", mul_run, ([(64, 2), (64, 2)], "float16")),
            ("mul_04", mul_run, ([(1,), (2,)], "float16")),
            ("mul_17", mul_run, ([(160, 1), (1024,)], "float16")),
            ("mul_19", mul_run, ([(2, 1024), (1,)], "float16")),
            ("mul_30", mul_run, ([(8, 128, 1), (8, 1, 128)], "float16")),
            ("mul_33", mul_run, ([(64, 1), (64, 2)], "float16")),
            ("mul_37", mul_run, ([(1,), (30522,)], "float16")),
            ("mul_58", mul_run, ([(2,), (1,)], "float16")),
            ("mul_60", mul_run, ([(4096,), (1,)], "float16")),
            ("mul_62", mul_run, ([(1,), (4096,)], "float16")),
            ("mul_26", mul_run, ([(30522,), (1,)], "float16")),
            ("mul_31", mul_run, ([(1,), (2, 1024)], "float16")),
            ("mul_34", mul_run, ([(1280, 1), (1024,)], "float16")),
            ("mul_38", mul_run, ([(8, 128, 1), (1024,)], "float16")),
            ("mul_40", mul_run, ([(64, 128, 1), (64, 1, 128)], "float16")),
            ("mul_46", mul_run, ([(160, 1), (160, 1024)], "float16")),
            ("mul_53", mul_run, ([(1024,), (1,)], "float16")),
            ("mul_59", mul_run, ([(8, 1, 128, 128), (1,)], "float16")),
            ("mul_64", mul_run, ([(1024, 1), (1024,)], "float16")),
            ("mul_67", mul_run, ([(1,), (160, 1024)], "float16")),
            ("mul_25", mul_run, ([(1,), (512, 1024)], "float16")),
            ("mul_39", mul_run, ([(64, 128, 1), (1024,)], "float16")),
            ("mul_42", mul_run, ([(64, 1, 128, 128), (1,)], "float16")),
        ]

        self.testarg_nightly = [
            ("mul_70", mul_run, ([(1,), (1,)], "float16"), ((1, 1),)),
            ("mul_71", mul_run, ([(1, 1, 384, 96), (1,)], "float16")),
            ("mul_72", mul_run, ([(1, 1, 1536, 1536), (1,)], "float16")),
            ("mul_74", mul_run, ([(3, 3, 32, 1), (1,)], "float16")),
            ("mul_78", mul_run, ([(1, 1, 256, 21), (1,)], "float16")),
            ("mul_79", mul_run, ([(64,), (1,)], "float16")),
            ("mul_80", mul_run, ([(1, 1, 256, 3), (1,)], "float16")),
            ("mul_83", mul_run, ([(1536,), (1,)], "float16")),
            ("mul_85", mul_run, ([(256,), (1,)], "float16")),
            ("mul_87", mul_run, ([(3, 3, 192, 1), (1,)], "float16")),
            ("mul_89", mul_run, ([(3, 3, 960, 1), (1,)], "float16")),
            ("mul_91", mul_run, ([(4, 33, 33, 256), (4, 33, 33, 256)], "float16")),
            ("mul_92", mul_run, ([(1, 1, 96, 576), (1,)], "float16")),
            ("mul_93", mul_run, ([(1052676, 1), (1052676, 21)], "float16")),
            ("mul_94", mul_run, ([(1, 1, 32, 192), (1,)], "float16")),
            ("mul_96", mul_run, ([(3, 3, 3, 32), (1,)], "float16")),
            ("mul_97", mul_run, ([(3, 3, 96, 1), (1,)], "float16")),
            ("mul_98", mul_run, ([(1, 1, 728, 728), (1,)], "float16")),
            ("mul_99", mul_run, ([(1, 1, 256, 256), (1,)], "float16")),
            ("mul_100", mul_run, ([(2048,), (1,)], "float16")),
            ("mul_101", mul_run, ([(1, 1, 64, 128), (1,)], "float16")),
            ("mul_102", mul_run, ([(1,), (4, 33, 33, 256)], "float16")),
            ("mul_103", mul_run, ([(1, 1, 320, 256), (1,)], "float16")),
            ("mul_104", mul_run, ([(1, 1, 128, 256), (1,)], "float16")),
            ("mul_105", mul_run, ([(1,), (1, 33, 33, 3)], "float16")),
            ("mul_106", mul_run, ([(1, 1, 64, 384), (1,)], "float16")),
            ("mul_108", mul_run, ([(1, 1, 32, 16), (1,)], "float16")),
            ("mul_109", mul_run, ([(2,), (1,)], "float16")),
            ("mul_110", mul_run, ([(3, 3, 384, 1), (1,)], "float16")),
            ("mul_111", mul_run, ([(1, 1, 256, 48), (1,)], "float16")),
            ("mul_112", mul_run, ([(1, 1, 728, 1024), (1,)], "float16")),
            ("mul_114", mul_run, ([(1, 33, 33, 3), (1,)], "float16")),
            ("mul_115", mul_run, ([(1, 1, 144, 32), (1,)], "float16")),
            ("mul_116", mul_run, ([(1,), (4, 513, 513, 3)], "float16")),
            ("mul_117", mul_run, ([(1, 1, 256, 728), (1,)], "float16")),
            ("mul_118", mul_run, ([(1, 1, 192, 32), (1,)], "float16")),
            ("mul_119", mul_run, ([(32,), (1,)], "float16")),
            ("mul_120", mul_run, ([(3, 3, 144, 1), (1,)], "float16")),
            ("mul_121", mul_run, ([(3, 3, 256, 1), (1,)], "float16")),
            ("mul_122", mul_run, ([(3, 3, 32, 64), (1,)], "float16")),
            ("mul_124", mul_run, ([(1, 1, 144, 24), (1,)], "float16")),
            ("mul_125", mul_run, ([(48,), (1,)], "float16")),
            ("mul_126", mul_run, ([(1, 1, 304, 256), (1,)], "float16")),
            ("mul_127", mul_run, ([(3, 3, 304, 1), (1,)], "float16")),
            ("mul_128", mul_run, ([(1, 1, 576, 96), (1,)], "float16")),
            ("mul_129", mul_run, ([(128,), (1,)], "float16")),
            ("mul_130", mul_run, ([(1, 1, 1024, 1536), (1,)], "float16")),
            ("mul_131", mul_run, ([(21,), (1,)], "float16")),
            ("mul_132", mul_run, ([(1, 1, 384, 64), (1,)], "float16")),
            ("mul_133", mul_run, ([(1,), (1, 513, 513, 3)], "float16")),
            ("mul_134", mul_run, ([(1, 1, 512, 256), (1,)], "float16")),
            ("mul_135", mul_run, ([(1, 1, 144, 48), (1,)], "float16")),
            ("mul_136", mul_run, ([(1, 1, 1536, 2048), (1,)], "float16")),
            ("mul_137", mul_run, ([(1, 1, 96, 24), (1,)], "float16")),
        ]
        self.testarg_cloud = [
            ("mul_04", mul_run, ([(1,), (2,)], "float32"), ((2, 2),)),
        ]

        self.testarg_rpc_cloud = [
            ("mul_70", mul_run, ([(1,), (64, 12, 128, 128)], "float16")),
            ("mul_71", mul_run, ([(1,), (8192, 3072)], "float16")),
            ("mul_72", mul_run, ([(1,), (8192, 768)], "float16")),
            ("mul_73", mul_run, ([(64, 1, 128, 128), (1,)], "float16")),
            ("mul_74", mul_run, ([(64, 12, 128, 128), (1,)], "float16")),
            ("mul_75", mul_run, ([(64, 12, 128, 128), (64, 12, 128, 128)], "float16")),
            ("mul_76", mul_run, ([(8192, 1), (768,)], "float16")),
            ("mul_77", mul_run, ([(8192, 1), (8192, 768)], "float16")),
            ("mul_78", mul_run, ([(8192, 3072), (1,)], "float16")),
            ("mul_79", mul_run, ([(8192, 3072), (8192, 3072)], "float16")),
            ("mul_80", mul_run, ([(8192, 768), (768,)], "float16")),

            ("mul_fp32_2", mul_run, ([(1,), (1,)], "float32")),
            ("mul_fp32_3", mul_run, ([(1,), (1280, 768)], "float32"), ),
            ("mul_fp32_4", mul_run, ([(1,), (2,)], "float32"), ((1, 1), (1, 1))),
            ("mul_fp32_5", mul_run, ([(1,), (2, 768)], "float32")),
            ("mul_fp32_6", mul_run, ([(1,), (21128,)], "float32")),
            ("mul_fp32_7", mul_run, ([(1,), (21128, 768)], "float32")),
            ("mul_fp32_9", mul_run, ([(1,), (3072, 768)], "float32")),
            ("mul_fp32_10", mul_run, ([(1,), (33, 64)], "float32")),
            ("mul_fp32_11", mul_run, ([(1,), (64, 128, 768)], "float32")),
            ("mul_fp32_12", mul_run, ([(1,), (768,)], "float32")),
            ("mul_fp32_13", mul_run, ([(1,), (768, 3072)], "float32")),
            ("mul_fp32_16", mul_run, ([(1280,), (1280,)], "float32")),
            ("mul_fp32_17", mul_run, ([(1280, 1), (1280, 21128)], "float32")),
            ("mul_fp32_18", mul_run, ([(1280, 1), (1280, 768)], "float32")),
            ("mul_fp32_19", mul_run, ([(1280, 1), (768,)], "float32")),
            ("mul_fp32_20", mul_run, ([(1280, 21128), (1280, 21128)], "float32")),
            ("mul_fp32_21", mul_run, ([(1280, 768), (1,)], "float32")),
            ("mul_fp32_23", mul_run, ([(1280, 768), (768,)], "float32")),
            ("mul_fp32_24", mul_run, ([(2,), (1,)], "float32")),
            ("mul_fp32_25", mul_run, ([(2,), (2,)], "float32")),
            ("mul_fp32_26", mul_run, ([(2, 768), (1,)], "float32")),
            ("mul_fp32_27", mul_run, ([(2, 768), (2, 768)], "float32")),
            ("mul_fp32_28", mul_run, ([(21128,), (1,)], "float32")),
            ("mul_fp32_29", mul_run, ([(21128,), (21128,)], "float32")),
            ("mul_fp32_32", mul_run, ([(3072,), (1,)], "float32")),
            ("mul_fp32_33", mul_run, ([(3072,), (3072,)], "float32")),
            ("mul_fp32_37", mul_run, ([(33, 64), (33, 64)], "float32")),
            ("mul_fp32_38", mul_run, ([(64, 1), (64, 2)], "float32")),
            ("mul_fp32_39", mul_run, ([(64, 128, 1), (64, 1, 128)], "float32")),
            ("mul_fp32_40", mul_run, ([(64, 128, 1), (64, 128, 768)], "float32")),
            ("mul_fp32_41", mul_run, ([(64, 128, 1), (768,)], "float32")),
            ("mul_fp32_42", mul_run, ([(64, 128, 768), (1,)], "float32")),
            ("mul_fp32_44", mul_run, ([(64, 128, 768), (768,)], "float32")),
            ("mul_fp32_49", mul_run, ([(768, 3072), (768, 3072)], "float32")),
        ]
        self.testarg_5d_rpc_cloud = [
            ("mul_fp16_01", mul_run, ([(32, 128, 1, 1, 16), (1, 1, 1, 16)], "float16")),
            ("mul_fp32_02", mul_run, ([(32, 128, 1, 1, 16), (1, 1, 1, 16)], "float32")),
            # mul_fp16_03 and mul_fp32_04 origin 4D shape is (1,1001,1,1), So tans to 5d, C1 = Ceil(1001/16) = 63
            ("mul_fp16_03", mul_run, ([(32, 63, 1, 1, 16), (1, 1, 1, 16)], "float16")),
            ("mul_fp32_04", mul_run, ([(32, 63, 1, 1, 16), (1, 1, 1, 16)], "float32")),
            ("mul_fp16_05", mul_run, ([(512, 128, 1, 1, 16), (1, 1, 1, 16)], "float16")),
            ("mul_fp32_06", mul_run, ([(512, 128, 1, 1, 16), (1, 1, 1, 16)], "float32")),
            ("mul_fp16_07", mul_run, ([(512, 32, 3, 3, 16), (1, 1, 1, 16)], "float16")),
            ("mul_fp32_08", mul_run, ([(512, 32, 3, 3, 16), (1, 1, 1, 16)], "float32")),
        ]
        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run_ci(self):
        """
        run case
        :return:
        """
        self.common_run(self.testarg_ci)

    def test_run_nightly(self):
        self.common_run(self.testarg_nightly)

    def test_run_cloud(self):
        """
        run case
        :return:
        """
        self.common_run(self.testarg_cloud)

    def test_run_rpc_cloud(self):
        """
        run case
        :return:
        """
        # for arg in self.testarg_5d_rpc_cloud:
        #     self.print_debug(arg)
        # assert self.caseresult
        self.common_run(self.testarg_5d_rpc_cloud)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
