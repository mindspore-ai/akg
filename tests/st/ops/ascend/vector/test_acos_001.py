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
acos test case
"""

import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.ascend.acos_run import acos_run


class TestAcos(TestBase):

    def setup(self):
        case_name = "test_akg_acos_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("========================{0}  Setup case=================".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("acos_f16_8_16", acos_run, ((8, 16), "float16")),
            #("acos_f16_1024_1024", acos_run, ((1024, 1024), "float16")),
            #("acos_f16_8_1024_1024", acos_run, ((8, 1024, 1024), "float16")),
            #("acos_f16_16_3_256_512", acos_run, ((16, 3, 256, 512), "float16")),
            #("acos_f16_32_16_512_512", acos_run, ((32, 16, 512, 512), "float16")),
            #("acos_f16_64_3125", acos_run, ((64, 3125), "float16")),
            #("acos_f16_96_3125", acos_run, ((96, 3125), "float16")),
            #("acos_f16_128_3125", acos_run, ((128, 3125), "float16")),
            #("acos_f16_64_1563", acos_run, ((64, 1563), "float16")),
            #("acos_f16_96_1563", acos_run, ((96, 1563), "float16")),
            #("acos_f16_128_1563", acos_run, ((128, 1563), "float16")),
            #("acos_f16_64_31250", acos_run, ((64, 31250), "float16")),
            #("acos_f16_96_31250", acos_run, ((96, 31250), "float16")),
            #("acos_f16_128_31250", acos_run, ((128, 31250), "float16")),
            #("acos_f16_64_15625", acos_run, ((64, 15625), "float16")),
            #("acos_f16_96_15625", acos_run, ((96, 15625), "float16")),
            #("acos_f16_128_15625", acos_run, ((128, 15625), "float16")),
            ("acos_f32_8_16", acos_run, ((8, 16), "float32")),
            #("acos_f32_1024_1024", acos_run, ((1024, 1024), "float32")),
            #("acos_f32_8_1024_1024", acos_run, ((8, 1024, 1024), "float32")),
            #("acos_f32_16_3_256_512", acos_run, ((16, 3, 256, 512), "float32")),
            #("acos_f32_32_16_512_512", acos_run, ((32, 16, 512, 512), "float32")),
            #("acos_f32_64_3125", acos_run, ((64, 3125), "float32")),
            #("acos_f32_96_3125", acos_run, ((96, 3125), "float32")),
            #("acos_f32_128_3125", acos_run, ((128, 3125), "float32")),
            #("acos_f32_64_1563", acos_run, ((64, 1563), "float32")),
            #("acos_f32_96_1563", acos_run, ((96, 1563), "float32")),
            #("acos_f32_128_1563", acos_run, ((128, 1563), "float32")),
            #("acos_f32_64_31250", acos_run, ((64, 31250), "float32")),
            #("acos_f32_96_31250", acos_run, ((96, 31250), "float32")),
            #("acos_f32_128_31250", acos_run, ((128, 31250), "float32")),
            #("acos_f32_64_15625", acos_run, ((64, 15625), "float32")),
            #("acos_f32_96_15625", acos_run, ((96, 15625), "float32")),
            #("acos_f32_128_15625", acos_run, ((128, 15625), "float32")),

        ]
        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
