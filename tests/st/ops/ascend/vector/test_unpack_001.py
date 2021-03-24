# Copyright 2020 Huawei Technologies Co., Ltd
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

"""unpack test case"""

import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.unpack_run import unpack_run


class TestUnpack(TestBase):
    def setup(self):
        case_name = "test_akg_unpack_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("=================%s Setup case=================", self.casename)
        self.testarg = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("unpack_su8_8_8", unpack_run, ((8, 8), "int8", "ND", 8, 1)),
            ("unpack_u8_8_8", unpack_run, ((8, 8), "uint8", "ND", 8, 1)),
            ("unpack_s16_8_8", unpack_run, ((8, 8), "int16", "ND", 8, 1)),
            ("unpack_u16_8_8", unpack_run, ((8, 8), "uint16", "ND", 8, 1)),
            ("unpack_s32_8_8", unpack_run, ((8, 8), "int32", "ND", 8, 1)),
            ("unpack_u32_8_8", unpack_run, ((8, 8), "uint32", "ND", 8, 1)),
            ("unpack_s64_8_8", unpack_run, ((8, 8), "int64", "ND", 8, 1)),
            ("unpack_u64_8_8", unpack_run, ((8, 8), "uint64", "ND", 8, 1)),

            ("unpack_f16_8_8", unpack_run, ((8, 8), "float16", "ND", 8, 1)),
            ("unpack_f32_8_8", unpack_run, ((8, 8), "float32", "ND", 8, 1)),

            ("unpack_f32_NC1HWC0", unpack_run, ((8, 8, 8, 8, 8), "float32", "NC1HWC0", 8, 2)),
            ("unpack_f32_NHWC", unpack_run, ((7, 5, 8, 8), "float32", "NHWC", 5, 1)),
            ("unpack_f32_NCHW", unpack_run, ((7, 5, 8, 8), "float32", "NCHW", 8, 2)),
            ("unpack_f32_HWCN", unpack_run, ((7, 5, 8, 8), "float32", "HWCN", 7, 0)),
        ]
        self.testarg_cloud = [
            ("unpack_s64_big", unpack_run, ((127,), "int64", "ND", None, 0)),
            ("unpack_u64_big", unpack_run, ((127,), "uint64", "ND", None, 0)),
            ("unpack_s8_big", unpack_run, ((127,), "int8", "ND", None, 0)),
            ("unpack_u8_big", unpack_run, ((127,), "uint8", "ND", None, 0)),
            ("unpack_s32_big", unpack_run, ((127,), "int32", "ND", None, 0)),
            ("unpack_u32_big", unpack_run, ((127,), "uint32", "ND", None, 0)),
            ("unpack_fp16_big", unpack_run, ((127,), "float16", "ND", None, 0)),
            ("unpack_fp32_big", unpack_run, ((127,), "float32", "ND", None, 0)),
            ("unpack_f16_8_big", unpack_run, ((8, 127), "float16", "ND", None, 0)),
        ]

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_mini_run(self):
        """mini run case"""
        self.common_run(self.testarg)

    def test_cloud_run(self):
        """cloud run case"""
        self.common_run(self.testarg_cloud)

    def teardown(self):
        """clean environment"""
        self._log.info("=============%s Teardown============", self.casename)
