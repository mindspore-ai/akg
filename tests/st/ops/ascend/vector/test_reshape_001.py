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
import os
import pytest
from tests.common.base import TestBase


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_reshape_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("========================{0}  Setup case=================".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, dimArgs
            # SSD
            ("reshape_004", "reshape_run", [(8, 1, 1, 24), (8, 4, 6), "float32"]),
            ("reshape_004", "reshape_run", [(8, 3, 3, 24), (8, 36, 6), "float32"]),
            ("reshape_004", "reshape_run", [(8, 5, 5, 36), (8, 150, 6), "float32"]),
            ("reshape_004", "reshape_run", [(8, 5, 5, 36), (8, 150, 6), "float32"]),
            ("reshape_004", "reshape_run", [(8, 10, 10, 36), (8, 600, 6), "float32"]),
            ("reshape_004", "reshape_run", [(8, 19, 19, 36), (8, 2166, 6), "float32"]),
            ("reshape_004", "reshape_run", [(8, 38, 38, 24), (8, 5776, 6), "float32"]),

            ("reshape_004", "reshape_run", [(1,), (1,), "float32"]),
            ("reshape_004", "reshape_run", [(8,), (8, 1), "float32"]),
            ("reshape_004", "reshape_run", [(8, 4, 6), (8, 1, 1, 24), "float32"]),

            ("reshape_004", "reshape_run", [(8, 1, 1, 16), (8, 4, 4), "float32"]),
            ("reshape_004", "reshape_run", [(8, 3, 3, 16), (8, 36, 4), "float32"]),
            ("reshape_004", "reshape_run", [(8, 5, 5, 24), (8, 150, 4), "float32"]),
            ("reshape_004", "reshape_run", [(8, 10, 10, 24), (8, 600, 4), "float32"]),
            ("reshape_004", "reshape_run", [(8, 19, 19, 24), (8, 2166, 4), "float32"]),

            ("reshape_004", "reshape_run", [(8, 38, 38, 16), (8, 5776, 4), "float32"]),

            ("reshape_004", "reshape_run", [(8, 4, 4), (8, 1, 1, 16), "float32"]),

            ("reshape_004", "reshape_run", [(8, 36, 4), (8, 3, 3, 16), "float32"]),
            ("reshape_004", "reshape_run", [(8, 36, 6), (8, 3, 3, 24), "float32"]),

            ("reshape_004", "reshape_run", [(8, 150, 4), (8, 5, 5, 24), "float32"]),
            ("reshape_004", "reshape_run", [(8, 150, 6), (8, 5, 5, 36), "float32"]),
            ("reshape_004", "reshape_run", [(8, 600, 6), (8, 10, 10, 36), "float32"]),
            ("reshape_004", "reshape_run", [(8, 600, 4), (8, 10, 10, 24), "float32"]),
            ("reshape_004", "reshape_run", [(8, 2166, 6), (8, 19, 19, 36), "float32"]),
            ("reshape_004", "reshape_run", [(8, 2166, 4), (8, 19, 19, 24), "float32"]),
            ("reshape_004", "reshape_run", [(8, 5776, 6), (8, 38, 38, 24), "float32"]),
            ("reshape_004", "reshape_run", [(8, 5776, 4), (8, 38, 38, 16), "float32"]),

            ###
            ("reshape_001", "reshape_run", [(6, 5), (30,), "int32"]),
            ("reshape_002", "reshape_run", [(18,), (6, 3), "int32"]),
            ("reshape_003", "reshape_run", [(6, 5), (5, 6), "int32"]),
            ("reshape_004", "reshape_run", [(6, 5), (3, 10), "float16"]),
            ("reshape_005", "reshape_run", [(6, 5, 5), (150,), "int32"]),
            ("reshape_006", "reshape_run", [(30, 1), (30,), "int32"]),
            ("reshape_007", "reshape_run", [(6, 5), (3, -1), "float16"]),
            ("reshape_008", "reshape_run", [(2048, 1024), (16, 128, 16, 64), "float32"],),
            ("reshape_009", "reshape_run", [(4096, 1024), (32, 128, 16, 64), "float32"],),

            # resnet50:
            ("reshape_010", "reshape_run", [(32, 2048, 1, 1), (32, 2048), "float16"],),
            ("reshape_011", "reshape_run", [(1001, 2048, 1, 1), (1001, 2048), "float16"],),
            ("reshape_012", "reshape_run", [(1, 1001, 1, 1), (1001,), "float16"],),
            ("reshape_013", "reshape_run", [(32, 1001, 1, 1), (32, 1001), "float16"],),
            ("reshape_014", "reshape_run", [(32, 1001), (32, 1001, 1, 1), "float16"],),
            ("reshape_015", "reshape_run", [(1001, 2048), (1001, 2048, 1, 1), "float16"],),
            ("reshape_016", "reshape_run", [(32, 2048), (32, 2048, 1, 1), "float16"],),
        ]
        self.testarg_cloud = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("reshape_004", "reshape_run", [(6, 5), (3, 10), "float32"], [(1, 1), (5, 5)]),
        ]

        self.testarg_rpc_cloud = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            # float - int32:[] - [1] = float:[1]
            ("reshape_001", "reshape_run", [(1,), (1,), "float32"]),
            # float - int32:[8192] - [2] = float:[8192, 1]
            ("reshape_003", "reshape_run", [(8192,), (8192, 1), "float32"]),
            ("reshape_004", "reshape_run", [(8192, 1), (8192,), "float32"]),
            # int32 - int32:[8, 128] - [1] = int32:[1024]
            ("reshape_005", "reshape_run", [(8, 128), (1024,), "int32"]),
            ("reshape_006", "reshape_run", [(1024,), (8, 128), "int32"]),
            # int32 - int32:[64, 1] - [1] = int32:[64]
            ("reshape_007", "reshape_run", [(64, 1), (64,), "int32"]),
            ("reshape_008", "reshape_run", [(64,), (64, 1), "int32"]),
            # int32 - int32:[64, 128, 1] - [1] = int32:[8192]
            ("reshape_009", "reshape_run", [(64, 128, 1), (8192,), "int32"]),
            ("reshape_010", "reshape_run", [(8192,), (64, 128, 1), "int32"]),
            # float - int32:[8192, 1024] - [4] = float:[64, 128, 16, 64]
            ("reshape_011", "reshape_run", [(8192, 1024), (64, 128, 16, 64), "float32"]),
            ("reshape_012", "reshape_run", [(64, 128, 16, 64), (8192, 1024), "float32"]),
            # float - int32:[8, 128, 1024] - [2] = float:[1024, 1024]
            ("reshape_013", "reshape_run", [(8, 128, 1024), (1024, 1024), "float32"]),
            ("reshape_014", "reshape_run", [(1024, 1024), (8, 128, 1024), "float32"]),
            # string - int32:[] - [1] = string:[1]
            # float - int32:[64, 128, 1024] - [2] = float:[8192, 1024]
            ("reshape_015", "reshape_run", [(64, 128, 1024), (8192, 1024), "float32"]),
            ("reshape_016", "reshape_run", [(8192, 1024), (64, 128, 1024), "float32"]),
            # float - int32:[64, 20] - [1] = float:[1280]
            ("reshape_017", "reshape_run", [(64, 20), (1280,), "float32"]),
            ("reshape_018", "reshape_run", [(1280,), (64, 20), "float32"]),
            # float - int32:[8, 128, 16, 64] - [2] = float:[1024, 1024]
            # ("reshape_019", "reshape_run", [(8, 128, 16, 64), (1024, 1024), "float32"]),
            ("reshape_020", "reshape_run", [(1024, 1024), (8, 128, 16, 64), "float32"]),
            # int32 - int32:[8, 20] - [1] = int32:[160]
            ("reshape_021", "reshape_run", [(8, 20), (160,), "int32"]),
            ("reshape_022", "reshape_run", [(160,), (8, 20), "int32"]),
            # float - int32:[8, 20] - [1] = float:[160]
            ("reshape_023", "reshape_run", [(8, 20), (160,), "float32"]),
            ("reshape_024", "reshape_run", [(160,), (8, 20), "float32"]),

            # float - int32:[64, 128] - [3] = float:[64, 128, 1]
            ("reshape_025", "reshape_run", [(64, 128), (64, 128, 1), "float32"]),
            ("reshape_026", "reshape_run", [(64, 128, 1), (64, 128), "float32"]),
            # float - int32:[128, 1024] - [3] = float:[1, 128, 1024]
            ("reshape_027", "reshape_run", [(128, 1024), (1, 128, 1024), "float32"]),
            ("reshape_028", "reshape_run", [(1, 128, 1024), (128, 1024), "float32"]),

            # int32 - int32:[8, 1] - [1] = int32:[8]
            ("reshape_029", "reshape_run", [(8, 1), (8,), "float32"]),
            ("reshape_030", "reshape_run", [(8,), (8, 1), "float32"]),
            # int32 - int32:[64, 128] - [1] = int32:[8192]
            ("reshape_031", "reshape_run", [(64, 128), (8192,), "int32"]),
            ("reshape_032", "reshape_run", [(8192,), (64, 128), "int32"]),
            # int32 - int32:[8, 128, 1] - [1] = int32:[1024]
            ("reshape_033", "reshape_run", [(8, 128, 1), (1024,), "int32"]),
            ("reshape_034", "reshape_run", [(1024,), (8, 128, 1), "int32"]),
            # int32 - int32:[64, 20] - [1] = int32:[1280]
            ("reshape_035", "reshape_run", [(64, 20), (1280,), "int32"]),
            ("reshape_036", "reshape_run", [(1280,), (64, 20), "int32"]),

            # float - int32:[64, 128, 16, 64] - [2] = float:[8192, 1024]
            ("reshape_037", "reshape_run", [(64, 128, 16, 64), (8192, 1024), "float32"]),
            ("reshape_038", "reshape_run", [(8192, 1024), (64, 128, 16, 64), "float32"]),
            # float - int32:[1024, 1024] - [4] = float:[8, 128, 16, 64]
            ("reshape_039", "reshape_run", [(1024, 1024), (8, 128, 16, 64), "float32"]),
            ("reshape_040", "reshape_run", [(8, 128, 16, 64), (1024, 1024), "float32"]),
            # int32 - int32:[8, 128] - [3] = int32:[8, 1, 128]
            ("reshape_041", "reshape_run", [(8, 128), (8, 1, 128), "int32"]),
            ("reshape_042", "reshape_run", [(8, 1, 128), (8, 128), "int32"]),
            # int32 - int32:[2] - [2] = int32:[2, 1]
            ("reshape_043", "reshape_run", [(2,), (2, 1), "int32"]),
            ("reshape_044", "reshape_run", [(2, 1), (2,), "int32"]),
            # int32 - int32:[64, 128] - [3] = int32:[64, 1, 128]
            ("reshape_045", "reshape_run", [(64, 128), (64, 1, 128), "int32"]),
            ("reshape_046", "reshape_run", [(64, 1, 128), (64, 128), "int32"]),
            # float - int32:[1280] - [2] = float:[1280, 1]
            ("reshape_047", "reshape_run", [(1280,), (1280, 1), "float32"]),
            ("reshape_048", "reshape_run", [(1280, 1), (1280,), "float32"]),
            # float - int32:[1024, 1024] - [3] = float:[8, 128, 1024]
            ("reshape_049", "reshape_run", [(1024, 1024), (8, 128, 1024), "float32"]),
            ("reshape_050", "reshape_run", [(8, 128, 1024), (1024, 1024), "float32"]),

            # float - int32:[8192, 1024] - [3] = float:[64, 128, 1024]
            ("reshape_051", "reshape_run", [(8192, 1024), (64, 128, 1024), "float32"]),
            # ("reshape_052", "reshape_run", [(64, 128, 1024), (8192, 1024), "float32"]),
            # float - int32:[64, 1024] - [3] = float:[64, 1, 1024]
            ("reshape_053", "reshape_run", [(64, 1024), (64, 1, 1024), "float32"]),
            ("reshape_054", "reshape_run", [(64, 1, 1024), (64, 1024), "float32"]),
            # half - int32:[128, 768, 128] - [4] = half:[128, 64, 12, 128]
            ("reshape_055", "reshape_run", [(128, 768, 128), (128, 64, 12, 128), "float16"]),
            ("reshape_056", "reshape_run", [(128, 64, 12, 128), (128, 768, 128), "float16"]),
            # int32 - int32:[64, 1] - [1] = int32:[64]
            ("reshape_057", "reshape_run", [(64, 1), (64,), "float16"]),
            ("reshape_058", "reshape_run", [(64,), (64, 1), "float16"]),
            # float - int32:[128, 128, 64] - [2] = float:[16384, 64]
            ("reshape_059", "reshape_run", [(128, 128, 64), (16384, 64), "float16"]),
            ("reshape_060", "reshape_run", [(16384, 64), (128, 128, 64), "float16"]),
            # half - int32:[8192, 768] - [3] = half:[64, 128, 768]
            ("reshape_061", "reshape_run", [(8192, 768), (64, 128, 768), "float16"]),
            ("reshape_062", "reshape_run", [(64, 128, 768), (8192, 768), "float16"]),
            # float - int32:[8192, 768] - [3] = float:[64, 128, 768]
            ("reshape_063", "reshape_run", [(8192, 768), (64, 128, 768), "float32"]),
            ("reshape_064", "reshape_run", [(64, 128, 768), (8192, 768), "float32"]),
            # half - int32:[128, 64, 12, 128] - [3] = half:[128, 768, 128]
            ("reshape_065", "reshape_run", [(128, 64, 12, 128), (128, 768, 128), "float16"]),
            ("reshape_066", "reshape_run", [(128, 768, 128), (128, 64, 12, 128), "float16"]),
            # half - int32:[128, 768, 64] - [4] = half:[128, 64, 12, 64]
            ("reshape_067", "reshape_run", [(128, 768, 64), (128, 64, 12, 64), "float16"]),
            ("reshape_068", "reshape_run", [(128, 64, 12, 64), (128, 768, 64), "float16"]),
            # float - int32:[64, 20] - [1] = float:[1280]
            ("reshape_069", "reshape_run", [(64, 20), (1280,), "float32"]),
            ("reshape_070", "reshape_run", [(1280,), (64, 20), "float32"]),
            # int32 - int32:[64, 20] - [1] = int32:[1280]
            ("reshape_071", "reshape_run", [(64, 20), (1280,), "int32"]),
            ("reshape_072", "reshape_run", [(1280,), (64, 20), "int32"]),
            # int32 - int32:[64, 128] - [3] = int32:[64, 1, 128]
            ("reshape_073", "reshape_run", [(64, 128), (64, 1, 128), "int32"]),
            ("reshape_074", "reshape_run", [(64, 1, 128), (64, 128), "int32"]),
            # int32 - int32:[64, 128, 1] - [1] = int32:[8192]
            ("reshape_075", "reshape_run", [(64, 128, 1), (8192,), "int32"]),
            ("reshape_076", "reshape_run", [(8192,), (64, 128, 1), "int32"]),
            # half - int32:[8192] - [2] = half:[8192, 1]
            ("reshape_077", "reshape_run", [(8192,), (8192, 1), "float16"]),
            ("reshape_078", "reshape_run", [(8192, 1), (8192,), "float16"]),
            # float - int32:[16384, 64] - [3] = float:[128, 128, 64]
            ("reshape_079", "reshape_run", [(16384, 64), (128, 128, 64), "float32"]),
            ("reshape_080", "reshape_run", [(128, 128, 64), (16384, 64), "float32"]),
            # string - int32:[] - [1] = string:[1]

            # float - int32:[64, 768] - [3] = float:[64, 1, 768]
            ("reshape_083", "reshape_run", [(64, 768), (64, 1, 768), "float32"]),
            ("reshape_084", "reshape_run", [(64, 1, 768), (64, 768), "float32"]),
            # float - int32:[1280] - [2] = float:[1280, 1]
            ("reshape_085", "reshape_run", [(1280,), (1280, 1), "float32"]),
            ("reshape_086", "reshape_run", [(1280, 1), (1280,), "float32"]),
            # half - int32:[64, 128, 12, 64] - [2] = half:[8192, 768]
            ("reshape_087", "reshape_run", [(64, 128, 12, 64), (8192, 768), "float16"]),
            ("reshape_088", "reshape_run", [(8192, 768), (64, 128, 12, 64), "float16"]),
            # half - int32:[64, 128, 768] - [2] = half:[8192, 768]
            ("reshape_087", "reshape_run", [(64, 128, 12, 64), (8192, 768), "float16"]),
            ("reshape_088", "reshape_run", [(8192, 768), (64, 128, 12, 64), "float16"]),
            # float - int32:[] - [1] = float:[1]
            ("reshape_089", "reshape_run", [(1,), (1,), "float32"]),
            ("reshape_090", "reshape_run", [(1,), (1,), "float32"]),
            # float - int32:[64, 128, 768] - [2] = float:[8192, 768]
            ("reshape_091", "reshape_run", [(64, 128, 768), (8192, 768), "float32"]),
            ("reshape_092", "reshape_run", [(8192, 768), (64, 128, 768), "float32"]),
            # half - int32:[128, 64, 12, 64] - [3] = half:[128, 768, 64]
            ("reshape_093", "reshape_run", [(128, 64, 12, 64), (128, 768, 64), "float16"]),
            ("reshape_094", "reshape_run", [(128, 768, 64), (128, 64, 12, 64), "float16"]),
            # int32 - int32:[16384] - [2] = int32:[128, 128]
            ("reshape_095", "reshape_run", [(16384,), (128, 128), "int32"]),
            ("reshape_096", "reshape_run", [(128, 128), (16384,), "int32"]),
            # float - int32:[64, 128] - [3] = float:[64, 128, 1]
            ("reshape_097", "reshape_run", [(64, 128), (64, 128, 1), "float32"]),
            ("reshape_098", "reshape_run", [(64, 128, 1), (64, 128), "float32"]),

            # int32 - int32:[64, 128] - [1] = int32:[8192]
            ("reshape_099", "reshape_run", [(64, 128), (8192,), "int32"]),
            ("reshape_100", "reshape_run", [(8192,), (64, 128), "int32"]),
            # half - int32:[8192, 768] - [4] = half:[64, 128, 12, 64]
            ("reshape_101", "reshape_run", [(64, 128), (8192,), "float16"]),
            ("reshape_102", "reshape_run", [(8192,), (64, 128), "float16"]),
            ("reshape_010", "reshape_run", [(1,), (1,), "float32"]),
            ("reshape_011", "reshape_run", [(8192,), (8192, 1), "float32"]),
            ("reshape_012", "reshape_run", [(8, 128), (1024,), "int32"]),
            ("reshape_013", "reshape_run", [(64, 1), (64,), "int32"]),
            ("reshape_014", "reshape_run", [(64, 128, 1), (8192,), "int32"]),
            ("reshape_015", "reshape_run", [(8192, 1024), (64, 128, 16, 64), "float32"]),
            ("reshape_019", "reshape_run", [(64, 20), (1280,), "float32"]),
            ("reshape_021", "reshape_run", [(8, 20), (160,), "int32"]),
            ("reshape_022", "reshape_run", [(8, 20), (160,), "float32"]),
            ("reshape_023", "reshape_run", [(64, 128), (64, 128, 1), "float32"]),
            ("reshape_024", "reshape_run", [(128, 1024), (1, 128, 1024), "float32"]),
            ("reshape_025", "reshape_run", [(8, 1), (8,), "float32"]),
            ("reshape_026", "reshape_run", [(64, 128), (8192,), "int32"]),
            ("reshape_027", "reshape_run", [(8, 128, 1), (1024,), "int32"]),
            ("reshape_028", "reshape_run", [(64, 20), (1280,), "int32"]),
            ("reshape_030", "reshape_run", [(1024, 1024), (8, 128, 16, 64), "float32"]),
            ("reshape_031", "reshape_run", [(8, 128), (8, 1, 128), "int32"]),
            ("reshape_032", "reshape_run", [(2,), (2, 1), "int32"]),
            ("reshape_033", "reshape_run", [(64, 128), (64, 1, 128), "int32"]),
            ("reshape_034", "reshape_run", [(1280, ), (1280, 1), "float32"]),
            ("reshape_035", "reshape_run", [(1024, 1024), (8, 128, 1024), "float32"]),
            ("reshape_036", "reshape_run", [(8192, 1024), (64, 128, 1024), "float32"]),
            ("reshape_037", "reshape_run", [(64, 1024), (64, 1, 1024), "float32"]),

            ("reshape_40", "reshape_run", [(128, 64, 12, 128), (128, 768, 128), "float16"]),
            ("reshape_41", "reshape_run", [(64,), (64, 1), "int32"]),
            ("reshape_42", "reshape_run", [(16384, 64), (128, 128, 64), "float32"]),
            ("reshape_43", "reshape_run", [(64, 128, 768), (8192, 768), "float16"]),
            ("reshape_45", "reshape_run", [(128, 768, 128), (128, 64, 12, 128), "float16"]),
            ("reshape_46", "reshape_run", [(128, 64, 12, 64), (128, 768, 64), "float16"]),
            ("reshape_47", "reshape_run", [(1280,), (64, 20), "float32"]),
            ("reshape_48", "reshape_run", [(1280,), (64, 20), "int32"]),
            ("reshape_49", "reshape_run", [(64, 1, 128), (64, 128), "int32"]),
            ("reshape_50", "reshape_run", [(8192,), (64, 128, 1), "int32"]),
            ("reshape_51", "reshape_run", [(8192, 1), (8192,), "float16"]),
            ("reshape_52", "reshape_run", [(128, 128, 64), (16384, 64), "float32"]),
            ("reshape_54", "reshape_run", [(64, 1, 768), (64, 768), "float32"]),
            ("reshape_55", "reshape_run", [(1280, 1), (1280,), "float32"]),
            ("reshape_56", "reshape_run", [(8192, 768), (64, 128, 12, 64), "float16"]),
            ("reshape_57", "reshape_run", [(8192, 768), (64, 128, 768), "float16"]),
            ("reshape_59", "reshape_run", [(8192, 768), (64, 128, 768), "float32"]),
            ("reshape_60", "reshape_run", [(128, 768, 64), (128, 64, 12, 64), "float16"]),
            ("reshape_61", "reshape_run", [(128, 128), (16384,), "int32"]),
            ("reshape_62", "reshape_run", [(64, 128, 1), (64, 128), "float32"]),
            ("reshape_63", "reshape_run", [(8192,), (64, 128), "int32"]),
            ("reshape_64", "reshape_run", [(64, 128, 12, 64), (8192, 768), "float16"]),
            ("reshape_016", "reshape_run", [(8, 128, 1024), (1024, 1024), "float32"], ((16, 1), (256, 1))),
            ("reshape_018", "reshape_run", [(64, 128, 1024), (8192, 1024), "float32"], ((16, 1), (256, 1))),
            ("reshape_020", "reshape_run", [(8, 128, 16, 64), (1024, 1024), "float32"], ((16, 1), (256, 1))),
            ("reshape_029", "reshape_run", [(64, 128, 16, 64), (8192, 1024), "float32"], ((16, 1), (256, 1))),
            ("reshape_44", "reshape_run", [(64, 128, 768), (8192, 768), "float32"], ((16, 1), (256, 1))),
            ("reshape_ssd_001", "reshape_run", [(8, 38, 38, 24), (8, 5776, 6), "float32"]),
            # dont support string for now:
            # ("reshape_017", "reshape_run", [(1,), (1,), "string"])

            # resnet50:
            ("reshape_resnet50_010", "reshape_run", [(32, 2048, 1, 1), (32, 2048), "float32"],),
            ("reshape_resnet50_011", "reshape_run", [(1001, 2048, 1, 1), (1001, 2048), "float32"],),
            ("reshape_resnet50_012", "reshape_run", [(1, 1001, 1, 1), (1001,), "float32"],),
            ("reshape_resnet50_013", "reshape_run", [(32, 1001, 1, 1), (32, 1001), "float32"],),
            ("reshape_resnet50_014", "reshape_run", [(32, 1001), (32, 1001, 1, 1), "float32"],),
            ("reshape_resnet50_015", "reshape_run", [(1001, 2048), (1001, 2048, 1, 1), "float32"],),
            ("reshape_resnet50_016", "reshape_run", [(32, 2048), (32, 2048, 1, 1), "float32"],),
            # lenet:
            ("reshape_lenet", "reshape_run", [(1, 16, 7, 7), (1, 784), "float16"],),

            # bert
            ("reshape_1", "reshape_run", [(128, 1024), (1, 16, 128, 64), "float32"]),
        ]

        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_run_cloud(self):
        self.common_run(self.testarg_cloud)

    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    def teardown(self):
        """
          clean environment
          :return:
          """
        self._log.info("========================{0} Teardown case=================".format(self.casename))


if __name__ == "__main__":
    t = TestCase()
    t.setup()
    t.test_run()
    t.teardown()
