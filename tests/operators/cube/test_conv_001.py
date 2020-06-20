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

"""testcase for conv op"""

import os
import pytest
from base import TestBase
from test_run.conv_run import conv_run


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_conv_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag, opfuncname, testRunArgs,
            # testflag, opfuncname, fmap_shape               , filter_shape          , pad_                                       , stride_              , dilation_                , use_bias, b Tile
            #(testflag, opfuncname, ((in_n, in_c, in_h, in_w), (cout, in_c, w_h, w_w), (pad_left, pad_right, pad_top, pad_bottom), (stride_h, stride_w), (dilation_h, dilation_w), bias,  [cutH, cutCo, cutM, cutK, cutN]))
            ('conv-perf_yolov3_convquant_fp16_3113.param', conv_run, ((1, 256, 26, 26), (512, 256, 3, 3), (1, 1, 1, 1), (1, 1), [1, 1], False)),

            # dilation testcase
            ("conv_run_dilation_000", conv_run, ((1, 128, 36, 36), (128, 128, 3, 3), (0, 0, 0, 0), (1, 1), (2, 2), False)),
            ("conv_run_dilation_001", conv_run, ((1, 128, 34, 34), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (2, 2), True)),
            ("conv_run_dilation_002", conv_run, ((1, 128, 28, 28), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (2, 2), True)),
            ("conv_run_dilation_003", conv_run, ((1, 256, 14, 14), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), (2, 2), True)),
            # ("conv_run_dilation_004", conv_run,((1, 3, 224, 224), (64, 3, 7, 7), (1, 1, 1, 1), (1, 1), (2, 2), True, [65, 64, 448, 32, 64])),
            ("conv_run_dilation_005", conv_run, ((1, 512, 7, 7), (512, 512, 3, 3), (1, 1, 1, 1), (1, 1), (2, 2), True)),

            # resnet50_wkl
            # cutw case
            # comment this failed case temporarily
            #("conv_run001", conv_run, ((1, 1024, 14, 14), (2048, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), True), [14, 2048, 64, 96, 128, 14]),
            ("conv_run001", conv_run, ((1, 1024, 14, 14), (2048, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), True)),
            ("conv_run002", conv_run, ((1, 1024, 14, 14), (256, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            ("conv_run003", conv_run, ((1, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), True)),
            ("conv_run004", conv_run, ((1, 128, 28, 28), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), True)),
            ("conv_run005", conv_run, ((1, 128, 28, 28), (512, 128, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            ("conv_run006", conv_run, ((1, 2048, 7, 7), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            ("conv_run007", conv_run, ((1, 256, 14, 14), (1024, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            ("conv_run008", conv_run, ((1, 256, 14, 14), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), True)),
            ("conv_run009", conv_run, ((1, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), True)),
            ("conv_run010", conv_run, ((1, 256, 56, 56), (64, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            ("conv_run011", conv_run, ((1, 3, 224, 224), (64, 3, 7, 7), (3, 3, 3, 3), (2, 2), (1, 1), True)),
            ("conv_run012", conv_run, ((1, 512, 28, 28), (128, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            ("conv_run013", conv_run, ((1, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), True)),
            ("conv_run014", conv_run, ((1, 512, 7, 7), (2048, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            ("conv_run015", conv_run, ((1, 512, 7, 7), (512, 512, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), True)),
            ("conv_run016", conv_run, ((1, 64, 56, 56), (256, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            ("conv_run017", conv_run, ((1, 64, 56, 56), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            ("conv_run018", conv_run, ((1, 64, 56, 56), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), True)),
            ("conv_run019", conv_run, ((1, 256, 56, 56), (512, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), True)),
            ("conv_run020", conv_run, ((1, 512, 28, 28), (1024, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), True)),

            # reid
            ("conv_reid_run001", conv_run, ((1, 3, 112, 112), (64, 3, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)),
            ("conv_reid_run002", conv_run, ((1, 64, 112, 112), (64, 64, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)),
            ("conv_reid_run003", conv_run, ((1, 64, 112, 112), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)),
            # fail
            # ("conv_reid_run004", conv_run, ((1, 64,  112, 112), (64,  64,  1, 1), (1, 1, 1, 1), (2, 2), (1, 1), False)),
            ("conv_reid_run005", conv_run, ((1, 64, 56, 56), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)),
            # ("conv_reid_run006", conv_run, ((1, 64,  56,   56), (64,  64,  3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)),
            ("conv_reid_run007", conv_run, ((1, 64, 56, 56), (128, 64, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)),
            ("conv_reid_run008", conv_run, ((1, 64, 56, 56), (128, 64, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)),
            ("conv_reid_run009", conv_run, ((1, 128, 56, 56), (128, 128, 3, 3), (1, 1, 1, 1), (2, 2), (1, 1), False)),
            ("conv_reid_run010", conv_run, ((1, 128, 28, 28), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)),
            # ("conv_reid_run011", conv_run, ((1, 128, 28,   28), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)),
            ("conv_reid_run012", conv_run, ((1, 128, 28, 28), (256, 128, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)),
            ("conv_reid_run013", conv_run, ((1, 128, 28, 28), (256, 128, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)),
            ("conv_reid_run014", conv_run, ((1, 256, 28, 28), (256, 256, 3, 3), (1, 1, 1, 1), (2, 2), (1, 1), False)),
            ("conv_reid_run015", conv_run, ((1, 256, 14, 14), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)),
            # ("conv_reid_run016", conv_run, ((1, 256, 14,   14), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)),
            ("conv_reid_run017", conv_run, ((1, 256, 14, 14), (512, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)),
            ("conv_reid_run018", conv_run, ((1, 256, 14, 14), (512, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)),
            ("conv_reid_run019", conv_run, ((1, 512, 14, 14), (512, 512, 3, 3), (1, 1, 1, 1), (2, 2), (1, 1), False)),
            ("conv_reid_run020", conv_run, ((1, 512, 7, 7), (512, 512, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)),
            ## ("conv_reid_run021", conv_run, ((1, 512, 7,    7),  (512, 512, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)),
        ]

        self.testarg_level1 = [
            # testflag, opfuncname, testRunArgs,
            # fmap_shape, filter_shape, pad_, stride_, dilation_, use_bias, bypass_l1 , dump_data, Tile

            # deeplabv3
            # cases take long time(>100s) are commented
            ("conv_run_1", conv_run, ((1, 256, 65, 65), (256, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            ("conv_run_2", conv_run, ((1, 160, 3, 3), (960, 160, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            ("conv_run_3", conv_run, ((1, 1024, 33, 33), (1024, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            #("conv_run_4",   conv_run, ((4, 256, 65, 65),    (728, 256, 1, 1),     (0, 0, 0, 0),  (1, 1),  (1, 1), True)),
            ("conv_run_5", conv_run, ((1, 728, 33, 33), (728, 728, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            #("conv_run_6",   conv_run, ((4, 128, 129, 129),  (128, 128, 1, 1),     (0, 0, 0, 0),  (1, 1),  (1, 1), True)),
            #("conv_run_7",   conv_run, ((4, 1024, 33, 33),   (1024, 1024, 1, 1),   (0, 0, 0, 0),  (1, 1),  (1, 1), True)),
            ("conv_run_8", conv_run, ((1, 384, 3, 3), (64, 384, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            #("conv_run_9",   conv_run, ((1, 256, 129, 129),  (256, 256, 1, 1),     (0, 0, 0, 0),  (1, 1),  (1, 1), True)),
            ("conv_run_10", conv_run, ((1, 24, 9, 9), (144, 24, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            ("conv_run_11", conv_run, ((1, 728, 33, 33), (1024, 728, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            #("conv_run_12",  conv_run, ((1, 1536, 33, 33),   (1536, 1536, 1, 1),   (0, 0, 0, 0),  (1, 1),  (1, 1), True)),
            #("conv_run_13",  conv_run, ((4, 256, 65, 65),    (728, 256, 1, 1),     (0, 0, 0, 0),  (2, 2),  (1, 1), True)),
            #("conv_run_14",  conv_run, ((4, 256, 129, 129),  (256, 256, 1, 1),     (0, 0, 0, 0),  (1, 1),  (1, 1), True)),
            #("conv_run_15",  conv_run, ((4, 1280, 33, 33),   (256, 1280, 1, 1),    (0, 0, 0, 0),  (1, 1),  (1, 1), True)),
            ("conv_run_16", conv_run, ((1, 64, 33, 33), (384, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            #("conv_run_18",  conv_run, ((1, 128, 129, 129),  (256, 128, 1, 1),     (0, 0, 0, 0),  (1, 1),  (1, 1), True)),
            ("conv_run_19", conv_run, ((1, 512, 3, 3), (256, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            #("conv_run_20",  conv_run, ((4, 3, 515, 515),    (32, 3, 3, 3),        (0, 0, 0, 0),  (2, 2),  (1, 1), True)),
            #("conv_run_21",  conv_run, ((4, 728, 33, 33),    (1024, 728, 1, 1),    (0, 0, 0, 0),  (1, 1),  (1, 1), True)),
            #("conv_run_22",  conv_run, ((4, 32, 257, 257),   (64, 32, 3, 3),       (0, 0, 0, 0),  (1, 1),  (1, 1), True)),
            ("conv_run_23", conv_run, ((1, 32, 5, 5), (192, 32, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            ("conv_run_24", conv_run, ((1, 32, 17, 17), (16, 32, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            #("conv_run_25",  conv_run, ((4, 728, 33, 33),    (728, 728, 1, 1),     (0, 0, 0, 0),  (1, 1),  (1, 1), True)),
            ("conv_run_26", conv_run, ((1, 320, 3, 3), (256, 320, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            ("conv_run_27", conv_run, ((1, 1280, 33, 33), (256, 1280, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            ("conv_run_28", conv_run, ((1, 576, 3, 3), (160, 576, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            #("conv_run_30",  conv_run, ((1, 256, 129, 129),  (48, 256, 1, 1),      (0, 0, 0, 0),  (1, 1),  (1, 1), True)),
            #("conv_run_32",  conv_run, ((1, 304, 129, 129),  (256, 304, 1, 1),     (0, 0, 0, 0),  (1, 1),  (1, 1), True)),
            #("conv_run_33",  conv_run, ((4, 1024, 33, 33),   (1536, 1024, 1, 1),   (0, 0, 0, 0),  (1, 1),  (1, 1), True)),
            #("conv_run_37",  conv_run, ((1, 1024, 33, 33),   (1536, 1024, 1, 1),   (0, 0, 0, 0),  (1, 1),  (1, 1), True)),
            ("conv_run_38", conv_run, ((1, 304, 9, 9), (256, 304, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            #("conv_run_39",  conv_run, ((4, 1536, 33, 33),   (2048, 1536, 1, 1),   (0, 0, 0, 0),  (1, 1),  (1, 1), True)),
            #("conv_run_40",  conv_run, ((1, 32, 257, 257),   (64, 32, 3, 3),       (1, 1, 1, 1),  (1, 1),  (1, 1), True)),
            #("conv_run_41",  conv_run, ((4, 128, 129, 129),  (256, 128, 1, 1),     (0, 0, 0, 0),  (2, 2),  (1, 1), True)),
            ("conv_run_42", conv_run, ((1, 144, 9, 9), (48, 144, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            ("conv_run_43", conv_run, ((1, 256, 9, 9), (256, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            #("conv_run_44",  conv_run, ((4, 2048, 33, 33),   (256, 2048, 1, 1),    (0, 0, 0, 0),  (1, 1),  (1, 1), True)),
            #("conv_run_45",  conv_run, ((1, 256, 65, 65),    (728, 256, 1, 1),     (0, 0, 0, 0),  (1, 1),  (1, 1), True)),
            ("conv_run_46", conv_run, ((1, 96, 6, 6), (24, 96, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            ("conv_run_47", conv_run, ((1, 128, 129, 129), (256, 128, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), True)),
            #("conv_run_48",  conv_run, ((4, 728, 65, 65),    (728, 728, 1, 1),     (0, 0, 0, 0),  (1, 1),  (1, 1), True)),
            #("conv_run_49",  conv_run, ((1, 128, 129, 129),  (128, 128, 1, 1),     (0, 0, 0, 0),  (1, 1),  (1, 1), True)),
            ("conv_run_50", conv_run, ((1, 728, 65, 65), (728, 728, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            ("conv_run_51", conv_run, ((1, 960, 3, 3), (320, 960, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            ("conv_run_52", conv_run, ((1, 576, 3, 3), (96, 576, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            #("conv_run_54",  conv_run, ((4, 256, 129, 129),  (21, 256, 1, 1),      (0, 0, 0, 0),  (1, 1),  (1, 1), True)),
            ("conv_run_55", conv_run, ((1, 96, 3, 3), (576, 96, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            ("conv_run_56", conv_run, ((1, 192, 3, 3), (64, 192, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            ("conv_run_57", conv_run, ((1, 2048, 33, 33), (256, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            #("conv_run_58",  conv_run, ((4, 256, 129, 129),  (48, 256, 1, 1),      (0, 0, 0, 0),  (1, 1),  (1, 1), True)),
            ("conv_run_59", conv_run, ((1, 144, 9, 9), (24, 144, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            #("conv_run_61",  conv_run, ((4, 128, 129, 129),  (256, 128, 1, 1),     (0, 0, 0, 0),  (1, 1),  (1, 1), True)),
            ("conv_run_62", conv_run, ((1, 384, 3, 3), (96, 384, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            #("conv_run_63",  conv_run, ((4, 1536, 33, 33),   (1536, 1536, 1, 1),   (0, 0, 0, 0),  (1, 1),  (1, 1), True)),
            #("conv_run_64",  conv_run, ((4, 256, 65, 65),    (256, 256, 1, 1),     (0, 0, 0, 0),  (1, 1),  (1, 1), True)),
            ("conv_run_65", conv_run, ((1, 960, 3, 3), (160, 960, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            ("conv_run_66", conv_run, ((1, 192, 5, 5), (32, 192, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            #("conv_run_67",  conv_run, ((4, 304, 129, 129),  (256, 304, 1, 1),     (0, 0, 0, 0),  (1, 1),  (1, 1), True)),
            #("conv_run_68",  conv_run, ((1, 1536, 33, 33),   (2048, 1536, 1, 1),   (0, 0, 0, 0),  (1, 1),  (1, 1), True)),
            #("conv_run_69",  conv_run, ((1, 3, 515, 515),    (32, 3, 3, 3),        (0, 0, 0, 0),  (2, 2),  (1, 1), True)),
            ("conv_run_71", conv_run, ((1, 256, 9, 9), (3, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            ("conv_run_72", conv_run, ((1, 256, 129, 129), (21, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            ("conv_run_74", conv_run, ((1, 144, 5, 5), (32, 144, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True)),
            ("conv_run_75", conv_run, ((1, 256, 65, 65), (728, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), True)),
            ("conv_run_76", conv_run, ((1, 3, 33, 33), (32, 3, 3, 3), (0, 0, 0, 0), (2, 2), (1, 1), True)),
        ]
        self.testlenet_rpc_cloud = [
            # testflag,opfuncname,testRunArgs, dimArgs
            #("conv_run001", conv_run,((1  , 1024 , 14  , 14  ), (2048, 1024 , 1  , 1  ), (0 , 0 , 0 , 0 ), ( 2 , 2 ), (1, 1), True), [14, 2048, 64, 96, 128, 14]),
            ("conv_run_1", conv_run, ((1, 3, 32, 32), (6, 3, 3, 3), (0, 0, 0, 0), (2, 2), (1, 1), True)),
            ("conv_run_2", conv_run, ((1, 6, 15, 15), (16, 6, 3, 3), (0, 0, 0, 0), (2, 2), (1, 1), True)),
            ("conv_run_3", conv_run, ((32, 128, 28, 28), (512, 128, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),

            ("test_resnet50_conv_024", conv_run, ((32, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
            ("test_resnet50_conv_025", conv_run, ((32, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
            ("test_resnet50_conv_026", conv_run, ((32, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
            ("test_resnet50_conv_027", conv_run,  ((32, 128, 56, 56), (128, 128, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1), False)),
            ("test_resnet50_conv_028", conv_run, ((32, 256, 28, 28), (256, 256, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1), False)),
            ("test_resnet50_conv_029", conv_run, ((32, 512, 14, 14), (512, 512, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1), False)),
            ("test_resnet50_conv_030", conv_run, ((32, 3, 224, 224), (64, 3, 7, 7), (2, 3, 2, 3), (2, 2), (1, 1), False))

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
        self.common_run(self.testarg, is_conv=True)

    @pytest.mark.level1
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_level1(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_level1, is_conv=True)

    @pytest.mark.rpc_cloud
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_rpc_cloud(self):
        """
        run case.#
        :return:
        """
        # self.common_run(self.testarg_rpc_cloud)
        self.common_run(self.testlenet_rpc_cloud)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return


if __name__ == "__main__":
    t = TestCase()
    t.setup()
    t.test_run()
