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
import json
import os
import sys

sys.path.append(os.getcwd())

from base_all_run import BaseCaseRun


class TestBert001(BaseCaseRun):
    def setup(self):
        """
        testcase preparcondition
        :return:
        """
        case_name = "test_bert_all_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        if not super(TestBert001, self).setup():
            return False

        self.test_args = [
            # TILE OP
            # float - int32:[64, 128, 1] - [3] = float:[64, 128, 1024]
            ("tile_001", "tile_run", ((64, 128, 1), "float32", (1, 1, 1024))),
            # float - int32:[1280, 1] - [2] = float:[1280, 1024]
            ("tile_002", "tile_run", ((1280, 1), "float32", (1, 1024))),
            # float - int32:[8192, 1] - [2] = float:[8192, 1024]
            ("tile_003", "tile_run", ((8192, 1), "float32", (1, 1024))),
            # float - int32:[1280, 1] - [2] = float:[1280, 30522]
            ("tile_004", "tile_run", ((1280, 1), "float32", (1, 30522))),
            # float - int32:[1] - [1] = float:[1280]
            ("tile_005", "tile_run", ((1,), "float32", (1280,))),
            # float - int32:[1280, 1] - [2] = float:[1280, 21128]
            ("tile_006", "tile_run", ((1280, 1), "float32", (1, 21128))),
            # float - int32:[1280, 1] - [2] = float:[1280, 768]
            ("tile_007", "tile_run", ((1280, 1), "float32", (1, 768))),
            # float - int32:[64, 128, 1] - [3] = float:[64, 128, 768]
            ("tile_008", "tile_run", ((64, 128, 1), "float32", (1, 1, 768))),
            # float - int32:[8192, 1] - [2] = float:[8192, 768]
            ("tile_009", "tile_run", ((8192, 1), "float32", (1, 768))),
            # int32 - int32:[128] - [1] = int32:[16384]
            ("tile_010", "tile_run", ((128,), "int32", (128,))),

            # float - int32:[1] - [1] = float:[20]
            ("tile_011", "tile_run", ((1,), "float32", (20,))),
            # float - int32:[20, 1] - [2] = float:[20, 2]
            ("tile_012", "tile_run", ((20, 1), "float32", (1, 2))),
            # float - int32:[20, 1] - [2] = float:[20, 32000]
            ("tile_013", "tile_run", ((20, 1), "float32", (1, 32000))),

            # transpose op  transpose
            # float - int32:[64, 16, 128, 64] - [4] = float:[64, 128, 16, 64]
            ("transpose_001", "transpose_run", ((64, 16, 128, 64), (0, 2, 1, 3), "float32")),
            # float - int32:[8, 16, 128, 64] - [4] = float:[8, 128, 16, 64]
            ("transpose_002", "transpose_run", ((8, 16, 128, 64), (0, 2, 1, 3), "float32")),
            # float - int32:[8, 128, 16, 64] - [4] = float:[8, 16, 128, 64]
            ("transpose_003", "transpose_run", ((8, 128, 16, 64), (0, 2, 1, 3), "float32")),
            # float - int32:[64, 128, 16, 64] - [4] = float:[64, 16, 128, 64]
            ("transpose_004", "transpose_run", ((64, 128, 16, 64), (0, 2, 1, 3), "float32")),
            # half - int32:[64, 12, 128, 64] - [4] = half:[64, 128, 12, 64]
            ("transpose_005", "transpose_run", ((64, 12, 128, 64), (0, 2, 1, 3), "float16")),
            # half - int32:[64, 12, 128, 128] - [4] = half:[128, 64, 12, 128]
            ("transpose_006", "transpose_run", ((64, 12, 128, 128), (2, 0, 1, 3), "float16")),
            # half - int32:[128, 64, 12, 64] - [4] = half:[64, 12, 128, 64]
            ("transpose_007", "transpose_run", ((128, 64, 12, 64), (1, 2, 0, 3), "float16")),
            # half - int32:[128, 64, 12, 128] - [4] = half:[64, 12, 128, 128]
            ("transpose_008", "transpose_run", ((128, 64, 12, 128), (1, 2, 0, 3), "float16")),
            # half - int32:[64, 128, 12, 64] - [4] = half:[64, 12, 128, 64]
            ("transpose_009", "transpose_run", ((64, 128, 12, 64), (0, 2, 1, 3), "float16")),
            # half - int32:[64, 12, 128, 64] - [4] = half:[128, 64, 12, 64]
            ("transpose_010", "transpose_run", ((64, 12, 128, 64), (2, 0, 1, 3), "float16")),

            # float - int32:[1, 128, 12, 64] - [4] = float:[1, 12, 128, 64]
            ("transpose_011", "transpose_run", ((1, 128, 12, 64), (0, 2, 1, 3), "float32")),
            # float - int32:[1, 12, 128, 64] - [4] = float:[1, 128, 12, 64]
            ("transpose_012", "transpose_run", ((1, 12, 128, 64), (0, 2, 1, 3), "float32")),
            # float - int32:[20, 768] - [2] = float:[20, 768]
            ("transpose_013", "transpose_run", ((20, 768), (0, 1), "float32")),
            # float - int32:[128, 768] - [2] = float:[128, 768]
            ("transpose_014", "transpose_run", ((128, 768), (0, 1), "float32")),
            # float - int32:[16, 768] - [2] = float:[16, 768]
            ("transpose_015", "transpose_run", ((16, 768), (0, 1), "float32")),
            # float - int32:[32000, 768] - [2] = float:[32000, 768]
            ("transpose_016", "transpose_run", ((32000, 768), (0, 1), "float32")),

            # UnsortedSegmentSum OP
            # float - int32 - int32:[38714, 1024] - [38714] - [] = float:[30522, 1024]
            ("unsortedsegmentsum_001", "unsortedsegmentsum_run", ([38714, 1024], [38714], 30522, "float32")),
            # float - int32 - int32:[1280, 1024] - [1280] - [] = float:[8192, 1024]
            ("unsortedsegmentsum_002", "unsortedsegmentsum_run", ([1280, 1024], [1280], 8192, "float32")),
            # float - int32 - int32:[1280, 768] - [1280] - [] = float:[8192, 768]
            ("unsortedsegmentsum_003", "unsortedsegmentsum_run", ([1280, 768], [1280], 8192, "float32")),

            # float - int32 - int32:[20, 768] - [20] - [128] = float:[128, 768]
            ("unsortedsegmentsum_004", "unsortedsegmentsum_run", ([20, 768], [20], 128, "float32")),
            # float - int32 - int32:[128, 768] - [128] - [16] = float:[16, 768]
            ("unsortedsegmentsum_005", "unsortedsegmentsum_run", ([128, 768], [128], 16, "float32")),
            # float - int32 - int32:[128, 768] - [128] - [32000] = float:[32000, 768]
            ("unsortedsegmentsum_006", "unsortedsegmentsum_run", ([128, 768], [128], 32000, "float32")),

            # gelu OP ###
            #  float16:[64 * 128, 4096] = float16:[64 * 128, 4096]
            ("gelu_001_input_8192_4096", "gelu_run", ((64 * 128, 4096), "float32")),
            # float16:[64 * 20, 1024] = float:[64 * 20, 1024]
            ("gelu_002_input_1280_1024", "gelu_run", ((1280, 1024), "float32")),

            # float32:[128, 3072] = float:[128, 3072]
            ("gelu_003_input_128_3072", "gelu_run", ((128, 3072), "float32")),
            # float32:[20, 768] = float:[20, 768]
            ("gelu_004_input_20_768", "gelu_run", ((20, 768), "float32")),

            ## gelu_grad OP ###
            #  float16:[64 * 128, 4096] = float16:[64 * 128, 4096]
            ("gelu_grad_001_input_8192_4096", "gelu_grad_run", ((64 * 128, 4096), "float32")),
            # float16:[64 * 20, 1024] = float:[64 * 20, 1024]
            ("gelu_grad_002_input_8192_1024", "gelu_grad_run", ((64 * 20, 1024), "float32")),

            # float32:[128, 3072] = float:[128, 3072]
            ("gelu_grad_003_input_128_3072", "gelu_grad_run", ((128, 3072), "float32")),
            # float32:[20, 768] = float:[20, 768]
            ("gelu_grad_004_input_20_768", "gelu_grad_run", ((20, 768), "float32")),

            # LayerNorm OP ###
            # float16:[64 * 128, 1024] = float16:[64 * 128, 1024]
            ("fused_layernorm_001_8192_1024", "fused_layernorm_run", ((64 * 128, 1024), 1, -1, 'float16')),
            # float16:[64 * 20, 1024] = float:[64 * 20, 1024]
            ("fused_layernorm_002_1280_1024", "fused_layernorm_run", ((64 * 20, 1024), 1, -1, 'float16')),

            # float32:[1, 128, 768] = float:[1, 128, 768]
            ("fused_layernorm_003_1_128_768", "fused_layernorm_run", ((1, 128, 768), -1, -1, 'float32')),
            # float32:[20, 768] = float:[20, 768]
            ("fused_layernorm_004_20_768", "fused_layernorm_run", ((20, 768), 1, -1, 'float32')),
            # float32:[128, 768] = float:[128, 768]
            ("fused_layernorm_005_128_768", "fused_layernorm_run", ((128, 768), 1, -1, 'float32')),

            # LayerNormGrad ###
            # float16:[64 * 128, 1024] = float16:[64 * 128, 1024]
            ("fused_layer_norm_grad_01", "fused_layer_norm_grad_run", ((8192, 1024), -1, -1, "float16")),

            # float32:[1, 128, 768] = float:[1, 128, 768]
            ("fused_layer_norm_grad_02", "fused_layer_norm_grad_run", ((1, 128, 768), -1, -1, 'float32')),
            # float32:[20, 768] = float:[20, 768]
            ("fused_layer_norm_grad_03", "fused_layer_norm_grad_run", ((20, 768), -1, -1, 'float32')),
            # float32:[128, 768] = float:[128, 768]
            ("fused_layer_norm_grad_04", "fused_layer_norm_grad_run", ((128, 768), -1, -1, 'float32')),

            # float32:[128, 1024] = float:[128, 1024]
            ("fused_layer_norm_grad_05", "fused_layer_norm_grad_run", ((128, 1024), -1, -1, 'float32')),
            # float32:[1, 128, 1024] = float:[1, 128, 1024]
            ("fused_layer_norm_grad_06", "fused_layer_norm_grad_run", ((1, 128, 1024), -1, -1, 'float32')),

            # FusedMinimumOrMaximumGrad ###
            ("fused_min_or_max_grad_01", "fused_minimum_or_maximum_grad_run",
             ((64,), (64,), (64,), True, True, "GE", "float16", "cce_min_max_grad_fp16")),
            ("fused_min_or_max_grad_02", "fused_minimum_or_maximum_grad_run",
             ((64,), (64,), (64,), True, True, "LE", "float16", "cce_min_max_grad_fp16")),
            ("fused_min_or_max_grad_03", "fused_minimum_or_maximum_grad_run",
             ((64,), (64,), (64,), False, True, "GE", "float16", "cce_min_max_grad_fp16")),
            ("fused_min_or_max_grad_04", "fused_minimum_or_maximum_grad_run",
             ((64, 64), (64, 64), (64, 64), True, False, "GE", "float16", "cce_min_max_grad_fp16")),
            ("fused_min_or_max_grad_05", "fused_minimum_or_maximum_grad_run",
             ((128,), (128,), (128,), True, False, "LE", "float16", "cce_min_max_grad_fp16")),
            ("fused_min_or_max_grad_06", "fused_minimum_or_maximum_grad_run",
             ((128,), (128,), (128,), False, True, "LE", "float16", "cce_min_max_grad_fp16")),

            #  dropout  OP ###
            # float16:[64, 128, 1024] = float:[64, 128, 1024]
            ("dropout_001", "dropout_run", ((64, 128, 1024), 1.0, "float16", "cce_dropout_do_mask")),
            # # float16:[64, 16, 128, 128] = float:[64, 16, 128, 128]
            ("dropout_002", "dropout_run", ((64, 16, 128, 128), 1.0, "float16", "cce_dropout_do_mask")),
            # # float16:[64 * 128, 1024] = float:[64 * 128, 1024]
            ("dropout_003", "dropout_run", ((64 * 128, 1024), 1.0, "float16", "cce_dropout_do_mask")),

            # reduce mean ###
            # float - int32:[64, 128, 1024] - [1]
            ("mean_001", "mean_run", ((64, 128, 1024), "float32", (2,), False, "cce_mean_64_128_1024_fp32")),
            # float - int32:[8] - [1]
            ("mean_002", "mean_run", ((8,), "float32", (0,), True, "cce_mean_8_fp32")),
            # float - int32:[1024, 1024] - [1]
            ("mean_003", "mean_run", ((1024, 1024), "float32", (1,), False, "cce_mean_1024_1024_fp32")),
            # float - int32:[64] - [1]
            ("mean_004", "mean_run", ((64,), "float32", (0,), False, "cce_mean_64_fp32")),
            # float - int32:[8, 128, 1024] - [1]
            ("mean_005", "mean_run", ((8, 128, 1024), "float32", (2,), True, "cce_mean_64_128_1024_fp32")),
            # float - int32:[1280, 1024] - [1]
            ("mean_006", "mean_run", ((1280, 1024), "float32", (1,), False, "cce_mean_1280_1024_fp32")),
            # float - int32:[8192, 1024] - [1]
            ("mean_007", "mean_run", ((8192, 1024), "float32", (1,), False, "cce_mean_8192_1024_fp32")),
            # float - int32:[160, 1024] - [1]
            ("mean_008", "mean_run", ((160, 1024), "float32", (0,), True, "cce_mean_160_1024_fp32")),
            # float - int32:[64, 128, 768] - [1] = float:[64, 128, 1]
            ("mean_009", "mean_run", ((64, 128, 768), "float32", (0,), True, "cce_mean_64_128_1024_fp32")),
            # float - int32:[1280, 768] - [1] = float:[1280, 1]
            ("mean_010", "mean_run", ((1280, 768), "float32", (0,), True, "cce_mean_64_128_1024_fp32")),
            # float - int32:[8192, 768] - [1] = float:[8192, 1]
            ("mean_011", "mean_run", ((8192, 768), "float32", (1,), True, "cce_mean_64_128_1024_fp32")),
            # float - int32:[64] - [1] = float:[]
            ("mean_012", "mean_run", ((64,), "float32", (0,), False, "cce_mean_64_128_1024_fp32")),

            # LessEqual OP ###
            # float - float:[64, 128, 768] - [] = bool:[64, 128, 768]
            ("less_equal_001", "less_equal_run", (((64, 128, 768), (64, 128, 768)), "float32", "less_equal_run")),
            # float - float:[128, 128, 64] - [] = bool:[128, 128, 64]
            ("less_equal_002", "less_equal_run", (((128, 128, 64), (128, 128, 64)), "float32", "less_equal_run")),
            # int32 - int32:[] - []
            ("less_equal_003", "less_equal_run", (((1,), (1,)), "int32", "less_equal_run")),

            # addn op
            # ("20_addn_input_1280_1024_2_dim_2_01", "addn_run", ((1280, 1024), "float16", 2)),
            # float - float:[1280, 1024] - [1280, 1024] = float:[1280, 1024]
            ("addn_001_input_fp32", "addn_run", ((1280, 1024), "float32", 2)),
            # float - float:[64, 128, 1024] - [64, 128, 1024] = float:[64, 128, 1024]
            ("addn_002_input_fp32", "addn_run", ((64, 128, 1024), "float32", 2)),
            # float - float:[8, 128, 1024] - [8, 128, 1024] = float:[8, 128, 1024]
            ("addn_003_input_fp32", "addn_run", ((8, 128, 1024), "float32", 2)),
            # float - float - float:[8192, 1024] - [8192, 1024] - [8192, 1024] = float:[8192, 1024]
            ("addn_004_input_fp32", "addn_run", ((8192, 1024), "float32", 3)),
            # float - float:[8192, 1024] - [8192, 1024] = float:[8192, 1024]
            ("addn_005_input_fp32", "addn_run", ((8192, 1024), "float32", 2)),
            # float - float - float - float:[8192, 1024] - [8192, 1024] - [8192, 1024] - [8192, 1024] = float:[8192, 1024]
            ("addn_006_input_fp32", "addn_run", ((8192, 1024), "float32", 4)),
            # float - float - float:[64, 128, 1024] - [64, 128, 1024] - [64, 128, 1024] = float:[64, 128, 1024]
            ("addn_007_input_fp32", "addn_run", ((64, 128, 1024), "float32", 3)),
            # float - float - float:[8192, 4096] - [8192, 4096] - [8192, 4096] = float:[8192, 4096]
            ("addn_008_input_fp32", "addn_run", ((8192, 4096), "float32", 3)),
            # float - float - float:[1280, 1024] - [1280, 1024] - [1280, 1024] = float:[1280, 1024]
            ("addn_009_input_fp32", "addn_run", ((1280, 1024), "float32", 3)),
            # half - half - half - half:[8192, 768] - [8192, 768] - [8192, 768] - [8192, 768] = half:[8192, 768]
            ("addn_010_input_fp32", "addn_run", ((8192, 768), "float16", 4)),
            # half - half:[8192, 768] - [8192, 768] = half:[8192, 768]
            ("addn_011_input_fp32", "addn_run", ((8192, 768), "float16", 2)),
            # float - float:[64, 128, 768] - [64, 128, 768] = float:[64, 128, 768]
            ("addn_012_input_fp32", "addn_run", ((8192, 768), "float32", 2)),
            # float - float:[21128, 768] - [21128, 768] = float:[21128, 768]
            ("addn_013_input_fp32", "addn_run", ((21128, 768), "float32", 2)),
            # half - half - half:[8192, 3072] - [8192, 3072] - [8192, 3072] = half:[8192, 3072]
            ("addn_014_input_fp32", "addn_run", ((8192, 3072), "float16", 3)),
            # half - half:[64, 12, 128, 64] - [64, 12, 128, 64] = half:[64, 12, 128, 64]
            ("addn_015_input_fp32", "addn_run", ((64, 12, 128, 64), "float16", 2)),
            # float - float:[1280, 768] - [1280, 768] = float:[1280, 768]
            ("addn_016_input_fp32", "addn_run", ((1280, 768), "float32", 2)),
            # float - float:[8192, 768] - [8192, 768] = float:[8192, 768]
            ("addn_017_input_fp32", "addn_run", ((8192, 768), "float32", 2)),
            # float - float - float:[64, 128, 768] - [64, 128, 768] - [64, 128, 768] = float:[64, 128, 768]
            ("addn_018_input_fp32", "addn_run", ((64, 128, 768), "float32", 3)),
            # half - half:[64, 12, 128, 128] - [64, 12, 128, 128] = half:[64, 12, 128, 128]
            ("addn_019_input_fp32", "addn_run", ((8192, 768), "float16", 2)),
            # float - float - float:[1280, 768] - [1280, 768] - [1280, 768] = float:[1280, 768]
            ("addn_020_input_fp32", "addn_run", ((1280, 768), "float32", 3)),

            # BactchMatmul OP ###
            # float - float:[64, 16, 128, 64] - [64, 16, 128, 64] = float:[64, 16, 128, 128]
            ("batch_matmul_4D_001", "batchmatmul_run",
             ((64, 16), 128, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[8, 16, 128, 64] - [8, 16, 128, 64] = float:[8, 16, 128, 128]
            ("batch_matmul_4D_002", "batchmatmul_run",
             ((8, 16), 128, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[64, 16, 128, 128] - [64, 16, 128, 64] = float:[64, 16, 128, 64]
            ("batch_matmul_4D_003", "batchmatmul_run",
             ((64, 16), 128, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[8, 16, 128, 128] - [8, 16, 128, 64] = float:[8, 16, 128, 64]
            ("batch_matmul_4D_004", "batchmatmul_run",
             ((8, 16), 128, 64, 128, (), "float32", False, False, "batch_matmul_output")),

            # half - half:[128, 768, 128] - [128, 768, 64] = half:[128, 128, 64]
            ("batch_matmul_3D_005", "batchmatmul_run",
             ((128,), 128, 64, 768, (), "float16", True, False, "batch_matmul_output")),
            # half - half:[64, 12, 128, 128] - [64, 12, 128, 64] = half:[64, 12, 128, 64]
            ("batch_matmul_4D_006", "batchmatmul_run",
             ((64, 12), 128, 64, 128, (), "float16", False, False, "batch_matmul_output")),

            # half - half:[128, 768, 64] - [128, 128, 64] = half:[128, 768, 128]
            ("batch_matmul_3D_007", "batchmatmul_run",
             ((128,), 768, 128, 64, (), "float16", False, True, "batch_matmul_output")),
            # half - half:[64, 12, 128, 64] - [64, 12, 128, 64] = half:[64, 12, 128, 128]
            ("batch_matmul_4D_008", "batchmatmul_run",
             ((64, 12), 128, 128, 64, (), "float16", False, True, "batch_matmul_output")),
            # half - half:[128, 768, 128] - [128, 128, 64] = half:[128, 768, 64]
            ("batch_matmul_4D_009", "batchmatmul_run",
             ((128,), 768, 64, 128, (), "float16", False, False, "batch_matmul_output")),

            # LogSoftMax OP ###
            ("logsoftmax_01_fp32", "logsoftmax_run", ((64, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[64, 2] = float:[64, 2]
            ("logsoftmax_001_fp32", "logsoftmax_run", ((64, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[160, 30522] = float:[160, 30522]
            ("logsoftmax_002_fp32", "logsoftmax_run", ((160, 30522), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[8, 2] = float:[8, 2]
            ("logsoftmax_003_fp32", "logsoftmax_run", ((8, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[1280, 30522] = float:[1280, 30522]
            ("logsoftmax_004_fp32", "logsoftmax_run", ((1280, 30522), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[1280, 21128] = float:[1280, 21128]
            ("logsoftmax_005_fp32", "logsoftmax_run", ((1280, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[64, 2] = float:[64, 2]
            ("logsoftmax_006_fp32", "logsoftmax_run", ((64, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[1, 2] = float:[1, 2]
            ("logsoftmax_007_fp32", "logsoftmax_run", ((1, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[20, 32000] = float:[20, 32000]
            ("logsoftmax_008_fp32", "logsoftmax_run", ((20, 32000), "float32", -1, "cce_logsoftmax_fp32")),

            # LogSoftMaxDrad OP ###
            # float:[64, 2] = float:[64, 2]
            ("logsoftmax_grad_001", "logsoftmax_grad_run", ((64, 2), "float32", -1, "cce_logsoftmax_fp16")),
            # float:[160, 30522] = float:[160, 30522]
            ("logsoftmax_grad_002", "logsoftmax_grad_run", ((160, 30522), "float32", -1, "cce_logsoftmax_fp16")),
            # float:[8, 2] = float:[8, 2]
            ("logsoftmax_grad_003", "logsoftmax_grad_run", ((8, 2), "float32", -1, "cce_logsoftmax_fp16")),
            # float:[1280, 30522] = float:[1280, 30522]
            ("logsoftmax_grad_004", "logsoftmax_grad_run", ((1280, 30522), "float32", -1, "cce_logsoftmax_fp16")),

            # float:[1, 2] = float:[1, 2]
            ("logsoftmax_grad_005", "logsoftmax_grad_run", ((1, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[20, 32000] = float:[20, 32000]
            ("logsoftmax_grad_006", "logsoftmax_grad_run", ((20, 32000), "float32", -1, "cce_logsoftmax_fp32")),

            # matmul op
            # # float - float:[160, 1024] - [1024, 1024] = float:[160, 1024]
            ("matmul_0001", "batchmatmul_run",
             ((), 160, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[8192, 4096] - [4096, 1024] = float:[8192, 1024]
            ("matmul_0002", "batchmatmul_run",
             ((), 8192, 1024, 4096, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[8192, 1024] - [8192, 1024] = float:[1024, 1024]
            ("matmul_0003", "batchmatmul_run",
             ((), 1024, 1024, 8192, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[64, 1024] - [2, 1024] = float:[64, 2]
            ("matmul_0004", "batchmatmul_run", ((), 64, 2, 1024, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[1024, 2] - [2, 1024] = float:[1024, 1024]
            ("matmul_0005", "batchmatmul_run", ((), 1024, 1024, 2, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[1024, 4096] - [4096, 1024] = float:[1024, 1024]
            ("matmul_0006", "batchmatmul_run",
             ((), 1024, 1024, 4096, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[8192, 2] - [8192, 1024] = float:[2, 1024]
            ("matmul_0007", "batchmatmul_run", ((), 2, 1024, 8192, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[1280, 30522] - [1280, 1024] = float:[30522, 1024]
            ("matmul_0008", "batchmatmul_run",
             ((), 30522, 1024, 1280, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[8, 1024] - [2, 1024] = float:[8, 2]
            ("matmul_0009", "batchmatmul_run", ((), 8, 2, 1024, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[8192, 1024] - [1024, 1024] = float:[8192, 1024]
            ("matmul_0010", "batchmatmul_run",
             ((), 8192, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[1280, 1024] - [1280, 1024] = float:[1024, 1024]
            ("matmul_0011", "batchmatmul_run",
             ((), 1280, 1280, 1024, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[1024, 1024] - [1024, 4096] = float:[1024, 4096]
            ("matmul_0012", "batchmatmul_run",
             ((), 1024, 4096, 1024, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[8192, 2] - [2, 1024] = float:[8192, 1024]
            ("matmul_0013", "batchmatmul_run", ((), 8192, 1024, 2, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[8192, 1024] - [8192, 4096] = float:[1024, 4096]
            ("matmul_0014", "batchmatmul_run",
             ((), 1024, 4096, 8192, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[64, 1024] - [1024, 1024] = float:[64, 1024]
            (
                "matmul_0015", "batchmatmul_run",
                ((), 64, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[64, 2] - [64, 1024] = float:[2, 1024]
            ("matmul_0016", "batchmatmul_run", ((), 2, 1024, 64, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[160, 1024] - [30522, 1024] = float:[160, 30522]
            ("matmul_0017", "batchmatmul_run",
             ((), 160, 30522, 1024, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[64, 1024] - [64, 1024] = float:[1024, 1024]
            ("matmul_0018", "batchmatmul_run", ((), 1024, 1024, 64, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[8192, 4096] - [8192, 1024] = float:[4096, 1024]
            ("matmul_0019", "batchmatmul_run",
             ((), 4096, 1024, 8192, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[1280, 30522] - [30522, 1024] = float:[1280, 1024]
            ("matmul_0020", "batchmatmul_run",
             ((), 1280, 1024, 30522, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[8192, 4096] - [1024, 4096] = float:[8192, 1024]
            ("matmul_0021", "batchmatmul_run",
             ((), 8192, 1024, 4096, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[8, 1024] - [1024, 1024] = float:[8, 1024]
            ("matmul_0022", "batchmatmul_run",
             ((), 160, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[1024, 1024] - [1024, 1024] = float:[1024, 1024]
            ("matmul_0023", "batchmatmul_run",
             ((), 1024, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[64, 2] - [2, 1024] = float:[64, 1024]
            ("matmul_0024", "batchmatmul_run", ((), 64, 1024, 2, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[1280, 1024] - [30522, 1024] = float:[1280, 30522]
            ("matmul_0025", "batchmatmul_run",
             ((), 1280, 30522, 1024, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[1280, 1024] - [1024, 1024] = float:[1280, 1024]
            ("matmul_0026", "batchmatmul_run",
             ((), 1280, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[8192, 1024] - [1024, 4096] = float:[8192, 4096]
            ("matmul_0027", "batchmatmul_run",
             ((), 8192, 4096, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[8192, 1024] - [4096, 1024] = float:[8192, 4096]
            ("matmul_0028", "batchmatmul_run",
             ((), 8192, 4096, 1024, (), "float32", False, True, "batch_matmul_output")),
            # half - half:[8192, 768] - [8192, 768] = half:[768, 768]
            ("matmul_0029", "batchmatmul_run", ((), 768, 768, 8192, (), "float16", True, False, "batch_matmul_output")),
            # float - float:[16384, 33] - [33, 64] = float:[16384, 64]
            ("matmul_0030", "batchmatmul_run", ((), 16384, 64, 33, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[1280, 21128] - [21128, 768] = float:[1280, 768]
            ("matmul_0031", "batchmatmul_run",
             ((), 1280, 768, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[1280, 768] - [768, 768] = float:[1280, 768]
            (
                "matmul_0032", "batchmatmul_run",
                ((), 1280, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # half - half:[8192, 3072] - [8192, 768] = half:[3072, 768]
            (
                "matmul_0033", "batchmatmul_run",
                ((), 3072, 768, 8192, (), "float16", True, False, "batch_matmul_output")),
            # float - float:[64, 2] - [64, 768] = float:[2, 768]
            ("matmul_0034", "batchmatmul_run", ((), 2, 768, 64, (), "float32", True, False, "batch_matmul_output")),
            # half - half:[8192, 768] - [768, 3072] = half:[8192, 3072]
            ("matmul_0035", "batchmatmul_run",
             ((), 8192, 3072, 768, (), "float16", False, False, "batch_matmul_output")),
            # half - half:[8192, 768] - [8192, 3072] = half:[768, 3072]
            (
                "matmul_0036", "batchmatmul_run",
                ((), 768, 3072, 8192, (), "float16", True, False, "batch_matmul_output")),
            # float - float:[16384, 33] - [16384, 64] = float:[33, 64]
            ("matmul_0037", "batchmatmul_run", ((), 33, 64, 16384, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[1280, 21128] - [1280, 768] = float:[21128, 768]
            ("matmul_0038", "batchmatmul_run",
             ((), 21128, 768, 1280, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[1280, 768] - [21128, 768] = float:[1280, 21128]
            ("matmul_0039", "batchmatmul_run",
             ((), 160, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[64, 768] - [64, 768] = float:[768, 768]
            ("matmul_0040", "batchmatmul_run", ((), 768, 768, 64, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[64, 768] - [2, 768] = float:[64, 2]
            ("matmul_0041", "batchmatmul_run", ((), 64, 2, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[8192, 21128] - [21128, 768] = float:[8192, 768]
            ("matmul_0042", "batchmatmul_run",
             ((), 8192, 768, 21128, (), "float32", False, False, "batch_matmul_output")),
            # half - half:[8192, 3072] - [768, 3072] = half:[8192, 768]
            (
                "matmul_0043", "batchmatmul_run",
                ((), 8192, 768, 3072, (), "float16", False, True, "batch_matmul_output")),
            # half - half:[8192, 768] - [3072, 768] = half:[8192, 3072]
            (
                "matmul_0044", "batchmatmul_run",
                ((), 8192, 3072, 768, (), "float16", False, True, "batch_matmul_output")),
            # float - float:[8192, 2] - [8192, 768] = float:[2, 768]
            ("matmul_0045", "batchmatmul_run", ((), 2, 768, 8192, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[64, 2] - [2, 768] = float:[64, 768]
            ("matmul_0046", "batchmatmul_run", ((), 64, 768, 2, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[8192, 2] - [2, 768] = float:[8192, 768]
            ("matmul_0047", "batchmatmul_run", ((), 8192, 768, 2, (), "float32", False, False, "batch_matmul_output")),
            # half - half:[8192, 768] - [768, 768] = half:[8192, 768]
            (
                "matmul_0048", "batchmatmul_run",
                ((), 8192, 768, 768, (), "float16", False, False, "batch_matmul_output")),
            # float - float:[64, 768] - [768, 768] = float:[64, 768]
            ("matmul_0049", "batchmatmul_run", ((), 64, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[8192, 21128] - [8192, 768] = float:[21128, 768]
            ("matmul_0050", "batchmatmul_run",
             ((), 21128, 768, 8192, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[1280, 768] - [1280, 768] = float:[768, 768]
            ("matmul_0051", "batchmatmul_run", ((), 768, 1024, 768, (), "float32", True, False, "batch_matmul_output")),
            # half - half:[8192, 3072] - [3072, 768] = half:[8192, 768]
            ("matmul_0052", "batchmatmul_run",
             ((), 8192, 768, 3072, (), "float16", False, False, "batch_matmul_output")),

            # float - float:[1, 768] - [2, 768] = float:[1, 2]
            ("matmul_0053", "batchmatmul_run", ((), 1, 2, 768, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[1, 2] - [1, 768] = float:[2, 768]
            ("matmul_0054", "batchmatmul_run", ((), 2, 768, 1, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[1, 768] - [1, 768] = float:[768, 768]
            ("matmul_0055", "batchmatmul_run", ((), 768, 768, 768, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[20, 768] - [32000, 768] = float:[20, 32000]
            ("matmul_0056", "batchmatmul_run", ((), 20, 32000, 768, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[20, 32000] - [32000, 768] = float:[20, 768]
            (
                "matmul_0057", "batchmatmul_run",
                ((), 20, 768, 32000, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[128, 768] - [128, 3072] = float:[768, 3072]
            ("matmul_0058", "batchmatmul_run", ((), 768, 3072, 128, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[128, 768] - [768, 3072] = float:[128, 3072]
            (
                "matmul_0059", "batchmatmul_run",
                ((), 128, 3072, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[1, 12, 128, 128] - [1, 12, 128, 64] = float:[1, 12, 128, 64]
            ("matmul_0060", "batchmatmul_run",
             ((1, 12), 128, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[128, 768] - [768, 3072] = float:[128, 768]
            ("matmul_0061", "batchmatmul_run", ((), 128, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[20, 32000] - [20, 768] = float:[32000, 768]
            ("matmul_0062", "batchmatmul_run", ((), 32000, 768, 20, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[1, 12, 128, 64] - [1, 12, 128, 64] = float:[1, 12, 128, 128]
            ("matmul_0063", "batchmatmul_run",
             ((1, 12), 128, 128, 64, (), "float32", False, True, "batch_matmul_output")),

            # Neg OP ###
            # float:[64] = float:[64]
            ('neg_001_64_128_1024_fp32', "neg_run", ((64,), 'float32')),
            # float:[8192, 1024] = float:[8192, 1024]
            ('neg_002_64_128_1024_fp32', "neg_run", ((8192, 1024), 'float32')),
            # float:[160] = float:[160]
            ('neg_003_64_128_1024_fp32', "neg_run", ((160,), 'float32')),
            # float:[1280, 1024] = float:[1280, 1024]
            ('neg_004_64_128_1024_fp32', "neg_run", ((1280, 1024), 'float32')),
            # float:[1280] = float:[1280]
            ('neg_005_64_128_1024_fp32', "neg_run", ((1280,), 'float32')),
            # float:[8] = float:[8]
            ('neg_006_64_128_1024_fp32', "neg_run", ((8,), 'float32')),
            # float:[64, 128, 1024] = float:[64, 128, 1024]
            ('neg_007_64_128_1024_fp32', "neg_run", ((64, 128, 1024), 'float32')),
            # half:[8192, 768] = half:[8192, 768]
            ('neg_008_64_128_1024_fp32', "neg_run", ((8192, 768), 'float16')),
            # float:[64] = float:[64]
            ('neg_009_64_128_1024_fp32', "neg_run", ((64,), 'float32')),
            # float:[64, 128, 768] = float:[64, 128, 768]
            ('neg_010_64_128_1024_fp32', "neg_run", ((64, 128, 768), 'float32')),
            # float:[1280, 768] = float:[1280, 768]
            ('neg_011_64_128_1024_fp32', "neg_run", ((1280, 768), 'float32')),
            # float:[1280] = float:[1280]
            ('neg_012_64_128_1024_fp32', "neg_run", ((1280,), 'float32')),
            # float:[20] = float:[20]
            ('neg_013_64_128_1024_fp32', "neg_run", ((20,), 'float32')),

            # onehot OP
            # int32 - int32 - float - float:[160] - [] - [] - [] = float:[160, 30522]
            ("test_one_hot_001", "one_hot_run", ((160,), 30522, "int32", 1, 0, -1)),
            # int32 - int32 - float - float:[8192] - [] - [] - [] = float:[8192, 2]
            ("test_one_hot_002", "one_hot_run", ((8192,), 2, "int32", 1, 0, -1)),
            # int32 - int32 - float - float:[1024] - [] - [] - [] = float:[1024, 2]
            ("test_one_hot_003", "one_hot_run", ((1024,), 2, "int32", 1, 0, -1)),
            # int32 - int32 - float - float:[1280] - [] - [] - [] = float:[1280, 30522]
            ("test_one_hot_004", "one_hot_run", ((1280,), 30522, "int32", 1, 0, -1)),
            # int32 - int32 - float - float:[8] - [] - [] - [] = float:[8, 2]
            ("test_one_hot_005", "one_hot_run", ((8,), 2, "int32", 1, 0, -1)),
            # int32 - int32 - float - float:[64] - [] - [] - [] = float:[64, 2]
            ("test_one_hot_006", "one_hot_run", ((64,), 2, "int32", 1, 0, -1)),
            # int32 - int32 - float - float:[8192] - [] - [] - [] = float:[8192, 21128]
            ("test_one_hot_007", "one_hot_run", ((8192,), 21128, "int32", 1, 0, -1)),
            # int32 - int32 - float - float:[8192] - [] - [] - [] = float:[8192, 2]
            ("test_one_hot_008", "one_hot_run", ((8192,), 2, "int32", 1, 0, -1)),
            # int32 - int32 - float - float:[1280] - [] - [] - [] = float:[1280, 21128]
            ("test_one_hot_009", "one_hot_run", ((1280,), 21128, "int32", 1, 0, -1)),
            # int32 - int32 - float - float:[64] - [] - [] - [] = float:[64, 2]
            ("test_one_hot_010", "one_hot_run", ((64,), 2, "int32", 1, 0, -1)),

            # int32 - int32 - float - float:[20] - [] - [] - [] = float:[20, 2]
            ("test_one_hot_011", "one_hot_run", ((20,), 2, "int32", 1, 0, -1)),
            # int32 - int32 - float - float:[1] - [32000] - [] - [] = float:[32000]
            ("test_one_hot_012", "one_hot_run", ((64,), 32000, "int32", 1, 0, -1)),

            # square OP
            # float:[4096]
            ("square_001", "square_run", ((4096,), "float32", "cce_mod_fp32")),
            # float:[1280, 1024]
            ("square_002", "square_run", ((1280, 1024), "float32", "cce_mod_fp32")),
            # float:[1024, 1024]
            ("square_003", "square_run", ((1024, 1024), "float32", "cce_mod_fp32")),
            # float:[2, 1024]
            ("square_004", "square_run", ((2, 1024), "float32", "cce_mod_fp32")),
            # # float:[4096, 1024]
            ("square_005", "square_run", ((4096, 1024), "float32", "cce_mod_fp32")),
            # # float:[8192, 4096]
            ("square_006", "square_run", ((8192, 4096), "float32", "cce_mod_fp32")),
            # float:[1024]
            ("square_007", "square_run", ((1024,), "float32", "cce_mod_fp32")),
            # float:[1024, 4096]
            ("square_008", "square_run", ((1024, 4096), "float32", "cce_mod_fp32")),
            # float:[30522]
            ("square_009", "square_run", ((30522,), "float32", "cce_mod_fp32")),
            # float:[30522, 1024]
            ("square_010", "square_run", ((30522, 1024), "float32", "cce_mod_fp32")),
            # float:[2]
            ("square_011", "square_run", ((2,), "float32", "cce_mod_fp32")),
            # float:[512, 1024]
            ("square_012", "square_run", ((512, 1024), "float32", "cce_mod_fp32")),
            # float:[768, 3072] = float:[768, 3072]
            ("square_013", "square_run", ((768, 3072), "float32", "cce_mod_fp32")),
            # half:[8192, 3072] = half:[8192, 3072]
            ("square_014", "square_run", ((8192, 3072), "float16", "cce_mod_fp32")),
            # float:[1280, 768] = float:[1280, 768]
            ("square_015", "square_run", ((1280, 768), "float32", "cce_mod_fp32")),
            # float:[768, 768] = float:[768, 768]
            ("square_016", "square_run", ((768, 768), "float32", "cce_mod_fp32")),
            # float:[3072] = float:[3072]
            ("square_017", "square_run", ((3072,), "float32", "cce_mod_fp32")),
            # float:[3072, 768] = float:[3072, 768]
            ("square_018", "square_run", ((3072, 768), "float32", "cce_mod_fp32")),
            # float:[21128, 768] = float:[21128, 768]
            ("square_019", "square_run", ((21128, 768), "float32", "cce_mod_fp32")),
            # float:[21128] = float:[21128]
            ("square_020", "square_run", ((21128,), "float32", "cce_mod_fp32")),
            # float:[2] = float:[2]
            ("square_021", "square_run", ((2,), "float32", "cce_mod_fp32")),
            # float:[33, 64] = float:[33, 64]
            ("square_022", "square_run", ((33, 64), "float32", "cce_mod_fp32")),
            # float:[768] = float:[768]
            ("square_023", "square_run", ((768,), "float32", "cce_mod_fp32")),
            # float:[2, 768] = float:[2, 768]
            ("square_024", "square_run", ((2, 768), "float32", "cce_mod_fp32")),

            # sub OP
            # float - float:[30522] - [30522] = float:[30522]
            ("001_sub_30522_30522_fp32", "sub_run", [(30522,), (30522,), "float32"]),
            # float - float:[1024] - [1024, 1024] = float:[1024, 1024]
            ("002_sub_1024_1024_1024_fp32", "sub_run", [(1024,), (1024, 1024), "float32"]),
            # float - float:[1024, 1024] - [1024, 1024] = float:[1024, 1024]
            ("003_sub_1024_1024_1024_1024_fp32", "sub_run", [(1024, 1024), (1024, 1024), "float32"]),
            # float - float:[1024, 4096] - [1024, 4096] = float:[1024, 4096]
            ("004_sub_1024_4096_1024_4096_fp32", "sub_run", [(1024, 4096), (1024, 4096), "float32"]),
            # # float - float:[64, 128, 1024] - [64, 128, 1] = float:[64, 128, 1024]
            ("005_sub_64_128_1024_64_128_1024_fp32", "sub_run", [(64, 128, 1024), (64, 128, 1), "float32"]),
            # float - float:[4096] - [4096] = float:[4096]
            ("006_sub_4096_4096_fp32", "sub_run", [(4096,), (4096,), "float32"]),
            # float - float:[30522, 1024] - [30522, 1024] = float:[30522, 1024]
            ("007_sub_30522_1024_30522_1024_fp32", "sub_run", [(30522, 1024), (30522, 1024), "float32"]),
            # float - float:[] - [] = float:[]
            ("008_sub_1_1_fp32", "sub_run", [(1,), (1,), "float32"]),
            # float - float:[1280, 1024] - [1280, 1] = float:[1280, 1024]
            ("009_sub_1280_1024_1280_1_fp32", "sub_run", [(1280, 1024), (1280, 1), "float32"]),
            # float - float:[1024] - [1024] = float:[1024]
            ("010_sub_1024_1024_fp32", "sub_run", [(1024,), (1024,), "float32"]),
            # float - float:[1280, 30522] - [1280, 30522] = float:[1280, 30522]
            ("011_sub_1280_30522_1280_30522_fp32", "sub_run", [(1280, 30522), (1280, 30522), "float32"]),
            # float - float:[64, 2] - [64, 2] = float:[64, 2]
            ("012_sub_64_2_64_2_fp32", "sub_run", [(64, 2), (64, 2), "float32"]),
            # float - float:[1024] - [8, 128, 1024] = float:[8, 128, 1024]
            ("013_sub_1024_8_128_1024_fp32", "sub_run", [(1024,), (8, 128, 1024), "float32"]),
            # float - float:[8192, 1024] - [8192, 1] = float:[8192, 1024]
            ("014_sub_8192_1024_8192_1_fp32", "sub_run", [(8192, 1024), (8192, 1), "float32"]),
            # float - float:[1024] - [64, 128, 1024] = float:[64, 128, 1024]
            ("015_sub_1024_64_128_1024_fp32", "sub_run", [(1024,), (64, 128, 1024), "float32"]),
            # float - float:[2, 1024] - [2, 1024] = float:[2, 1024]
            ("016_sub_2_1024_2_1024_fp32", "sub_run", [(2, 1024), (2, 1024), "float32"]),
            # float - float:[2] - [2] = float:[2]
            ("017_sub_2_2_fp32", "sub_run", [(2,), (2,), "float32"]),
            # int32 - int32:[2] - [2] = int32:[2]
            ("018_sub_2_2_int32", "sub_run", [(2,), (2,), "int32"]),
            # float - float:[1024] - [8192, 1024] = float:[8192, 1024]
            ("019_sub_1024_8192_1024_fp32", "sub_run", [(1024,), (8192, 1024), "float32"]),
            # float - float:[512, 1024] - [512, 1024] = float:[512, 1024]
            ("020_sub_512_1024_512_1024_fp32", "sub_run", [(512, 1024), (512, 1024), "float32"]),
            # float - float:[1024] - [160, 1024] = float:[160, 1024]
            ("021_sub_1024_160_1024_fp32", "sub_run", [(1024,), (160, 1024), "float32"]),
            # float - float:[4096, 1024] - [4096, 1024] = float:[4096, 1024]
            ("022_sub_4096_1024_4096_1024_fp32", "sub_run", [(4096, 1024), (4096, 1024), "float32"]),
            # float - float:[1024] - [1280, 1024] = float:[1280, 1024]
            ("023_sub_1024_1280_1024_fp32", "sub_run", [(1024,), (1280, 1024), "float32"]),
            # float - float:[] - [8, 1, 128, 128] = float:[8, 1, 128, 128]
            ("024_sub_1_8_1_128_128_fp32", "sub_run", [(1,), (8, 1, 128, 128), "float32"]),
            # float - float:[64, 16, 128, 128] - [64, 16, 128, 1] = float:[64, 16, 128, 128]
            ("025_sub_64_16_128_128_64_16_128_1_fp32", "sub_run", [(64, 16, 128, 128), (64, 16, 128, 1), "float32"]),
            # float - float:[] - [64, 1, 128, 128] = float:[64, 1, 128, 128]
            ("026_sub_1_64_1_128_128_fp32", "sub_run", [(1,), (64, 1, 128, 128), "float32"]),
            # half - half:[] - [64, 1, 128, 128] = half:[64, 1, 128, 128]
            ("027_sub_1_64_1_128_128_fp32", "sub_run", [(1,), (64, 1, 128, 128), "float16"]),
            # float - float:[2, 768] - [2, 768] = float:[2, 768]
            ("028_sub_2_768_2_768_fp32", "sub_run", [(2, 768), (2, 768), "float32"]),
            # float - float:[768, 768] - [768, 768] = float:[768, 768]
            ("029_sub_768_768_768_768_fp32", "sub_run", [(768, 768), (768, 768), "float32"]),
            # float - float:[768] - [1280, 768] = float:[1280, 768]
            ("030_sub_768_1280_768_fp32", "sub_run", [(768,), (1280, 768), "float32"]),
            # float - float:[1280, 768] - [1280, 1] = float:[1280, 768]
            ("031_sub_1280_768_1280_1_fp32", "sub_run", [(1280, 768), (1280, 1), "float32"]),
            # float - float:[8192, 768] - [8192, 1] = float:[8192, 768]
            ("032_sub_8192_768_8192_1_fp32", "sub_run", [(8192, 768), (8192, 1), "float32"]),
            # float - float:[21128] - [21128] = float:[21128]
            ("033_sub_21128_21128_1_fp32", "sub_run", [(21128,), (21128,), "float32"]),
            # float - float:[3072] - [3072] = float:[3072]
            ("034_sub_3072_3072_fp32", "sub_run", [(3072,), (3072,), "float32"]),
            # float - float:[3072, 768] - [3072, 768] = float:[3072, 768]
            ("035_sub_3072_768_3072_768_fp32", "sub_run", [(3072, 768), (3072, 768), "float32"]),
            # # float - float:[1280, 21128] - [1280, 21128] = float:[1280, 21128]
            ("036_sub_1280_21128_1280_21128_fp32", "sub_run", [(1280, 21128), (1280, 21128), "float32"]),
            # # float - float:[64, 128, 768] - [64, 128, 1] = float:[64, 128, 768]
            ("037_sub_64_128_768_64_128_1_fp32", "sub_run", [(64, 128, 768), (64, 128, 1), "float32"]),
            # # float - float:[21128, 768] - [21128, 768] = float:[21128, 768]
            ("038_sub_21128_768_21128_768_fp32", "sub_run", [(21128, 768), (21128, 768), "float32"]),
            # float - float:[768] - [768] = float:[768]
            ("039_sub_768_768_fp32", "sub_run", [(768,), (768,), "float32"]),
            # float - float:[768] - [64, 128, 768] = float:[64, 128, 768]
            ("040_sub_768_64_128_768_fp32", "sub_run", [(768,), (64, 128, 768), "float32"]),
            # half - half:[64, 12, 128, 128] - [64, 12, 128, 1] = half:[64, 12, 128, 128]
            ("041_sub_64_12_128_128_64_12_128_1_fp32", "sub_run", [(64, 12, 128, 128), (64, 12, 128, 1), "float16"]),
            # float - float:[2] - [2] = float:[2]
            ("042_sub_2_2_fp32", "sub_run", [(2,), (2,), "float32"]),
            # float - float:[] - [] = float:[]
            ("043_sub_1_1_fp32", "sub_run", [(1,), (1,), "float32"]),
            # float - float:[768, 3072] - [768, 3072] = float:[768, 3072]
            ("044_sub_768_3072_768_3072_fp32", "sub_run", [(768, 3072), (768, 3072), "float32"]),
            # float - float:[64, 2] - [64, 2] = float:[64, 2]
            ("045_sub_64_2_64_2_fp32", "sub_run", [(64, 2), (64, 2), "float32"]),
            # float - float:[33, 64] - [33, 64] = float:[33, 64]
            ("046_sub_33_64_33_64_fp32", "sub_run", [(33, 64), (33, 64), "float32"]),
            # half - half:[768] - [8192, 768] = half:[8192, 768]
            ("047_sub_768_8192_768_fp32", "sub_run", [(768,), (8192, 768), "float16"]),

            # float - float:[2] - [1, 2] = float:[1, 2]
            ("048_sub_2_1_2_fp32", "sub_run", [(2,), (1, 2), "float32"]),
            # float - float:[768] - [1, 768] = float:[1, 768]
            ("049_sub_768_1_768_fp32", "sub_run", [(1,), (1, 768), "float32"]),
            # float - float:[768] - [768, 768] = float:[768, 768]
            ("050_sub_768_768_768_fp32", "sub_run", [(768,), (768, 768), "float32"]),
            # float - float:[2, 768] - [2, 768] = float:[2, 768]
            ("051_sub_2_768_2_768_fp32", "sub_run", [(2, 768), (2, 768), "float32"]),
            # float - float:[32000] - [1, 32000] = float:[1, 32000]
            ("052_sub_32000_1_32000_fp32", "sub_run", [(32000,), (1, 32000), "float32"]),
            # float - float:[768, 3072] - [768, 3072] = float:[768, 3072]
            ("053_sub_768_3072_768_3072_fp32", "sub_run", [(768, 3072), (768, 3072), "float32"]),
            # float - float:[3072] - [1, 3072] = float:[1, 3072]
            ("054_sub_3072_1_3072_fp32", "sub_run", [(3072,), (1, 3072), "float32"]),
            # float - float:[3072, 768] - [3072, 768] = float:[3072, 768]
            ("055_sub_3072_768_3072_768_fp32", "sub_run", [(3072, 768), (3072, 768), "float32"]),
            # float - float:[33, 64] - [33, 64] = float:[33, 64]
            ("056_sub_33_64_33_64_fp32", "sub_run", [(33, 64), (33, 64), "float32"]),
            # float - float:[16, 768] - [16, 768] = float:[16, 768]
            ("057_sub_16_768_16_768_fp32", "sub_run", [(16, 768), (16, 768), "float32"]),
            # float - float:[32000, 768] - [32000, 768] = float:[32000, 768]
            ("058_sub_32000_768_32000_768_fp32", "sub_run", [(32000, 768), (32000, 768), "float32"]),
            # float - float:[1, 12, 128, 128] - [1, 12, 128, 1] = float:[1, 12, 128, 128]
            ("059_sub_1_12_128_1_1_12_128_1_fp32", "sub_run", [(1, 12, 128, 128), (1, 12, 128, 1), "float32"]),
            # float - float:[1] - [1, 1, 128, 128] = float:[1, 1, 128, 128]
            ("060_sub_1_1_1_128_128_fp32", "sub_run", [(1,), (1, 1, 128, 128), "float32"]),

            # sum OP
            # float - int32:[64, 128, 1024] - [2] = float:[1024]:ok
            ("001_sum", "sum_run", ((64, 128, 1024), (0, 1), False, "float32")),
            # float - int32:[8] - [1] = float:[]:
            ("002_sum", "sum_run", ((8,), (0,), False, "float32")),
            # float - int32:[64, 128, 1024] - [1] = float:[64, 128]
            ("003_sum", "sum_run", ((64, 128, 1024), (2,), False, "float32")),
            # float - int32:[1280, 30522] - [1] = float:[1280]
            ("004_sum", "sum_run", ((1280, 30522), (1,), False, "float32")),
            # float - int32:[64, 2] - [] = float:[64, 1]
            ("005_sum", "sum_run", ((64, 2), (1,), True, "float32")),
            # float - int32:[1280] - [1] = float:[]
            ("006_sum", "sum_run", ((1280,), (0,), False, "float32")),
            # float - int32:[64, 128, 1] - [1] = float:[64, 128]
            ("007_sum", "sum_run", ((64, 128, 1), (2,), False, "float32")),
            # float - int32:[8192, 1024] - [1] = float:[8192]:ok
            ("008_sum", "sum_run", ((8192, 1024), (1,), False, "float32")),
            # float - int32:[1280, 30522] - [] = float:[1280, 1]:ok
            ("009_sum", "sum_run", ((1280, 30522), (1,), True, "float32")),
            # float - int32:[398] - [1] = float:[]
            ("010_sum", "sum_run", ((398,), (0,), True, "float32")),
            # float - int32:[1280, 1] - [1] = float:[1280]:ok
            ("011_sum", "sum_run", ((1280, 1), (1,), False, "float32")),
            # float - int32:[160, 30522] - [1] = float:[160]:ok
            ("012_sum", "sum_run", ((160, 30522), (1,), False, "float32")),
            # float - int32:[1280, 1024] - [1] = float:[1280]:ok
            ("013_sum", "sum_run", ((1280, 1024), (1,), False, "float32")),
            # float - int32:[64, 2] - [] = float:[64]:ok
            ("014_sum", "sum_run", ((64, 2), (1,), False, "float32")),
            # float - int32:[64, 128, 1024] - [1] = float:[128, 1024]:
            ("015_sum", "sum_run", ((64, 128, 1024), (0,), False, "float32")),
            # float - int32:[64, 16, 128, 128] - [] = float:[64, 16, 128, 1]:ok
            ("016_sum", "sum_run", ((64, 16, 128, 128), (3,), True, "float32")),
            # float - int32:[160] - [1] = float:[]
            ("017_sum", "sum_run", ((160,), (0,), True, "float32")),
            # float - int32:[8, 2] - [] = float:[8]
            ("018_sum", "sum_run", ((8, 2), (1,), True, "float32")),
            # float - int32:[8192, 1024] - [1] = float:[1024]:ok
            ("019_sum", "sum_run", ((8192, 1024), (0,), False, "float32")),
            # float - int32:[8192, 1] - [1] = float:[8192]:ok
            ("020_sum", "sum_run", ((8192, 1), (1,), False, "float32")),
            # float - int32:[1280, 1024] - [1] = float:[1024]:ok
            ("021_sum", "sum_run", ((1280, 1024), (0,), False, "float32")),
            # float - int32:[3072] - [1] = float:[1]
            ("022_sum", "sum_run", ((3072,), (0,), True, "float32")),
            # float - int32:[64, 128, 768] - [2] = float:[768]
            ("023_sum", "sum_run", ((64, 128, 768), (0, 1), False, "float32")),
            # half - int32:[8192, 1] - [1] = half:[8192]
            ("024_sum", "sum_run", ((8092, 1), (1,), False, "float16")),
            # float - int32:[1280, 768] - [1] = float:[1280]
            ("025_sum", "sum_run", ((1280, 768), (1,), False, "float32")),
            # float - int32:[3072, 768] - [2] = float:[1, 1]
            ("026_sum", "sum_run", ((3072, 768), (0, 1), True, "float32")),
            # float - int32:[768, 3072] - [2] = float:[1, 1]
            ("027_sum", "sum_run", ((768, 3072), (0, 1), True, "float32")),
            # float - int32:[1280, 768] - [1] = float:[768]
            ("028_sum", "sum_run", ((1280, 768), (0,), False, "float32")),
            # float - int32:[1280] - [1] = float:[]
            ("029_sum", "sum_run", ((1280,), (0,), True, "float32")),
            # float - int32:[21128] - [1] = float:[1]
            ("030_sum", "sum_run", ((21128,), (0,), True, "float32")),
            # float - int32:[21128, 768] - [2] = float:[1, 1]
            ("031_sum", "sum_run", ((21128, 768), (0, 1), True, "float32")),
            # float - int32:[1280, 21128] - [1] = float:[1280]
            ("032_sum", "sum_run", ((21128, 768), (1,), True, "float32")),
            # half - int32:[64, 12, 128, 128] - [] = half:[64, 12, 128, 1]
            ("033_sum", "sum_run", ((64, 12, 128, 128), (3,), True, "float16")),
            # half - int32:[8192, 768] - [1] = half:[768]
            ("034_sum", "sum_run", ((8192, 768), (0,), False, "float16")),
            # float - int32:[64, 2] - [] = float:[64]
            ("035_sum", "sum_run", ((64, 2), (1,), False, "float32")),
            # float - int32:[768, 768] - [2] = float:[1, 1]
            ("036_sum", "sum_run", ((768, 768), (0, 1), False, "float32")),
            # half - int32:[8192, 768] - [1] = half:[8192]
            ("037_sum", "sum_run", ((8192, 768), (1,), False, "float16")),
            # float - int32:[2] - [1] = float:[1]
            ("038_sum", "sum_run", ((2, 1), (0,), False, "float32")),
            # float - int32:[1280, 1] - [1] = float:[1280]
            ("039_sum", "sum_run", ((1280, 1), (1,), False, "float32")),
            # float - int32:[33, 64] - [2] = float:[1, 1]
            ("040_sum", "sum_run", ((33, 64), (0, 1), True, "float32")),
            # float - int32:[2, 768] - [2] = float:[1, 1]
            ("041_sum", "sum_run", ((2, 768), (0, 1), True, "float32")),
            # float - int32:[768] - [1] = float:[1]
            ("042_sum", "sum_run", ((768, 1), (0,), False, "float32")),
            # float - int32:[64, 128, 1] - [1] = float:[64, 128]
            ("043_sum", "sum_run", ((64, 128, 1), (2,), False, "float32")),
            # float - int32:[64, 128, 768] - [1] = float:[64, 128]
            ("044_sum", "sum_run", ((64, 128, 768), (2,), False, "float32")),
            # float - int32:[64, 2] - [] = float:[64, 1]
            ("045_sum", "sum_run", ((64, 2), (1,), True, "float32")),
            # float - int32:[1280, 21128] - [] = float:[1280, 1]
            ("046_sum", "sum_run", ((1280, 21128), (1,), False, "float32")),
            # float - int32:[20, 2] - [0] = float:[2]
            ("047_sum", "sum_run", ((20, 2), (1,), False, "float32")),
            # float - int32:[2, 768] - [1] = float:[2]
            ("048_sum", "sum_run", ((2, 768), (1,), False, "float32")),
            # float - int32:[768, 768] - [1] = float:[768]
            ("049_sum", "sum_run", ((768, 768), (0,), False, "float32")),
            # float - int32:[32000] - [1] = float:[1]
            ("050_sum", "sum_run", ((32000,), (0,), False, "float32")),
            # float - int32:[768, 3072] - [1] = float:[768]
            ("051_sum", "sum_run", ((768, 3072), (1,), False, "float32")),
            # float - int32:[3072, 768] - [1] = float:[3072]
            ("052_sum", "sum_run", ((768, 3072), (1,), False, "float32")),
            # float - int32:[33, 64] - [1] = float:[33]
            ("053_sum", "sum_run", ((33, 64), (1,), False, "float32")),
            # float - int32:[16, 768] - [1] = float:[16]
            ("054_sum", "sum_run", ((16, 768), (1,), False, "float32")),
            # float - int32:[32000, 768] - [1] = float:[32000]
            ("055_sum", "sum_run", ((32000, 768), (1,), False, "float32")),

            # StridedSlice OP
            # int32 - int32 - int32 - int32:[1] - [1] - [1] - [1] = int32:[0]
            ("strided_slice_001_fp32", "strided_slice_run",
             ((1,), [0, ], [0, ], [1, ], 0, 0, 0, 0, 0, "float32")),
            # int32 - int32 - int32 - int32:[2] - [1] - [1] - [1] = int32:[1]
            ("strided_slice_002_fp32", "strided_slice_run",
             ((2,), [0, ], [1, ], [1, ], 0, 0, 0, 0, 0, "float32")),
            ("strided_slice_003_fp32", "strided_slice_run",
             ((2,), [1, ], [2, ], [1, ], 0, 0, 0, 0, 0, "float32")),
            # int32 - int32 - int32 - int32:[1] - [1] - [1] - [1] = int32:[1]
            ("strided_slice_004_fp32", "strided_slice_run",
             ((1,), [0, ], [1, ], [1, ], 0, 0, 0, 0, 0, "float32")),
            # float - int32 - int32 - int32:[64, 128, 1024] - [3] - [3] - [3] = float:[64, 1, 1024]
            ("strided_slice_005_fp32", "strided_slice_run",
             ((64, 128, 1024), [0, 0, 0], [64, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),
            # float - int32 - int32 - int32:[8, 128, 1024] - [3] - [3] - [3] = float:[8, 1, 1024]
            ("strided_slice_006_fp32", "strided_slice_run",
             ((8, 128, 1024), [0, 0, 0], [8, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),
            # int32 - int32 - int32 - int32:[2] - [1] - [1] - [1] = int32:[]
            ("strided_slice_007_fp32", "strided_slice_run",
             ((2,), [0, ], [1, ], [1, ], 0, 0, 0, 0, 0, "float32")),
            # float - int32 - int32 - int32:[64, 128, 768] - [3] - [3] - [3] = float:[64, 1, 768]
            ("strided_slice_008_fp32", "strided_slice_run",
             ((64, 128, 768), [0, 0, 0], [64, 1, 768], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),
            ("strided_slice_009_fp32", "strided_slice_run",
             ((64, 128, 768), [0, 1, 0], [64, 2, 768], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),
            ("strided_slice_010_fp32", "strided_slice_run",
             ((64, 128, 768), [0, 2, 0], [64, 3, 768], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),
            # int32 - int32 - int32 - int32:[2] - [1] - [1] - [1] = int32:[1]
            ("strided_slice_011_fp32", "strided_slice_run",
             ((2,), [0, ], [1, ], [1, ], 0, 0, 0, 0, 0, "float32")),
            # int32 - int32 - int32 - int32:[1] - [1] - [1] - [1] = int32:[0]
            ("strided_slice_012_fp32", "strided_slice_run",
             ((1,), [0, ], [1, ], [1, ], 0, 0, 0, 0, 0, "float32")),
            # int32 - int32 - int32 - int32:[1] - [1] - [1] - [1] = int32:[1]
            ("strided_slice_013_fp32", "strided_slice_run",
             ((1,), [0, ], [1, ], [1, ], 0, 0, 0, 0, 0, "float32")),
            # int32 - int32 - int32 - int32:[2] - [1] - [1] - [1] = int32:[]
            ("strided_slice_014_fp32", "strided_slice_run",
             ((2,), [0, ], [1, ], [1, ], 0, 0, 0, 0, 0, "float32")),

            # StridedSliceGrad OP
            # int32 - int32 - int32 - int32 - float:[3] - [3] - [3] - [3] - [64, 1, 1024] = float:[64, 128, 1024]
            ("strided_slice_grad_dim3_int32_001", "strided_slice_grad_run",
             [(64, 128, 1024), [0, 0, 0], [64, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, (64, 1, 1024), "int32"]),
            # int32 - int32 - int32 - int32 - float:[3] - [3] - [3] - [3] - [64, 1, 768] = float:[64, 128, 768]
            ("strided_slice_grad_dim3_int32_002", "strided_slice_grad_run",
             [(64, 128, 768), [0, 0, 0], [64, 1, 768], [1, 1, 1], 0, 0, 0, 0, 0, (64, 1, 768), "int32"]),
            # float - int32 - int32 - int32 - float:[3] - [3] - [3] - [3] - [3] - [1, 1, 768] = float:[1, 128, 768]
            ("strided_slice_grad_dim3_int32_003", "strided_slice_grad_run",
             [(1, 128, 768), [0, 0, 0], [1, 1, 768], [1, 1, 1], 0, 0, 0, 0, 0, (1, 1, 768), "int32"]),
            # float - int32 - int32 - int32 - float:[3] - [3] - [3] - [3] - [3] - [1, 128, 768] = float:[1, 128, 768]
            ("strided_slice_grad_dim3_int32_004", "strided_slice_grad_run",
             [(1, 1, 768), [0, 0, 0], [1, 128, 768], [1, 1, 1], 0, 0, 0, 0, 0, (1, 1, 768), "int32"]),

            # Tanh OP
            # float:[1280, 1024] = float:[1280, 1024]
            ('tanh_001_1280_1024_fp32', "tanh_run", ((1280, 1024), 'float32')),
            # float:[8, 1024] = float:[8, 1024]
            ('tanh_002_8_1024_fp32', "tanh_run", ((8, 1024), 'float32')),
            # float:[64, 1024] = float:[64, 1024]
            ('tanh_003_64_1024_fp32', "tanh_run", ((64, 1024), 'float32')),
            # float:[1024, 4096] = float:[1024, 4096]
            ('tanh_004_1024_4096_fp32', "tanh_run", ((1024, 4096), 'float32')),
            # float:[8192, 4096] = float:[8192, 4096]
            ('tanh_005_8192_4096_fp32', "tanh_run", ((8192, 4096), 'float32')),
            # float:[160, 1024] = float:[160, 1024]
            ('tanh_006_160_1024_fp32', "tanh_run", ((160, 1024), 'float32')),
            # float:[64, 768] = float:[64, 768]
            ('tanh_007_64_768_fp32', "tanh_run", ((64, 768), 'float32')),
            # float:[1280, 768] = float:[1280, 768]
            ('tanh_008_1280_768_fp32', "tanh_run", ((1280, 768), 'float32')),
            # half:[8192, 3072] = half:[8192, 3072]
            ('tanh_009_8192_3072_fp32', "tanh_run", ((8192, 3072), 'float16')),
            # float:[1, 768] = float:[1, 768]
            ('tanh_010_1_768_fp32', "tanh_run", ((1, 768), 'float32')),

            # TanhGrad op
            # float:[1280, 1024] = float:[1280, 1024]
            ("tanh_grad_001_fp32", "tanh_grad_run", ([1280, 1024], "float32")),
            # float:[8, 1024] = float:[8, 1024]
            ("tanh_grad_002_fp32", "tanh_grad_run", ([8, 1024], "float32")),
            # float:[64, 1024] = float:[64, 1024]
            ("tanh_grad_003_fp32", "tanh_grad_run", ([64, 1024], "float32")),
            # float:[1024, 4096] = float:[1024, 4096]
            ("tanh_grad_004_fp32", "tanh_grad_run", ([1024, 4096], "float32")),

            # float:[8192, 4096] = float:[8192, 4096]
            ("tanh_grad_005_fp32", "tanh_grad_run", ([8192, 4096], "float32")),
            # float:[160, 1024] = float:[160, 1024]
            ("tanh_grad_006_fp32", "tanh_grad_run", ([160, 1024], "float32")),

            # half - half:[8192, 3072] - [8192, 3072] = half:[8192, 3072]
            ("tanh_grad_007_fp32", "tanh_grad_run", ([8192, 3072], "float16")),
            # float - float:[1280, 768] - [1280, 768] = float:[1280, 768]
            ("tanh_grad_008_fp32", "tanh_grad_run", ([1280, 768], "float32")),
            # float - float:[64, 768] - [64, 768] = float:[64, 768]
            ("tanh_grad_009_fp32", "tanh_grad_run", ([64, 768], "float32")),
            # float - float:[1, 768] - [1, 768] = float:[1, 768]
            ("tanh_grad_010_fp32", "tanh_grad_run", ([1, 768], "float32")),

            # RESHAPE OP
            # float - int32:[] - [1] = float:[1]
            ("reshape_001", "reshape_run", [(1,), (1,), "float32"]),
            # float - int32:[8192] - [2] = float:[8192, 1]
            ("reshape_002", "reshape_run", [(8192,), (8192, 1), "float32"]),
            ("reshape_003", "reshape_run", [(8192, 1), (8192,), "float32"]),
            # int32 - int32:[8, 128] - [1] = int32:[1024]
            ("reshape_004", "reshape_run", [(8, 128), (1024,), "int32"]),
            ("reshape_005", "reshape_run", [(1024,), (8, 128), "int32"]),
            # int32 - int32:[64, 1] - [1] = int32:[64]
            ("reshape_006", "reshape_run", [(64, 1), (64,), "int32"]),
            ("reshape_007", "reshape_run", [(64,), (64, 1), "int32"]),
            # int32 - int32:[64, 128, 1] - [1] = int32:[8192]
            ("reshape_008", "reshape_run", [(64, 128, 1), (8192,), "int32"]),
            ("reshape_009", "reshape_run", [(8192,), (64, 128, 1), "int32"]),
            # float - int32:[8192, 1024] - [4] = float:[64, 128, 16, 64]
            ("reshape_010", "reshape_run", [(8192, 1024), (64, 128, 16, 64), "float32"]),
            ("reshape_011", "reshape_run", [(64, 128, 16, 64), (8192, 1024), "float32"]),
            # float - int32:[8, 128, 1024] - [2] = float:[1024, 1024]
            ("reshape_012", "reshape_run", [(8, 128, 1024), (1024, 1024), "float32"]),
            ("reshape_013", "reshape_run", [(1024, 1024), (8, 128, 1024), "float32"]),
            # string - int32:[] - [1] = string:[1]
            # float - int32:[64, 128, 1024] - [2] = float:[8192, 1024]
            ("reshape_015", "reshape_run", [(64, 128, 1024), (8192, 1024), "float32"]),
            ("reshape_016", "reshape_run", [(8192, 1024), (64, 128, 1024), "float32"]),
            # float - int32:[64, 20] - [1] = float:[1280]
            ("reshape_017", "reshape_run", [(64, 20), (1280,), "float32"]),
            ("reshape_018", "reshape_run", [(1280,), (64, 20), "float32"]),
            # float - int32:[8, 128, 16, 64] - [2] = float:[1024, 1024]
            ("reshape_019", "reshape_run", [(8, 128, 16, 64), (1024, 1024), "float32"]),
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
            ("reshape_052", "reshape_run", [(64, 128, 1024), (8192, 1024), "float32"]),
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
            # # float - int32:[8192, 768] - [3] = float:[64, 128, 768]
            ("reshape_063", "reshape_run", [(8192, 768), (64, 128, 768), "float32"]),
            ("reshape_064", "reshape_run", [(64, 128, 768), (8192, 768), "float32"]),
            # # half - int32:[128, 64, 12, 128] - [3] = half:[128, 768, 128]
            ("reshape_065", "reshape_run", [(128, 64, 12, 128), (128, 768, 128), "float16"]),
            ("reshape_066", "reshape_run", [(128, 768, 128), (128, 64, 12, 128), "float16"]),
            # # half - int32:[128, 768, 64] - [4] = half:[128, 64, 12, 64]
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
            # float - int32:[] - [1] = float:[1]
            ("reshape_089", "reshape_run", [(1,), (1,), "float32"]),
            ("reshape_090", "reshape_run", [(1,), (1,), "float32"]),
            # float - int32:[64, 128, 768] - [2] = float:[8192, 768]
            ("reshape_091", "reshape_run", [(64, 128, 768), (8192, 768), "float32"]),
            ("reshape_092", "reshape_run", [(8192, 768), (64, 128, 768), "float32"]),
            # # half - int32:[128, 64, 12, 64] - [3] = half:[128, 768, 64]
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
            # int32 - int32:[128] - [1] = int32:[128]
            ("reshape_103", "reshape_run", [(128,), (128,), "float16"]),
            ("reshape_104", "reshape_run", [(128,), (128,), "float16"]),
            # float - int32:[1, 128, 768] - [3] = float:[1, 128, 768]
            ("reshape_105", "reshape_run", [(1, 128, 768), (1, 128, 768), "float32"]),
            ("reshape_106", "reshape_run", [(1, 128, 768), (1, 128, 768), "float32"]),
            # float - int32:[1, 128, 768] - [2] = float:[128, 768]
            ("reshape_107", "reshape_run", [(1, 128, 768), (128, 768), "float32"]),
            ("reshape_108", "reshape_run", [(128, 768), (1, 128, 768), "float32"]),
            # float - int32:[128, 768] - [2] = float:[128, 768]
            ("reshape_109", "reshape_run", [(128, 768), (128, 768), "float32"]),
            ("reshape_110", "reshape_run", [(128, 768), (128, 768), "float32"]),
            # float - int32:[128, 768] - [4] = float:[1, 128, 12, 64]
            ("reshape_111", "reshape_run", [(128, 768), (1, 128, 12, 64), "float32"]),
            ("reshape_112", "reshape_run", [(1, 128, 12, 64), (128, 768), "float32"]),
            # float - int32:[128, 768] - [4] = float:[1, 128, 12, 64]
            ("reshape_113", "reshape_run", [(128, 768), (1, 128, 12, 64), "float32"]),
            ("reshape_114", "reshape_run", [(1, 128, 12, 64), (128, 768), "float32"]),
            # float - int32:[128] - [3] = float:[1, 1, 128]
            ("reshape_115", "reshape_run", [(128,), (1, 1, 128), "float32"]),
            ("reshape_116", "reshape_run", [(1, 1, 128), (128,), "float32"]),
            # float - int32:[1, 128, 12, 64] - [2] = float:[128, 768]
            ("reshape_117", "reshape_run", [(1, 128, 12, 64), (128, 768), "float32"]),
            ("reshape_118", "reshape_run", [(128, 768), (1, 128, 12, 64), "float32"]),
            # float - int32:[128, 768] - [3] = float:[1, 128, 768]
            ("reshape_119", "reshape_run", [(128, 768), (1, 128, 768), "float32"]),
            ("reshape_120", "reshape_run", [(1, 128, 768), (128, 768), "float32"]),
            # float - int32:[20] - [] = float:[20]
            ("reshape_121", "reshape_run", [(20,), (20,), "float32"]),
            # float - int32:[1] - [] = float:[1]
            ("reshape_122", "reshape_run", [(1,), (1,), "float32"]),
            # float - int32:[20] - [2] = float:[20, 1]
            ("reshape_123", "reshape_run", [(20,), (20, 1), "float32"]),
            ("reshape_124", "reshape_run", [(20, 1), (20,), "float32"]),
            # float - int32:[1, 2] - [2] = float:[1, 2]
            ("reshape_125", "reshape_run", [(1, 2), (1, 2), "float32"]),
            ("reshape_126", "reshape_run", [(1, 2), (1, 2), "float32"]),
            # float - int32:[1] - [2] = float:[1, 1]
            ("reshape_127", "reshape_run", [(1,), (1, 1), "float32"]),
            ("reshape_128", "reshape_run", [(1, 1), (1,), "float32"]),
            # float - int32:[1, 768] - [3] = float:[1, 1, 768]
            ("reshape_129", "reshape_run", [(1, 768), (1, 1, 768), "float32"]),
            ("reshape_130", "reshape_run", [(1, 1, 768), (1, 768), "float32"]),
            # half - int32:[64, 128, 768] - [2] = half:[8192, 768]
            ("reshape_131", "reshape_run", [(64, 128, 768), (8192, 768), "float16"]),
            ("reshape_132", "reshape_run", [(8192, 768), (64, 128, 768), "float16"]),

            # softmax OP
            # float:[64, 16, 128, 128] = float:[64, 16, 128, 128]
            ("softmax_001", "softmax_run", ((64, 16, 128, 128), "float32", -1, "cce_softmax_fp32")),
            # float:[8, 16, 128, 128] = float:[8, 16, 128, 128]
            ("softmax_002", "softmax_run", ((8, 16, 128, 128), "float32", -1, "cce_softmax_fp32")),
            # float:[20, 32000] = float:[20, 32000]
            ("softmax_003", "softmax_run", ((20, 32000), "float32", -1, "cce_softmax_fp32")),
            # float:[1, 12, 128, 128] = float:[1, 12, 128,128]
            ("softmax_004", "softmax_run", ((1, 12, 128, 128), "float32", -1, "cce_softmax_fp32")),

            # pow OP
            # float - float:[1280, 768] - [] = float:[1280, 768]
            ("pow_001", "pow_run", ((1280, 768), (1,), 'float32')),
            # float - float:[] - [] = float:[]
            ("pow_002", "pow_run", ((1,), (1,), 'float32')),
            # half - half:[8192, 3072] - [] = half:[8192, 3072]
            ("pow_003", "pow_run", ((8192, 3072), (8192, 3072), 'float16')),

            # reciprocal OP
            # float - float:[160, 1024] = float:[160, 1024]
            ("reciprocal_001", "reciprocal_run", ((160, 1024), 'float32'),),
            # float - float:[] = float:[]
            ("reciprocal_002", "reciprocal_run", ((1,), 'float32'),),

            # bert联调base新增shape
            # Tile OP
            # float-int32:[10240, 1]-[2]=float:[10240, 21128]
            ("tile_001", "tile_run", ((10240, 1), "float32", (1, 21128))),
            # float-int32:[1024, 1]-[2]=float:[1024, 2]
            ("tile_002", "tile_run", ((1024, 1), "float32", (1, 2))),
            # float-int32:[1, 1]-[2]=float:[1, 2]
            ("tile_003", "tile_run", ((1, 1), "float32", (2,))),
            # float-int32:[1]-[1]=float:[1]
            ("tile_004", "tile_run", ((1,), "float32", (1,))),
            # float-int32:[1]-[1]=float:[1024]
            ("tile_005", "tile_run", ((1,), "float32", (1024,))),
            # float-int32:[1]-[1]=float:[10240]
            ("tile_006", "tile_run", ((1,), "float32", (10240,))),
            # float-int32:[1]-[1]=float:[128]
            ("tile_007", "tile_run", ((1,), "float32", (128,))),
            # float-int32:[1]-[1]=float:[1280]
            ("tile_008", "tile_run", ((1,), "float32", (1280,))),
            # float-int32:[1]-[1]=float:[16]
            ("tile_009", "tile_run", ((1,), "float32", (16,))),
            # float-int32:[1]-[1]=float:[160]
            ("tile_010", "tile_run", ((1,), "float32", (160,))),
            # float-int32:[1]-[1]=float:[2]
            ("tile_011", "tile_run", ((1,), "float32", (2,))),
            # float-int32:[1]-[1]=float:[20]
            ("tile_012", "tile_run", ((1,), "float32", (20,))),
            # float-int32:[1]-[1]=float:[20480]
            ("tile_013", "tile_run", ((1,), "float32", (20480,))),
            # float-int32:[1]-[1]=float:[256]
            ("tile_014", "tile_run", ((1,), "float32", (256,))),
            # float-int32:[1]-[1]=float:[2560]
            ("tile_015", "tile_run", ((1,), "float32", (2560,))),
            # float-int32:[1]-[1]=float:[32]
            ("tile_016", "tile_run", ((1,), "float32", (32,))),
            # float-int32:[1]-[1]=float:[320]
            ("tile_017", "tile_run", ((1,), "float32", (320,))),
            # float-int32:[1]-[1]=float:[4]
            ("tile_018", "tile_run", ((1,), "float32", (4,))),
            # float-int32:[1]-[1]=float:[40]
            ("tile_019", "tile_run", ((1,), "float32", (40,))),
            # float-int32:[1]-[1]=float:[512]
            ("tile_020", "tile_run", ((1,), "float32", (512,))),
            # float-int32:[1]-[1]=float:[5120]
            ("tile_021", "tile_run", ((1,), "float32", (5120,))),
            # float-int32:[1]-[1]=float:[64]
            ("tile_022", "tile_run", ((1,), "float32", (64,))),
            # float-int32:[1]-[1]=float:[640]
            ("tile_023", "tile_run", ((1,), "float32", (640,))),
            # float-int32:[1]-[1]=float:[8]
            ("tile_024", "tile_run", ((1,), "float32", (8,))),
            # float-int32:[1]-[1]=float:[80]
            ("tile_025", "tile_run", ((1,), "float32", (80,))),
            # float-int32:[1280, 1]-[2]=float:[1280, 21128]
            ("tile_026", "tile_run", ((1280, 1), "float32", (1, 21128))),
            # float-int32:[128, 1]-[2]=float:[128, 2]
            ("tile_027", "tile_run", ((128, 1), "float32", (1, 2))),
            # float-int32:[160, 1]-[2]=float:[160, 21128]
            ("tile_028", "tile_run", ((160, 1), "float32", (1, 21128))),
            # float-int32:[16, 1]-[2]=float:[16, 2]
            ("tile_029", "tile_run", ((16, 1), "float32", (1, 2))),
            # float-int32:[20, 1]-[2]=float:[20, 21128]
            ("tile_030", "tile_run", ((20, 1), "float32", (1, 21128))),
            # float-int32:[20480, 1]-[2]=float:[20480, 21128]
            ("tile_031", "tile_run", ((20480, 1), "float32", (1, 21128))),
            # float-int32:[2, 1]-[2]=float:[2, 2]
            ("tile_032", "tile_run", ((2, 1), "float32", (1, 2))),
            # float-int32:[2560, 1]-[2]=float:[2560, 21128]
            ("tile_033", "tile_run", ((2560, 1), "float32", (1, 21128))),
            # float-int32:[256, 1]-[2]=float:[256, 2]
            ("tile_034", "tile_run", ((256, 1), "float32", (1, 2))),
            # float-int32:[320, 1]-[2]=float:[320, 21128]
            ("tile_035", "tile_run", ((320, 1), "float32", (1, 21128))),
            # float-int32:[32, 1]-[2]=float:[32, 2]
            ("tile_036", "tile_run", ((32, 1), "float32", (1, 2))),
            # float-int32:[40, 1]-[2]=float:[40, 21128]
            ("tile_037", "tile_run", ((40, 1), "float32", (1, 21128))),
            # float-int32:[4, 1]-[2]=float:[4, 2]
            ("tile_038", "tile_run", ((4, 1), "float32", (1, 2))),
            # float-int32:[5120, 1]-[2]=float:[5120, 21128]
            ("tile_039", "tile_run", ((5120, 1), "float32", (1, 21128))),
            # float-int32:[512, 1]-[2]=float:[512, 2]
            ("tile_040", "tile_run", ((512, 1), "float32", (1, 2))),
            # float-int32:[640, 1]-[2]=float:[640, 21128]
            ("tile_041", "tile_run", ((640, 1), "float32", (1, 21128))),
            # float-int32:[64, 1]-[2]=float:[64, 2]
            ("tile_042", "tile_run", ((64, 1), "float32", (1, 2))),
            # float-int32:[80, 1]-[2]=float:[80, 21128]
            ("tile_043", "tile_run", ((80, 1), "float32", (1, 21128))),
            # float-int32:[8, 1]-[2]=float:[8, 2]
            ("tile_044", "tile_run", ((8, 1), "float32", (1, 2))),
            # int32 - int32:[128] - [1] = int32:[16384]
            ("tile_045", "tile_run", ((128,), "int32", (128,))),

            # Transpose OP
            # float-int32:[10240, 768]-[2]=float:[10240, 768]
            ("transpose_0001", "transpose_run", ((10240, 768), (0, 1,), "float32")),
            # float-int32:[1024, 12, 128, 128]-[4]=float:[128, 1024, 12, 128]
            ("transpose_0002", "transpose_run", ((1024, 12, 128, 128), (2, 0, 1, 3), "float32")),
            # float-int32:[1024, 12, 128, 64]-[4]=float:[1024, 128, 12, 64]
            ("transpose_0003", "transpose_run", ((1024, 12, 128, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[1024, 12, 128, 64]-[4]=float:[128, 1024, 12, 64]
            ("transpose_0004", "transpose_run", ((1024, 12, 128, 64), (2, 0, 1, 3), "float32")),
            # float-int32:[1024, 128, 12, 64]-[4]=float:[1024, 12, 128, 64]
            ("transpose_0005", "transpose_run", ((1024, 128, 12, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[1024, 768]-[2]=float:[1024, 768]
            ("transpose_0006", "transpose_run", ((1024, 768), (0, 1,), "float32")),
            # float-int32:[1, 12, 128, 128]-[4]=float:[128, 1, 12, 128]
            ("transpose_0007", "transpose_run", ((1, 12, 128, 128), (2, 0, 1, 3), "float32")),
            # float-int32:[1, 12, 128, 64]-[4]=float:[1, 128, 12, 64]
            ("transpose_0008", "transpose_run", ((1, 12, 128, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[1, 12, 128, 64]-[4]=float:[128, 1, 12, 64]
            ("transpose_0009", "transpose_run", ((1, 12, 128, 64), (2, 0, 1, 3), "float32")),
            # float-int32:[1, 128, 12, 64]-[4]=float:[1, 12, 128, 64]
            ("transpose_0010", "transpose_run", ((1, 128, 12, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[1280, 768]-[2]=float:[1280, 768]
            ("transpose_0011", "transpose_run", ((1280, 768), (0, 1,), "float32")),
            # float-int32:[128, 1024, 12, 128]-[4]=float:[1024, 12, 128, 128]
            ("transpose_0012", "transpose_run", ((128, 1024, 12, 128), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 1024, 12, 64]-[4]=float:[1024, 12, 128, 64]
            ("transpose_0013", "transpose_run", ((128, 1024, 12, 64), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 1, 12, 128]-[4]=float:[1, 12, 128, 128]
            ("transpose_0014", "transpose_run", ((128, 1, 12, 128), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 1, 12, 64]-[4]=float:[1, 12, 128, 64]
            ("transpose_0015", "transpose_run", ((128, 1, 12, 64), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 12, 128, 128]-[4]=float:[128, 128, 12, 128]
            ("transpose_0016", "transpose_run", ((128, 12, 128, 128), (2, 0, 1, 3), "float32")),
            # float-int32:[128, 12, 128, 64]-[4]=float:[128, 128, 12, 64]
            ("transpose_0017", "transpose_run", ((128, 12, 128, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[128, 12, 128, 64]-[4]=float:[128, 128, 12, 64]
            ("transpose_0018", "transpose_run", ((128, 12, 128, 64), (2, 0, 1, 3), "float32")),
            # float-int32:[128, 128, 12, 128]-[4]=float:[128, 12, 128, 128]
            ("transpose_0019", "transpose_run", ((128, 128, 12, 128), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 128, 12, 64]-[4]=float:[128, 12, 128, 64]
            ("transpose_0020", "transpose_run", ((128, 128, 12, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[128, 128, 12, 64]-[4]=float:[128, 12, 128, 64]
            ("transpose_0021", "transpose_run", ((128, 128, 12, 64), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 128, 64]-[3]=float:[128, 128, 64]
            ("transpose_0022", "transpose_run", ((128, 128, 64), (0, 1, 2,), "float32")),
            # float-int32:[128, 16, 12, 128]-[4]=float:[16, 12, 128, 128]
            ("transpose_0023", "transpose_run", ((128, 16, 12, 128), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 16, 12, 64]-[4]=float:[16, 12, 128, 64]
            ("transpose_0024", "transpose_run", ((128, 16, 12, 64), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 2, 12, 128]-[4]=float:[2, 12, 128, 128]
            ("transpose_0025", "transpose_run", ((128, 2, 12, 128), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 2, 12, 64]-[4]=float:[2, 12, 128, 64]
            ("transpose_0026", "transpose_run", ((128, 2, 12, 64), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 256, 12, 128]-[4]=float:[256, 12, 128, 128]
            ("transpose_0027", "transpose_run", ((128, 256, 12, 128), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 256, 12, 64]-[4]=float:[256, 12, 128, 64]
            ("transpose_0028", "transpose_run", ((128, 256, 12, 64), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 32, 12, 128]-[4]=float:[32, 12, 128, 128]
            ("transpose_0029", "transpose_run", ((128, 32, 12, 128), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 32, 12, 64]-[4]=float:[32, 12, 128, 64]
            ("transpose_0030", "transpose_run", ((128, 32, 12, 64), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 4, 12, 128]-[4]=float:[4, 12, 128, 128]
            ("transpose_0031", "transpose_run", ((128, 4, 12, 128), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 4, 12, 64]-[4]=float:[4, 12, 128, 64]
            ("transpose_0032", "transpose_run", ((128, 4, 12, 64), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 512, 12, 128]-[4]=float:[512, 12, 128, 128]
            ("transpose_0033", "transpose_run", ((128, 512, 12, 128), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 512, 12, 64]-[4]=float:[512, 12, 128, 64]
            ("transpose_0034", "transpose_run", ((128, 512, 12, 64), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 64, 12, 128]-[4]=float:[64, 12, 128, 128]
            ("transpose_0035", "transpose_run", ((128, 64, 12, 128), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 64, 12, 64]-[4]=float:[64, 12, 128, 64]
            ("transpose_0036", "transpose_run", ((128, 64, 12, 64), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 768]-[2]=float:[128, 768]
            ("transpose_0037", "transpose_run", ((128, 768), (0, 1,), "float32")),
            # float-int32:[128, 8, 12, 128]-[4]=float:[8, 12, 128, 128]
            ("transpose_0038", "transpose_run", ((128, 8, 12, 128), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 8, 12, 64]-[4]=float:[8, 12, 128, 64]
            ("transpose_0039", "transpose_run", ((128, 8, 12, 64), (1, 2, 0, 3), "float32")),
            # float-int32:[131072, 768]-[2]=float:[131072, 768]
            ("transpose_0040", "transpose_run", ((131072, 768), (0, 1,), "float32")),
            # float-int32:[160, 768]-[2]=float:[160, 768]
            ("transpose_0041", "transpose_run", ((160, 768), (0, 1,), "float32")),
            # float-int32:[16, 12, 128, 128]-[4]=float:[128, 16, 12, 128]
            ("transpose_0042", "transpose_run", ((16, 12, 128, 128), (2, 0, 1, 3), "float32")),
            # float-int32:[16, 12, 128, 64]-[4]=float:[128, 16, 12, 64]
            ("transpose_0043", "transpose_run", ((16, 12, 128, 64), (2, 0, 1, 3), "float32")),
            # float-int32:[16, 12, 128, 64]-[4]=float:[16, 128, 12, 64]
            ("transpose_0044", "transpose_run", ((16, 12, 128, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[16, 128, 12, 64]-[4]=float:[16, 12, 128, 64]
            ("transpose_0045", "transpose_run", ((16, 128, 12, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[16384, 768]-[2]=float:[16384, 768]
            ("transpose_0046", "transpose_run", ((16384, 768), (0, 1,), "float32")),
            # float-int32:[20480, 768]-[2]=float:[20480, 768]
            ("transpose_0047", "transpose_run", ((20480, 768), (0, 1,), "float32")),
            # float-int32:[2048, 768]-[2]=float:[2048, 768]
            ("transpose_0048", "transpose_run", ((2048, 768), (0, 1,), "float32")),
            # float-int32:[20, 768]-[2]=float:[20, 768]
            ("transpose_0049", "transpose_run", ((20, 768), (0, 1,), "float32")),
            # float-int32:[21128, 768]-[2]=float:[21128, 768]
            ("transpose_0050", "transpose_run", ((21128, 768), (0, 1,), "float32")),
            # float-int32:[2, 12, 128, 128]-[4]=float:[128, 2, 12, 128]
            ("transpose_0051", "transpose_run", ((2, 12, 128, 128), (2, 0, 1, 3), "float32")),
            # float-int32:[2, 12, 128, 64]-[4]=float:[128, 2, 12, 64]
            ("transpose_0052", "transpose_run", ((2, 12, 128, 64), (2, 0, 1, 3), "float32")),
            # float-int32:[2, 12, 128, 64]-[4]=float:[2, 128, 12, 64]
            ("transpose_0053", "transpose_run", ((2, 12, 128, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[2, 128, 12, 64]-[4]=float:[2, 12, 128, 64]
            ("transpose_0054", "transpose_run", ((2, 128, 12, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[2560, 768]-[2]=float:[2560, 768]
            ("transpose_0055", "transpose_run", ((2560, 768), (0, 1,), "float32")),
            # float-int32:[256, 12, 128, 128]-[4]=float:[128, 256, 12, 128]
            ("transpose_0056", "transpose_run", ((256, 12, 128, 128), (2, 0, 1, 3), "float32")),
            # float-int32:[256, 12, 128, 64]-[4]=float:[128, 256, 12, 64]
            ("transpose_0057", "transpose_run", ((256, 12, 128, 64), (2, 0, 1, 3), "float32")),
            # float-int32:[256, 12, 128, 64]-[4]=float:[256, 128, 12, 64]
            ("transpose_0058", "transpose_run", ((256, 12, 128, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[256, 128, 12, 64]-[4]=float:[256, 12, 128, 64]
            ("transpose_0059", "transpose_run", ((256, 128, 12, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[256, 768]-[2]=float:[256, 768]
            ("transpose_0060", "transpose_run", ((256, 768), (0, 1,), "float32")),
            # float-int32:[2, 768]-[2]=float:[2, 768]
            ("transpose_0061", "transpose_run", ((2, 768), (0, 1,), "float32")),
            # float-int32:[320, 768]-[2]=float:[320, 768]
            ("transpose_0062", "transpose_run", ((320, 768), (0, 1,), "float32")),
            # float-int32:[32, 12, 128, 128]-[4]=float:[128, 32, 12, 128]
            ("transpose_0063", "transpose_run", ((32, 12, 128, 128), (2, 0, 1, 3), "float32")),
            # float-int32:[32, 12, 128, 64]-[4]=float:[128, 32, 12, 64]
            ("transpose_0064", "transpose_run", ((32, 12, 128, 64), (2, 0, 1, 3), "float32")),
            # float-int32:[32, 12, 128, 64]-[4]=float:[32, 128, 12, 64]
            ("transpose_0065", "transpose_run", ((32, 12, 128, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[32, 128, 12, 64]-[4]=float:[32, 12, 128, 64]
            ("transpose_0066", "transpose_run", ((32, 128, 12, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[32768, 768]-[2]=float:[32768, 768]
            ("transpose_0067", "transpose_run", ((32768, 768), (0, 1,), "float32")),
            # float-int32:[33, 64]-[2]=float:[33, 64]
            ("transpose_0068", "transpose_run", ((33, 64), (0, 1,), "float32")),
            # float-int32:[40, 768]-[2]=float:[40, 768]
            ("transpose_0069", "transpose_run", ((40, 768), (0, 1,), "float32")),
            # float-int32:[4096, 768]-[2]=float:[4096, 768]
            ("transpose_0070", "transpose_run", ((4096, 768), (0, 1,), "float32")),
            # float-int32:[4, 12, 128, 128]-[4]=float:[128, 4, 12, 128]
            ("transpose_0071", "transpose_run", ((4, 12, 128, 128), (2, 0, 1, 3), "float32")),
            # float-int32:[4, 12, 128, 64]-[4]=float:[128, 4, 12, 64]
            ("transpose_0072", "transpose_run", ((4, 12, 128, 64), (2, 0, 1, 3), "float32")),
            # float-int32:[4, 12, 128, 64]-[4]=float:[4, 128, 12, 64]
            ("transpose_0073", "transpose_run", ((4, 12, 128, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[4, 128, 12, 64]-[4]=float:[4, 12, 128, 64]
            ("transpose_0074", "transpose_run", ((4, 128, 12, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[5120, 768]-[2]=float:[5120, 768]
            ("transpose_0075", "transpose_run", ((5120, 768), (0, 1,), "float32")),
            # float-int32:[512, 12, 128, 128]-[4]=float:[128, 512, 12, 128]
            ("transpose_0076", "transpose_run", ((512, 12, 128, 128), (2, 0, 1, 3), "float32")),
            # float-int32:[512, 12, 128, 64]-[4]=float:[128, 512, 12, 64]
            ("transpose_0077", "transpose_run", ((512, 12, 128, 64), (2, 0, 1, 3), "float32")),
            # float-int32:[512, 12, 128, 64]-[4]=float:[512, 128, 12, 64]
            ("transpose_0078", "transpose_run", ((512, 12, 128, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[512, 128, 12, 64]-[4]=float:[512, 12, 128, 64]
            ("transpose_0079", "transpose_run", ((512, 128, 12, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[512, 768]-[2]=float:[512, 768]
            ("transpose_0080", "transpose_run", ((512, 768), (0, 1,), "float32")),
            # float-int32:[640, 768]-[2]=float:[640, 768]
            ("transpose_0081", "transpose_run", ((640, 768), (0, 1,), "float32")),
            # float-int32:[64, 12, 128, 128]-[4]=float:[128, 64, 12, 128]
            ("transpose_0082", "transpose_run", ((64, 12, 128, 128), (2, 0, 1, 3), "float32")),
            # float-int32:[64, 12, 128, 64]-[4]=float:[128, 64, 12, 64]
            ("transpose_0083", "transpose_run", ((64, 12, 128, 64), (2, 0, 1, 3), "float32")),
            # float-int32:[64, 12, 128, 64]-[4]=float:[64, 128, 12, 64]
            ("transpose_0084", "transpose_run", ((64, 12, 128, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[64, 128, 12, 64]-[4]=float:[64, 12, 128, 64]
            ("transpose_0085", "transpose_run", ((64, 128, 12, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[65536, 768]-[2]=float:[65536, 768]
            ("transpose_0086", "transpose_run", ((65536, 768), (0, 1,), "float32")),
            # float-int32:[80, 768]-[2]=float:[80, 768]
            ("transpose_0087", "transpose_run", ((80, 768), (0, 1,), "float32")),
            # float-int32:[8, 12, 128, 128]-[4]=float:[128, 8, 12, 128]
            ("transpose_0088", "transpose_run", ((8, 12, 128, 128), (2, 0, 1, 3), "float32")),
            # float-int32:[8, 12, 128, 64]-[4]=float:[128, 8, 12, 64]
            ("transpose_0089", "transpose_run", ((8, 12, 128, 64), (2, 0, 1, 3), "float32")),
            # float-int32:[8, 12, 128, 64]-[4]=float:[8, 128, 12, 64]
            ("transpose_0090", "transpose_run", ((8, 12, 128, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[8, 128, 12, 64]-[4]=float:[8, 12, 128, 64]
            ("transpose_0091", "transpose_run", ((8, 128, 12, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[8192, 768]-[2]=float:[8192, 768]
            ("transpose_0092", "transpose_run", ((8192, 768), (0, 1,), "float32")),
            # int32 - int32:[128, 128] - [2] = int32:[128, 128]
            ("transpose_0093", "transpose_run", ((128, 128), (1, 0,), "int32")),

            # UnsortedSegmentSum OP
            # float-int32-int32:[10240, 768]-[10240]-[]=float:[65536, 768]
            ("unsortedsegmentsum_001", "unsortedsegmentsum_run", ([10240, 768], [10240], 65536, "float32")),
            # float-int32-int32:[1024, 768]-[1024]-[]=float:[21128, 768]
            ("unsortedsegmentsum_002", "unsortedsegmentsum_run", ([1024, 768], [1024], 21128, "float32")),
            # float-int32-int32:[1024, 768]-[1024]-[]=float:[2, 768]
            ("unsortedsegmentsum_003", "unsortedsegmentsum_run", ([1024, 768], [1024], 2, "float32")),
            # float-int32-int32:[1280, 768]-[1280]-[]=float:[8192, 768]
            ("unsortedsegmentsum_004", "unsortedsegmentsum_run", ([1280, 768], [1280], 8192, "float32")),
            # float-int32-int32:[128, 128, 64]-[128]-[]=float:[33, 64]
            ("unsortedsegmentsum_005", "unsortedsegmentsum_run", ([128, 128, 64], [128], 33, "float32")),
            # float-int32-int32:[128, 768]-[128]-[]=float:[21128, 768]
            ("unsortedsegmentsum_006", "unsortedsegmentsum_run", ([128, 768], [128], 21128, "float32")),
            # float-int32-int32:[128, 768]-[128]-[]=float:[2, 768]
            ("unsortedsegmentsum_007", "unsortedsegmentsum_run", ([128, 768], [128], 2, "float32")),
            # float-int32-int32:[131072, 768]-[131072]-[]=float:[21128, 768]
            ("unsortedsegmentsum_008", "unsortedsegmentsum_run", ([131072, 768], [131072], 21128, "float32")),
            # float-int32-int32:[131072, 768]-[131072]-[]=float:[2, 768]
            ("unsortedsegmentsum_009", "unsortedsegmentsum_run", ([131072, 768], [131072], 2, "float32")),
            # float-int32-int32:[160, 768]-[160]-[]=float:[1024, 768]
            ("unsortedsegmentsum_010", "unsortedsegmentsum_run", ([160, 768], [160], 1024, "float32")),
            # float-int32-int32:[16384, 768]-[16384]-[]=float:[21128, 768]
            ("unsortedsegmentsum_011", "unsortedsegmentsum_run", ([16384, 768], [16384], 21128, "float32")),
            # float-int32-int32:[16384, 768]-[16384]-[]=float:[2, 768]
            ("unsortedsegmentsum_012", "unsortedsegmentsum_run", ([16384, 768], [16384], 2, "float32")),
            # float-int32-int32:[20480, 768]-[20480]-[]=float:[131072, 768]
            ("unsortedsegmentsum_013", "unsortedsegmentsum_run", ([20480, 768], [20480], 131072, "float32")),
            # float-int32-int32:[2048, 768]-[2048]-[]=float:[21128, 768]
            ("unsortedsegmentsum_014", "unsortedsegmentsum_run", ([2048, 768], [2048], 21128, "float32")),
            # float-int32-int32:[2048, 768]-[2048]-[]=float:[2, 768]
            ("unsortedsegmentsum_015", "unsortedsegmentsum_run", ([2048, 768], [2048], 2, "float32")),
            # float-int32-int32:[20, 768]-[20]-[]=float:[128, 768]
            ("unsortedsegmentsum_016", "unsortedsegmentsum_run", ([20, 768], [20], 128, "float32")),
            # float-int32-int32:[2560, 768]-[2560]-[]=float:[16384, 768]
            ("unsortedsegmentsum_017", "unsortedsegmentsum_run", ([2560, 768], [2560], 16384, "float32")),
            # float-int32-int32:[256, 768]-[256]-[]=float:[21128, 768]
            ("unsortedsegmentsum_018", "unsortedsegmentsum_run", ([256, 768], [256], 21128, "float32")),
            # float-int32-int32:[256, 768]-[256]-[]=float:[2, 768]
            ("unsortedsegmentsum_019", "unsortedsegmentsum_run", ([256, 768], [256], 2, "float32")),
            # float-int32-int32:[320, 768]-[320]-[]=float:[2048, 768]
            ("unsortedsegmentsum_020", "unsortedsegmentsum_run", ([320, 768], [320], 2048, "float32")),
            # float-int32-int32:[32768, 768]-[32768]-[]=float:[21128, 768]
            ("unsortedsegmentsum_021", "unsortedsegmentsum_run", ([32768, 768], [32768], 21128, "float32")),
            # float-int32-int32:[32768, 768]-[32768]-[]=float:[2, 768]
            ("unsortedsegmentsum_022", "unsortedsegmentsum_run", ([32768, 768], [32768], 2, "float32")),
            # float-int32-int32:[40, 768]-[40]-[]=float:[256, 768]
            ("unsortedsegmentsum_023", "unsortedsegmentsum_run", ([40, 768], [40], 256, "float32")),
            # float-int32-int32:[4096, 768]-[4096]-[]=float:[21128, 768]
            ("unsortedsegmentsum_024", "unsortedsegmentsum_run", ([4096, 768], [4096], 21128, "float32")),
            # float-int32-int32:[4096, 768]-[4096]-[]=float:[2, 768]
            ("unsortedsegmentsum_025", "unsortedsegmentsum_run", ([4096, 768], [4096], 2, "float32")),
            # float-int32-int32:[5120, 768]-[5120]-[]=float:[32768, 768]
            ("unsortedsegmentsum_026", "unsortedsegmentsum_run", ([5120, 768], [5120], 32768, "float32")),
            # float-int32-int32:[512, 768]-[512]-[]=float:[21128, 768]
            ("unsortedsegmentsum_027", "unsortedsegmentsum_run", ([512, 768], [512], 21128, "float32")),
            # float-int32-int32:[512, 768]-[512]-[]=float:[2, 768]
            ("unsortedsegmentsum_028", "unsortedsegmentsum_run", ([512, 768], [512], 2, "float32")),
            # float-int32-int32:[640, 768]-[640]-[]=float:[4096, 768]
            ("unsortedsegmentsum_029", "unsortedsegmentsum_run", ([640, 768], [640], 4096, "float32")),
            # float-int32-int32:[65536, 768]-[65536]-[]=float:[21128, 768]
            ("unsortedsegmentsum_030", "unsortedsegmentsum_run", ([65536, 768], [65536], 21128, "float32")),
            # float-int32-int32:[65536, 768]-[65536]-[]=float:[2, 768]
            ("unsortedsegmentsum_031", "unsortedsegmentsum_run", ([65536, 768], [65536], 2, "float32")),
            # float-int32-int32:[80, 768]-[80]-[]=float:[512, 768]
            ("unsortedsegmentsum_032", "unsortedsegmentsum_run", ([80, 768], [80], 512, "float32")),
            # float-int32-int32:[8192, 768]-[8192]-[]=float:[21128, 768]
            ("unsortedsegmentsum_033", "unsortedsegmentsum_run", ([8192, 768], [8192], 21128, "float32")),
            # float-int32-int32:[8192, 768]-[8192]-[]=float:[2, 768]
            ("unsortedsegmentsum_034", "unsortedsegmentsum_run", ([8192, 768], [8192], 2, "float32")),

            # gelu OP
            # float32:[1280, 768]=float:[1280, 768]
            ("gelu_001", "gelu_run", ((1280, 768), "float32")),
            # float32:[160, 768]=float:[160, 768]
            ("gelu_002", "gelu_run", ((160, 768), "float32")),
            # float32:[16384, 3072]=float:[16384, 3072]
            ("gelu_003", "gelu_run", ((16384, 3072), "float32")),
            # float32:[2048, 3072]=float:[2048, 3072]
            ("gelu_004", "gelu_run", ((2048, 3072), "float32")),
            # float32:[2560, 768]=float:[2560, 768]
            ("gelu_005", "gelu_run", ((2560, 768), "float32")),
            # float32:[256, 3072]=float:[256, 3072]
            ("gelu_006", "gelu_run", ((256, 3072), "float32")),
            # float32:[320, 768]=float:[320, 768]
            ("gelu_007", "gelu_run", ((320, 768), "float32")),
            # float32:[40, 768]=float:[40, 768]
            ("gelu_008", "gelu_run", ((40, 768), "float32")),
            # float32:[4096, 3072]=float:[4096, 3072]
            ("gelu_009", "gelu_run", ((4096, 3072), "float32")),
            # float32:[512, 3072]=float:[512, 3072]
            ("gelu_010", "gelu_run", ((512, 3072), "float32")),
            # float32:[640, 768]=float:[640, 7682]
            ("gelu_011", "gelu_run", ((640, 768), "float32")),
            # float32:[80, 768]=float:[80, 768]
            ("gelu_012", "gelu_run", ((80, 768), "float32")),
            # float32:[8192, 3072]=float:[8192, 3072]
            ("gelu_013", "gelu_run", ((8192, 3072), "float32")),
            # float:[65536, 3072]=float:[65536, 3072]
            ("gelu_014", "gelu_run", ((65536, 3072), "float32")),
            # float:[10240, 768]=float:[10240, 768]
            ("gelu_015", "gelu_run", ((10240, 768), "float32")),
            # float:[128, 3072]=float:[128, 3072]
            ("gelu_016", "gelu_run", ((128, 3072), "float32")),
            # float:[131072, 3072]=float:[131072, 3072]
            ("gelu_017", "gelu_run", ((131072, 3072), "float32")),
            # float:[20480, 768]=float:[20480, 768]
            ("gelu_018", "gelu_run", ((20480, 768), "float32")),
            # float:[20, 768]=float:[20, 768]
            ("gelu_019", "gelu_run", ((20, 768), "float32")),
            # float:[32768, 3072]=float:[32768, 3072]
            ("gelu_020", "gelu_run", ((32768, 3072), "float32")),
            # float:[5120, 768]=float:[5120, 768]
            ("gelu_021", "gelu_run", ((5120, 768), "float32")),

            # gelu_grad OP
            # float32:[1024, 3072] = float:[1024, 3072]
            ("gelu_grad_001", "gelu_grad_run", ((1024, 3072), "float32")),
            # float32:[1280, 768]=float:[1280, 768]
            ("gelu_grad_002", "gelu_grad_run", ((1280, 768), "float32")),
            # float32:[160, 768]=float:[160, 768]
            ("gelu_grad_003", "gelu_grad_run", ((160, 768), "float32")),
            # float32:[16384, 3072]=float:[16384, 3072]
            ("gelu_grad_004", "gelu_grad_run", ((16384, 3072), "float32")),
            # float32:[2048, 3072]=float:[2048, 3072]
            ("gelu_grad_005", "gelu_grad_run", ((2048, 3072), "float32")),
            # float32:[2560, 768]=float:[2560, 768]
            ("gelu_grad_006", "gelu_grad_run", ((2560, 768), "float32")),
            # float32:[256, 3072]=float:[256, 3072]
            ("gelu_grad_007", "gelu_grad_run", ((256, 3072), "float32")),
            # float32:[320, 768]=float:[320, 768]
            ("gelu_grad_008", "gelu_grad_run", ((320, 768), "float32")),
            # float32:[40, 768]=float:[40, 768]
            ("gelu_grad_009", "gelu_grad_run", ((40, 768), "float32")),
            # float32:[4096, 3072]=float:[4096, 3072]
            ("gelu_grad_010", "gelu_grad_run", ((4096, 3072), "float32")),
            # float32:[512, 3072]=float:[512, 3072]
            ("gelu_grad_011", "gelu_grad_run", ((512, 3072), "float32")),
            # float32:[640, 768]=float:[640, 768]
            ("gelu_grad_012", "gelu_grad_run", ((640, 768), "float32")),
            # float32:[80, 768]=float:[80, 768]
            ("gelu_grad_013", "gelu_grad_run", ((80, 768), "float32")),
            # float32:[8192, 3072]=float:[8192, 3072]
            ("gelu_grad_014", "gelu_grad_run", ((8192, 3072), "float32")),
            # float:[10240, 768]=float:[10240, 768]
            ("gelu_grad_015", "gelu_grad_run", ((10240, 768), "float32")),
            # float:[128, 3072]=float:[128, 3072]
            ("gelu_grad_016", "gelu_grad_run", ((128, 3072), "float32")),
            # float:[131072, 3072]=float:[131072, 3072]
            ("gelu_grad_017", "gelu_grad_run", ((131072, 3072), "float32")),
            # float:[20480, 768]=float:[20480, 768]
            ("gelu_grad_018", "gelu_grad_run", ((20480, 768), "float32")),
            # float:[20, 768]=float:[20, 768]
            ("gelu_grad_019", "gelu_grad_run", ((20, 768), "float32")),
            # float:[32768, 3072]=float:[32768, 3072]
            ("gelu_grad_020", "gelu_grad_run", ((32768, 3072), "float32")),
            # float:[5120, 768]=float:[5120, 768]
            ("gelu_grad_021", "gelu_grad_run", ((5120, 768), "float32")),
            # float:[65536, 3072]=float:[65536, 3072]
            ("gelu_grad_022", "gelu_grad_run", ((65536, 3072), "float32")),

            # LayerNorm OP
            # float32:[1024, 768] = float32:[1024, 768]
            ("fused_layernorm_001", "fused_layernorm_run", ((1024, 768), -1, -1, "float32")),
            # float32:[1280, 768] = float32:[1280, 768]
            ("fused_layernorm_002", "fused_layernorm_run", ((1280, 768), -1, -1, "float32")),
            # float32:[128, 128, 768] = float32:[128, 128, 768]
            ("fused_layernorm_003", "fused_layernorm_run", ((128, 128, 768), -1, -1, "float32")),
            # float32:[160, 768] = float32:[160, 768]
            ("fused_layernorm_004", "fused_layernorm_run", ((160, 768), -1, -1, "float32")),
            # float32:[16, 128, 768] = float32:[16, 128, 768]
            ("fused_layernorm_005", "fused_layernorm_run", ((16, 128, 768), -1, -1, "float32")),
            # float32:[16384, 768] = float32:[16384, 768]
            ("fused_layernorm_006", "fused_layernorm_run", ((16384, 768), -1, -1, "float32")),
            # float32:[2048, 768] = float32:[2048, 768]
            ("fused_layernorm_007", "fused_layernorm_run", ((2048, 768), -1, -1, "float32")),
            # float32:[2, 128, 768] = float32:[2, 128, 768]
            ("fused_layernorm_008", "fused_layernorm_run", ((2, 128, 768), -1, -1, "float32")),
            # float32:[2560, 768] = float32:[2560, 768]
            ("fused_layernorm_009", "fused_layernorm_run", ((2560, 768), -1, -1, "float32")),
            # float32:[256, 768] = float32:[256, 768]
            ("fused_layernorm_010", "fused_layernorm_run", ((256, 768), -1, -1, "float32")),
            # float32:[320, 768] = float32:[320, 768]
            ("fused_layernorm_011", "fused_layernorm_run", ((320, 768), -1, -1, "float32")),
            # float32:[32, 128, 768] = float32:[32, 128, 768]
            ("fused_layernorm_012", "fused_layernorm_run", ((32, 128, 768), -1, -1, "float32")),
            # float32:[40, 768] = float32:[40, 768]
            ("fused_layernorm_013", "fused_layernorm_run", ((40, 768), -1, -1, "float32")),
            # float32:[4096, 768] = float32:[4096, 768]
            ("fused_layernorm_014", "fused_layernorm_run", ((4096, 768), -1, -1, "float32")),
            # float32:[4, 128, 768] = float32:[4, 128, 768]
            ("fused_layernorm_015", "fused_layernorm_run", ((4, 128, 768), -1, -1, "float32")),
            # float32:[512, 768] = float32:[512, 768]
            ("fused_layernorm_016", "fused_layernorm_run", ((512, 768), -1, -1, "float32")),
            # float32:[640, 768] = float32:[640, 768]
            ("fused_layernorm_017", "fused_layernorm_run", ((640, 768), -1, -1, "float32")),
            # float32:[64, 128, 768] = float32:[64, 128, 768]
            ("fused_layernorm_018", "fused_layernorm_run", ((64, 128, 768), -1, -1, "float32")),
            # float32:[80, 768] = float32:[80, 768]
            ("fused_layernorm_019", "fused_layernorm_run", ((80, 768), -1, -1, "float32")),
            # float32:[8, 128, 768] = float32:[8, 128, 768]
            ("fused_layernorm_020", "fused_layernorm_run", ((8, 128, 768), -1, -1, "float32")),
            # float32:[8192, 768] = float32:[8192, 768]
            ("fused_layernorm_021", "fused_layernorm_run", ((8192, 768), -1, -1, "float32")),
            # float:[512, 128, 768]=float:[512, 128, 768]
            ("fused_layernorm_022", "fused_layernorm_run", ((512, 128, 768), -1, -1, "float32")),
            # float:[65536, 768]=float:[65536, 768]
            ("fused_layernorm_023", "fused_layernorm_run", ((65536, 768), -1, -1, "float32")),
            # float:[1, 128, 768]=float:[1, 128, 768]
            ("fused_layernorm_024", "fused_layernorm_run", ((1, 128, 768), -1, -1, "float32")),
            # float:[128, 768]=float:[128, 768]
            ("fused_layernorm_025", "fused_layernorm_run", ((128, 768), -1, -1, "float32")),
            # float:[131072, 768]=float:[131072, 768]
            ("fused_layernorm_026", "fused_layernorm_run", ((131072, 768), -1, -1, "float32")),
            # float:[20480, 768]=float:[20480, 768]
            ("fused_layernorm_027", "fused_layernorm_run", ((20480, 768), -1, -1, "float32")),
            # float:[256, 128, 768]=float:[256, 128, 768]
            ("fused_layernorm_028", "fused_layernorm_run", ((256, 128, 768), -1, -1, "float32")),
            # float:[20, 768]=float:[20, 768]
            ("fused_layernorm_029", "fused_layernorm_run", ((20, 768), -1, -1, "float32")),
            # float:[32768, 768]=float:[32768, 768]
            ("fused_layernorm_030", "fused_layernorm_run", ((32768, 768), -1, -1, "float32")),
            # float:[5120, 768]=float:[5120, 768]
            ("fused_layernorm_031", "fused_layernorm_run", ((5120, 768), -1, -1, "float32")),

            # LayerNormGrad
            # float32:[1024, 768] = float32:[1024, 768]
            ("fused_layer_norm_grad_001", "fused_layer_norm_grad_run", ((1024, 768), -1, -1, "float32")),
            # float32:[1280, 768] = float32:[1280, 768]
            ("fused_layer_norm_grad_002", "fused_layer_norm_grad_run", ((1280, 768), -1, -1, "float32")),
            # float32:[128, 128, 768] = float32:[128, 128, 768]
            ("fused_layer_norm_grad_003", "fused_layer_norm_grad_run", ((128, 128), -1, -1, "float32")),
            # float32:[160, 768] = float32:[160, 768]
            ("fused_layer_norm_grad_004", "fused_layer_norm_grad_run", ((160, 768), -1, -1, "float32")),
            # float32:[16, 128, 768] = float32:[16, 128, 768]
            ("fused_layer_norm_grad_005", "fused_layer_norm_grad_run", ((16, 128, 768), -1, -1, "float32")),
            # float32:[16384, 768] = float32:[16384, 768]
            ("fused_layer_norm_grad_006", "fused_layer_norm_grad_run", ((16384, 768), -1, -1, "float32")),
            # float32:[2048, 768] = float32:[2048, 768]
            ("fused_layer_norm_grad_007", "fused_layer_norm_grad_run", ((2048, 768), -1, -1, "float32")),
            # float32:[2, 128, 768] = float32:[2, 128, 768]
            ("fused_layer_norm_grad_008", "fused_layer_norm_grad_run", ((2, 128), -1, -1, "float32")),
            # float32:[2560, 768] = float32:[2560, 768]
            ("fused_layer_norm_grad_009", "fused_layer_norm_grad_run", ((2560, 768), -1, -1, "float32")),
            # float32:[256, 768] = float32:[256, 768]
            ("fused_layer_norm_grad_010", "fused_layer_norm_grad_run", ((256, 768), -1, -1, "float32")),
            # float32:[320, 768] = float32:[320, 768]
            ("fused_layer_norm_grad_011", "fused_layer_norm_grad_run", ((320, 768), -1, -1, "float32")),
            # float32:[32, 128, 768] = float32:[32, 128, 768]
            ("fused_layer_norm_grad_012", "fused_layer_norm_grad_run", ((32, 128, 768), -1, -1, "float32")),
            # float32:[40, 768] = float32:[40, 768]
            ("fused_layer_norm_grad_013", "fused_layer_norm_grad_run", ((40, 768), -1, -1, "float32")),
            # float32:[4096, 768] = float32:[4096, 768]
            ("fused_layer_norm_grad_014", "fused_layer_norm_grad_run", ((4096, 768), -1, -1, "float32")),
            # float32:[4, 128, 768] = float32:[4, 128, 768]
            ("fused_layer_norm_grad_015", "fused_layer_norm_grad_run", ((4, 128), -1, -1, "float32")),
            # float32:[512, 768] = float32:[512, 768]
            ("fused_layer_norm_grad_016", "fused_layer_norm_grad_run", ((512, 768), -1, -1, "float32")),
            # float32:[640, 768] = float32:[640, 768]
            ("fused_layer_norm_grad_017", "fused_layer_norm_grad_run", ((640, 768), -1, -1, "float32")),
            # float32:[64, 128, 768] = float32:[64, 128, 768]
            ("fused_layer_norm_grad_018", "fused_layer_norm_grad_run", ((64, 128), -1, -1, "float32")),
            # float32:[80, 768] = float32:[80, 768]
            ("fused_layer_norm_grad_019", "fused_layer_norm_grad_run", ((80, 768), -1, -1, "float32")),
            # float32:[8, 128, 768] = float32:[8, 128, 768]
            ("fused_layer_norm_grad_020", "fused_layer_norm_grad_run", ((8, 128), -1, -1, "float32")),
            # float32:[8192, 768] = float32:[8192, 768]
            ("fused_layer_norm_grad_021", "fused_layer_norm_grad_run", ((8192, 768), -1, -1, "float32")),
            # float:[10240, 768]=float:[10240, 768]
            ("fused_layer_norm_grad_022", "fused_layer_norm_grad_run", ((10240, 768), -1, -1, "float32")),
            # float:[1024, 128, 768]=float:[1024, 128, 768]
            ("fused_layer_norm_grad_023", "fused_layer_norm_grad_run", ((1024, 128, 768), -1, -1, "float32")),
            # float:[1, 128, 768]=float:[1, 128, 768]
            ("fused_layer_norm_grad_024", "fused_layer_norm_grad_run", ((1, 128, 768), -1, -1, "float32")),
            # float:[128, 768]=float:[128, 768]
            ("fused_layer_norm_grad_025", "fused_layer_norm_grad_run", ((128, 768), -1, -1, "float32")),
            # float:[131072, 768]=float:[131072, 768]
            ("fused_layer_norm_grad_026", "fused_layer_norm_grad_run", ((131072, 768), -1, -1, "float32")),
            # float:[20480, 768]=float:[20480, 768]
            ("fused_layer_norm_grad_027", "fused_layer_norm_grad_run", ((20480, 768), -1, -1, "float32")),
            # float:[256, 128, 768]=float:[256, 128, 768]
            ("fused_layer_norm_grad_028", "fused_layer_norm_grad_run", ((256, 128, 768), -1, -1, "float32")),
            # float:[20, 768]=float:[20, 768]
            ("fused_layer_norm_grad_029", "fused_layer_norm_grad_run", ((20, 768), -1, -1, "float32")),
            # float:[32768, 768]=float:[32768, 768]
            ("fused_layer_norm_grad_030", "fused_layer_norm_grad_run", ((32768, 768), -1, -1, "float32")),
            # float:[5120, 768]=float:[5120, 768]
            ("fused_layer_norm_grad_031", "fused_layer_norm_grad_run", ((5120, 768), -1, -1, "float32")),
            # float:[512, 128, 768]=float:[512, 128, 768]
            ("fused_layer_norm_grad_032", "fused_layer_norm_grad_run", ((512, 128, 768), -1, -1, "float32")),
            # float:[65536, 768]=float:[65536, 768]
            ("fused_layer_norm_grad_033", "fused_layer_norm_grad_run", ((65536, 768), -1, -1, "float32")),

            # dropout OP
            # float32:[1024, 768] = float32:[1024, 768]
            ("dropout_001", "dropout_run", ((1024, 768), 1.0, "float32", "cce_dropout_do_mask")),
            # float32:[1280, 768] = float32:[1280, 768]
            ("dropout_002", "dropout_run", ((1280, 768), 1.0, "float32", "cce_dropout_do_mask")),
            # float32:[128, 128, 768] = float32:[128, 128, 768]
            ("dropout_003", "dropout_run", ((128, 128, 768), 1.0, "float32", "cce_dropout_do_mask")),
            # float32:[160, 768] = float32:[160, 768]
            ("dropout_004", "dropout_run", ((160, 768), 1.0, "float32", "cce_dropout_do_mask")),
            # float32:[16, 128, 768] = float32:[16, 128, 768]
            ("dropout_005", "dropout_run", ((16, 128, 768), 1.0, "float32", "cce_dropout_do_mask")),
            # float32:[16384, 768] = float32:[16384, 768]
            ("dropout_006", "dropout_run", ((16384, 768), 1.0, "float32", "cce_dropout_do_mask")),
            # float32:[2048, 768] = float32:[2048, 768]
            ("dropout_007", "dropout_run", ((2048, 768), 1.0, "float32", "cce_dropout_do_mask")),
            # float32:[2, 128, 768] = float32:[2, 128, 768]
            ("dropout_008", "dropout_run", ((2, 128, 768), 1.0, "float32", "cce_dropout_do_mask")),
            # float32:[2560, 768] = float32:[2560, 768]
            ("dropout_009", "dropout_run", ((2560, 768), 1.0, "float32", "cce_dropout_do_mask")),
            # float32:[256, 768] = float32:[256, 768]
            ("dropout_010", "dropout_run", ((256, 768), 1.0, "float32", "cce_dropout_do_mask")),
            # float32:[320, 768] = float32:[320, 768]
            ("dropout_011", "dropout_run", ((320, 768), 1.0, "float32", "cce_dropout_do_mask")),
            # float32:[32, 128, 768] = float32:[32, 128, 768]
            ("dropout_012", "dropout_run", ((32, 128, 768), 1.0, "float32", "cce_dropout_do_mask")),
            # float32:[40, 768] = float32:[40, 768]
            ("dropout_013", "dropout_run", ((40, 768), 1.0, "float32", "cce_dropout_do_mask")),
            # float32:[4096, 768] = float32:[4096, 768]
            ("dropout_014", "dropout_run", ((4096, 768), 1.0, "float32", "cce_dropout_do_mask")),
            # float32:[4, 128, 768] = float32:[4, 128, 768]
            ("dropout_015", "dropout_run", ((4, 128, 768), 1.0, "float32", "cce_dropout_do_mask")),
            # float32:[512, 768] = float32:[512, 768]
            ("dropout_016", "dropout_run", ((512, 768), 1.0, "float32", "cce_dropout_do_mask")),
            # float32:[640, 768] = float32:[640, 768]
            ("dropout_017", "dropout_run", ((640, 768), 1.0, "float32", "cce_dropout_do_mask")),
            # float32:[64, 128, 768] = float32:[64, 128, 768]
            ("dropout_018", "dropout_run", ((64, 128, 768), 1.0, "float32", "cce_dropout_do_mask")),
            # float32:[80, 768] = float32:[80, 768]
            ("dropout_019", "dropout_run", ((80, 768), 1.0, "float32", "cce_dropout_do_mask")),
            # float32:[8, 128, 768] = float32:[8, 128, 768]
            ("dropout_020", "dropout_run", ((8, 128, 768), 1.0, "float32", "cce_dropout_do_mask")),
            # float32:[8192, 768] = float32:[8192, 768]
            ("dropout_021", "dropout_run", ((8192, 768), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[1024, 12, 128, 128]=float:[1024, 12, 128, 128]
            ("dropout_022", "dropout_run", ((1024, 12, 128, 128), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[1024, 128, 768]=float:[1024, 128, 768]
            ("dropout_023", "dropout_run", ((1024, 128, 768), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[131072, 768]=float:[131072, 768]
            ("dropout_024", "dropout_run", ((131072, 768), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[256, 12, 128, 128]=float:[256, 12, 128, 128]
            ("dropout_025", "dropout_run", ((256, 12, 128, 128), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[256, 128, 768]=float:[256, 128, 768]
            ("dropout_026", "dropout_run", ((256, 128, 768), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[32768, 768]=float:[32768, 768]
            ("dropout_027", "dropout_run", ((32768, 768), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[512, 12, 128, 128]=float:[512, 12, 128, 128]
            ("dropout_028", "dropout_run", ((512, 12, 128, 128), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[512, 128, 768]=float:[512, 128, 768]
            ("dropout_029", "dropout_run", ((512, 128, 768), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[65536, 768]=float:[65536, 768]
            ("dropout_030", "dropout_run", ((65536, 768), 1.0, "float32", "cce_dropout_do_mask")),

            # addn OP
            # float-float:[1024, 12, 128, 128]-[1024, 12, 128, 128]=float:[1024, 12, 128, 128]
            ("addn_001", "addn_run", ((1024, 12, 128, 128), "float32", 2)),
            # float-float:[1024, 12, 128, 64]-[1024, 12, 128, 64]=float:[1024, 12, 128, 64]
            ("addn_002", "addn_run", ((1024, 12, 128, 64), "float32", 2)),
            # float-float:[1024, 128, 768]-[1024, 128, 768]=float:[1024, 128, 768]
            ("addn_003", "addn_run", ((1024, 128, 768), "float32", 2)),
            # float-float:[1024, 768]-[1024, 768]=float:[1024, 768]
            ("addn_004", "addn_run", ((1024, 768), "float32", 2)),
            # float-float:[1, 12, 128, 128]-[1, 12, 128, 128]=float:[1, 12, 128, 128]
            ("addn_005", "addn_run", ((1, 12, 128, 128), "float32", 2)),
            # float-float:[1, 12, 128, 64]-[1, 12, 128, 64]=float:[1, 12, 128, 64]
            ("addn_006", "addn_run", ((1, 12, 128, 64), "float32", 2)),
            # float-float:[1, 128, 768]-[1, 128, 768]=float:[1, 128, 768]
            ("addn_007", "addn_run", ((1, 128, 768), "float32", 2)),
            # float-float:[128, 12, 128, 128]-[128, 12, 128, 128]=float:[128, 12, 128, 128]
            ("addn_008", "addn_run", ((128, 12, 128, 128), "float32", 2)),
            # float-float:[128, 12, 128, 64]-[128, 12, 128, 64]=float:[128, 12, 128, 64]
            ("addn_009", "addn_run", ((128, 12, 128, 64), "float32", 2)),
            # float-float:[128, 128, 768]-[128, 128, 768]=float:[128, 128, 768]
            ("addn_010", "addn_run", ((128, 128, 768), "float32", 2)),
            # float-float:[128, 768]-[128, 768]=float:[128, 768]
            ("addn_011", "addn_run", ((128, 768), "float32", 2)),
            # float-float:[131072, 768]-[131072, 768]=float:[131072, 768]
            ("addn_012", "addn_run", ((131072, 768), "float32", 2)),
            # float-float:[16, 12, 128, 128]-[16, 12, 128, 128]=float:[16, 12, 128, 128]
            ("addn_013", "addn_run", ((16, 12, 128, 128), "float32", 2)),
            # float-float:[16, 12, 128, 64]-[16, 12, 128, 64]=float:[16, 12, 128, 64]
            ("addn_014", "addn_run", ((16, 12, 128, 64), "float32", 2)),
            # float-float:[16, 128, 768]-[16, 128, 768]=float:[16, 128, 768]
            ("addn_015", "addn_run", ((16, 128, 768), "float32", 2)),
            # float-float:[16384, 768]-[16384, 768]=float:[16384, 768]
            ("addn_016", "addn_run", ((16384, 768), "float32", 2)),
            # float-float:[2048, 768]-[2048, 768]=float:[2048, 768]
            ("addn_017", "addn_run", ((2048, 768), "float32", 2)),
            # float-float:[21128, 768]-[21128, 768]=float:[21128, 768]
            ("addn_018", "addn_run", ((21128, 768), "float32", 2)),
            # float-float:[2, 12, 128, 128]-[2, 12, 128, 128]=float:[2, 12, 128, 128]
            ("addn_019", "addn_run", ((2, 12, 128, 128), "float32", 2)),
            # float-float:[2, 12, 128, 64]-[2, 12, 128, 64]=float:[2, 12, 128, 64]
            ("addn_020", "addn_run", ((2, 12, 128, 64), "float32", 2)),
            # float-float:[2, 128, 768]-[2, 128, 768]=float:[2, 128, 768]
            ("addn_021", "addn_run", ((2, 128, 768), "float32", 2)),
            # float-float:[256, 12, 128, 128]-[256, 12, 128, 128]=float:[256, 12, 128, 128]
            ("addn_022", "addn_run", ((256, 12, 128, 128), "float32", 2)),
            # float-float:[256, 12, 128, 64]-[256, 12, 128, 64]=float:[256, 12, 128, 64]
            ("addn_023", "addn_run", ((256, 12, 128, 64), "float32", 2)),
            # float-float:[256, 128, 768]-[256, 128, 768]=float:[256, 128, 768]
            ("addn_024", "addn_run", ((256, 128, 768), "float32", 2)),
            # float-float:[256, 768]-[256, 768]=float:[256, 768]
            ("addn_025", "addn_run", ((256, 768), "float32", 2)),
            # float-float:[32, 12, 128, 128]-[32, 12, 128, 128]=float:[32, 12, 128, 128]
            ("addn_026", "addn_run", ((32, 12, 128, 128), "float32", 2)),
            # float-float:[32, 12, 128, 64]-[32, 12, 128, 64]=float:[32, 12, 128, 64]
            ("addn_027", "addn_run", ((32, 12, 128, 64), "float32", 2)),
            # float-float:[32, 128, 768]-[32, 128, 768]=float:[32, 128, 768]
            ("addn_028", "addn_run", ((32, 128, 768), "float32", 2)),
            # float-float:[32768, 768]-[32768, 768]=float:[32768, 768]
            ("addn_029", "addn_run", ((32768, 768), "float32", 2)),
            # float-float:[33, 64]-[33, 64]=float:[33, 64]
            ("addn_030", "addn_run", ((33, 64), "float32", 2)),
            # float-float:[4096, 768]-[4096, 768]=float:[4096, 768]
            ("addn_031", "addn_run", ((4096, 768), "float32", 2)),
            # float-float:[4, 12, 128, 128]-[4, 12, 128, 128]=float:[4, 12, 128, 128]
            ("addn_032", "addn_run", ((4, 12, 128, 128), "float32", 2)),
            # float-float:[4, 12, 128, 64]-[4, 12, 128, 64]=float:[4, 12, 128, 64]
            ("addn_033", "addn_run", ((4, 12, 128, 64), "float32", 2)),
            # float-float:[4, 128, 768]-[4, 128, 768]=float:[4, 128, 768]
            ("addn_034", "addn_run", ((4, 128, 768), "float32", 2)),
            # float-float:[512, 12, 128, 128]-[512, 12, 128, 128]=float:[512, 12, 128, 128]
            ("addn_035", "addn_run", ((512, 12, 128, 128), "float32", 2)),
            # float-float:[512, 12, 128, 64]-[512, 12, 128, 64]=float:[512, 12, 128, 64]
            ("addn_036", "addn_run", ((512, 12, 128, 64), "float32", 2)),
            # float-float:[512, 128, 768]-[512, 128, 768]=float:[512, 128, 768]
            ("addn_037", "addn_run", ((512, 128, 768), "float32", 2)),
            # float-float:[512, 768]-[512, 768]=float:[512, 768]
            ("addn_038", "addn_run", ((512, 768), "float32", 2)),
            # float-float:[64, 12, 128, 128]-[64, 12, 128, 128]=float:[64, 12, 128, 128]
            ("addn_039", "addn_run", ((64, 12, 128, 128), "float32", 2)),
            # float-float:[64, 12, 128, 64]-[64, 12, 128, 64]=float:[64, 12, 128, 64]
            ("addn_040", "addn_run", ((64, 12, 128, 64), "float32", 2)),
            # float-float:[64, 128, 768]-[64, 128, 768]=float:[64, 128, 768]
            ("addn_041", "addn_run", ((64, 128, 768), "float32", 2)),
            # float-float:[65536, 768]-[65536, 768]=float:[65536, 768]
            ("addn_042", "addn_run", ((65536, 768), "float32", 2)),
            # float-float:[8, 12, 128, 128]-[8, 12, 128, 128]=float:[8, 12, 128, 128]
            ("addn_043", "addn_run", ((8, 12, 128, 128), "float32", 2)),
            # float-float:[8, 12, 128, 64]-[8, 12, 128, 64]=float:[8, 12, 128, 64]
            ("addn_044", "addn_run", ((8, 12, 128, 64), "float32", 2)),
            # float-float:[8, 128, 768]-[8, 128, 768]=float:[8, 128, 768]
            ("addn_045", "addn_run", ((8, 128, 768), "float32", 2)),
            # float-float:[8192, 768]-[8192, 768]=float:[8192, 768]
            ("addn_046", "addn_run", ((8192, 768), "float32", 2)),
            # float-float-float:[1024, 768]-[1024, 768]-[1024, 768]=float:[1024, 768]
            ("addn_047", "addn_run", ((1024, 768), "float32", 3)),
            # float-float-float:[128, 768]-[128, 768]-[128, 768]=float:[128, 768]
            ("addn_048", "addn_run", ((128, 768), "float32", 3)),
            # float-float-float:[131072, 768]-[131072, 768]-[131072, 768]=float:[131072, 768]
            ("addn_049", "addn_run", ((131072, 768), "float32", 3)),
            # float-float-float:[16384, 768]-[16384, 768]-[16384, 768]=float:[16384, 768]
            ("addn_050", "addn_run", ((16384, 768), "float32", 3)),
            # float-float-float:[2048, 768]-[2048, 768]-[2048, 768]=float:[2048, 768]
            ("addn_051", "addn_run", ((2048, 768), "float32", 3)),
            # float-float-float:[256, 768]-[256, 768]-[256, 768]=float:[256, 768]
            ("addn_052", "addn_run", ((256, 768), "float32", 3)),
            # float-float-float:[32768, 768]-[32768, 768]-[32768, 768]=float:[32768, 768]
            ("addn_053", "addn_run", ((32768, 768), "float32", 3)),
            # float-float-float:[4096, 768]-[4096, 768]-[4096, 768]=float:[4096, 768]
            ("addn_054", "addn_run", ((4096, 768), "float32", 3)),
            # float-float-float:[512, 768]-[512, 768]-[512, 768]=float:[512, 768]
            ("addn_055", "addn_run", ((512, 768), "float32", 3)),
            # float-float-float:[65536, 768]-[65536, 768]-[65536, 768]=float:[65536, 768]
            ("addn_056", "addn_run", ((65536, 768), "float32", 3)),
            # float-float-float:[8192, 768]-[8192, 768]-[8192, 768]=float:[8192, 768]
            ("addn_057", "addn_run", ((8192, 768), "float32", 3)),

            # LogSoftMax OP
            # float:[128, 2]=float:[128, 2]
            ("logsoftmax_001", "logsoftmax_run", ((128, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[160, 21128]=float:[160, 21128]
            ("logsoftmax_002", "logsoftmax_run", ((160, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[16, 2]=float:[16, 2]
            ("logsoftmax_003", "logsoftmax_run", ((16, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[20, 21128]=float:[20, 21128]
            ("logsoftmax_004", "logsoftmax_run", ((20, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[2, 2]=float:[2, 2]
            ("logsoftmax_005", "logsoftmax_run", ((2, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[2560, 21128]=float:[2560, 21128]
            ("logsoftmax_006", "logsoftmax_run", ((2560, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[320, 21128]=float:[320, 21128]
            ("logsoftmax_007", "logsoftmax_run", ((320, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[32, 2]=float:[32, 2]
            ("logsoftmax_008", "logsoftmax_run", ((32, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[40, 21128]=float:[40, 21128]
            ("logsoftmax_009", "logsoftmax_run", ((40, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[4, 2]=float:[4, 2]
            ("logsoftmax_010", "logsoftmax_run", ((4, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[640, 21128]=float:[640, 21128]
            ("logsoftmax_011", "logsoftmax_run", ((640, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[80, 21128]=float:[80, 21128]
            ("logsoftmax_012", "logsoftmax_run", ((80, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[10240, 21128]=float:[10240, 21128]
            ("logsoftmax_013", "logsoftmax_run", ((10240, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[1024, 2]=float:[1024, 2]
            ("logsoftmax_014", "logsoftmax_run", ((1024, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[1280, 21128]=float:[1280, 21128]
            ("logsoftmax_015", "logsoftmax_run", ((1280, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[1, 2]=float:[1, 2]
            ("logsoftmax_016", "logsoftmax_run", ((1, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[20480, 21128]=float:[20480, 21128]
            ("logsoftmax_017", "logsoftmax_run", ((20480, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[256, 2]=float:[256, 2]
            ("logsoftmax_018", "logsoftmax_run", ((256, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[5120, 21128]=float:[5120, 21128]
            ("logsoftmax_019", "logsoftmax_run", ((5120, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[512, 2]=float:[512, 2]
            ("logsoftmax_020", "logsoftmax_run", ((512, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[64, 2]=float:[64, 2]
            ("logsoftmax_021", "logsoftmax_run", ((64, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[8, 2]=float:[8, 2]
            ("logsoftmax_022", "logsoftmax_run", ((8, 2), "float32", -1, "cce_logsoftmax_fp32")),

            # LogSoftMaxGrad OP
            # float:[128, 2]=float:[128, 2]
            ("logsoftmax_grad_001", "logsoftmax_grad_run", ((128, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[160, 21128]=float:[160, 21128]
            ("logsoftmax_grad_002", "logsoftmax_grad_run", ((160, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[16, 2]=float:[16, 2]
            ("logsoftmax_grad_003", "logsoftmax_grad_run", ((16, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[20, 21128]=float:[20, 21128]
            ("logsoftmax_grad_004", "logsoftmax_grad_run", ((20, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[2, 2]=float:[2, 2]
            ("logsoftmax_grad_005", "logsoftmax_grad_run", ((2, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[2560, 21128]=float:[2560, 21128]
            ("logsoftmax_grad_006", "logsoftmax_grad_run", ((2560, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[320, 21128]=float:[320, 21128]
            ("logsoftmax_grad_007", "logsoftmax_grad_run", ((320, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[32, 2]=float:[32, 2]
            ("logsoftmax_grad_008", "logsoftmax_grad_run", ((32, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[40, 21128]=float:[40, 21128]
            ("logsoftmax_grad_009", "logsoftmax_grad_run", ((40, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[4, 2]=float:[4, 2]
            ("logsoftmax_grad_010", "logsoftmax_grad_run", ((4, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[640, 21128]=float:[640, 21128]
            ("logsoftmax_grad_011", "logsoftmax_grad_run", ((640, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[80, 21128]=float:[80, 21128]
            ("logsoftmax_grad_012", "logsoftmax_grad_run", ((80, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[10240, 21128]=float:[10240, 21128]
            ("logsoftmax_grad_013", "logsoftmax_grad_run", ((10240, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[1024, 2]=float:[1024, 2]
            ("logsoftmax_grad_014", "logsoftmax_grad_run", ((1024, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[1280, 21128]=float:[1280, 21128]
            ("logsoftmax_grad_015", "logsoftmax_grad_run", ((1280, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[1, 2]=float:[1, 2]
            ("logsoftmax_grad_016", "logsoftmax_grad_run", ((1, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[20480, 21128]=float:[20480, 21128]
            ("logsoftmax_grad_017", "logsoftmax_grad_run", ((20480, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[256, 2]=float:[256, 2]
            ("logsoftmax_grad_018", "logsoftmax_grad_run", ((256, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[5120, 21128]=float:[5120, 21128]
            ("logsoftmax_grad_019", "logsoftmax_grad_run", ((5120, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[512, 2]=float:[512, 2]
            ("logsoftmax_grad_020", "logsoftmax_grad_run", ((512, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[64, 2]=float:[64, 2]
            ("logsoftmax_grad_021", "logsoftmax_grad_run", ((64, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[8, 2]=float:[8, 2]
            ("logsoftmax_grad_022", "logsoftmax_grad_run", ((8, 2), "float32", -1, "cce_logsoftmax_fp32")),

            # matmul op
            # float - float:[1024, 768] - [768, 3072] = float:[1024, 3072]
            ("matmul_0001", "batchmatmul_run",
             ((), 1024, 3072, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[1024, 3072] - [3072, 768] = float:[1024, 768]
            ("matmul_0002", "batchmatmul_run",
             ((), 1024, 768, 3072, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[1024, 768] - [768, 768] = float:[1024, 768]
            (
                "matmul_0003", "batchmatmul_run",
                ((), 1024, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[1, 128, 1] - [1, 1, 128] = float:[1, 128, 128]
            ("matmul_0004", "batchmatmul_run", ((1,), 128, 128, 1, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[128, 12, 128, 64] - [128, 12, 128, 64] = float:[128, 12, 128, 128]
            ("matmul_0005", "batchmatmul_run",
             ((128, 12), 128, 128, 128, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[128, 12, 128, 128] - [128, 12, 128, 64] = float:[128, 12, 128, 64]
            ("matmul_0006", "batchmatmul_run",
             ((128, 12), 128, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[128, 12, 64] - [128, 128, 64] = float:[128, 12, 128]
            (
                "matmul_0007", "batchmatmul_run",
                ((128,), 12, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[128, 12, 128] - [128, 128, 64] = float:[128, 12, 64]
            ("matmul_0008", "batchmatmul_run",
             ((128,), 12, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[128, 128, 1] - [128, 1, 128] = float:[128, 128, 128]
            ("matmul_0009", "batchmatmul_run",
             ((128,), 128, 128, 1, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[128, 12, 128] - [128, 12, 64] = float:[128, 128, 64]
            (
                "matmul_0010", "batchmatmul_run",
                ((128,), 128, 64, 12, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[128, 1536, 128] - [128, 1536, 64] = float:[128, 128, 64]
            ("matmul_0011", "batchmatmul_run",
             ((128,), 128, 64, 1536, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[128, 192, 128] - [128, 192, 64] = float:[128, 128, 64]
            ("matmul_0012", "batchmatmul_run",
             ((128,), 128, 64, 192, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[128, 24, 128] - [128, 24, 64] = float:[128, 128, 64]
            (
                "matmul_0013", "batchmatmul_run",
                ((128,), 128, 64, 24, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[128, 384, 128] - [128, 384, 64] = float:[128, 128, 64]
            ("matmul_0014", "batchmatmul_run",
             ((128,), 128, 64, 384, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[128, 48, 128] - [128, 48, 64] = float:[128, 128, 64]
            (
                "matmul_0015", "batchmatmul_run",
                ((128,), 128, 64, 48, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[128, 768, 128] - [128, 768, 64] = float:[128, 128, 64]
            ("matmul_0016", "batchmatmul_run",
             ((128,), 128, 64, 768, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[128, 96, 128] - [128, 96, 64] = float:[128, 128, 64]
            (
                "matmul_0017", "batchmatmul_run",
                ((128,), 128, 64, 96, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[128, 1536, 64] - [128, 128, 64] = float:[128, 1536, 128]
            ("matmul_0018", "batchmatmul_run",
             ((128,), 1536, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[128, 1536, 128] - [128, 128, 64] = float:[128, 1536, 64]
            (
                "matmul_0019", "batchmatmul_run",
                ((128,), 1536, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[128, 192, 64] - [128, 128, 64] = float:[128, 192, 128]
            ("matmul_0020", "batchmatmul_run",
             ((128,), 192, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[128, 192, 128] - [128, 128, 64] = float:[128, 192, 64]
            ("matmul_0021", "batchmatmul_run",
             ((128,), 192, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[128, 24, 64] - [128, 128, 64] = float:[128, 24, 128]
            (
                "matmul_0022", "batchmatmul_run",
                ((128,), 24, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[128, 24, 128] - [128, 128, 64] = float:[128, 24, 64]
            ("matmul_0023", "batchmatmul_run",
             ((128,), 24, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[128, 768] - [2, 768] = float:[128, 2]
            ("matmul_0024", "batchmatmul_run", ((), 128, 2, 768, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[128, 384, 64] - [128, 128, 64] = float:[128, 384, 128]
            ("matmul_0025", "batchmatmul_run",
             ((128,), 384, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[128, 384, 128] - [128, 128, 64] = float:[128, 384, 64]
            ("matmul_0026", "batchmatmul_run",
             ((128,), 384, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[128, 48, 64] - [128, 128, 64] = float:[128, 48, 128]
            (
                "matmul_0027", "batchmatmul_run",
                ((128,), 48, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[128, 48, 128] - [128, 128, 64] = float:[128, 48, 64]
            ("matmul_0028", "batchmatmul_run",
             ((128,), 48, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[128, 768, 64] - [128, 128, 64] = float:[128, 768, 128]
            ("matmul_0029", "batchmatmul_run",
             ((128,), 768, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[128, 768, 128] - [128, 128, 64] = float:[128, 768, 64]
            ("matmul_0030", "batchmatmul_run",
             ((128,), 768, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[128, 2] - [2, 768] = float:[128, 768]
            ("matmul_0031", "batchmatmul_run", ((), 128, 768, 2, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[128, 3072] - [3072, 768] = float:[128, 768]
            (
                "matmul_0032", "batchmatmul_run",
                ((), 128, 768, 3072, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[128, 768] - [768, 768] = float:[128, 768]
            ("matmul_0033", "batchmatmul_run", ((), 128, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[128, 96, 64] - [128, 128, 64] = float:[128, 96, 128]
            (
                "matmul_0034", "batchmatmul_run",
                ((128,), 96, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[128, 96, 128] - [128, 128, 64] = float:[128, 96, 64]
            ("matmul_0035", "batchmatmul_run",
             ((128,), 96, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[160, 768] - [21128, 768] = float:[160, 21128]
            (
                "matmul_0036", "batchmatmul_run",
                ((), 160, 21128, 768, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[160, 21128] - [21128, 768] = float:[160, 768]
            ("matmul_0037", "batchmatmul_run",
             ((), 160, 768, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[160, 768] - [768, 768] = float:[160, 768]
            ("matmul_0038", "batchmatmul_run", ((), 160, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[16, 12, 128, 64] - [16, 12, 128, 64] = float:[16, 12, 128, 128]
            ("matmul_0039", "batchmatmul_run",
             ((16, 12), 128, 128, 128, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[16, 12, 128, 128] - [16, 12, 128, 64] = float:[16, 12, 128, 64]
            ("matmul_0040", "batchmatmul_run",
             ((16, 12), 128, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[16, 128, 1] - [16, 1, 128] = float:[16, 128, 128]
            (
                "matmul_0041", "batchmatmul_run",
                ((16,), 128, 128, 1, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[16, 768] - [2, 768] = float:[16, 2]
            ("matmul_0042", "batchmatmul_run", ((), 16, 2, 768, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[16384, 768] - [768, 3072] = float:[16384, 3072]
            ("matmul_0043", "batchmatmul_run",
             ((), 16384, 3072, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[16384, 3072] - [3072, 768] = float:[16384, 768]
            ("matmul_0044", "batchmatmul_run",
             ((), 16384, 768, 3072, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[16384, 768] - [768, 768] = float:[16384, 768]
            ("matmul_0045", "batchmatmul_run",
             ((), 16384, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[16, 2] - [2, 768] = float:[16, 768]
            ("matmul_0046", "batchmatmul_run", ((), 16, 768, 2, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[16, 768] - [768, 768] = float:[16, 768]
            ("matmul_0047", "batchmatmul_run", ((), 16, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[1, 2] - [2, 768] = float:[1, 768]
            ("matmul_0048", "batchmatmul_run", ((), 1, 768, 2, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[1, 768] - [768, 768] = float:[1, 768]
            ("matmul_0049", "batchmatmul_run", ((), 1, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[20, 768] - [21128, 768] = float:[20, 21128]
            ("matmul_0050", "batchmatmul_run", ((), 20, 21128, 768, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[2048, 768] - [768, 3072] = float:[2048, 3072]
            ("matmul_0051", "batchmatmul_run",
             ((), 2048, 3072, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[2048, 3072] - [3072, 768] = float:[2048, 768]
            ("matmul_0052", "batchmatmul_run",
             ((), 2048, 768, 3072, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[2048, 768] - [768, 768] = float:[2048, 768]
            (
                "matmul_0053", "batchmatmul_run",
                ((), 2048, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[20, 21128] - [21128, 768] = float:[20, 768]
            (
                "matmul_0054", "batchmatmul_run",
                ((), 20, 768, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[20, 768] - [768, 768] = float:[20, 768]
            ("matmul_0055", "batchmatmul_run", ((), 20, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[160, 21128] - [160, 768] = float:[21128, 768]
            ("matmul_0056", "batchmatmul_run",
             ((), 21128, 768, 21128, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[20, 21128] - [20, 768] = float:[21128, 768]
            ("matmul_0057", "batchmatmul_run",
             ((), 21128, 768, 21128, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[2560, 21128] - [2560, 768] = float:[21128, 768]
            ("matmul_0058", "batchmatmul_run",
             ((), 21128, 768, 21128, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[320, 21128] - [320, 768] = float:[21128, 768]
            ("matmul_0059", "batchmatmul_run",
             ((), 21128, 768, 21128, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[40, 21128] - [40, 768] = float:[21128, 768]
            ("matmul_0060", "batchmatmul_run",
             ((), 21128, 768, 21128, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[640, 21128] - [640, 768] = float:[21128, 768]
            ("matmul_0061", "batchmatmul_run",
             ((), 21128, 768, 21128, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[80, 21128] - [80, 768] = float:[21128, 768]
            ("matmul_0062", "batchmatmul_run",
             ((), 21128, 768, 21128, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[2, 12, 128, 64] - [2, 12, 128, 64] = float:[2, 12, 128, 128]
            ("matmul_0063", "batchmatmul_run",
             ((2, 12), 128, 128, 128, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[2, 12, 128, 128] - [2, 12, 128, 64] = float:[2, 12, 128, 64]
            ("matmul_0064", "batchmatmul_run",
             ((2, 12), 128, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[2, 128, 1] - [2, 1, 128] = float:[2, 128, 128]
            ("matmul_0065", "batchmatmul_run", ((2,), 128, 128, 1, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[2, 768] - [2, 768] = float:[2, 2]
            ("matmul_0066", "batchmatmul_run", ((), 2, 2, 768, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[2560, 768] - [21128, 768] = float:[2560, 21128]
            ("matmul_0067", "batchmatmul_run",
             ((), 2560, 21128, 768, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[2560, 21128] - [21128, 768] = float:[2560, 768]
            ("matmul_0068", "batchmatmul_run",
             ((), 2560, 768, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[2560, 768] - [768, 768] = float:[2560, 768]
            (
                "matmul_0069", "batchmatmul_run",
                ((), 2560, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[256, 768] - [768, 3072] = float:[256, 3072]
            (
                "matmul_0070", "batchmatmul_run",
                ((), 256, 3072, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[256, 3072] - [3072, 768] = float:[256, 768]
            (
                "matmul_0071", "batchmatmul_run",
                ((), 256, 768, 3072, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[256, 768] - [768, 768] = float:[256, 768]
            ("matmul_0072", "batchmatmul_run", ((), 256, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[2, 2] - [2, 768] = float:[2, 768]
            ("matmul_0073", "batchmatmul_run", ((), 2, 768, 2, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[2, 768] - [768, 768] = float:[2, 768]
            ("matmul_0074", "batchmatmul_run", ((), 2, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[128, 2] - [128, 768] = float:[2, 768]
            ("matmul_0075", "batchmatmul_run", ((), 2, 768, 2, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[16, 2] - [16, 768] = float:[2, 768]
            ("matmul_0076", "batchmatmul_run", ((), 2, 768, 2, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[32, 2] - [32, 768] = float:[2, 768]
            ("matmul_0077", "batchmatmul_run", ((), 2, 768, 2, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[4, 2] - [4, 768] = float:[2, 768]
            ("matmul_0078", "batchmatmul_run", ((), 2, 768, 2, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[8, 2] - [8, 768] = float:[2, 768]
            ("matmul_0079", "batchmatmul_run", ((), 2, 768, 2, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[320, 768] - [21128, 768] = float:[320, 21128]
            (
                "matmul_0080", "batchmatmul_run",
                ((), 320, 21128, 768, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[320, 21128] - [21128, 768] = float:[320, 768]
            ("matmul_0081", "batchmatmul_run",
             ((), 320, 768, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[320, 768] - [768, 768] = float:[320, 768]
            ("matmul_0082", "batchmatmul_run", ((), 320, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[32, 12, 128, 64] - [32, 12, 128, 64] = float:[32, 12, 128, 128]
            ("matmul_0083", "batchmatmul_run",
             ((32, 12), 128, 128, 128, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[32, 12, 128, 128] - [32, 12, 128, 64] = float:[32, 12, 128, 64]
            ("matmul_0084", "batchmatmul_run",
             ((32, 12), 128, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[32, 128, 1] - [32, 1, 128] = float:[32, 128, 128]
            (
                "matmul_0085", "batchmatmul_run",
                ((32,), 128, 128, 1, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[32, 768] - [2, 768] = float:[32, 2]
            ("matmul_0086", "batchmatmul_run", ((), 32, 2, 768, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[32, 2] - [2, 768] = float:[32, 768]
            ("matmul_0087", "batchmatmul_run", ((), 32, 768, 2, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[32, 768] - [768, 768] = float:[32, 768]
            ("matmul_0088", "batchmatmul_run", ((), 32, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[40, 768] - [21128, 768] = float:[40, 21128]
            ("matmul_0089", "batchmatmul_run", ((), 40, 21128, 768, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[40, 21128] - [21128, 768] = float:[40, 768]
            (
                "matmul_0090", "batchmatmul_run",
                ((), 40, 768, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[40, 768] - [768, 768] = float:[40, 768]
            ("matmul_0091", "batchmatmul_run", ((), 40, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[4096, 768] - [768, 3072] = float:[4096, 3072]
            ("matmul_0092", "batchmatmul_run",
             ((), 4096, 3072, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[4096, 3072] - [3072, 768] = float:[4096, 768]
            ("matmul_0093", "batchmatmul_run",
             ((), 4096, 768, 3072, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[4096, 768] - [768, 768] = float:[4096, 768]
            (
                "matmul_0094", "batchmatmul_run",
                ((), 4096, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[4, 12, 128, 64] - [4, 12, 128, 64] = float:[4, 12, 128, 128]
            ("matmul_0095", "batchmatmul_run",
             ((4, 12), 128, 128, 128, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[4, 12, 128, 128] - [4, 12, 128, 64] = float:[4, 12, 128, 64]
            ("matmul_0096", "batchmatmul_run",
             ((4, 12), 128, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[4, 128, 1] - [4, 1, 128] = float:[4, 128, 128]
            ("matmul_0097", "batchmatmul_run", ((4,), 128, 128, 1, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[4, 768] - [2, 768] = float:[4, 2]
            ("matmul_0098", "batchmatmul_run", ((), 4, 2, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[4, 2] - [2, 768] = float:[4, 768]
            ("matmul_0099", "batchmatmul_run", ((), 4, 768, 2, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[4, 768] - [768, 768] = float:[4, 768]
            ("matmul_0100", "batchmatmul_run", ((), 4, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[512, 768] - [768, 3072] = float:[512, 3072]
            (
                "matmul_0101", "batchmatmul_run",
                ((), 512, 3072, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[512, 3072] - [3072, 768] = float:[512, 768]
            (
                "matmul_0102", "batchmatmul_run",
                ((), 512, 768, 3072, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[512, 768] - [768, 768] = float:[512, 768]
            ("matmul_0103", "batchmatmul_run", ((), 512, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[640, 768] - [21128, 768] = float:[640, 21128]
            ("matmul_0104", "batchmatmul_run",
             ((), 640, 21128, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[640, 21128] - [21128, 768] = float:[640, 768]
            ("matmul_0105", "batchmatmul_run",
             ((), 640, 768, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[640, 768] - [768, 768] = float:[640, 768]
            ("matmul_0106", "batchmatmul_run", ((), 640, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[64, 12, 128, 64] - [64, 12, 128, 64] = float:[64, 12, 128, 128]
            ("matmul_0107", "batchmatmul_run",
             ((64, 12), 128, 128, 128, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[64, 12, 128, 128] - [64, 12, 128, 64] = float:[64, 12, 128, 64]
            ("matmul_0108", "batchmatmul_run",
             ((64, 12), 128, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[64, 128, 1] - [64, 1, 128] = float:[64, 128, 128]
            (
                "matmul_0109", "batchmatmul_run",
                ((64,), 128, 128, 1, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[80, 768] - [21128, 768] = float:[80, 21128]
            (
                "matmul_0110", "batchmatmul_run",
                ((), 80, 21128, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[80, 21128] - [21128, 768] = float:[80, 768]
            (
                "matmul_0111", "batchmatmul_run",
                ((), 80, 768, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[80, 768] - [768, 768] = float:[80, 768]
            ("matmul_0112", "batchmatmul_run", ((), 80, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[8, 12, 128, 64] - [8, 12, 128, 64] = float:[8, 12, 128, 128]
            ("matmul_0113", "batchmatmul_run",
             ((8, 12), 128, 128, 128, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[8, 12, 128, 128] - [8, 12, 128, 64] = float:[8, 12, 128, 64]
            ("matmul_0114", "batchmatmul_run",
             ((8, 12), 128, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[8, 128, 1] - [8, 1, 128] = float:[8, 128, 128]
            ("matmul_0115", "batchmatmul_run", ((8,), 128, 128, 1, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[8192, 768] - [768, 3072] = float:[8192, 3072]
            ("matmul_0116", "batchmatmul_run",
             ((), 8192, 3072, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[8192, 3072] - [3072, 768] = float:[8192, 768]
            ("matmul_0117", "batchmatmul_run",
             ((), 8192, 768, 3072, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[8192, 768] - [768, 768] = float:[8192, 768]
            (
                "matmul_0118", "batchmatmul_run",
                ((), 8192, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[8, 768] - [2, 768] = float:[8, 2]
            ("matmul_0119", "batchmatmul_run", ((), 8, 2, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[8, 2] - [2, 768] = float:[8, 768]
            ("matmul_0120", "batchmatmul_run", ((), 8, 768, 2, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[8, 768] - [768, 768]) = float:[8, 768]
            ("matmul_0121", "batchmatmul_run", ((), 8, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[128, 768]-[2, 768]=float:[128, 2]
            ("matmul_0122", "batchmatmul_run", ((), 128, 2, 768, (), "float32", False, True, "batch_matmul_output")),
            # float-float:[16, 768]-[2, 768]=float:[16, 2]
            ("matmul_0123", "batchmatmul_run", ((), 16, 2, 768, (), "float32", False, True, "batch_matmul_output")),
            # float-float:[2, 768]-[2, 768]=float:[2, 2]
            ("matmul_0124", "batchmatmul_run", ((), 2, 2, 768, (), "float32", False, True, "batch_matmul_output")),
            # float-float:[128, 2]-[128, 768]=float:[2, 768]
            ("matmul_0125", "batchmatmul_run", ((), 2, 768, 128, (), "float32", True, False, "batch_matmul_output")),
            # float-float:[16, 2]-[16, 768]=float:[2, 768]
            ("matmul_0126", "batchmatmul_run", ((), 2, 768, 16, (), "float32", True, False, "batch_matmul_output")),
            # float-float:[32, 2]-[32, 768]=float:[2, 768]
            ("matmul_0127", "batchmatmul_run", ((), 2, 768, 32, (), "float32", True, False, "batch_matmul_output")),
            # float-float:[4, 2]-[4, 768]=float:[2, 768]
            ("matmul_0128", "batchmatmul_run", ((), 2, 768, 4, (), "float32", True, False, "batch_matmul_output")),
            # float-float:[8, 2]-[8, 768]=float:[2, 768]
            ("matmul_0129", "batchmatmul_run", ((), 2, 768, 8, (), "float32", True, False, "batch_matmul_output")),
            # float-float:[32, 768]-[2, 768]=float:[32, 2]
            ("matmul_0130", "batchmatmul_run", ((), 32, 2, 768, (), "float32", False, True, "batch_matmul_output")),
            # float-float:[4, 768]-[2, 768]=float:[4, 2]
            ("matmul_0131", "batchmatmul_run", ((), 4, 2, 768, (), "float32", False, True, "batch_matmul_output")),
            # float-float:[8, 768]-[2, 768]=float:[8, 2]
            ("matmul_0132", "batchmatmul_run", ((), 8, 2, 768, (), "float32", False, True, "batch_matmul_output")),

            # Neg OP
            # float:[128]=float:[128]
            ("neg_001", "neg_run", ((128,), "float32")),
            # float:[16]=float:[16]
            ("neg_002", "neg_run", ((16,), "float32")),
            # float:[1]=float:[1]
            ("neg_003", "neg_run", ((1,), "float32")),
            # float:[2560]=float:[2560]
            ("neg_004", "neg_run", ((2560,), "float32")),
            # float:[2]=float:[2]
            ("neg_005", "neg_run", ((2,), "float32")),
            # float:[320]=float:[320]
            ("neg_006", "neg_run", ((320,), "float32")),
            # float:[32]=float:[32]
            ("neg_007", "neg_run", ((32,), "float32")),
            # float:[40]=float:[40]
            ("neg_008", "neg_run", ((40,), "float32")),
            # float:[4]=float:[4]
            ("neg_009", "neg_run", ((4,), "float32")),
            # float:[640]=float:[640]
            ("neg_010", "neg_run", ((640,), "float32")),
            # float:[80]=float:[80]
            ("neg_011", "neg_run", ((80,), "float32")),
            # float:[10240] = float:[10240]
            ("neg_012", "neg_run", ((10240,), "float32")),
            # float:[1024]=float:[1024]
            ("neg_013", "neg_run", ((1024,), "float32")),
            # float:[1280]=float:[1280]
            ("neg_014", "neg_run", ((1280,), "float32")),
            # float:[160]=float:[160]
            ("neg_015", "neg_run", ((160,), "float32")),
            # float:[20480]=float:[20480]
            ("neg_016", "neg_run", ((20480,), "float32")),
            # float:[20]=float:[20]
            ("neg_017", "neg_run", ((20,), "float32")),
            # float:[256]=float:[256]
            ("neg_018", "neg_run", ((256,), "float32")),
            # float:[5120]=float:[5120]
            ("neg_019", "neg_run", ((5120,), "float32")),
            # float:[512]=float:[512]
            ("neg_020", "neg_run", ((512,), "float32")),
            # float:[64]=float:[64]
            ("neg_021", "neg_run", ((64,), "float32")),
            # float:[8]=float:[8]
            ("neg_022", "neg_run", ((8,), "float32")),

            # onehot OP
            # int32-int32-float-float:[128]-[]-[]-[]=float:[128, 2]
            ("one_hot_001", "one_hot_run", ((128,), 2, "int32", 1, 0, -1)),
            # int32-int32-float-float:[160]-[]-[]-[]=float:[160, 21128]
            ("one_hot_002", "one_hot_run", ((160,), 21128, "int32", 1, 0, -1)),
            # int32-int32-float-float:[16]-[]-[]-[]=float:[16, 2]
            ("one_hot_003", "one_hot_run", ((16,), 2, "int32", 1, 0, -1)),
            # int32-int32-float-float:[20]-[]-[]-[]=float:[20, 21128]
            ("one_hot_004", "one_hot_run", ((20,), 21128, "int32", 1, 0, -1)),
            # int32-int32-float-float:[2]-[]-[]-[]=float:[2, 2]
            ("one_hot_005", "one_hot_run", ((2,), 2, "int32", 1, 0, -1)),
            # int32-int32-float-float:[2560]-[]-[]-[]=float:[2560, 21128]
            ("one_hot_006", "one_hot_run", ((2560,), 21128, "int32", 1, 0, -1)),
            # int32-int32-float-float:[1]-[]-[]-[]=float:[2]
            ("one_hot_007", "one_hot_run", ((1,), 2, "int32", 1, 0, -1)),
            # int32-int32-float-float:[320]-[]-[]-[]=float:[320, 21128]
            ("one_hot_008", "one_hot_run", ((320,), 21128, "int32", 1, 0, -1)),
            # int32-int32-float-float:[32]-[]-[]-[]=float:[32, 2]
            ("one_hot_009", "one_hot_run", ((32,), 2, "int32", 1, 0, -1)),
            # int32-int32-float-float:[40]-[]-[]-[]=float:[40, 21128]
            ("one_hot_010", "one_hot_run", ((40,), 21128, "int32", 1, 0, -1)),
            # int32-int32-float-float:[4]-[]-[]-[]=float:[4, 2]
            ("one_hot_011", "one_hot_run", ((4,), 2, "int32", 1, 0, -1)),
            # int32-int32-float-float:[640]-[]-[]-[]=float:[640, 21128]
            ("one_hot_012", "one_hot_run", ((640,), 21128, "int32", 1, 0, -1)),
            # int32-int32-float-float:[80]-[]-[]-[]=float:[80, 21128]
            ("one_hot_013", "one_hot_run", ((80,), 21128, "int32", 1, 0, -1)),
            # int32-int32-float-float:[8]-[]-[]-[]=float:[8, 2]
            ("one_hot_014", "one_hot_run", ((8,), 2, "int32", 1, 0, -1)),
            # int32-int32-float-float:[64]-[]-[]-[]=float:[64, 2]
            ("one_hot_015", "one_hot_run", ((64,), 2, "int32", 1, 0, -1)),
            # int32-int32-float-float:[5120]-[]-[]-[]=float:[5120, 21128]
            ("one_hot_016", "one_hot_run", ((5120,), 21128, "int32", 1, 0, -1)),
            # int32-int32-float-float:[512]-[]-[]-[]=float:[512, 2]
            ("one_hot_017", "one_hot_run", ((512,), 2, "int32", 1, 0, -1)),
            # int32-int32-float-float:[256]-[]-[]-[]=float:[256, 2]
            ("one_hot_018", "one_hot_run", ((256,), 2, "int32", 1, 0, -1)),
            # int32-int32-float-float:[20480]-[]-[]-[]=float:[20480, 21128]
            ("one_hot_019", "one_hot_run", ((20480,), 21128, "int32", 1, 0, -1)),
            # int32-int32-float-float:[10240]-[]-[]-[]=float:[10240, 21128]
            ("one_hot_020", "one_hot_run", ((10240,), 21128, "int32", 1, 0, -1)),
            # int32-int32-float-float:[1024]-[]-[]-[]=float:[1024, 2]
            ("one_hot_021", "one_hot_run", ((1024,), 2, "int32", 1, 0, -1)),
            # int32-int32-float-float:[1280]-[]-[]-[]=float:[1280, 21128]
            ("one_hot_022", "one_hot_run", ((1280,), 21128, "int32", 1, 0, -1)),

            # sub OP
            # float - float:[1] - [128, 1, 128, 128] = float:[128, 1, 128, 128]
            ("sub_001", "sub_run", [(1,), (128, 1, 128, 128), "float32"]),
            # float - float:[1] - [16, 1, 128, 128] = float:[16, 1, 128, 128]
            ("sub_002", "sub_run", [(1,), (16, 1, 128, 128), "float32"]),
            # float - float:[1] - [1] = float:[1]
            ("sub_003", "sub_run", [(1,), (1,), "float32"]),
            # float - float:[1] - [2, 1, 128, 128] = float:[2, 1, 128, 128]
            ("sub_004", "sub_run", [(1,), (2, 1, 128, 128), "float32"]),
            # float - float:[1] - [32, 1, 128, 128] = float:[32, 1, 128, 128]
            ("sub_005", "sub_run", [(1,), (32, 1, 128, 128), "float32"]),
            # float - float:[1] - [4, 1, 128, 128] = float:[4, 1, 128, 128]
            ("sub_006", "sub_run", [(1,), (4, 1, 128, 128), "float32"]),
            # float - float:[1] - [64, 1, 128, 128] = float:[64, 1, 128, 128]
            ("sub_007", "sub_run", [(1,), (64, 1, 128, 128), "float32"]),
            # float - float:[1] - [8, 1, 128, 128] = float:[8, 1, 128, 128]
            ("sub_008", "sub_run", [(1,), (8, 1, 128, 128), "float32"]),
            # int32 - int32:[128, 128] - [128, 128] = int32:[128, 128]
            ("sub_010", "sub_run", [(128, 128), (128, 128), "int32"]),
            # float - float:[8, 12, 128, 128] - [8, 12, 128, 1] = float:[8, 12, 128, 128]
            ("sub_011", "sub_run", [(8, 12, 128, 128), (8, 12, 128, 1), "float32"]),
            # float - float:[128, 12, 128, 128] - [128, 12, 128, 1] = float:[128, 12, 128, 128]
            ("sub_012", "sub_run", [(128, 12, 128, 128), (128, 12, 128, 1), "float32"]),
            # float - float:[16, 12, 128, 128] - [16, 12, 128, 1] = float:[16, 12, 128, 128]
            ("sub_013", "sub_run", [(16, 12, 128, 128), (16, 12, 128, 1), "float32"]),
            # float - float:[2, 12, 128, 128] - [2, 12, 128, 1] = float:[2, 12, 128, 128]
            ("sub_014", "sub_run", [(2, 12, 128, 128), (2, 12, 128, 1), "float32"]),
            # float - float:[32, 12, 128, 128] - [32, 12, 128, 1] = float:[32, 12, 128, 128]
            ("sub_015", "sub_run", [(32, 12, 128, 128), (32, 12, 128, 1), "float32"]),
            # float - float:[4, 12, 128, 128] - [4, 12, 128, 1] = float:[4, 12, 128, 128]
            ("sub_016", "sub_run", [(4, 12, 128, 128), (4, 12, 128, 1), "float32"]),
            # float - float:[64, 12, 128, 128] - [64, 12, 128, 1] = float:[64, 12, 128, 128]
            ("sub_017", "sub_run", [(64, 12, 128, 128), (64, 12, 128, 1), "float32"]),
            # float-float:[768, 3072]-[768, 3072]=float:[768, 3072]
            ("sub_018", "sub_run", [(768, 3072), (768, 3072), "float32"]),
            # float-float:[768, 768]-[768, 768]=float:[768, 768]
            ("sub_019", "sub_run", [(768, 768), (768, 768), "float32"]),
            # float-float:[768]-[768]=float:[768]
            ("sub_020", "sub_run", [(768,), (768,), "float32"]),
            # float-float:[512, 12, 128, 128]-[512, 12, 128, 1]=float:[512, 12, 128, 128]
            ("sub_021", "sub_run", [(512, 12, 128, 128), (512, 12, 128, 128), "float32"]),
            # float-float:[33, 64]-[33, 64]=float:[33, 64]
            ("sub_022", "sub_run", [(33, 64), (33, 64), "float32"]),
            # float-float:[2]-[2]=float:[2]
            ("sub_023", "sub_run", [(2,), (2,), "float32"]),
            # float-float:[256, 12, 128, 128]-[256, 12, 128, 1]=float:[256, 12, 128, 128]
            ("sub_024", "sub_run", [(256, 12, 128, 128), (256, 12, 128, 128), "float32"]),
            # float-float:[2, 768]-[2, 768]=float:[2, 768]
            ("sub_025", "sub_run", [(2, 768), (2, 768), "float32"]),
            # float-float:[3072]-[3072]=float:[3072]
            ("sub_026", "sub_run", [(3072,), (3072,), "float32"]),
            # float-float:[3072, 768]-[3072, 768]=float:[3072, 768]
            ("sub_027", "sub_run", [(3072, 768), (3072, 768), "float32"]),
            # float-float:[21128]-[21128]=float:[21128]
            ("sub_028", "sub_run", [(21128,), (21128,), "float32"]),
            # float-float:[21128, 768]-[21128, 768]=float:[21128, 768]
            ("sub_029", "sub_run", [(21128, 768), (21128, 768), "float32"]),
            # float-float:[1]-[512, 1, 128, 128]=float:[512, 1, 128, 128]
            ("sub_030", "sub_run", [(1,), (512, 1, 128, 128), "float32"]),
            # float-float:[1]-[256, 1, 128, 128]=float:[256, 1, 128, 128]
            ("sub_031", "sub_run", [(1,), (256, 1, 128, 128), "float32"]),
            # float-float:[1024, 12, 128, 128]-[1024, 12, 128, 1]=float:[1024, 12, 128, 128]
            ("sub_032", "sub_run", [(1024, 12, 128, 128), (1024, 12, 128, 128), "float32"]),
            # float-float:[1]-[1024, 1, 128, 128]=float:[1024, 1, 128, 128]
            ("sub_033", "sub_run", [(1,), (1024, 1, 128, 128), "float32"]),
            # float-float:[1]-[1, 1, 128, 128]=float:[1, 1, 128, 128]
            ("sub_034", "sub_run", [(1,), (1, 1, 128, 128), "float32"]),
            # float-float:[1, 12, 128, 128]-[1, 12, 128, 1]=float:[1, 12, 128, 128]
            ("sub_035", "sub_run", [(1, 12, 128, 128), (1, 12, 128, 128), "float32"]),

            # sub OP
            # float - int32:[1, 12, 128, 128] - [-1] = float:[1, 12, 128, 1]
            ("sum_001", "sum_run", ((1, 12, 128, 128), (-1,), False, "float32")),
            # float - int32:[128, 12, 128, 128] - [-1] = float:[128, 12, 128, 1]
            ("sum_002", "sum_run", ((128, 12, 128, 128), (-1,), False, "float32")),
            # float - int32:[16, 12, 128, 128] - [-1] = float:[16, 12, 128, 1]
            ("sum_003", "sum_run", ((16, 12, 128, 128), (-1,), False, "float32")),
            # float - int32:[1280] - [-1] = float:[1]
            ("sum_004", "sum_run", ((1280,), (-1,), False, "float32")),
            # float - int32:[160] - [-1] = float:[1]
            ("sum_005", "sum_run", ((160,), (-1,), False, "float32")),
            # float - int32:[20] - [-1] = float:[1]
            ("sum_006", "sum_run", ((20,), (-1,), False, "float32")),
            # float - int32:[21128] - [-1] = float:[1]
            ("sum_007", "sum_run", ((21128,), (-1,), False, "float32")),
            # float - int32:[2560] - [-1] = float:[1]
            ("sum_008", "sum_run", ((2560,), (-1,), False, "float32")),
            # float - int32:[2] - [-1] = float:[1]
            ("sum_009", "sum_run", ((2,), (-1,), False, "float32")),
            # float - int32:[3072] - [-1] = float:[1]
            ("sum_010", "sum_run", ((3072,), (-1,), False, "float32")),
            # float - int32:[320] - [-1] = float:[1]
            ("sum_011", "sum_run", ((320,), (-1,), False, "float32")),
            # float - int32:[40] - [-1] = float:[1]
            ("sum_012", "sum_run", ((40,), (-1,), False, "float32")),
            # float - int32:[640] - [-1] = float:[1]
            ("sum_013", "sum_run", ((640,), (-1,), False, "float32")),
            # float - int32:[768] - [-1] = float:[1]
            ("sum_014", "sum_run", ((768,), (-1,), False, "float32")),
            # float - int32:[80] - [-1] = float:[1]
            ("sum_015", "sum_run", ((80,), (-1,), False, "float32")),
            # float - int32:[21128, 768] - [-1] = float:[21128]
            ("sum_016", "sum_run", ((21128,), (-1,), False, "float32")),
            # float - int32:[2, 12, 128, 128] - [-1] = float:[2, 12, 128, 1]
            ("sum_017", "sum_run", ((2, 12, 128, 128,), (-1,), False, "float32")),
            # float - int32:[2, 768] - [-1] = float:[2]
            ("sum_018", "sum_run", ((2, 768), (-1,), False, "float32")),
            # float - int32:[3072, 768] - [-1] = float:[3072]
            ("sum_019", "sum_run", ((3072, 768), (-1,), False, "float32")),
            # float - int32:[32, 12, 128, 128] - [-1] = float:[32, 12, 128, 1]
            ("sum_020", "sum_run", ((32, 12, 128, 128), (-1,), False, "float32")),
            # float - int32:[33, 64] - [-1] = float:[33]
            ("sum_021", "sum_run", ((33, 64), (-1,), False, "float32")),
            # float - int32:[4, 12, 128, 128] - [-1] = float:[4, 12, 128, 1]
            ("sum_022", "sum_run", ((4, 12, 128, 128), (-1,), False, "float32")),
            # float - int32:[64, 12, 128, 128] - [-1] = float:[64, 12, 128, 1]
            ("sum_023", "sum_run", ((64, 12, 128, 128), (-1,), False, "float32")),
            # float - int32:[768, 3072] - [-1] = float:[768]
            ("sum_024", "sum_run", ((768, 3072), (-1,), False, "float32")),
            # float - int32:[768, 768] - [-1] = float:[768]
            ("sum_025", "sum_run", ((768, 768), (-1,), False, "float32")),
            # float - int32:[8, 12, 128, 128] - [-1] = float:[8, 12, 128, 1]
            ("sum_026", "sum_run", ((8, 12, 128, 128), (-1,), False, "float32")),
            # float-int32:[80, 21128]-[-1]=float:[21128]
            ("sum_027", "sum_run", ((80, 21128), (0,), False, "float32")),
            # float-int32:[640, 21128]-[-1]=float:[21128]
            ("sum_028", "sum_run", ((640, 21128), (0,), False, "float32")),
            # float-int32:[5120]-[-1]=float:[1]
            ("sum_029", "sum_run", ((5120,), (0,), False, "float32")),
            # float-int32:[5120, 21128]-[-1]=float:[21128]
            ("sum_030", "sum_run", ((5120, 21128), (0,), False, "float32")),
            # float-int32:[512, 12, 128, 128]-[-1]=float:[512, 12, 128, 1]
            ("sum_031", "sum_run", ((512, 12, 128, 128), (-1,), False, "float32")),
            # float-int32:[40, 21128]-[-1]=float:[21128]
            ("sum_032", "sum_run", ((40, 21128), (0,), False, "float32")),
            # float-int32:[320, 21128]-[-1]=float:[21128]
            ("sum_033", "sum_run", ((320, 21128), (0,), False, "float32")),
            # float-int32:[2560, 21128]-[-1]=float:[21128]
            ("sum_034", "sum_run", ((2560, 21128), (0,), False, "float32")),
            # float-int32:[256, 12, 128, 128]-[-1]=float:[256, 12, 128, 1]
            ("sum_035", "sum_run", ((256, 12, 128, 128), (-1,), False, "float32")),
            # float-int32:[20, 21128]-[-1]=float:[21128]
            ("sum_036", "sum_run", ((20, 21128), (0,), False, "float32")),
            # float-int32:[20480]-[-1]=float:[1]
            ("sum_037", "sum_run", ((20480,), (0,), False, "float32")),
            # float-int32:[20480, 21128]-[-1]=float:[21128]
            ("sum_038", "sum_run", ((20480, 21128), (0,), False, "float32")),
            # float-int32:[160, 21128]-[-1]=float:[21128]
            ("sum_039", "sum_run", ((160, 21128), (0,), False, "float32")),
            # float-int32:[1280, 21128]-[-1]=float:[21128]
            ("sum_040", "sum_run", ((1280, 21128), (0,), False, "float32")),
            # float-int32:[10240]-[-1]=float:[1]
            ("sum_041", "sum_run", ((10240,), (0,), False, "float32")),
            # float-int32:[10240, 21128]-[-1]=float:[21128]
            ("sum_042", "sum_run", ((10240, 21128), (0,), False, "float32")),
            # float-int32:[1024, 12, 128, 128]-[-1]=float:[1024, 12, 128, 1]
            ("sum_043", "sum_run", ((1024, 12, 128, 128), (-1,), False, "float32")),

            # StridedSlice OP
            # float - int32 - int32 - int32 - int32:[1, 128, 768] - [3] - [3] - [3] - [3] = float:[1, 1, 768]
            ("strided_slice_001", "strided_slice_run",
             ((1, 128, 768), [0, 0, 0], [1, 1, 768], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),
            # float - int32 - int32 - int32 - int32:[128, 128, 768] - [3] - [3] - [3] - [3] = float:[128, 1, 768]
            ("strided_slice_002", "strided_slice_run",
             ((128, 128, 768), [0, 0, 0], [128, 1, 768], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),
            # float - int32 - int32 - int32 - int32:[16, 128, 768] - [3] - [3] - [3] - [3] = float:[16, 1, 768]
            ("strided_slice_003", "strided_slice_run",
             ((16, 128, 768), [0, 0, 0], [16, 1, 768], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),
            # float - int32 - int32 - int32 - int32:[2, 128, 768] - [3] - [3] - [3] - [3] = float:[2, 1, 768]
            ("strided_slice_004", "strided_slice_run",
             ((2, 128, 768), [0, 0, 0], [2, 1, 768], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),
            # float - int32 - int32 - int32 - int32:[32, 128, 768] - [3] - [3] - [3] - [3] = float:[32, 1, 768]
            ("strided_slice_005", "strided_slice_run",
             ((32, 128, 768), [0, 0, 0], [32, 1, 768], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),
            # float - int32 - int32 - int32 - int32:[4, 128, 768] - [3] - [3] - [3] - [3] = float:[4, 1, 768]
            ("strided_slice_006", "strided_slice_run",
             ((4, 128, 768), [0, 0, 0], [4, 1, 768], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),
            # float - int32 - int32 - int32 - int32:[64, 128, 768] - [3] - [3] - [3] - [3] = float:[64, 1, 768]
            ("strided_slice_007", "strided_slice_run",
             ((64, 128, 768), [0, 0, 0], [64, 1, 768], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),
            # float - int32 - int32 - int32 - int32:[8, 128, 768] - [3] - [3] - [3] - [3] = float:[8, 1, 768]
            ("strided_slice_008", "strided_slice_run",
             ((8, 128, 768), [0, 0, 0], [8, 1, 768], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),
            # float - int32 - int32 - int32:[512, 128, 768] - [3] - [3] - [3] = float:[512, 1, 768]
            ("strided_slice_009", "strided_slice_run",
             ((512, 128, 768), [0, 0, 0], [512, 1, 768], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),
            # float - int32 - int32 - int32:[256, 128, 768] - [3] - [3] - [3] = float:[256, 1, 768]
            ("strided_slice_010", "strided_slice_run",
             ((256, 128, 768), [0, 0, 0], [256, 1, 768], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),
            # float - int32 - int32 - int32:[1024, 128, 768] - [3] - [3] - [3] = float:[1024, 1, 768]
            ("strided_slice_011", "strided_slice_run",
             ((1024, 128, 768), [0, 0, 0], [1024, 1, 768], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),

            # StridedSliceGrad OP
            # float - int32 - int32 - int32 - int32:[1, 1, 768] - [3] - [3] - [3] - [3] = float:[1, 128, 768]
            ("strided_slice_grad_001", "strided_slice_grad_run",
             [(1, 128, 768), [0, 0, 0], [1, 1, 768], [1, 1, 1], 0, 0, 0, 0, 0, (1, 1, 768), "int32"]),
            # float - int32 - int32 - int32 - int32:[128, 1, 768] - [3] - [3] - [3] - [3] = float:[128, 128, 768]
            ("strided_slice_grad_002", "strided_slice_grad_run",
             [(128, 128, 768), [0, 0, 0], [128, 1, 768], [1, 1, 1], 0, 0, 0, 0, 0, (128, 1, 768), "int32"]),
            # float - int32 - int32 - int32 - int32:[16, 1, 768] - [3] - [3] - [3] - [3] = float:[16, 128, 768]
            ("strided_slice_grad_003", "strided_slice_grad_run",
             [(16, 128, 768), [0, 0, 0], [16, 1, 768], [1, 1, 1], 0, 0, 0, 0, 0, (16, 1, 768), "int32"]),
            # float - int32 - int32 - int32 - int32:[2, 1, 768] - [3] - [3] - [3] - [3] = float:[2, 128, 768]
            ("strided_slice_grad_004", "strided_slice_grad_run",
             [(2, 128, 768), [0, 0, 0], [2, 1, 768], [1, 1, 1], 0, 0, 0, 0, 0, (2, 1, 768), "int32"]),
            # float - int32 - int32 - int32 - int32:[32, 1, 768] - [3] - [3] - [3] - [3] = float:[32, 128, 768]
            ("strided_slice_grad_005", "strided_slice_grad_run",
             [(32, 128, 768), [0, 0, 0], [32, 1, 768], [1, 1, 1], 0, 0, 0, 0, 0, (32, 1, 768), "int32"]),
            # float - int32 - int32 - int32 - int32:[4, 1, 768] - [3] - [3] - [3] - [3] = float:[4, 128, 768]
            ("strided_slice_grad_006", "strided_slice_grad_run",
             [(4, 128, 768), [0, 0, 0], [4, 1, 768], [1, 1, 1], 0, 0, 0, 0, 0, (4, 1, 768), "int32"]),
            # float - int32 - int32 - int32 - int32:[64, 1, 768] - [3] - [3] - [3] - [3] = float:[64, 128, 768]
            ("strided_slice_grad_007", "strided_slice_grad_run",
             [(64, 128, 768), [0, 0, 0], [64, 1, 768], [1, 1, 1], 0, 0, 0, 0, 0, (64, 1, 768), "int32"]),
            # float - int32 - int32 - int32 - int32:[8, 1, 768] - [3] - [3] - [3] - [3] = float:[8, 128, 768]
            ("strided_slice_grad_008", "strided_slice_grad_run",
             [(8, 128, 768), [0, 0, 0], [8, 1, 768], [1, 1, 1], 0, 0, 0, 0, 0, (8, 1, 768), "int32"]),
            # float - int32 - int32 - int32:[512, 128, 768] - [3] - [3] - [3] = float:[512, 1, 768]
            ("strided_slice_grad_009", "strided_slice_grad_run",
             ((512, 128, 768), [0, 0, 0], [512, 1, 768], [1, 1, 1], 0, 0, 0, 0, 0, (512, 1, 768), "int32")),
            # float - int32 - int32 - int32:[256, 128, 768] - [3] - [3] - [3] = float:[256, 1, 768]
            ("strided_slice_grad_010", "strided_slice_grad_run",
             ((256, 128, 768), [0, 0, 0], [256, 1, 768], [1, 1, 1], 0, 0, 0, 0, 0, (256, 1, 768), "int32")),
            # float - int32 - int32 - int32:[1024, 128, 768] - [3] - [3] - [3] = float:[1024, 1, 768]
            ("strided_slice_grad_011", "strided_slice_grad_run",
             ((1024, 128, 768), [0, 0, 0], [1024, 1, 768], [1, 1, 1], 0, 0, 0, 0, 0, (1024, 1, 768), "int32")),

            # Tanh OP
            # float:[1024, 768]=float:[1024, 768]
            ("tanh_001", "tanh_run", ((1024, 768), "float32")),
            # float:[128, 768]=float:[128, 768]
            ("tanh_002", "tanh_run", ((128, 768), "float32")),
            # float:[16, 768]=float:[16, 768]
            ("tanh_003", "tanh_run", ((16, 768), "float32")),
            # float:[1, 768]=float:[1, 768]
            ("tanh_004", "tanh_run", ((1, 768), "float32")),
            # float:[256, 768]=float:[256, 768]
            ("tanh_005", "tanh_run", ((256, 768), "float32")),
            # float:[2, 768]=float:[2, 768]
            ("tanh_006", "tanh_run", ((2, 768), "float32")),
            # float:[32, 768]=float:[32, 768]
            ("tanh_007", "tanh_run", ((32, 768), "float32")),
            # float:[4, 768]=float:[4, 768]
            ("tanh_008", "tanh_run", ((4, 768), "float32")),
            # float:[512, 768]=float:[512, 768]
            ("tanh_009", "tanh_run", ((512, 768), "float32")),
            # float:[64, 768]=float:[64, 768]
            ("tanh_010", "tanh_run", ((64, 768), "float32")),
            # float:[8, 768]=float:[8, 768]
            ("tanh_011", "tanh_run", ((8, 768), "float32")),

            # TanhGrad OP
            # float:[1024, 768]=float:[1024, 768]
            ("tanh_001", "tanh_grad_run", ((1024, 768), "float32")),
            # float:[128, 768]=float:[128, 768]
            ("tanh_002", "tanh_grad_run", ((128, 768), "float32")),
            # float:[16, 768]=float:[16, 768]
            ("tanh_003", "tanh_grad_run", ((16, 768), "float32")),
            # float:[1, 768]=float:[1, 768]
            ("tanh_004", "tanh_grad_run", ((1, 768), "float32")),
            # float:[256, 768]=float:[256, 768]
            ("tanh_005", "tanh_grad_run", ((256, 768), "float32")),
            # float:[2, 768]=float:[2, 768]
            ("tanh_006", "tanh_grad_run", ((2, 768), "float32")),
            # float:[32, 768]=float:[32, 768]
            ("tanh_007", "tanh_grad_run", ((32, 768), "float32")),
            # float:[4, 768]=float:[4, 768]
            ("tanh_008", "tanh_grad_run", ((4, 768), "float32")),
            # float:[512, 768]=float:[512, 768]
            ("tanh_009", "tanh_grad_run", ((512, 768), "float32")),
            # float:[64, 768]=float:[64, 768]
            ("tanh_010", "tanh_grad_run", ((64, 768), "float32")),
            # float:[8, 768]=float:[8, 768]
            ("tanh_011", "tanh_grad_run", ((8, 768), "float32")),

            # reshape OP
            # float-int32:[10240]-[2]=float:[10240, 1]
            ("reshape_0001", "reshape_run", [(10240,), (10240, 1), "float32"]),
            ("reshape_0002", "reshape_run", [(10240, 1), (10240,), "float32"]),
            # float-int32:[10240]-[]=float:[512, 20]
            ("reshape_0003", "reshape_run", [(10240,), (512, 20), "float32"]),
            ("reshape_0004", "reshape_run", [(512, 20), (10240,), "float32"]),
            # float-int32:[1024, 128, 12, 64]-[2]=float:[131072, 768]
            ("reshape_0005", "reshape_run", [(1024, 128, 12, 64), (131072, 768), "float32"]),
            ("reshape_0006", "reshape_run", [(131072, 768), (1024, 128, 12, 64), "float32"]),
            # float-int32:[1024, 128, 768]-[2]=float:[131072, 768]
            ("reshape_0007", "reshape_run", [(1024, 128, 768), (131072, 768), "float32"]),
            ("reshape_0008", "reshape_run", [(131072, 768), (1024, 128, 768), "float32"]),
            # float-int32:[1024, 20]-[]=float:[20480]
            ("reshape_0009", "reshape_run", [(1024, 20), (20480,), "float32"]),
            ("reshape_0010", "reshape_run", [(20480,), (1024, 20), "float32"]),
            # float-int32:[1024]-[2]=float:[1024, 1]
            ("reshape_0011", "reshape_run", [(1024,), (1024, 1), "float32"]),
            ("reshape_0012", "reshape_run", [(1024, 1), (1024,), "float32"]),
            # float-int32:[1024, 768]-[2]=float:[1024, 768]
            ("reshape_0013", "reshape_run", [(1024, 768), (1024, 768), "float32"]),
            # float-int32:[1024, 768]-[3]=float:[1024, 1, 768]
            ("reshape_0014", "reshape_run", [(1024, 768), (1024, 1, 768), "float32"]),
            ("reshape_0015", "reshape_run", [(1024, 1, 768), (1024, 768), "float32"]),
            # float-int32:[1024, 768]-[3]=float:[8, 128, 768]
            ("reshape_0016", "reshape_run", [(1024, 768), (8, 128, 768), "float32"]),
            ("reshape_0017", "reshape_run", [(8, 128, 768), (1024, 768), "float32"]),
            # float-int32:[1024, 768]-[4]=float:[8, 128, 12, 64]
            ("reshape_0018", "reshape_run", [(1024, 768), (8, 128, 12, 64), "float32"]),
            ("reshape_0019", "reshape_run", [(8, 128, 12, 64), (1024, 768), "float32"]),
            # float-int32:[1, 128, 12, 64]-[2]=float:[128, 768]
            ("reshape_0020", "reshape_run", [(1, 128, 12, 64), (128, 768), "float32"]),
            ("reshape_0021", "reshape_run", [(128, 768), (1, 128, 12, 64), "float32"]),
            # float-int32:[1, 128, 768]-[2]=float:[128, 768]
            ("reshape_0022", "reshape_run", [(1, 128, 768), (128, 768), "float32"]),
            ("reshape_0023", "reshape_run", [(128, 768), (1, 128, 768), "float32"]),
            # float-int32:[1, 20]-[]=float:[20]
            ("reshape_0024", "reshape_run", [(1, 20), (20,), "float32"]),
            ("reshape_0025", "reshape_run", [(20,), (1, 20), "float32"]),
            # float-int32:[1, 21128]-[]=float:[21128]
            ("reshape_0026", "reshape_run", [(1, 21128), (21128,), "float32"]),
            ("reshape_0027", "reshape_run", [(21128,), (1, 21128), "float32"]),
            # float-int32:[1280]-[2]=float:[1280, 1]
            ("reshape_0028", "reshape_run", [(1280,), (1280, 1), "float32"]),
            ("reshape_0029", "reshape_run", [(1280, 1), (1280,), "float32"]),
            # float-int32:[1280]-[]=float:[64, 20]
            ("reshape_0030", "reshape_run", [(1280,), (64, 20), "float32"]),
            ("reshape_0031", "reshape_run", [(64, 20), (1280,), "float32"]),
            # float-int32:[128, 1024, 12, 128]-[3]=float:[128, 12288, 128]
            ("reshape_0032", "reshape_run", [(128, 1024, 12, 128), (128, 12288, 128), "float32"]),
            ("reshape_0033", "reshape_run", [(128, 12288, 128), (128, 1024, 12, 128), "float32"]),
            # float-int32:[128, 1024, 12, 64]-[3]=float:[128, 12288, 64]
            ("reshape_0034", "reshape_run", [(128, 1024, 12, 64), (128, 12288, 64), "float32"]),
            ("reshape_0035", "reshape_run", [(128, 12288, 64), (128, 1024, 12, 64), "float32"]),
            # float-int32:[128, 1, 12, 128]-[3]=float:[128, 12, 128]
            ("reshape_0036", "reshape_run", [(128, 1, 12, 128), (128, 12, 128), "float32"]),
            ("reshape_0037", "reshape_run", [(128, 12, 128), (128, 1, 12, 128), "float32"]),
            # float-int32:[128, 1, 12, 64]-[3]=float:[128, 12, 64]
            ("reshape_0038", "reshape_run", [(128, 1, 12, 64), (128, 12, 64), "float32"]),
            ("reshape_0039", "reshape_run", [(128, 12, 64), (128, 1, 12, 64), "float32"]),
            # float-int32:[128, 128, 12, 128]-[3]=float:[128, 1536, 128]
            ("reshape_0040", "reshape_run", [(128, 128, 12, 128), (128, 1536, 128), "float32"]),
            ("reshape_0041", "reshape_run", [(128, 1536, 128), (128, 128, 12, 128), "float32"]),
            # float-int32:[128, 128, 12, 64]-[2]=float:[16384, 768]
            ("reshape_0042", "reshape_run", [(128, 128, 12, 64), (16384, 768), "float32"]),
            ("reshape_0043", "reshape_run", [(16384, 768), (128, 128, 12, 64), "float32"]),
            # float-int32:[128, 128, 12, 64]-[3]=float:[128, 1536, 64]
            ("reshape_0044", "reshape_run", [(128, 128, 12, 64), (128, 1536, 64), "float32"]),
            ("reshape_0045", "reshape_run", [(128, 1536, 64), (128, 128, 12, 64), "float32"]),
            # float-int32:[128, 128, 768]-[2]=float:[16384, 768]
            ("reshape_0046", "reshape_run", [(128, 128, 768), (16384, 768), "float32"]),
            ("reshape_0047", "reshape_run", [(16384, 768), (128, 128, 768), "float32"]),
            # float-int32:[128, 16, 12, 128]-[3]=float:[128, 192, 128]
            ("reshape_0048", "reshape_run", [(128, 16, 12, 128), (128, 192, 128), "float32"]),
            ("reshape_0049", "reshape_run", [(128, 192, 128), (128, 16, 12, 128), "float32"]),
            # float-int32:[128, 16, 12, 64]-[3]=float:[128, 192, 64]
            ("reshape_0050", "reshape_run", [(128, 16, 12, 64), (128, 192, 64), "float32"]),
            ("reshape_0051", "reshape_run", [(128, 192, 64), (128, 16, 12, 64), "float32"]),
            # float-int32:[128, 20]-[]=float:[2560]
            ("reshape_0052", "reshape_run", [(128, 20), (2560,), "float32"]),
            ("reshape_0053", "reshape_run", [(2560,), (128, 20), "float32"]),
            # float-int32:[128, 2, 12, 128]-[3]=float:[128, 24, 128]
            ("reshape_0054", "reshape_run", [(128, 2, 12, 128), (128, 24, 128), "float32"]),
            ("reshape_0055", "reshape_run", [(128, 24, 128), (128, 2, 12, 128), "float32"]),
            # float-int32:[128, 2, 12, 64]-[3]=float:[128, 24, 64]
            ("reshape_0056", "reshape_run", [(128, 2, 12, 64), (128, 24, 64), "float32"]),
            ("reshape_0057", "reshape_run", [(128, 24, 64), (128, 2, 12, 64), "float32"]),
            # float-int32:[128, 256, 12, 128]-[3]=float:[128, 3072, 128]
            ("reshape_0058", "reshape_run", [(128, 256, 12, 128), (128, 3072, 128), "float32"]),
            ("reshape_0059", "reshape_run", [(128, 3072, 128), (128, 256, 12, 128), "float32"]),
            # float-int32:[128, 256, 12, 64]-[3]=float:[128, 3072, 64]
            ("reshape_0060", "reshape_run", [(128, 256, 12, 64), (128, 3072, 64), "float32"]),
            ("reshape_0061", "reshape_run", [(128, 3072, 64), (128, 256, 12, 64), "float32"]),
            # float-int32:[128]-[2]=float:[128, 1]
            ("reshape_0062", "reshape_run", [(128,), (128, 1), "float32"]),
            ("reshape_0063", "reshape_run", [(128, 1), (128,), "float32"]),
            # float-int32:[128, 32, 12, 128]-[3]=float:[128, 384, 128]
            ("reshape_0064", "reshape_run", [(128, 32, 12, 128), (128, 384, 128), "float32"]),
            ("reshape_0065", "reshape_run", [(128, 384, 128), (128, 32, 12, 128), "float32"]),
            # float-int32:[128, 32, 12, 64]-[3]=float:[128, 384, 64]
            ("reshape_0066", "reshape_run", [(128, 32, 12, 64), (128, 384, 64), "float32"]),
            ("reshape_0067", "reshape_run", [(128, 384, 64), (128, 32, 12, 64), "float32"]),
            # float-int32:[128, 4, 12, 128]-[3]=float:[128, 48, 128]
            ("reshape_0068", "reshape_run", [(128, 4, 12, 128), (128, 48, 128), "float32"]),
            ("reshape_0069", "reshape_run", [(128, 48, 128), (128, 4, 12, 128), "float32"]),
            # float-int32:[128, 4, 12, 64]-[3]=float:[128, 48, 64]
            ("reshape_0070", "reshape_run", [(128, 4, 12, 64), (128, 48, 64), "float32"]),
            ("reshape_0071", "reshape_run", [(128, 48, 64), (128, 4, 12, 64), "float32"]),
            # float-int32:[128, 512, 12, 128]-[3]=float:[128, 6144, 128]
            ("reshape_0072", "reshape_run", [(128, 512, 12, 128), (128, 6144, 128), "float32"]),
            ("reshape_0073", "reshape_run", [(128, 6144, 128), (128, 512, 12, 128), "float32"]),
            # float-int32:[128, 512, 12, 64]-[3]=float:[128, 6144, 64]
            ("reshape_0074", "reshape_run", [(128, 512, 12, 64), (128, 6144, 64), "float32"]),
            ("reshape_0075", "reshape_run", [(128, 6144, 64), (128, 512, 12, 64), "float32"]),
            # float-int32:[128, 64, 12, 128]-[3]=float:[128, 768, 128]
            ("reshape_0076", "reshape_run", [(128, 64, 12, 128), (128, 768, 128), "float32"]),
            ("reshape_0077", "reshape_run", [(128, 768, 128), (128, 64, 12, 128), "float32"]),
            # float-int32:[128, 64, 12, 64]-[3]=float:[128, 768, 64]
            ("reshape_0078", "reshape_run", [(128, 64, 12, 64), (128, 768, 64), "float32"]),
            ("reshape_0079", "reshape_run", [(128, 768, 64), (128, 64, 12, 64), "float32"]),
            # float-int32:[128, 768]-[2]=float:[128, 768]
            ("reshape_0080", "reshape_run", [(128, 768), (128, 768), "float32"]),
            # float-int32:[128, 768]-[3]=float:[128, 1, 768]
            ("reshape_0081", "reshape_run", [(128, 768), (128, 1, 768), "float32"]),
            ("reshape_0082", "reshape_run", [(128, 1, 768), (128, 768), "float32"]),
            # float-int32:[128, 8, 12, 128]-[3]=float:[128, 96, 128]
            ("reshape_0083", "reshape_run", [(128, 8, 12, 128), (128, 96, 128), "float32"]),
            ("reshape_0084", "reshape_run", [(128, 96, 128), (128, 8, 12, 128), "float32"]),
            # float-int32:[128, 8, 12, 64]-[3]=float:[128, 96, 64]
            ("reshape_0085", "reshape_run", [(128, 8, 12, 64), (128, 96, 64), "float32"]),
            ("reshape_0086", "reshape_run", [(128, 96, 64), (128, 8, 12, 64), "float32"]),
            # float-int32:[1]-[2]=float:[1, 1]
            ("reshape_0087", "reshape_run", [(1,), (1, 1), "float32"]),
            ("reshape_0088", "reshape_run", [(1, 1), (1,), "float32"]),
            # float-int32:[1, 2]-[]=float:[2]
            ("reshape_0089", "reshape_run", [(1, 2), (2,), "float32"]),
            ("reshape_0090", "reshape_run", [(2,), (1, 2), "float32"]),
            # float-int32:[1, 3072]-[]=float:[3072]
            ("reshape_0091", "reshape_run", [(1, 3072), (3072,), "float32"]),
            ("reshape_0092", "reshape_run", [(3072,), (1, 3072), "float32"]),
            # float-int32:[131072, 768]-[2]=float:[131072, 768]
            ("reshape_0093", "reshape_run", [(131072, 768), (131072, 768), "float32"]),
            # float-int32:[160]-[2]=float:[160, 1]
            ("reshape_0094", "reshape_run", [(160,), (160, 1), "float32"]),
            ("reshape_0095", "reshape_run", [(160, 1), (160,), "float32"]),
            # float-int32:[160]-[]=float:[8, 20]
            ("reshape_0096", "reshape_run", [(160,), (8, 20), "float32"]),
            ("reshape_0097", "reshape_run", [(8, 20), (160,), "float32"]),
            # float-int32:[16, 128, 12, 64]-[2]=float:[2048, 768]
            ("reshape_0098", "reshape_run", [(16, 128, 12, 64), (2048, 768), "float32"]),
            ("reshape_0099", "reshape_run", [(2048, 768), (16, 128, 12, 64), "float32"]),
            # float-int32:[16, 128, 768]-[2]=float:[2048, 768]
            ("reshape_0100", "reshape_run", [(16, 128, 768), (2048, 768), "float32"]),
            ("reshape_0101", "reshape_run", [(2048, 768), (16, 128, 768), "float32"]),
            # float-int32:[16, 20]-[]=float:[320]
            ("reshape_0102", "reshape_run", [(16, 20), (320,), "float32"]),
            ("reshape_0103", "reshape_run", [(320,), (16, 20), "float32"]),
            # float-int32:[16]-[2]=float:[16, 1]
            ("reshape_0104", "reshape_run", [(16,), (16, 1), "float32"]),
            ("reshape_0105", "reshape_run", [(16, 1), (16,), "float32"]),
            # float-int32:[16384, 768]-[2]=float:[16384, 768]
            ("reshape_0106", "reshape_run", [(16384, 768), (16384, 768), "float32"]),
            # float-int32:[16, 768]-[3]=float:[16, 1, 768]
            ("reshape_0107", "reshape_run", [(16, 768), (16, 1, 768), "float32"]),
            ("reshape_0108", "reshape_run", [(16, 1, 768), (16, 768), "float32"]),
            # float-int32:[1, 768]-[3]=float:[1, 1, 768]
            ("reshape_0109", "reshape_run", [(1, 768), (1, 1, 768), "float32"]),
            ("reshape_0110", "reshape_run", [(1, 1, 768), (1, 768), "float32"]),
            # float-int32:[1, 768]-[]=float:[768]
            ("reshape_0111", "reshape_run", [(1, 768), (768,), "float32"]),
            ("reshape_0112", "reshape_run", [(768,), (1, 768), "float32"]),
            # float-int32:[1]-[]=float:[1]
            ("reshape_0113", "reshape_run", [(1,), (1,), "float32"]),
            # float-int32:[20]-[2]=float:[20, 1]
            ("reshape_0114", "reshape_run", [(20,), (20, 1), "float32"]),
            ("reshape_0115", "reshape_run", [(20, 1), (20,), "float32"]),
            # float-int32:[20480]-[2]=float:[20480, 1]
            ("reshape_0116", "reshape_run", [(20480,), (20480, 1), "float32"]),
            ("reshape_0117", "reshape_run", [(20480, 1), (20480,), "float32"]),
            # float-int32:[2048, 768]-[2]=float:[2048, 768]
            ("reshape_0118", "reshape_run", [(2048, 768), (2048, 768), "float32"]),
            # float-int32:[21128, 768]-[2]=float:[21128, 768]
            ("reshape_0119", "reshape_run", [(21128, 768), (21128, 768), "float32"]),
            # float-int32:[21128]-[]=float:[21128]
            ("reshape_0120", "reshape_run", [(21128,), (21128,), "float32"]),
            # float-int32:[2, 128, 12, 64]-[2]=float:[256, 768]
            ("reshape_0121", "reshape_run", [(2, 128, 12, 64), (256, 768), "float32"]),
            ("reshape_0122", "reshape_run", [(256, 768), (2, 128, 12, 64), "float32"]),
            # float-int32:[2, 128, 768]-[2]=float:[256, 768]
            ("reshape_0123", "reshape_run", [(2, 128, 768), (256, 768), "float32"]),
            ("reshape_0124", "reshape_run", [(256, 768), (2, 128, 768), "float32"]),
            # float-int32:[2, 20]-[]=float:[40]
            ("reshape_0125", "reshape_run", [(2, 20), (40,), "float32"]),
            ("reshape_0126", "reshape_run", [(40,), (2, 20), "float32"]),
            # float-int32:[2]-[2]=float:[2, 1]
            ("reshape_0127", "reshape_run", [(2,), (2, 1), "float32"]),
            ("reshape_0128", "reshape_run", [(2, 1), (2,), "float32"]),
            # float-int32:[2560]-[2]=float:[2560, 1]
            ("reshape_0129", "reshape_run", [(2560,), (2560, 1), "float32"]),
            ("reshape_0130", "reshape_run", [(2560, 1), (2560,), "float32"]),
            # float-int32:[256, 128, 12, 64]-[2]=float:[32768, 768]
            ("reshape_0131", "reshape_run", [(256, 128, 12, 64), (32768, 768), "float32"]),
            ("reshape_0132", "reshape_run", [(32768, 768), (256, 128, 12, 64), "float32"]),
            # float-int32:[256, 128, 768]-[2]=float:[32768, 768]
            ("reshape_0133", "reshape_run", [(256, 128, 768), (32768, 768), "float32"]),
            ("reshape_0134", "reshape_run", [(32768, 768), (256, 128, 768), "float32"]),
            # float-int32:[256, 20]-[]=float:[5120]
            ("reshape_0135", "reshape_run", [(256, 20), (5120,), "float32"]),
            ("reshape_0136", "reshape_run", [(5120,), (256, 20), "float32"]),
            # float-int32:[256]-[2]=float:[256, 1]
            ("reshape_0137", "reshape_run", [(256,), (256, 1), "float32"]),
            ("reshape_0138", "reshape_run", [(256, 1), (256,), "float32"]),
            # float-int32:[256, 768]-[2]=float:[256, 768]
            ("reshape_0139", "reshape_run", [(256, 768), (256, 768), "float32"]),
            # float-int32:[256, 768]-[3]=float:[256, 1, 768]
            ("reshape_0140", "reshape_run", [(256, 768), (256, 1, 768), "float32"]),
            ("reshape_0141", "reshape_run", [(256, 1, 768), (256, 768), "float32"]),
            # float-int32:[2, 768]-[2]=float:[2, 768]
            ("reshape_0142", "reshape_run", [(2, 768), (2, 768), "float32"]),
            # float-int32:[2, 768]-[3]=float:[2, 1, 768]
            ("reshape_0143", "reshape_run", [(2, 768), (2, 1, 768), "float32"]),
            ("reshape_0144", "reshape_run", [(2, 1, 768), (2, 768), "float32"]),
            # float-int32:[2]-[]=float:[2]
            ("reshape_0145", "reshape_run", [(2,), (2,), "float32"]),
            # float-int32:[3072, 768]-[2]=float:[3072, 768]
            ("reshape_0146", "reshape_run", [(3072, 768), (3072, 768), "float32"]),
            # float-int32:[3072]-[]=float:[3072]
            ("reshape_0147", "reshape_run", [(3072,), (3072,), "float32"]),
            # float-int32:[320]-[2]=float:[320, 1]
            ("reshape_0148", "reshape_run", [(320,), (320, 1), "float32"]),
            ("reshape_0149", "reshape_run", [(320, 1), (320,), "float32"]),
            # float-int32:[32, 128, 12, 64]-[2]=float:[4096, 768]
            ("reshape_0150", "reshape_run", [(32, 128, 12, 64), (4096, 768), "float32"]),
            ("reshape_0151", "reshape_run", [(4096, 768), (32, 128, 12, 64), "float32"]),
            # float-int32:[32, 128, 768]-[2]=float:[4096, 768]
            ("reshape_0152", "reshape_run", [(32, 128, 768), (4096, 768), "float32"]),
            ("reshape_0153", "reshape_run", [(4096, 768), (32, 128, 768), "float32"]),
            # float-int32:[32, 20]-[]=float:[640]
            ("reshape_0154", "reshape_run", [(32, 20), (640,), "float32"]),
            ("reshape_0155", "reshape_run", [(640,), (32, 20), "float32"]),
            # float-int32:[32]-[2]=float:[32, 1]
            ("reshape_0156", "reshape_run", [(32,), (32, 1), "float32"]),
            ("reshape_0157", "reshape_run", [(32, 1), (32,), "float32"]),
            # float-int32:[32, 768]-[3]=float:[32, 1, 768]
            ("reshape_0158", "reshape_run", [(32, 768), (32, 1, 768), "float32"]),
            ("reshape_0159", "reshape_run", [(32, 1, 768), (32, 768), "float32"]),
            # float-int32:[32768, 768]-[2]=float:[32768, 768]
            ("reshape_0160", "reshape_run", [(32768, 768), (32768, 768), "float32"]),
            # float-int32:[33, 64]-[2]=float:[33, 64]
            ("reshape_0161", "reshape_run", [(33, 64), (33, 64), "float32"]),
            # float-int32:[40]-[2]=float:[40, 1]
            ("reshape_0162", "reshape_run", [(40,), (40, 1), "float32"]),
            ("reshape_0163", "reshape_run", [(40, 1), (40,), "float32"]),
            # float-int32:[4096, 768]-[2]=float:[4096, 768]
            ("reshape_0164", "reshape_run", [(4096, 768), (4096, 768), "float32"]),
            # float-int32:[4, 128, 12, 64]-[2]=float:[512, 768]
            ("reshape_0165", "reshape_run", [(4, 128, 12, 64), (512, 768), "float32"]),
            ("reshape_0166", "reshape_run", [(512, 768), (4, 128, 12, 64), "float32"]),
            # float-int32:[4, 128, 768]-[2]=float:[512, 768]
            ("reshape_0167", "reshape_run", [(4, 128, 768), (512, 768), "float32"]),
            ("reshape_0168", "reshape_run", [(512, 768), (4, 128, 768), "float32"]),
            # float-int32:[4, 20]-[]=float:[80]
            ("reshape_0169", "reshape_run", [(4, 20), (80,), "float32"]),
            ("reshape_0170", "reshape_run", [(80,), (4, 20), "float32"]),
            # float-int32:[4]-[2]=float:[4, 1]
            ("reshape_0171", "reshape_run", [(4,), (4, 1), "float32"]),
            ("reshape_0172", "reshape_run", [(4, 1), (4,), "float32"]),
            # float-int32:[4, 768]-[3]=float:[4, 1, 768]
            ("reshape_0173", "reshape_run", [(4, 768), (4, 1, 768), "float32"]),
            ("reshape_0174", "reshape_run", [(4, 1, 768), (4, 768), "float32"]),
            # float-int32:[5120]-[2]=float:[5120, 1]
            ("reshape_0175", "reshape_run", [(5120,), (5120, 1), "float32"]),
            ("reshape_0176", "reshape_run", [(5120, 1), (5120,), "float32"]),
            # float-int32:[512, 128, 12, 64]-[2]=float:[65536, 768]
            ("reshape_0177", "reshape_run", [(512, 128, 12, 64), (65536, 768), "float32"]),
            ("reshape_0178", "reshape_run", [(65536, 768), (512, 128, 12, 64), "float32"]),
            # float-int32:[512, 128, 768]-[2]=float:[65536, 768]
            ("reshape_0179", "reshape_run", [(512, 128, 768), (65536, 768), "float32"]),
            ("reshape_0180", "reshape_run", [(65536, 768), (512, 128, 768), "float32"]),
            # float-int32:[512]-[2]=float:[512, 1]
            ("reshape_0181", "reshape_run", [(512,), (512, 1), "float32"]),
            ("reshape_0182", "reshape_run", [(512, 1), (512,), "float32"]),
            # float-int32:[512, 768]-[2]=float:[512, 768]
            ("reshape_0183", "reshape_run", [(512, 768), (512, 768), "float32"]),
            # float-int32:[512, 768]-[3]=float:[512, 1, 768]
            ("reshape_0184", "reshape_run", [(512, 768), (512, 1, 768), "float32"]),
            ("reshape_0185", "reshape_run", [(512, 1, 768), (512, 768), "float32"]),
            # float-int32:[640]-[2]=float:[640, 1]
            ("reshape_0186", "reshape_run", [(640,), (640, 1), "float32"]),
            ("reshape_0187", "reshape_run", [(640, 1), (640,), "float32"]),
            # float-int32:[64, 128, 12, 64]-[2]=float:[8192, 768]
            ("reshape_0188", "reshape_run", [(64, 128, 12, 64), (8192, 768), "float32"]),
            ("reshape_0189", "reshape_run", [(8192, 768), (64, 128, 12, 64), "float32"]),
            # float-int32:[64, 128, 768]-[2]=float:[8192, 768]
            ("reshape_0190", "reshape_run", [(64, 128, 768), (8192, 768), "float32"]),
            ("reshape_0191", "reshape_run", [(8192, 768), (64, 128, 768), "float32"]),
            # float-int32:[64]-[2]=float:[64, 1]
            ("reshape_0192", "reshape_run", [(64,), (64, 1), "float32"]),
            ("reshape_0193", "reshape_run", [(64, 1), (64,), "float32"]),
            # float-int32:[64, 768]-[3]=float:[64, 1, 768]
            ("reshape_0194", "reshape_run", [(64, 768), (64, 1, 768), "float32"]),
            ("reshape_0195", "reshape_run", [(64, 1, 768), (64, 768), "float32"]),
            # float-int32:[65536, 768]-[2]=float:[65536, 768]
            ("reshape_0196", "reshape_run", [(65536, 768), (65536, 768), "float32"]),
            # float-int32:[768, 3072]-[2]=float:[768, 3072]
            ("reshape_0197", "reshape_run", [(768, 3072), (768, 3072), "float32"]),
            # float-int32:[768, 768]-[2]=float:[768, 768]
            ("reshape_0198", "reshape_run", [(768, 768), (768, 768), "float32"]),
            # float-int32:[768]-[]=float:[768]
            ("reshape_0199", "reshape_run", [(768,), (768,), "float32"]),
            # float-int32:[80]-[2]=float:[80, 1]
            ("reshape_0200", "reshape_run", [(80,), (80, 1), "float32"]),
            ("reshape_0201", "reshape_run", [(80, 1), (80,), "float32"]),
            # float-int32:[8192, 768]-[2]=float:[8192, 768]
            ("reshape_0202", "reshape_run", [(8192, 768), (8192, 768), "float32"]),
            # float-int32:[8]-[2]=float:[8, 1]
            ("reshape_0203", "reshape_run", [(8,), (8, 1), "float32"]),
            ("reshape_0204", "reshape_run", [(8, 1), (8,), "float32"]),
            # float-int32:[8, 768]-[3]=float:[8, 1, 768]
            ("reshape_0205", "reshape_run", [(8, 768), (8, 1, 768), "float32"]),
            ("reshape_0206", "reshape_run", [(8, 1, 768), (8, 768), "float32"]),
            # int32-int32:[10240]-[]=int32:[512, 20]
            ("reshape_0207", "reshape_run", [(10240,), (512, 20), "int32"]),
            ("reshape_0208", "reshape_run", [(512, 20), (10240,), "int32"]),
            # int32-int32:[1024, 1, 128]-[3]=int32:[1024, 128]
            ("reshape_0209", "reshape_run", [(1024, 1, 128), (1024, 128), "int32"]),
            ("reshape_0210", "reshape_run", [(1024, 128), (1024, 1, 128), "int32"]),
            # int32-int32:[1024, 128, 1]-[]=int32:[131072]
            ("reshape_0211", "reshape_run", [(1024, 128, 1), (131072,), "int32"]),
            ("reshape_0212", "reshape_run", [(131072,), (1024, 128, 1), "int32"]),
            # int32-int32:[1024, 128]-[]=int32:[131072]
            ("reshape_0213", "reshape_run", [(1024, 128), (131072,), "int32"]),
            ("reshape_0214", "reshape_run", [(131072,), (1024, 128), "int32"]),
            # int32-int32:[1024, 1]-[2]=int32:[1024]
            ("reshape_0215", "reshape_run", [(1024, 1), (1024,), "int32"]),
            ("reshape_0216", "reshape_run", [(1024,), (1024, 1), "int32"]),
            # int32-int32:[1024, 20]-[]=int32:[20480]
            ("reshape_0217", "reshape_run", [(1024, 20), (20480,), "int32"]),
            ("reshape_0218", "reshape_run", [(20480,), (1024, 20), "int32"]),
            # int32-int32:[1024]-[]=int32:[8, 128]
            ("reshape_0219", "reshape_run", [(1024,), (8, 128), "int32"]),
            ("reshape_0220", "reshape_run", [(8, 128), (1024,), "int32"]),
            # int32-int32:[1024]-[]=int32:[8, 128, 1]
            ("reshape_0221", "reshape_run", [(1024,), (8, 128, 1), "int32"]),
            ("reshape_0222", "reshape_run", [(8, 128, 1), (1024,), "int32"]),
            # int32-int32:[1, 1, 128]-[3]=int32:[1, 128]
            ("reshape_0223", "reshape_run", [(1, 1, 128), (1, 128), "int32"]),
            ("reshape_0224", "reshape_run", [(1, 128), (1, 1, 128), "int32"]),
            # int32-int32:[1, 128, 1]-[]=int32:[128]
            ("reshape_0225", "reshape_run", [(1, 128, 1), (128,), "int32"]),
            ("reshape_0226", "reshape_run", [(128,), (1, 128, 1), "int32"]),
            # int32-int32:[1, 128]-[]=int32:[128]
            ("reshape_0227", "reshape_run", [(1, 128), (128,), "int32"]),
            ("reshape_0228", "reshape_run", [(128,), (1, 128), "int32"]),
            # int32-int32:[1, 1]-[2]=int32:[1]
            ("reshape_0229", "reshape_run", [(1, 1), (1,), "int32"]),
            ("reshape_0230", "reshape_run", [(1,), (1, 1), "int32"]),
            # int32-int32:[1, 20]-[]=int32:[20]
            ("reshape_0231", "reshape_run", [(1, 20), (20,), "int32"]),
            ("reshape_0232", "reshape_run", [(20,), (1, 20), "int32"]),
            # int32-int32:[1280]-[]=int32:[64, 20]
            ("reshape_0233", "reshape_run", [(1280,), (64, 20), "int32"]),
            ("reshape_0234", "reshape_run", [(64, 20), (1280,), "int32"]),
            # int32-int32:[128, 1, 128]-[3]=int32:[128, 128]
            ("reshape_0235", "reshape_run", [(128, 1, 128), (128, 128), "int32"]),
            ("reshape_0236", "reshape_run", [(128, 128), (128, 1, 128), "int32"]),
            # int32-int32:[128, 128, 1]-[]=int32:[16384]
            ("reshape_0237", "reshape_run", [(128, 128, 1), (16384,), "int32"]),
            ("reshape_0238", "reshape_run", [(16384,), (128, 128, 1), "int32"]),
            # int32-int32:[128, 128]-[2]=int32:[16384]
            ("reshape_0239", "reshape_run", [(128, 128), (16384,), "int32"]),
            ("reshape_0240", "reshape_run", [(16384,), (128, 128), "int32"]),
            # int32-int32:[128, 1]-[2]=int32:[128]
            ("reshape_0241", "reshape_run", [(128, 1), (128,), "int32"]),
            ("reshape_0242", "reshape_run", [(128,), (128, 1), "int32"]),
            # int32-int32:[128, 20]-[]=int32:[2560]
            ("reshape_0243", "reshape_run", [(128, 20), (2560,), "int32"]),
            ("reshape_0244", "reshape_run", [(2560,), (128, 20), "int32"]),
            # int32-int32:[160]-[]=int32:[8, 20]
            ("reshape_0245", "reshape_run", [(160,), (8, 20), "int32"]),
            ("reshape_0246", "reshape_run", [(8, 20), (160,), "int32"]),
            # int32-int32:[16, 1, 128]-[3]=int32:[16, 128]
            ("reshape_0247", "reshape_run", [(16, 1, 128), (16, 128), "int32"]),
            ("reshape_0248", "reshape_run", [(16, 128), (16, 1, 128), "int32"]),
            # int32-int32:[16, 128, 1]-[]=int32:[2048]
            ("reshape_0249", "reshape_run", [(16, 128, 1), (2048,), "int32"]),
            ("reshape_0250", "reshape_run", [(2048,), (16, 128, 1), "int32"]),
            # int32-int32:[16, 128]-[]=int32:[2048]
            ("reshape_0251", "reshape_run", [(16, 128), (2048,), "int32"]),
            ("reshape_0252", "reshape_run", [(2048,), (16, 128), "int32"]),
            # int32-int32:[16, 1]-[2]=int32:[16]
            ("reshape_0253", "reshape_run", [(16, 1), (16,), "int32"]),
            ("reshape_0254", "reshape_run", [(16,), (16, 1), "int32"]),
            # int32-int32:[16, 20]-[]=int32:[320]
            ("reshape_0255", "reshape_run", [(16, 20), (320,), "int32"]),
            ("reshape_0256", "reshape_run", [(320,), (16, 20), "int32"]),
            # int32-int32:[2, 1, 128]-[3]=int32:[2, 128]
            ("reshape_0257", "reshape_run", [(2, 1, 128), (2, 128), "int32"]),
            ("reshape_0258", "reshape_run", [(2, 128), (2, 1, 128), "int32"]),
            # int32-int32:[2, 128, 1]-[]=int32:[256]
            ("reshape_0259", "reshape_run", [(2, 128, 1), (256,), "int32"]),
            ("reshape_0260", "reshape_run", [(256,), (2, 128, 1), "int32"]),
            # int32-int32:[2, 128]-[]=int32:[256]
            ("reshape_0261", "reshape_run", [(2, 128), (256,), "int32"]),
            ("reshape_0262", "reshape_run", [(256,), (2, 128), "int32"]),
            # int32-int32:[2, 1]-[2]=int32:[2]
            ("reshape_0263", "reshape_run", [(2, 1), (2,), "int32"]),
            ("reshape_0264", "reshape_run", [(2,), (2, 1), "int32"]),
            # int32-int32:[2, 20]-[]=int32:[40]
            ("reshape_0265", "reshape_run", [(2, 20), (40,), "int32"]),
            ("reshape_0266", "reshape_run", [(40,), (2, 20), "int32"]),
            # int32-int32:[256, 1, 128]-[3]=int32:[256, 128]
            ("reshape_0267", "reshape_run", [(256, 1, 128), (256, 128), "int32"]),
            ("reshape_0268", "reshape_run", [(256, 128), (256, 1, 128), "int32"]),
            # int32-int32:[256, 128, 1]-[]=int32:[32768]
            ("reshape_0269", "reshape_run", [(256, 128, 1), (32768,), "int32"]),
            ("reshape_0270", "reshape_run", [(32768,), (256, 128, 1), "int32"]),
            # int32-int32:[256, 128]-[]=int32:[32768]
            ("reshape_0271", "reshape_run", [(256, 128), (32768,), "int32"]),
            ("reshape_0272", "reshape_run", [(32768,), (256, 128), "int32"]),
            # int32-int32:[256, 1]-[2]=int32:[256]
            ("reshape_0273", "reshape_run", [(256, 1), (256,), "int32"]),
            ("reshape_0274", "reshape_run", [(256,), (256, 1), "int32"]),
            # int32-int32:[256, 20]-[]=int32:[5120]
            ("reshape_0275", "reshape_run", [(256, 20), (5120,), "int32"]),
            ("reshape_0276", "reshape_run", [(5120,), (256, 20), "int32"]),
            # int32-int32:[32, 1, 128]-[3]=int32:[32, 128]
            ("reshape_0277", "reshape_run", [(32, 1, 128), (32, 128), "int32"]),
            ("reshape_0278", "reshape_run", [(32, 128), (32, 1, 128), "int32"]),
            # int32-int32:[32, 128, 1]-[]=int32:[4096]
            ("reshape_0279", "reshape_run", [(32, 128, 1), (4096,), "int32"]),
            ("reshape_0280", "reshape_run", [(4096,), (32, 128, 1), "int32"]),
            # int32-int32:[32, 128]-[]=int32:[4096]
            ("reshape_0281", "reshape_run", [(32, 128), (4096,), "int32"]),
            ("reshape_0282", "reshape_run", [(4096,), (32, 128), "int32"]),
            # int32-int32:[32, 1]-[2]=int32:[32]
            ("reshape_0283", "reshape_run", [(32, 1), (32,), "int32"]),
            ("reshape_0284", "reshape_run", [(32,), (32, 1), "int32"]),
            # int32-int32:[32, 20]-[]=int32:[640]
            ("reshape_0285", "reshape_run", [(32, 20), (640,), "int32"]),
            ("reshape_0286", "reshape_run", [(640,), (32, 20), "int32"]),
            # int32-int32:[4, 1, 128]-[3]=int32:[4, 128]
            ("reshape_0287", "reshape_run", [(4, 1, 128), (4, 128), "int32"]),
            ("reshape_0288", "reshape_run", [(4, 128), (4, 1, 128), "int32"]),
            # int32-int32:[4, 128, 1]-[]=int32:[512]
            ("reshape_0289", "reshape_run", [(4, 128, 1), (512,), "int32"]),
            ("reshape_0290", "reshape_run", [(512,), (4, 128, 1), "int32"]),
            # int32-int32:[4, 128]-[]=int32:[512]
            ("reshape_0291", "reshape_run", [(4, 128), (512,), "int32"]),
            ("reshape_0292", "reshape_run", [(512,), (4, 128), "int32"]),
            # int32-int32:[4, 1]-[2]=int32:[4]
            ("reshape_0293", "reshape_run", [(4, 1), (4,), "int32"]),
            ("reshape_0294", "reshape_run", [(4,), (4, 1), "int32"]),
            # int32-int32:[4, 20]-[]=int32:[80]
            ("reshape_0295", "reshape_run", [(4, 20), (80,), "int32"]),
            ("reshape_0296", "reshape_run", [(80,), (4, 20), "int32"]),
            # int32-int32:[512, 1, 128]-[3]=int32:[512, 128]
            ("reshape_0297", "reshape_run", [(512, 1, 128), (512, 128), "int32"]),
            ("reshape_0298", "reshape_run", [(512, 128), (512, 1, 128), "int32"]),
            # int32-int32:[512, 128, 1]-[]=int32:[65536]
            ("reshape_0299", "reshape_run", [(512, 128, 1), (65536,), "int32"]),
            ("reshape_0300", "reshape_run", [(65536,), (512, 128, 1), "int32"]),
            # int32-int32:[512, 128]-[]=int32:[65536]
            ("reshape_0301", "reshape_run", [(512, 128), (65536,), "int32"]),
            ("reshape_0302", "reshape_run", [(65536,), (512, 128), "int32"]),
            # int32-int32:[512, 1]-[2]=int32:[512]
            ("reshape_0303", "reshape_run", [(512, 1), (512,), "int32"]),
            ("reshape_0304", "reshape_run", [(512,), (512, 1), "int32"]),
            # int32-int32:[64, 1, 128]-[3]=int32:[64, 128]
            ("reshape_0305", "reshape_run", [(64, 1, 128), (64, 128), "int32"]),
            ("reshape_0306", "reshape_run", [(64, 128), (64, 1, 128), "int32"]),
            # int32-int32:[64, 128, 1]-[]=int32:[8192]
            ("reshape_0307", "reshape_run", [(64, 128, 1), (8192,), "int32"]),
            ("reshape_0308", "reshape_run", [(8192,), (64, 128, 1), "int32"]),
            # int32-int32:[64, 128]-[]=int32:[8192]
            ("reshape_0309", "reshape_run", [(64, 128), (8192,), "int32"]),
            ("reshape_0310", "reshape_run", [(8192,), (64, 128), "int32"]),
            # int32-int32:[64, 1]-[2]=int32:[64]
            ("reshape_0311", "reshape_run", [(64, 1), (64,), "int32"]),
            ("reshape_0312", "reshape_run", [(64,), (64, 1), "int32"]),
            # int32-int32:[8, 1, 128]-[3]=int32:[8, 128]
            ("reshape_0313", "reshape_run", [(8, 1, 128), (8, 128), "int32"]),
            ("reshape_0314", "reshape_run", [(8, 128), (8, 1, 128), "int32"]),
            # int32-int32:[8, 1]-[2]=int32:[8]
            ("reshape_0315", "reshape_run", [(8, 1), (8,), "int32"]),
            ("reshape_0316", "reshape_run", [(8,), (8, 1), "int32"]),

            # float:[1024, 12, 128, 128]=float:[1024, 12, 128, 128]
            ("softmax_001", "softmax_run", ((1024, 12, 128, 128), "float32", -1, "cce_softmax_fp32")),
            # float:[1, 12, 128, 128]=float:[1, 12, 128, 128]
            ("softmax_002", "softmax_run", ((1, 12, 128, 128), "float32", -1, "cce_softmax_fp32")),
            # float:[128, 12, 128, 128]=float:[128, 12, 128, 128]
            ("softmax_003", "softmax_run", ((128, 12, 128, 128), "float32", -1, "cce_softmax_fp32")),
            # float:[16, 12, 128, 128]=float:[16, 12, 128, 128]
            ("softmax_004", "softmax_run", ((16, 12, 128, 128), "float32", -1, "cce_softmax_fp32")),
            # float:[2, 12, 128, 128]=float:[2, 12, 128, 128]
            ("softmax_005", "softmax_run", ((2, 12, 128, 128), "float32", -1, "cce_softmax_fp32")),
            # float:[256, 12, 128, 128]=float:[256, 12, 128, 128]
            ("softmax_006", "softmax_run", ((256, 12, 128, 128), "float32", -1, "cce_softmax_fp32")),
            # float:[32, 12, 128, 128]=float:[32, 12, 128, 128]
            ("softmax_007", "softmax_run", ((32, 12, 128, 128), "float32", -1, "cce_softmax_fp32")),
            # float:[4, 12, 128, 128]=float:[4, 12, 128, 128]
            ("softmax_008", "softmax_run", ((4, 12, 128, 128), "float32", -1, "cce_softmax_fp32")),
            # float:[512, 12, 128, 128]=float:[512, 12, 128, 128]
            ("softmax_009", "softmax_run", ((512, 12, 128, 128), "float32", -1, "cce_softmax_fp32")),
            # float:[64, 12, 128, 128]=float:[64, 12, 128, 128]
            ("softmax_0010", "softmax_run", ((64, 12, 128, 128), "float32", -1, "cce_softmax_fp32")),
            # float:[8, 12, 128, 128]=float:[8, 12, 128, 128]
            ("softmax_0011", "softmax_run", ((8, 12, 128, 128), "float32", -1, "cce_softmax_fp32")),

            # pow OP
            # float - float:[1280, 768] - [] = float:[1280, 768]
            ("pow_001", "pow_run", ((1280, 768), (1,), 'float32')),
            # float - float:[] - [] = float:[]
            ("pow_002", "pow_run", ((1,), (1,), 'float32')),
            # half - half:[8192, 3072] - [] = half:[8192, 3072]
            ("pow_003", "pow_run", ((8192, 3072), (8192, 3072), 'float16')),

            # reciprocal OP
            # float - float:[160, 1024] = float:[160, 1024]
            ("reciprocal_001", "reciprocal_run", ((160, 1024), 'float32'),),
            # float - float:[] = float:[]
            ("reciprocal_002", "reciprocal_run", ((1,), 'float32'),),

            # bert联调large新增shape
            # Tile OP
            # float-int32:[10240, 1]-[2]=float:[10240, 21128]
            ("tile_001", "tile_run", ((10240, 1), "float32", (1, 21128))),
            # float-int32:[1024, 1]-[2]=float:[1024, 2]
            ("tile_002", "tile_run", ((1024, 1), "float32", (1, 2))),
            # float-int32:[1, 1]-[2]=float:[1, 2]
            ("tile_003", "tile_run", ((1, 1), "float32", (1, 2))),
            # float-int32:[1]-[1]=float:[1]
            ("tile_004", "tile_run", ((1,), "float32", (1, 1))),
            # float-int32:[1]-[1]=float:[1024]
            ("tile_005", "tile_run", ((1,), "float32", (1, 1024))),
            # float-int32:[1]-[1]=float:[10240]
            ("tile_006", "tile_run", ((1,), "float32", (1, 10240))),
            # float-int32:[1]-[1]=float:[128]
            ("tile_007", "tile_run", ((1,), "float32", (1, 128))),
            # float-int32:[1]-[1]=float:[1280]
            ("tile_008", "tile_run", ((1,), "float32", (1, 1280))),
            # float-int32:[1]-[1]=float:[16]
            ("tile_009", "tile_run", ((1,), "float32", (1, 16))),
            # float-int32:[1]-[1]=float:[160]
            ("tile_010", "tile_run", ((1,), "float32", (1, 160))),
            # float-int32:[1]-[1]=float:[2]
            ("tile_011", "tile_run", ((1,), "float32", (1, 2))),
            # float-int32:[1]-[1]=float:[20]
            ("tile_012", "tile_run", ((1,), "float32", (1, 20))),
            # float-int32:[1]-[1]=float:[20480]
            ("tile_013", "tile_run", ((1,), "float32", (1, 20480))),
            # float-int32:[1]-[1]=float:[256]
            ("tile_014", "tile_run", ((1,), "float32", (1, 256))),
            # float-int32:[1]-[1]=float:[2560]
            ("tile_015", "tile_run", ((1,), "float32", (1, 2560))),
            # float-int32:[1]-[1]=float:[32]
            ("tile_016", "tile_run", ((1,), "float32", (1, 32))),
            # float-int32:[1]-[1]=float:[320]
            ("tile_017", "tile_run", ((1,), "float32", (1, 320))),
            # float-int32:[1]-[1]=float:[4]
            ("tile_018", "tile_run", ((1,), "float32", (1, 4))),
            # float-int32:[1]-[1]=float:[40]
            ("tile_019", "tile_run", ((1,), "float32", (1, 40))),
            # float-int32:[1]-[1]=float:[512]
            ("tile_020", "tile_run", ((1,), "float32", (1, 512))),
            # float-int32:[1]-[1]=float:[5120]
            ("tile_021", "tile_run", ((1,), "float32", (1, 5120))),
            # float-int32:[1]-[1]=float:[64]
            ("tile_022", "tile_run", ((1,), "float32", (1, 64))),
            # float-int32:[1]-[1]=float:[640]
            ("tile_023", "tile_run", ((1,), "float32", (1, 640))),
            # float-int32:[1]-[1]=float:[8]
            ("tile_024", "tile_run", ((1,), "float32", (1, 8))),
            # float-int32:[1]-[1]=float:[80]
            ("tile_025", "tile_run", ((1,), "float32", (1, 80))),
            # float-int32:[1280, 1]-[2]=float:[1280, 21128]
            ("tile_026", "tile_run", ((1280, 1), "float32", (1, 21128))),
            # float-int32:[128, 1]-[2]=float:[128, 2]
            ("tile_027", "tile_run", ((128, 1), "float32", (1, 2))),
            # float-int32:[160, 1]-[2]=float:[160, 21128]
            ("tile_028", "tile_run", ((160, 1), "float32", (1, 21128))),
            # float-int32:[16, 1]-[2]=float:[16, 2]
            ("tile_029", "tile_run", ((16, 1), "float32", (1, 2))),
            # float-int32:[20, 1]-[2]=float:[20, 21128]
            ("tile_030", "tile_run", ((20, 1), "float32", (1, 21128))),
            # float-int32:[20480, 1]-[2]=float:[20480, 21128]
            ("tile_031", "tile_run", ((20480, 1), "float32", (1, 21128))),
            # float-int32:[2, 1]-[2]=float:[2, 2]
            ("tile_032", "tile_run", ((2, 1), "float32", (1, 2))),
            # float-int32:[2560, 1]-[2]=float:[2560, 21128]
            ("tile_033", "tile_run", ((2560, 1), "float32", (1, 21128))),
            # float-int32:[256, 1]-[2]=float:[256, 2]
            ("tile_034", "tile_run", ((256, 1), "float32", (1, 2))),
            # float-int32:[320, 1]-[2]=float:[320, 21128]
            ("tile_035", "tile_run", ((320, 1), "float32", (1, 21128))),
            # float-int32:[32, 1]-[2]=float:[32, 2]
            ("tile_036", "tile_run", ((32, 1), "float32", (1, 2))),
            # float-int32:[40, 1]-[2]=float:[40, 21128]
            ("tile_037", "tile_run", ((40, 1), "float32", (1, 21128))),
            # float-int32:[4, 1]-[2]=float:[4, 2]
            ("tile_038", "tile_run", ((4, 1), "float32", (1, 2))),
            # float-int32:[5120, 1]-[2]=float:[5120, 21128]
            ("tile_039", "tile_run", ((5120, 1), "float32", (1, 21128))),
            # float-int32:[512, 1]-[2]=float:[512, 2]
            ("tile_040", "tile_run", ((512, 1), "float32", (1, 2))),
            # float-int32:[640, 1]-[2]=float:[640, 21128]
            ("tile_041", "tile_run", ((640, 1), "float32", (1, 21128))),
            # float-int32:[64, 1]-[2]=float:[64, 2]
            ("tile_042", "tile_run", ((64, 1), "float32", (1, 2))),
            # float-int32:[80, 1]-[2]=float:[80, 21128]
            ("tile_043", "tile_run", ((80, 1), "float32", (1, 21128))),
            # float-int32:[8, 1]-[2]=float:[8, 2]
            ("tile_044", "tile_run", ((8, 1), "float32", (1, 2))),

            # Transpose OP
            # float-int32:[10240, 1024]-[2]=float:[10240, 1024]
            ("transpose_001", "transpose_run", ((10240, 1024), (0, 1), "float32")),
            # float-int32:[1024, 1024]-[2]=float:[1024, 1024]
            ("transpose_002", "transpose_run", ((1024, 1024), (0, 1), "float32")),
            # float-int32:[1024, 128, 16, 64]-[4]=float:[1024, 16, 128, 64]
            ("transpose_003", "transpose_run", ((1024, 128, 16, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[1024, 16, 128, 128]-[4]=float:[128, 1024, 16, 128]
            ("transpose_004", "transpose_run", ((1024, 16, 128, 128), (2, 0, 1, 3), "float32")),
            # float-int32:[1024, 16, 128, 64]-[4]=float:[1024, 128, 16, 64]
            ("transpose_005", "transpose_run", ((1024, 16, 128, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[1024, 16, 128, 64]-[4]=float:[128, 1024, 16, 64]
            ("transpose_006", "transpose_run", ((1024, 16, 128, 64), (2, 0, 1, 3), "float32")),
            # float-int32:[1, 128, 16, 64]-[4]=float:[1, 16, 128, 64]
            ("transpose_007", "transpose_run", ((1, 128, 16, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[1, 16, 128, 128]-[4]=float:[128, 1, 16, 128]
            ("transpose_008", "transpose_run", ((1, 16, 128, 128), (2, 0, 1, 3), "float32")),
            # float-int32:[1, 16, 128, 64]-[4]=float:[1, 128, 16, 64]
            ("transpose_009", "transpose_run", ((1, 16, 128, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[1, 16, 128, 64]-[4]=float:[128, 1, 16, 64]
            ("transpose_0010", "transpose_run", ((1, 16, 128, 64), (2, 0, 1, 3), "float32")),
            # float-int32:[128, 1024, 16, 128]-[4]=float:[1024, 16, 128, 128]
            ("transpose_0011", "transpose_run", ((128, 1024, 16, 128), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 1024, 16, 64]-[4]=float:[1024, 16, 128, 64]
            ("transpose_0012", "transpose_run", ((128, 1024, 16, 64), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 1024]-[2]=float:[128, 1024]
            ("transpose_0013", "transpose_run", ((128, 1024), (0, 1), "float32")),
            # float-int32:[128, 1, 16, 128]-[4]=float:[1, 16, 128, 128]
            ("transpose_0014", "transpose_run", ((128, 1, 16, 128), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 1, 16, 64]-[4]=float:[1, 16, 128, 64]
            ("transpose_0015", "transpose_run", ((128, 1, 16, 64), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 128, 16, 128]-[4]=float:[128, 16, 128, 128]
            ("transpose_0016", "transpose_run", ((128, 128, 16, 128), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 128, 16, 64]-[4]=float:[128, 16, 128, 64]
            ("transpose_0017", "transpose_run", ((128, 128, 16, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[128, 128, 16, 64]-[4]=float:[128, 16, 128, 64]
            ("transpose_0018", "transpose_run", ((128, 128, 16, 64), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 128, 64]-[3]=float:[128, 128, 64]
            ("transpose_0019", "transpose_run", ((128, 128, 64), (0, 1, 2,), "float32")),
            # float-int32:[128, 16, 128, 128]-[4]=float:[128, 128, 16, 128]
            ("transpose_0020", "transpose_run", ((128, 16, 128, 128), (2, 0, 1, 3), "float32")),
            # float-int32:[128, 16, 128, 64]-[4]=float:[128, 128, 16, 64]
            ("transpose_0021", "transpose_run", ((128, 16, 128, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[128, 16, 128, 64]-[4]=float:[128, 128, 16, 64]
            ("transpose_0022", "transpose_run", ((128, 16, 128, 64), (2, 0, 1, 3), "float32")),
            # float-int32:[128, 16, 16, 128]-[4]=float:[16, 16, 128, 128]
            ("transpose_0023", "transpose_run", ((128, 16, 16, 128), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 16, 16, 64]-[4]=float:[16, 16, 128, 64]
            ("transpose_0024", "transpose_run", ((128, 16, 16, 64), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 2, 16, 128]-[4]=float:[2, 16, 128, 128]
            ("transpose_0025", "transpose_run", ((128, 2, 16, 128), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 2, 16, 64]-[4]=float:[2, 16, 128, 64]
            ("transpose_0026", "transpose_run", ((128, 2, 16, 64), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 256, 16, 128]-[4]=float:[256, 16, 128, 128]
            ("transpose_0027", "transpose_run", ((128, 256, 16, 128), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 256, 16, 64]-[4]=float:[256, 16, 128, 64]
            ("transpose_0028", "transpose_run", ((128, 256, 16, 64), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 32, 16, 128]-[4]=float:[32, 16, 128, 128]
            ("transpose_0029", "transpose_run", ((128, 32, 16, 128), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 32, 16, 64]-[4]=float:[32, 16, 128, 64]
            ("transpose_0030", "transpose_run", ((128, 32, 16, 64), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 4, 16, 128]-[4]=float:[4, 16, 128, 128]
            ("transpose_0031", "transpose_run", ((128, 4, 16, 128), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 4, 16, 64]-[4]=float:[4, 16, 128, 64]
            ("transpose_0032", "transpose_run", ((128, 4, 16, 64), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 512, 16, 128]-[4]=float:[512, 16, 128, 128]
            ("transpose_0033", "transpose_run", ((128, 512, 16, 128), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 512, 16, 64]-[4]=float:[512, 16, 128, 64]
            ("transpose_0034", "transpose_run", ((128, 512, 16, 64), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 8, 16, 128]-[4]=float:[8, 16, 128, 128]
            ("transpose_0035", "transpose_run", ((128, 8, 16, 128), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 8, 16, 64]-[4]=float:[8, 16, 128, 64]
            ("transpose_0036", "transpose_run", ((128, 8, 16, 64), (1, 2, 0, 3), "float32")),
            # float-int32:[131072, 1024]-[2]=float:[131072, 1024]
            ("transpose_0037", "transpose_run", ((131072, 1024), (0, 1), "float32")),
            # float-int32:[160, 1024]-[2]=float:[160, 1024]
            ("transpose_0038", "transpose_run", ((160, 1024), (0, 1), "float32")),
            # float-int32:[16, 128, 16, 64]-[4]=float:[16, 16, 128, 64]
            ("transpose_0039", "transpose_run", ((16, 128, 16, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[16, 16, 128, 128]-[4]=float:[128, 16, 16, 128]
            ("transpose_0040", "transpose_run", ((16, 16, 128, 128), (2, 0, 1, 3), "float32")),
            # float-int32:[16, 16, 128, 64]-[4]=float:[128, 16, 16, 64]
            ("transpose_0041", "transpose_run", ((16, 16, 128, 64), (2, 0, 1, 3), "float32")),
            # float-int32:[16, 16, 128, 64]-[4]=float:[16, 128, 16, 64]
            ("transpose_0042", "transpose_run", ((16, 16, 128, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[16384, 1024]-[2]=float:[16384, 1024]
            ("transpose_0043", "transpose_run", ((16384, 1024), (0, 1), "float32")),
            # float-int32:[20, 1024]-[2]=float:[20, 1024]
            ("transpose_0044", "transpose_run", ((20, 1024), (0, 1), "float32")),
            # float-int32:[20480, 1024]-[2]=float:[20480, 1024]
            ("transpose_0045", "transpose_run", ((20480, 1024), (0, 1), "float32")),
            # float-int32:[2048, 1024]-[2]=float:[2048, 1024]
            ("transpose_0046", "transpose_run", ((2048, 1024), (0, 1), "float32")),
            # float-int32:[2, 1024]-[2]=float:[2, 1024]
            ("transpose_0047", "transpose_run", ((2, 1024), (0, 1), "float32")),
            # float-int32:[21128, 1024]-[2]=float:[21128, 1024]
            ("transpose_0048", "transpose_run", ((21128, 1024), (0, 1), "float32")),
            # float-int32:[2, 128, 16, 64]-[4]=float:[2, 16, 128, 64]
            ("transpose_0049", "transpose_run", ((2, 128, 16, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[2, 16, 128, 128]-[4]=float:[128, 2, 16, 128]
            ("transpose_0050", "transpose_run", ((2, 16, 128, 128), (2, 0, 1, 3), "float32")),
            # float-int32:[2, 16, 128, 64]-[4]=float:[128, 2, 16, 64]
            ("transpose_0051", "transpose_run", ((2, 16, 128, 64), (2, 0, 1, 3), "float32")),
            # float-int32:[2, 16, 128, 64]-[4]=float:[2, 128, 16, 64]
            ("transpose_0052", "transpose_run", ((2, 16, 128, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[2560, 1024]-[2]=float:[2560, 1024]
            ("transpose_0053", "transpose_run", ((2560, 1024), (0, 1), "float32")),
            # float-int32:[256, 1024]-[2]=float:[256, 1024]
            ("transpose_0054", "transpose_run", ((256, 1024), (0, 1), "float32")),
            # float-int32:[256, 128, 16, 64]-[4]=float:[256, 16, 128, 64]
            ("transpose_0055", "transpose_run", ((256, 128, 16, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[256, 16, 128, 128]-[4]=float:[128, 256, 16, 128]
            ("transpose_0056", "transpose_run", ((256, 16, 128, 128), (2, 0, 1, 3), "float32")),
            # float-int32:[256, 16, 128, 64]-[4]=float:[128, 256, 16, 64]
            ("transpose_0057", "transpose_run", ((256, 16, 128, 64), (2, 0, 1, 3), "float32")),
            # float-int32:[256, 16, 128, 64]-[4]=float:[256, 128, 16, 64]
            ("transpose_0058", "transpose_run", ((256, 16, 128, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[320, 1024]-[2]=float:[320, 1024]
            ("transpose_0059", "transpose_run", ((320, 1024), (0, 1), "float32")),
            # float-int32:[32, 128, 16, 64]-[4]=float:[32, 16, 128, 64]
            ("transpose_0060", "transpose_run", ((32, 128, 16, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[32, 16, 128, 128]-[4]=float:[128, 32, 16, 128]
            ("transpose_0061", "transpose_run", ((32, 16, 128, 128), (2, 0, 1, 3), "float32")),
            # float-int32:[32, 16, 128, 64]-[4]=float:[128, 32, 16, 64]
            ("transpose_0062", "transpose_run", ((32, 16, 128, 64), (2, 0, 1, 3), "float32")),
            # float-int32:[32, 16, 128, 64]-[4]=float:[32, 128, 16, 64]
            ("transpose_0063", "transpose_run", ((32, 16, 128, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[32768, 1024]-[2]=float:[32768, 1024]
            ("transpose_0064", "transpose_run", ((32768, 1024), (0, 1), "float32")),
            # float-int32:[33, 64]-[2]=float:[33, 64]
            ("transpose_0065", "transpose_run", ((33, 64), (0, 1), "float32")),
            # float-int32:[40, 1024]-[2]=float:[40, 1024]
            ("transpose_0066", "transpose_run", ((40, 1024), (0, 1), "float32")),
            # float-int32:[4096, 1024]-[2]=float:[4096, 1024]
            ("transpose_0067", "transpose_run", ((4096, 1024), (0, 1), "float32")),
            # float-int32:[4, 128, 16, 64]-[4]=float:[4, 16, 128, 64]
            ("transpose_0068", "transpose_run", ((4, 128, 16, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[4, 16, 128, 128]-[4]=float:[128, 4, 16, 128]
            ("transpose_0069", "transpose_run", ((4, 16, 128, 128), (2, 0, 1, 3), "float32")),
            # float-int32:[4, 16, 128, 64]-[4]=float:[128, 4, 16, 64]
            ("transpose_0070", "transpose_run", ((4, 16, 128, 64), (2, 0, 1, 3), "float32")),
            # float-int32:[4, 16, 128, 64]-[4]=float:[4, 128, 16, 64]
            ("transpose_0071", "transpose_run", ((4, 16, 128, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[5120, 1024]-[2]=float:[5120, 1024]
            ("transpose_0072", "transpose_run", ((5120, 1024), (0, 1), "float32")),
            # float-int32:[512, 1024]-[2]=float:[512, 1024]
            ("transpose_0073", "transpose_run", ((512, 1024), (0, 1), "float32")),
            # float-int32:[512, 128, 16, 64]-[4]=float:[512, 16, 128, 64]
            ("transpose_0074", "transpose_run", ((512, 128, 16, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[512, 16, 128, 128]-[4]=float:[128, 512, 16, 128]
            ("transpose_0075", "transpose_run", ((512, 16, 128, 128), (2, 0, 1, 3), "float32")),
            # float-int32:[512, 16, 128, 64]-[4]=float:[128, 512, 16, 64]
            ("transpose_0076", "transpose_run", ((512, 16, 128, 64), (2, 0, 1, 3), "float32")),
            # float-int32:[512, 16, 128, 64]-[4]=float:[512, 128, 16, 64]
            ("transpose_0077", "transpose_run", ((512, 16, 128, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[640, 1024]-[2]=float:[640, 1024]
            ("transpose_0078", "transpose_run", ((640, 1024), (0, 1), "float32")),
            # float-int32:[65536, 1024]-[2]=float:[65536, 1024]
            ("transpose_0079", "transpose_run", ((65536, 1024), (0, 1), "float32")),
            # float-int32:[80, 1024]-[2]=float:[80, 1024]
            ("transpose_0080", "transpose_run", ((80, 1024), (0, 1), "float32")),
            # float-int32:[8, 128, 16, 64]-[4]=float:[8, 16, 128, 64]
            ("transpose_0081", "transpose_run", ((8, 128, 16, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[8, 16, 128, 128]-[4]=float:[128, 8, 16, 128]
            ("transpose_0082", "transpose_run", ((8, 16, 128, 128), (2, 0, 1, 3), "float32")),
            # float-int32:[8, 16, 128, 64]-[4]=float:[128, 8, 16, 64]
            ("transpose_0083", "transpose_run", ((8, 16, 128, 64), (2, 0, 1, 3), "float32")),
            # float-int32:[8, 16, 128, 64]-[4]=float:[8, 128, 16, 64]
            ("transpose_0084", "transpose_run", ((8, 16, 128, 64), (0, 2, 1, 3), "float32")),
            # int32-int32:[128, 128]-[2]=float:[128, 128]
            ("transpose_0085", "transpose_run", ((128, 128), (1, 0), "int32")),

            # unsortedsegmentsum OP
            # float-int32-int32:[10240, 1024]-[[10240]]-[]=float:[65536, 1024]
            ("unsortedsegmentsum_001", "unsortedsegmentsum_run", ([10240, 1024], [10240], 65536, "float32")),
            # float-int32-int32:[1024, 1024]-[[1024]]-[]=float:[2, 1024]
            ("unsortedsegmentsum_002", "unsortedsegmentsum_run", ([1024, 1024], [1024], 2, "float32")),
            # float-int32-int32:[1024, 1024]-[[1024]]-[]=float:[21128, 1024]
            ("unsortedsegmentsum_003", "unsortedsegmentsum_run", ([1024, 1024], [1024], 21128, "float32")),
            # float-int32-int32:[128, 1024]-[[128]]-[]=float:[2, 1024]
            ("unsortedsegmentsum_004", "unsortedsegmentsum_run", ([128, 1024], [128], 2, "float32")),
            # float-int32-int32:[128, 1024]-[[128]]-[]=float:[21128, 1024]
            ("unsortedsegmentsum_005", "unsortedsegmentsum_run", ([128, 1024], [128], 21128, "float32")),
            # float-int32-int32:[128, 128, 64]-[[128]-[]=float:[33, 64]
            ("unsortedsegmentsum_006", "unsortedsegmentsum_run", ([128, 128, 64], [128], 33, "float32")),
            # float-int32-int32:[131072, 1024]-[[131072]]-[]=float:[2, 1024]
            ("unsortedsegmentsum_007", "unsortedsegmentsum_run", ([131072, 1024], [131072], 2, "float32")),
            # float-int32-int32:[131072, 1024]-[[131072]]-[]=float:[21128, 1024]
            ("unsortedsegmentsum_008", "unsortedsegmentsum_run", ([131072, 1024], [131072], 21128, "float32")),
            # float-int32-int32:[160, 1024]-[[160]]-[]=float:[1024, 1024]
            ("unsortedsegmentsum_009", "unsortedsegmentsum_run", ([160, 1024], [160], 1024, "float32")),
            # float-int32-int32:[16384, 1024]-[[16384]]-[]=float:[2, 1024]
            ("unsortedsegmentsum_0010", "unsortedsegmentsum_run", ([16384, 1024], [16384], 2, "float32")),
            # float-int32-int32:[16384, 1024]-[[16384]]-[]=float:[21128, 1024]
            ("unsortedsegmentsum_0011", "unsortedsegmentsum_run", ([16384, 1024], [16384], 21128, "float32")),
            # float-int32-int32:[20, 1024]-[[20]]-[]=float:[128, 1024]
            ("unsortedsegmentsum_0012", "unsortedsegmentsum_run", ([20, 1024], [20], 128, "float32")),
            # float-int32-int32:[20480, 1024]-[[20480]]-[]=float:[131072, 1024]
            ("unsortedsegmentsum_0013", "unsortedsegmentsum_run", ([20480, 1024], [20480], 131072, "float32")),
            # float-int32-int32:[2048, 1024]-[[2048]]-[]=float:[2, 1024]
            ("unsortedsegmentsum_0014", "unsortedsegmentsum_run", ([2048, 1024], [2048], 2, "float32")),
            # float-int32-int32:[2048, 1024]-[[2048]]-[]=float:[21128, 1024]
            ("unsortedsegmentsum_0015", "unsortedsegmentsum_run", ([2048, 1024], [2048], 21128, "float32")),
            # float-int32-int32:[2560, 1024]-[[2560]]-[]=float:[16384, 1024]
            ("unsortedsegmentsum_0016", "unsortedsegmentsum_run", ([2560, 1024], [2560], 16384, "float32")),
            # float-int32-int32:[256, 1024]-[[256]]-[]=float:[2, 1024]
            ("unsortedsegmentsum_0017", "unsortedsegmentsum_run", ([256, 1024], [256], 2, "float32")),
            # float-int32-int32:[256, 1024]-[[256]]-[]=float:[21128, 1024]
            ("unsortedsegmentsum_0018", "unsortedsegmentsum_run", ([256, 1024], [256], 21128, "float32")),
            # float-int32-int32:[320, 1024]-[[320]]-[]=float:[2048, 1024]
            ("unsortedsegmentsum_0019", "unsortedsegmentsum_run", ([320, 1024], [320], 2048, "float32")),
            # float-int32-int32:[32768, 1024]-[[32768]]-[]=float:[2, 1024]
            ("unsortedsegmentsum_0020", "unsortedsegmentsum_run", ([32768, 1024], [32768], 2, "float32")),
            # float-int32-int32:[32768, 1024]-[[32768]]-[]=float:[21128, 1024]
            ("unsortedsegmentsum_0021", "unsortedsegmentsum_run", ([32768, 1024], [32768], 21128, "float32")),
            # float-int32-int32:[40, 1024]-[[40]]-[]=float:[256, 1024]
            ("unsortedsegmentsum_0022", "unsortedsegmentsum_run", ([40, 1024], [40], 256, "float32")),
            # float-int32-int32:[4096, 1024]-[[4096]]-[]=float:[2, 1024]
            ("unsortedsegmentsum_0023", "unsortedsegmentsum_run", ([4096, 1024], [4096], 2, "float32")),
            # float-int32-int32:[4096, 1024]-[[4096]]-[]=float:[21128, 1024]
            ("unsortedsegmentsum_0024", "unsortedsegmentsum_run", ([4096, 1024], [4096], 21128, "float32")),
            # float-int32-int32:[5120, 1024]-[[5120]]-[]=float:[32768, 1024]
            ("unsortedsegmentsum_0025", "unsortedsegmentsum_run", ([5120, 1024], [5120], 32768, "float32")),
            # float-int32-int32:[512, 1024]-[[512]]-[]=float:[2, 1024]
            ("unsortedsegmentsum_0026", "unsortedsegmentsum_run", ([512, 1024], [512], 2, "float32")),
            # float-int32-int32:[512, 1024]-[[512]]-[]=float:[21128, 1024]
            ("unsortedsegmentsum_0027", "unsortedsegmentsum_run", ([512, 1024], [512], 21128, "float32")),
            # float-int32-int32:[640, 1024]-[[640]]-[]=float:[4096, 1024]
            ("unsortedsegmentsum_0028", "unsortedsegmentsum_run", ([640, 1024], [640], 4096, "float32")),
            # float-int32-int32:[65536, 1024]-[[65536]]-[]=float:[2, 1024]
            ("unsortedsegmentsum_0029", "unsortedsegmentsum_run", ([65536, 1024], [65536], 2, "float32")),
            # float-int32-int32:[65536, 1024]-[[65536]]-[]=float:[21128, 1024]
            ("unsortedsegmentsum_0030", "unsortedsegmentsum_run", ([65536, 1024], [65536], 21128, "float32")),
            # float-int32-int32:[80, 1024]-[[80]]-[]=float:[512, 1024]
            ("unsortedsegmentsum_0031", "unsortedsegmentsum_run", ([80, 1024], [80], 512, "float32")),

            # gelu OP
            # float:[10240, 1024]=float:[10240, 1024]
            ("gelu_001", "gelu_run", ((10240, 1024), "float32")),
            # float:[1024, 4096]=float:[1024, 4096]
            ("gelu_002", "gelu_run", ((1024, 4096), "float32")),
            # float:[128, 4096]=float:[128, 4096]
            ("gelu_003", "gelu_run", ((128, 4096), "float32")),
            # float:[131072, 4096]=float:[131072, 4096]
            ("gelu_004", "gelu_run", ((131072, 4096), "float32")),
            # float:[160, 1024]=float:[160, 1024]
            ("gelu_005", "gelu_run", ((160, 1024), "float32")),
            # float:[16384, 4096]=float:[16384, 4096]
            ("gelu_006", "gelu_run", ((16384, 4096), "float32")),
            # float:[20, 1024]=float:[20, 1024]
            ("gelu_007", "gelu_run", ((20, 1024), "float32")),
            # float:[20480, 1024]=float:[20480, 1024]
            ("gelu_008", "gelu_run", ((20480, 1024), "float32")),
            # float:[2048, 4096]=float:[2048, 4096]
            ("gelu_009", "gelu_run", ((2048, 4096), "float32")),
            # float:[2560, 1024]=float:[2560, 1024]
            ("gelu_0010", "gelu_run", ((2560, 1024), "float32")),
            # float:[256, 4096]=float:[256, 4096]
            ("gelu_0011", "gelu_run", ((256, 4096), "float32")),
            # float:[320, 1024]=float:[320, 1024]
            ("gelu_0012", "gelu_run", ((320, 1024), "float32")),
            # float:[32768, 4096]=float:[32768, 4096]
            ("gelu_0013", "gelu_run", ((32768, 4096), "float32")),
            # float:[40, 1024]=float:[40, 1024]
            ("gelu_0014", "gelu_run", ((40, 1024), "float32")),
            # float:[4096, 4096]=float:[4096, 4096]
            ("gelu_0015", "gelu_run", ((4096, 4096), "float32")),
            # float:[5120, 1024]=float:[5120, 1024]
            ("gelu_0016", "gelu_run", ((5120, 1024), "float32")),
            # float:[512, 4096]=float:[512, 4096]
            ("gelu_0017", "gelu_run", ((512, 4096), "float32")),
            # float:[640, 1024]=float:[640, 1024]
            ("gelu_0018", "gelu_run", ((640, 1024), "float32")),
            # float:[65536, 4096]=float:[65536, 4096]
            ("gelu_0019", "gelu_run", ((65536, 4096), "float32")),
            # float:[80, 1024]=float:[80, 1024]
            ("gelu_0020", "gelu_run", ((80, 1024), "float32")),

            # gelu_grad OP
            # float:[10240, 1024]=float:[10240, 1024]
            ("gelu_grad_001", "gelu_grad_run", ((10240, 1024), "float32")),
            # float:[1024, 4096]=float:[1024, 4096]
            ("gelu_grad_002", "gelu_grad_run", ((1024, 4096), "float32")),
            # float:[128, 4096]=float:[128, 4096]
            ("gelu_grad_003", "gelu_grad_run", ((128, 4096), "float32")),
            # float:[131072, 4096]=float:[131072, 4096]
            ("gelu_grad_004", "gelu_grad_run", ((131072, 4096), "float32")),
            # float:[160, 1024]=float:[160, 1024]
            ("gelu_grad_005", "gelu_grad_run", ((160, 1024), "float32")),
            # float:[16384, 4096]=float:[16384, 4096]
            ("gelu_grad_006", "gelu_grad_run", ((16384, 4096), "float32")),
            # float:[20, 1024]=float:[20, 1024]
            ("gelu_grad_007", "gelu_grad_run", ((20, 1024), "float32")),
            # float:[20480, 1024]=float:[20480, 1024]
            ("gelu_grad_008", "gelu_grad_run", ((20480, 1024), "float32")),
            # float:[2048, 4096]=float:[2048, 4096]
            ("gelu_grad_009", "gelu_grad_run", ((2048, 4096), "float32")),
            # float:[2560, 1024]=float:[2560, 1024]
            ("gelu_grad_0010", "gelu_grad_run", ((2560, 1024), "float32")),
            # float:[256, 4096]=float:[256, 4096]
            ("gelu_grad_0011", "gelu_grad_run", ((256, 4096), "float32")),
            # float:[320, 1024]=float:[320, 1024]
            ("gelu_grad_0012", "gelu_grad_run", ((320, 1024), "float32")),
            # float:[32768, 4096]=float:[32768, 4096]
            ("gelu_grad_0013", "gelu_grad_run", ((32768, 4096), "float32")),
            # float:[40, 1024]=float:[40, 1024]
            ("gelu_grad_0014", "gelu_grad_run", ((40, 1024), "float32")),
            # float:[4096, 4096]=float:[4096, 4096]
            ("gelu_grad_0015", "gelu_grad_run", ((4096, 4096), "float32")),
            # float:[5120, 1024]=float:[5120, 1024]
            ("gelu_grad_0016", "gelu_grad_run", ((5120, 1024), "float32")),
            # float:[512, 4096]=float:[512, 4096]
            ("gelu_grad_0017", "gelu_grad_run", ((512, 4096), "float32")),
            # float:[640, 1024]=float:[640, 1024]
            ("gelu_grad_0018", "gelu_grad_run", ((640, 1024), "float32")),
            # float:[65536, 4096]=float:[65536, 4096]
            ("gelu_grad_0019", "gelu_grad_run", ((65536, 4096), "float32")),
            # float:[80, 1024]=float:[80, 1024]
            ("gelu_grad_0020", "gelu_run", ((80, 1024), "float32")),

            # LayerNorm OP
            # float:[10240, 1024]=float:[10240, 1024]
            ("fused_layernorm_001", "fused_layernorm_run", ((10240, 1024), -1, -1, "float32")),
            # float:[1024, 1024]=float:[1024, 1024]
            ("fused_layernorm_002", "fused_layernorm_run", ((1024, 1024), -1, -1, "float32")),
            # float:[1024, 128, 1024]=float:[1024, 128, 1024]
            ("fused_layernorm_003", "fused_layernorm_run", ((1024, 128, 1024), -1, -1, "float32")),
            # float:[1, 128, 1024]=float:[1, 128, 1024]
            ("fused_layernorm_004", "fused_layernorm_run", ((1, 128, 1024), -1, -1, "float32")),
            # float:[128, 1024]=float:[128, 1024]
            ("fused_layernorm_005", "fused_layernorm_run", ((128, 1024), -1, -1, "float32")),
            # float:[128, 128, 1024]=float:[128, 128, 1024]
            ("fused_layernorm_006", "fused_layernorm_run", ((128, 128, 1024), -1, -1, "float32")),
            # float:[131072, 1024]=float:[131072, 1024]
            ("fused_layernorm_007", "fused_layernorm_run", ((131072, 1024), -1, -1, "float32")),
            # float:[160, 1024]=float:[160, 1024]
            ("fused_layernorm_008", "fused_layernorm_run", ((160, 1024), -1, -1, "float32")),
            # float:[16, 128, 1024]=float:[16, 128, 1024]
            ("fused_layernorm_009", "fused_layernorm_run", ((16, 128, 1024), -1, -1, "float32")),
            # float:[16384, 1024]=float:[16384, 1024]
            ("fused_layernorm_0010", "fused_layernorm_run", ((16384, 1024), -1, -1, "float32")),
            # float:[20, 1024]=float:[20, 1024]
            ("fused_layernorm_0011", "fused_layernorm_run", ((20, 1024), -1, -1, "float32")),
            # float:[20480, 1024]=float:[20480, 1024]
            ("fused_layernorm_0012", "fused_layernorm_run", ((20480, 1024), -1, -1, "float32")),
            # float:[2048, 1024]=float:[2048, 1024]
            ("fused_layernorm_0013", "fused_layernorm_run", ((2048, 1024), -1, -1, "float32")),
            # float:[2, 128, 1024]=float:[2, 128, 1024]
            ("fused_layernorm_0014", "fused_layernorm_run", ((2, 128, 1024), -1, -1, "float32")),
            # float:[2560, 1024]=float:[2560, 1024]
            ("fused_layernorm_0015", "fused_layernorm_run", ((2560, 1024), -1, -1, "float32")),
            # float:[256, 1024]=float:[256, 1024]
            ("fused_layernorm_0016", "fused_layernorm_run", ((256, 1024), -1, -1, "float32")),
            # float:[256, 128, 1024]=float:[256, 128, 1024]
            ("fused_layernorm_0017", "fused_layernorm_run", ((256, 128, 1024), -1, -1, "float32")),
            # float:[320, 1024]=float:[320, 1024]
            ("fused_layernorm_0018", "fused_layernorm_run", ((320, 1024), -1, -1, "float32")),
            # float:[32, 128, 1024]=float:[32, 128, 1024]
            ("fused_layernorm_0019", "fused_layernorm_run", ((32, 128, 1024), -1, -1, "float32")),
            # float:[32768, 1024]=float:[32768, 1024]
            ("fused_layernorm_0020", "fused_layernorm_run", ((32768, 1024), -1, -1, "float32")),
            # float:[40, 1024]=float:[40, 1024]
            ("fused_layernorm_0021", "fused_layernorm_run", ((40, 1024), -1, -1, "float32")),
            # float:[4096, 1024]=float:[4096, 1024]
            ("fused_layernorm_0022", "fused_layernorm_run", ((4096, 1024), -1, -1, "float32")),
            # float:[4, 128, 1024]=float:[4, 128, 1024]
            ("fused_layernorm_0023", "fused_layernorm_run", ((4, 128, 1024), -1, -1, "float32")),
            # float:[5120, 1024]=float:[5120, 1024]
            ("fused_layernorm_0024", "fused_layernorm_run", ((5120, 1024), -1, -1, "float32")),
            # float:[512, 1024]=float:[512, 1024]
            ("fused_layernorm_0025", "fused_layernorm_run", ((512, 1024), -1, -1, "float32")),
            # float:[512, 128, 1024]=float:[512, 128, 1024]
            ("fused_layernorm_0026", "fused_layernorm_run", ((512, 128, 1024), -1, -1, "float32")),
            # float:[640, 1024]=float:[640, 1024]
            ("fused_layernorm_0027", "fused_layernorm_run", ((640, 1024), -1, -1, "float32")),
            # float:[65536, 1024]=float:[65536, 1024]
            ("fused_layernorm_0028", "fused_layernorm_run", ((65536, 1024), -1, -1, "float32")),
            # float:[80, 1024]=float:[80, 1024]
            ("fused_layernorm_0029", "fused_layernorm_run", ((80, 1024), -1, -1, "float32")),
            # float:[8, 128, 1024]=float:[8, 128, 1024]
            ("fused_layernorm_0030", "fused_layernorm_run", ((8, 128, 1024), -1, -1, "float32")),

            # LayerNormGrad
            # float:[10240, 1024]=float:[10240, 1024]
            ("fused_layer_norm_grad_001", "fused_layer_norm_grad_run", ((10240, 1024), -1, -1, "float32")),
            # float:[1024, 1024]=float:[1024, 1024]
            ("fused_layer_norm_grad_002", "fused_layer_norm_grad_run", ((1024, 1024), -1, -1, "float32")),
            # float:[1024, 128, 1024]=float:[1024, 128, 1024]
            ("fused_layer_norm_grad_003", "fused_layer_norm_grad_run", ((1024, 128, 1024), -1, -1, "float32")),
            # float:[1, 128, 1024]=float:[1, 128, 1024]
            ("fused_layer_norm_grad_004", "fused_layer_norm_grad_run", ((1, 128, 1024), -1, -1, "float32")),
            # float:[128, 1024]=float:[128, 1024]
            ("fused_layer_norm_grad_005", "fused_layer_norm_grad_run", ((128, 1024), -1, -1, "float32")),
            # float:[128, 128, 1024]=float:[128, 128, 1024]
            ("fused_layer_norm_grad_006", "fused_layer_norm_grad_run", ((128, 128, 1024), -1, -1, "float32")),
            # float:[131072, 1024]=float:[131072, 1024]
            ("fused_layer_norm_grad_007", "fused_layer_norm_grad_run", ((131072, 1024), -1, -1, "float32")),
            # float:[160, 1024]=float:[160, 1024]
            ("fused_layer_norm_grad_008", "fused_layer_norm_grad_run", ((160, 1024), -1, -1, "float32")),
            # float:[16, 128, 1024]=float:[16, 128, 1024]
            ("fused_layer_norm_grad_009", "fused_layer_norm_grad_run", ((16, 128, 1024), -1, -1, "float32")),
            # float:[16384, 1024]=float:[16384, 1024]
            ("fused_layer_norm_grad_0010", "fused_layer_norm_grad_run", ((16384, 1024), -1, -1, "float32")),
            # float:[20, 1024]=float:[20, 1024]
            ("fused_layer_norm_grad_0011", "fused_layer_norm_grad_run", ((20, 1024), -1, -1, "float32")),
            # float:[20480, 1024]=float:[20480, 1024]
            ("fused_layer_norm_grad_0012", "fused_layer_norm_grad_run", ((20480, 1024), -1, -1, "float32")),
            # float:[2048, 1024]=float:[2048, 1024]
            ("fused_layer_norm_grad_0013", "fused_layer_norm_grad_run", ((2048, 1024), -1, -1, "float32")),
            # float:[2, 128, 1024]=float:[2, 128, 1024]
            ("fused_layer_norm_grad_0014", "fused_layer_norm_grad_run", ((2, 128, 1024), -1, -1, "float32")),
            # float:[2560, 1024]=float:[2560, 1024]
            ("fused_layer_norm_grad_0015", "fused_layer_norm_grad_run", ((2560, 1024), -1, -1, "float32")),
            # float:[256, 1024]=float:[256, 1024]
            ("fused_layer_norm_grad_0016", "fused_layer_norm_grad_run", ((256, 1024), -1, -1, "float32")),
            # float:[256, 128, 1024]=float:[256, 128, 1024]
            ("fused_layer_norm_grad_0017", "fused_layer_norm_grad_run", ((256, 128, 1024), -1, -1, "float32")),
            # float:[320, 1024]=float:[320, 1024]
            ("fused_layer_norm_grad_0018", "fused_layer_norm_grad_run", ((320, 1024), -1, -1, "float32")),
            # float:[32, 128, 1024]=float:[32, 128, 1024]
            ("fused_layer_norm_grad_0019", "fused_layer_norm_grad_run", ((32, 128, 1024), -1, -1, "float32")),
            # float:[32768, 1024]=float:[32768, 1024]
            ("fused_layer_norm_grad_0020", "fused_layer_norm_grad_run", ((32768, 1024), -1, -1, "float32")),
            # float:[40, 1024]=float:[40, 1024]
            ("fused_layer_norm_grad_0021", "fused_layer_norm_grad_run", ((40, 1024), -1, -1, "float32")),
            # float:[4096, 1024]=float:[4096, 1024]
            ("fused_layer_norm_grad_0022", "fused_layer_norm_grad_run", ((4096, 1024), -1, -1, "float32")),
            # float:[4, 128, 1024]=float:[4, 128, 1024]
            ("fused_layer_norm_grad_0023", "fused_layer_norm_grad_run", ((4, 128, 1024), -1, -1, "float32")),
            # float:[5120, 1024]=float:[5120, 1024]
            ("fused_layer_norm_grad_0024", "fused_layer_norm_grad_run", ((5120, 1024), -1, -1, "float32")),
            # float:[512, 1024]=float:[512, 1024]
            ("fused_layer_norm_grad_0025", "fused_layer_norm_grad_run", ((512, 1024), -1, -1, "float32")),
            # float:[512, 128, 1024]=float:[512, 128, 1024]
            ("fused_layer_norm_grad_0026", "fused_layer_norm_grad_run", ((512, 128, 1024), -1, -1, "float32")),
            # float:[640, 1024]=float:[640, 1024]
            ("fused_layer_norm_grad_0027", "fused_layer_norm_grad_run", ((640, 1024), -1, -1, "float32")),
            # float:[65536, 1024]=float:[65536, 1024]
            ("fused_layer_norm_grad_0028", "fused_layer_norm_grad_run", ((65536, 1024), -1, -1, "float32")),
            # float:[80, 1024]=float:[80, 1024]
            ("fused_layer_norm_grad_0029", "fused_layer_norm_grad_run", ((80, 1024), -1, -1, "float32")),
            # float:[8, 128, 1024]=float:[8, 128, 1024]
            ("fused_layer_norm_grad_0030", "fused_layer_norm_grad_run", ((8, 128, 1024), -1, -1, "float32")),

            # dropout OP
            # float:[1024, 1024]=float:[1024, 1024]
            ("dropout_001", "dropout_run", ((1024, 1024), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[1024, 128, 1024]=float:[1024, 128, 1024]
            ("dropout_002", "dropout_run", ((1024, 128, 1024), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[1024, 16, 128, 128]=float:[1024, 16, 128, 128]
            ("dropout_003", "dropout_run", ((1024, 16, 128, 128), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[1, 128, 1024]=float:[1, 128, 1024]
            ("dropout_004", "dropout_run", ((1, 128, 1024), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[1, 16, 128, 128]=float:[1, 16, 128, 128]
            ("dropout_005", "dropout_run", ((1, 16, 128, 128), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[128, 1024]=float:[128, 1024]
            ("dropout_006", "dropout_run", ((128, 1024), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[128, 128, 1024]=float:[128, 128, 1024]
            ("dropout_007", "dropout_run", ((128, 128, 1024), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[128, 16, 128, 128]=float:[128, 16, 128, 128]
            ("dropout_008", "dropout_run", ((128, 16, 128, 128), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[131072, 1024]=float:[131072, 1024]
            ("dropout_009", "dropout_run", ((131072, 1024), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[16, 128, 1024]=float:[16, 128, 1024]
            ("dropout_0010", "dropout_run", ((16, 128, 1024), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[16, 16, 128, 128]=float:[16, 16, 128, 128]
            ("dropout_0011", "dropout_run", ((16, 16, 128, 128), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[16384, 1024]=float:[16384, 1024]
            ("dropout_0012", "dropout_run", ((16384, 1024), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[2048, 1024]=float:[2048, 1024]
            ("dropout_0013", "dropout_run", ((2048, 1024), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[2, 128, 1024]=float:[2, 128, 1024]
            ("dropout_0014", "dropout_run", ((2, 128, 1024), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[2, 16, 128, 128]=float:[2, 16, 128, 128]
            ("dropout_0015", "dropout_run", ((2, 16, 128, 128), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[256, 1024]=float:[256, 1024]
            ("dropout_0016", "dropout_run", ((256, 1024), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[256, 128, 1024]=float:[256, 128, 1024]
            ("dropout_0017", "dropout_run", ((256, 128, 1024), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[256, 16, 128, 128]=float:[256, 16, 128, 128]
            ("dropout_0018", "dropout_run", ((256, 16, 128, 128), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[32, 128, 1024]=float:[32, 128, 1024]
            ("dropout_0019", "dropout_run", ((32, 128, 1024), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[32, 16, 128, 128]=float:[32, 16, 128, 128]
            ("dropout_0020", "dropout_run", ((32, 16, 128, 128), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[32768, 1024]=float:[32768, 1024]
            ("dropout_0021", "dropout_run", ((32768, 1024), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[4096, 1024]=float:[4096, 1024]
            ("dropout_0022", "dropout_run", ((4096, 1024), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[4, 128, 1024]=float:[4, 128, 1024]
            ("dropout_0023", "dropout_run", ((4, 128, 1024), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[4, 16, 128, 128]=float:[4, 16, 128, 128]
            ("dropout_0024", "dropout_run", ((4, 16, 128, 128), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[512, 1024]=float:[512, 1024]
            ("dropout_0025", "dropout_run", ((512, 1024), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[512, 128, 1024]=float:[512, 128, 1024]
            ("dropout_0026", "dropout_run", ((512, 128, 1024), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[512, 16, 128, 128]=float:[512, 16, 128, 128]
            ("dropout_0027", "dropout_run", ((512, 16, 128, 128), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[65536, 1024]=float:[65536, 1024]
            ("dropout_0028", "dropout_run", ((65536, 1024), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[8, 128, 1024]=float:[8, 128, 1024]
            ("dropout_0029", "dropout_run", ((8, 128, 1024), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[8, 16, 128, 128]=float:[8, 16, 128, 128]
            ("dropout_0030", "dropout_run", ((8, 16, 128, 128), 1.0, "float32", "cce_dropout_do_mask")),

            # addn OP
            # float-float:[1024, 1024]-[1024, 1024]=float:[1024, 1024]
            ("addn_001", "addn_run", ((1024, 1024), "float32", 2)),
            # float-float:[1024, 128, 1024]-[1024, 128, 1024]=float:[1024, 128, 1024]
            ("addn_002", "addn_run", ((1024, 128, 1024), "float32", 2)),
            # float-float:[1024, 16, 128, 128]-[1024, 16, 128, 128]=float:[1024, 16, 128, 128]
            ("addn_003", "addn_run", ((1024, 16, 128, 128), "float32", 2)),
            # float-float:[1024, 16, 128, 64]-[1024, 16, 128, 64]=float:[1024, 16, 128, 64]
            ("addn_004", "addn_run", ((1024, 16, 128, 64), "float32", 2)),
            # float-float:[1, 128, 1024]-[1, 128, 1024]=float:[1, 128, 1024]
            ("addn_005", "addn_run", ((1, 128, 1024), "float32", 2)),
            # float-float:[1, 16, 128, 128]-[1, 16, 128, 128]=float:[1, 16, 128, 128]
            ("addn_006", "addn_run", ((1, 16, 128, 128), "float32", 2)),
            # float-float:[1, 16, 128, 64]-[1, 16, 128, 64]=float:[1, 16, 128, 64]
            ("addn_007", "addn_run", ((1, 16, 128, 64), "float32", 2)),
            # float-float:[128, 1024]-[128, 1024]=float:[128, 1024]
            ("addn_008", "addn_run", ((128, 1024), "float32", 2)),
            # float-float:[128, 128, 1024]-[128, 128, 1024]=float:[128, 128, 1024]
            ("addn_009", "addn_run", ((128, 128, 1024), "float32", 2)),
            # float-float:[128, 16, 128, 128]-[128, 16, 128, 128]=float:[128, 16, 128, 128]
            ("addn_010", "addn_run", ((128, 16, 128, 128), "float32", 2)),
            # float-float:[128, 16, 128, 64]-[128, 16, 128, 64]=float:[128, 16, 128, 64]
            ("addn_011", "addn_run", ((128, 16, 128, 64), "float32", 2)),
            # float-float:[131072, 1024]-[131072, 1024]=float:[131072, 1024]
            ("addn_012", "addn_run", ((131072, 1024), "float32", 2)),
            # float-float:[16, 128, 1024]-[16, 128, 1024]=float:[16, 128, 1024]
            ("addn_013", "addn_run", ((16, 128, 1024), "float32", 2)),
            # float-float:[16, 16, 128, 128]-[16, 16, 128, 128]=float:[16, 16, 128, 128]
            ("addn_014", "addn_run", ((16, 16, 128, 128), "float32", 2)),
            # float-float:[16, 16, 128, 64]-[16, 16, 128, 64]=float:[16, 16, 128, 64]
            ("addn_015", "addn_run", ((16, 16, 128, 64), "float32", 2)),
            # float-float:[16384, 1024]-[16384, 1024]=float:[16384, 1024]
            ("addn_016", "addn_run", ((16384, 1024), "float32", 2)),
            # float-float:[2048, 1024]-[2048, 1024]=float:[2048, 1024]
            ("addn_017", "addn_run", ((2048, 1024), "float32", 2)),
            # float-float:[21128, 1024]-[21128, 1024]=float:[21128, 1024]
            ("addn_018", "addn_run", ((21128, 1024), "float32", 2)),
            # float-float:[2, 128, 1024]-[2, 128, 1024]=float:[2, 128, 1024]
            ("addn_019", "addn_run", ((2, 128, 1024), "float32", 2)),
            # float-float:[2, 16, 128, 128]-[2, 16, 128, 128]=float:[2, 16, 128, 128]
            ("addn_020", "addn_run", ((2, 16, 128, 128), "float32", 2)),
            # float-float:[2, 16, 128, 64]-[2, 16, 128, 64]=float:[2, 16, 128, 64]
            ("addn_021", "addn_run", ((2, 16, 128, 64), "float32", 2)),
            # float-float:[256, 1024]-[256, 1024]=float:[256, 1024]
            ("addn_022", "addn_run", ((256, 1024), "float32", 2)),
            # float-float:[256, 128, 1024]-[256, 128, 1024]=float:[256, 128, 1024]
            ("addn_023", "addn_run", ((256, 128, 1024), "float32", 2)),
            # float-float:[256, 16, 128, 128]-[256, 16, 128, 128]=float:[256, 16, 128, 128]
            ("addn_024", "addn_run", ((256, 16, 128, 128), "float32", 2)),
            # float-float:[256, 16, 128, 64]-[256, 16, 128, 64]=float:[256, 16, 128, 64]
            ("addn_025", "addn_run", ((256, 16, 128, 64), "float32", 2)),
            # float-float:[32, 128, 1024]-[32, 128, 1024]=float:[32, 128, 1024]
            ("addn_026", "addn_run", ((32, 128, 1024), "float32", 2)),
            # float-float:[32, 16, 128, 128]-[32, 16, 128, 128]=float:[32, 16, 128, 128]
            ("addn_027", "addn_run", ((32, 16, 128, 128), "float32", 2)),
            # float-float:[32, 16, 128, 64]-[32, 16, 128, 64]=float:[32, 16, 128, 64]
            ("addn_028", "addn_run", ((32, 16, 128, 64), "float32", 2)),
            # float-float:[32768, 1024]-[32768, 1024]=float:[32768, 1024]
            ("addn_029", "addn_run", ((32768, 1024), "float32", 2)),
            # float-float:[33, 64]-[33, 64]=float:[33, 64]
            ("addn_030", "addn_run", ((33, 64), "float32", 2)),
            # float-float:[4096, 1024]-[4096, 1024]=float:[4096, 1024]
            ("addn_031", "addn_run", ((4096, 1024), "float32", 2)),
            # float-float:[4, 128, 1024]-[4, 128, 1024]=float:[4, 128, 1024]
            ("addn_032", "addn_run", ((4, 128, 1024), "float32", 2)),
            # float-float:[4, 16, 128, 128]-[4, 16, 128, 128]=float:[4, 16, 128, 128]
            ("addn_033", "addn_run", ((4, 16, 128, 128), "float32", 2)),
            # float-float:[4, 16, 128, 64]-[4, 16, 128, 64]=float:[4, 16, 128, 64]
            ("addn_034", "addn_run", ((4, 16, 128, 64), "float32", 2)),
            # float-float:[512, 1024]-[512, 1024]=float:[512, 1024]
            ("addn_035", "addn_run", ((512, 1024), "float32", 2)),
            # float-float:[512, 128, 1024]-[512, 128, 1024]=float:[512, 128, 1024]
            ("addn_036", "addn_run", ((512, 128, 1024), "float32", 2)),
            # float-float:[512, 16, 128, 128]-[512, 16, 128, 128]=float:[512, 16, 128, 128]
            ("addn_037", "addn_run", ((512, 16, 128, 128), "float32", 2)),
            # float-float:[512, 16, 128, 64]-[512, 16, 128, 64]=float:[512, 16, 128, 64]
            ("addn_038", "addn_run", ((512, 16, 128, 64), "float32", 2)),
            # float-float:[65536, 1024]-[65536, 1024]=float:[65536, 1024]
            ("addn_039", "addn_run", ((65536, 1024), "float32", 2)),
            # float-float:[8, 128, 1024]-[8, 128, 1024]=float:[8, 128, 1024]
            ("addn_040", "addn_run", ((8, 128, 1024), "float32", 2)),
            # float-float:[8, 16, 128, 128]-[8, 16, 128, 128]=float:[8, 16, 128, 128]
            ("addn_041", "addn_run", ((8, 16, 128, 128), "float32", 2)),
            # float-float:[8, 16, 128, 64]-[8, 16, 128, 64]=float:[8, 16, 128, 64]
            ("addn_042", "addn_run", ((8, 16, 128, 64), "float32", 2)),
            # float-float-float:[1024, 1024]-[1024, 1024]-[1024, 1024]=float:[1024, 1024]
            ("addn_043", "addn_run", ((1024, 1024), "float32", 3)),
            # float-float-float:[128, 1024]-[128, 1024]-[128, 1024]=float:[128, 1024]
            ("addn_044", "addn_run", ((128, 1024), "float32", 3)),
            # float-float-float:[131072, 1024]-[131072, 1024]-[131072, 1024]=float:[131072, 1024]
            ("addn_045", "addn_run", ((131072, 1024), "float32", 3)),
            # float-float-float:[16384, 1024]-[16384, 1024]-[16384, 1024]=float:[16384, 1024]
            ("addn_046", "addn_run", ((16384, 1024), "float32", 3)),
            # float-float-float:[2048, 1024]-[2048, 1024]-[2048, 1024]=float:[2048, 1024]
            ("addn_047", "addn_run", ((2048, 1024), "float32", 3)),
            # float-float-float:[256, 1024]-[256, 1024]-[256, 1024]=float:[256, 1024]
            ("addn_048", "addn_run", ((256, 1024), "float32", 3)),
            # float-float-float:[32768, 1024]-[32768, 1024]-[32768, 1024]=float:[32768, 1024]
            ("addn_049", "addn_run", ((32768, 1024), "float32", 3)),
            # float-float-float:[4096, 1024]-[4096, 1024]-[4096, 1024]=float:[4096, 1024]
            ("addn_050", "addn_run", ((4096, 1024), "float32", 3)),
            # float-float-float:[512, 1024]-[512, 1024]-[512, 1024]=float:[512, 1024]
            ("addn_051", "addn_run", ((512, 1024), "float32", 3)),
            # float-float-float:[65536, 1024]-[65536, 1024]-[65536, 1024]=float:[65536, 1024]
            ("addn_052", "addn_run", ((65536, 1024), "float32", 3)),

            # LogSoftMax OP
            # float:[10240, 21128]=float:[10240, 21128]
            ("logsoftmax_001", "logsoftmax_run", ((10240, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[1024, 2]=float:[1024, 2]
            ("logsoftmax_002", "logsoftmax_run", ((1024, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[128, 2]=float:[128, 2]
            ("logsoftmax_003", "logsoftmax_run", ((128, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[1, 2]=float:[1, 2]
            ("logsoftmax_004", "logsoftmax_run", ((1, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[160, 21128]=float:[160, 21128]
            ("logsoftmax_005", "logsoftmax_run", ((160, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[16, 2]=float:[16, 2]
            ("logsoftmax_006", "logsoftmax_run", ((16, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[20, 21128]=float:[20, 21128]
            ("logsoftmax_007", "logsoftmax_run", ((20, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[20480, 21128]=float:[20480, 21128]
            ("logsoftmax_008", "logsoftmax_run", ((20480, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[2, 2]=float:[2, 2]
            ("logsoftmax_009", "logsoftmax_run", ((2, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[2560, 21128]=float:[2560, 21128]
            ("logsoftmax_0010", "logsoftmax_run", ((2560, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[256, 2]=float:[256, 2]
            ("logsoftmax_0011", "logsoftmax_run", ((256, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[320, 21128]=float:[320, 21128]
            ("logsoftmax_0012", "logsoftmax_run", ((320, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[32, 2]=float:[32, 2]
            ("logsoftmax_0013", "logsoftmax_run", ((32, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[40, 21128]=float:[40, 21128]
            ("logsoftmax_0014", "logsoftmax_run", ((40, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[4, 2]=float:[4, 2]
            ("logsoftmax_0015", "logsoftmax_run", ((4, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[5120, 21128]=float:[5120, 21128]
            ("logsoftmax_0016", "logsoftmax_run", ((5120, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[512, 2]=float:[512, 2]
            ("logsoftmax_0017", "logsoftmax_run", ((512, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[640, 21128]=float:[640, 21128]
            ("logsoftmax_0018", "logsoftmax_run", ((640, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[80, 21128]=float:[80, 21128]
            ("logsoftmax_0019", "logsoftmax_run", ((80, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[8, 2]=float:[8, 2]
            ("logsoftmax_0020", "logsoftmax_run", ((8, 2), "float32", -1, "cce_logsoftmax_fp32")),

            # LogSoftMaxGrad OP
            # float:[10240, 21128]=float:[10240, 21128]
            ("logsoftmax_grad_001", "logsoftmax_grad_run", ((10240, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[1024, 2]=float:[1024, 2]
            ("logsoftmax_grad_002", "logsoftmax_grad_run", ((1024, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[128, 2]=float:[128, 2]
            ("logsoftmax_grad_003", "logsoftmax_grad_run", ((128, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[1, 2]=float:[1, 2]
            ("logsoftmax_grad_004", "logsoftmax_grad_run", ((1, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[160, 21128]=float:[160, 21128]
            ("logsoftmax_grad_005", "logsoftmax_grad_run", ((160, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[16, 2]=float:[16, 2]
            ("logsoftmax_grad_006", "logsoftmax_grad_run", ((16, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[20, 21128]=float:[20, 21128]
            ("logsoftmax_grad_007", "logsoftmax_grad_run", ((20, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[20480, 21128]=float:[20480, 21128]
            ("logsoftmax_grad_008", "logsoftmax_grad_run", ((20480, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[2, 2]=float:[2, 2]
            ("logsoftmax_grad_009", "logsoftmax_grad_run", ((2, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[2560, 21128]=float:[2560, 21128]
            ("logsoftmax_grad_0010", "logsoftmax_grad_run", ((2560, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[256, 2]=float:[256, 2]
            ("logsoftmax_grad_0011", "logsoftmax_grad_run", ((256, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[320, 21128]=float:[320, 21128]
            ("logsoftmax_grad_0012", "logsoftmax_grad_run", ((320, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[32, 2]=float:[32, 2]
            ("logsoftmax_grad_0013", "logsoftmax_grad_run", ((32, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[40, 21128]=float:[40, 21128]
            ("logsoftmax_grad_0014", "logsoftmax_grad_run", ((40, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[4, 2]=float:[4, 2]
            ("logsoftmax_grad_0015", "logsoftmax_grad_run", ((4, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[5120, 21128]=float:[5120, 21128]
            ("logsoftmax_grad_0016", "logsoftmax_grad_run", ((5120, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[512, 2]=float:[512, 2]
            ("logsoftmax_grad_0017", "logsoftmax_grad_run", ((512, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[640, 21128]=float:[640, 21128]
            ("logsoftmax_grad_0018", "logsoftmax_grad_run", ((640, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[80, 21128]=float:[80, 21128]
            ("logsoftmax_grad_0019", "logsoftmax_grad_run", ((80, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[8, 2]=float:[8, 2]
            ("logsoftmax_grad_0020", "logsoftmax_grad_run", ((8, 2), "float32", -1, "cce_logsoftmax_fp32")),

            # matmul op
            # float-float:[10240, 1024]-[1024, 1024]=float:[10240, 1024]
            ("matmul_01", "batchmatmul_run",
             ((), 10240, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[10240, 1024]-[21128, 1024]=float:[10240, 21128]
            ("matmul_02", "batchmatmul_run",
             ((), 10240, 21128, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[10240, 21128]-[10240, 1024]=float:[21128, 1024]
            ("matmul_03", "batchmatmul_run",
             ((), 21128, 1024, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[10240, 21128]-[21128, 1024]=float:[10240, 1024]
            ("matmul_04", "batchmatmul_run",
             ((), 10240, 1024, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[1024, 1024]-[1024, 1024]=float:[1024, 1024]
            (
                "matmul_05", "batchmatmul_run",
                ((), 1024, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[1024, 1024]-[1024, 4096]=float:[1024, 4096]
            (
                "matmul_06", "batchmatmul_run",
                ((), 1024, 4096, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[1024, 128, 1]-[1024, 1, 128]=float:[1024, 128, 128]
            (
                "matmul_07", "batchmatmul_run",
                ((1024,), 128, 128, 1, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[1024, 16, 128, 128]-[1024, 16, 128, 64]=float:[1024, 16, 128, 64]
            ("matmul_08", "batchmatmul_run",
             ((1024, 16), 128, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[1024, 16, 128, 64]-[1024, 16, 128, 64]=float:[1024, 16, 128, 128]
            ("matmul_09", "batchmatmul_run",
             ((1024, 16), 128, 128, 128, (), "float32", False, True, "batch_matmul_output")),
            # float-float:[1024, 2]-[2, 1024]=float:[1024, 1024]
            ("matmul_010", "batchmatmul_run", ((), 1024, 1024, 2, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[1024, 4096]-[4096, 1024]=float:[1024, 1024]
            ("matmul_011", "batchmatmul_run",
             ((), 1024, 1024, 4096, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[1, 1024]-[1024, 1024]=float:[1, 1024]
            ("matmul_012", "batchmatmul_run", ((), 1, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[1, 128, 1]-[1, 1, 128]=float:[1, 128, 128]
            ("matmul_013", "batchmatmul_run", ((1,), 128, 128, 1, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[1, 16, 128, 128]-[1, 16, 128, 64]=float:[1, 16, 128, 64]
            ("matmul_014", "batchmatmul_run",
             ((1, 16), 128, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[1, 16, 128, 64]-[1, 16, 128, 64]=float:[1, 16, 128, 128]
            ("matmul_015", "batchmatmul_run",
             ((1, 16), 128, 128, 128, (), "float32", False, True, "batch_matmul_output")),
            # float-float:[1, 2]-[2, 1024]=float:[1, 1024]
            ("matmul_016", "batchmatmul_run", ((), 1, 1024, 2, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[128, 1024]-[1024, 1024]=float:[128, 1024]
            (
                "matmul_017", "batchmatmul_run",
                ((), 128, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[128, 1024]-[1024, 4096]=float:[128, 4096]
            (
                "matmul_018", "batchmatmul_run",
                ((), 128, 4096, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[128, 128, 1]-[128, 1, 128]=float:[128, 128, 128]
            (
                "matmul_019", "batchmatmul_run",
                ((128,), 128, 128, 1, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[128, 128, 128]-[128, 128, 64]=float:[128, 128, 64]
            ("matmul_020", "batchmatmul_run",
             ((128,), 128, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[128, 128, 64]-[128, 128, 64]=float:[128, 128, 128]
            (
                "matmul_021", "batchmatmul_run",
                ((128,), 128, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float-float:[128, 16, 128, 128]-[128, 16, 128, 64]=float:[128, 16, 128, 64]
            ("matmul_022", "batchmatmul_run",
             ((128, 16), 128, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[128, 16, 128]-[128, 128, 64]=float:[128, 16, 64]
            (
                "matmul_023", "batchmatmul_run",
                ((128,), 16, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[128, 16, 128]-[128, 16, 64]=float:[128, 128, 64]
            ("matmul_024", "batchmatmul_run", ((128,), 128, 64, 16, (), "float32", True, False, "batch_matmul_output")),
            # float-float:[128, 16, 128, 64]-[128, 16, 128, 64]=float:[128, 16, 128, 128]
            ("matmul_025", "batchmatmul_run",
             ((128, 16), 128, 128, 128, (), "float32", False, True, "batch_matmul_output")),
            # float-float:[128, 16384, 128]-[128, 128, 64]=float:[128, 16384, 64]
            ("matmul_026", "batchmatmul_run",
             ((128,), 16384, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[128, 16384, 128]-[128, 16384, 64]=float:[128, 128, 64]
            ("matmul_027", "batchmatmul_run",
             ((128,), 128, 64, 16384, (), "float32", True, False, "batch_matmul_output")),
            # float-float:[128, 16384, 64]-[128, 128, 64]=float:[128, 16384, 128]
            ("matmul_028", "batchmatmul_run",
             ((128,), 16384, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float-float:[128, 16, 64]-[128, 128, 64]=float:[128, 16, 128]
            ("matmul_029", "batchmatmul_run", ((128,), 16, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float-float:[128, 2048, 128]-[128, 128, 64]=float:[128, 2048, 64]
            ("matmul_030", "batchmatmul_run",
             ((128,), 2048, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[128, 2048, 128]-[128, 2048, 64]=float:[128, 128, 64]
            ("matmul_031", "batchmatmul_run",
             ((128,), 128, 64, 2048, (), "float32", True, False, "batch_matmul_output")),
            # float-float:[128, 2048, 64]-[128, 128, 64]=float:[128, 2048, 128]
            ("matmul_032", "batchmatmul_run",
             ((128,), 2048, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float-float:[128, 2]-[2, 1024]=float:[128, 1024]
            ("matmul_033", "batchmatmul_run", ((), 128, 1024, 2, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[128, 256, 128]-[128, 128, 64]=float:[128, 256, 64]
            ("matmul_034", "batchmatmul_run",
             ((128,), 256, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[128, 256, 128]-[128, 256, 64]=float:[128, 128, 64]
            (
                "matmul_035", "batchmatmul_run",
                ((128,), 128, 64, 256, (), "float32", True, False, "batch_matmul_output")),
            # float-float:[128, 256, 64]-[128, 128, 64]=float:[128, 256, 128]
            (
                "matmul_036", "batchmatmul_run",
                ((128,), 256, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float-float:[128, 32, 128]-[128, 128, 64]=float:[128, 32, 64]
            (
                "matmul_037", "batchmatmul_run",
                ((128,), 32, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[128, 32, 128]-[128, 32, 64]=float:[128, 128, 64]
            ("matmul_038", "batchmatmul_run", ((128,), 128, 64, 32, (), "float32", True, False, "batch_matmul_output")),
            # float-float:[128, 32, 64]-[128, 128, 64]=float:[128, 32, 128]
            ("matmul_039", "batchmatmul_run", ((128,), 32, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float-float:[128, 4096, 128]-[128, 128, 64]=float:[128, 4096, 64]
            ("matmul_040", "batchmatmul_run",
             ((128,), 4096, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[128, 4096, 128]-[128, 4096, 64]=float:[128, 128, 64]
            ("matmul_041", "batchmatmul_run",
             ((128,), 128, 64, 4096, (), "float32", True, False, "batch_matmul_output")),
            # float-float:[128, 4096]-[4096, 1024]=float:[128, 1024]
            (
                "matmul_042", "batchmatmul_run",
                ((), 128, 1024, 4096, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[128, 4096, 64]-[128, 128, 64]=float:[128, 4096, 128]
            ("matmul_043", "batchmatmul_run",
             ((128,), 4096, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float-float:[128, 512, 128]-[128, 128, 64]=float:[128, 512, 64]
            ("matmul_044", "batchmatmul_run",
             ((128,), 512, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[128, 512, 128]-[128, 512, 64]=float:[128, 128, 64]
            (
                "matmul_045", "batchmatmul_run",
                ((128,), 128, 64, 512, (), "float32", True, False, "batch_matmul_output")),
            # float-float:[128, 512, 64]-[128, 128, 64]=float:[128, 512, 128]
            (
                "matmul_046", "batchmatmul_run",
                ((128,), 512, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float-float:[128, 64, 128]-[128, 128, 64]=float:[128, 64, 64]
            (
                "matmul_047", "batchmatmul_run",
                ((128,), 64, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[128, 64, 128]-[128, 64, 64]=float:[128, 128, 64]
            ("matmul_048", "batchmatmul_run", ((128,), 128, 64, 64, (), "float32", True, False, "batch_matmul_output")),
            # float-float:[128, 64, 64]-[128, 128, 64]=float:[128, 64, 128]
            ("matmul_049", "batchmatmul_run", ((128,), 64, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float-float:[128, 8192, 128]-[128, 128, 64]=float:[128, 8192, 64]
            ("matmul_050", "batchmatmul_run",
             ((128,), 8192, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[128, 8192, 128]-[128, 8192, 64]=float:[128, 128, 64]
            ("matmul_051", "batchmatmul_run",
             ((128,), 128, 64, 8192, (), "float32", True, False, "batch_matmul_output")),
            # float-float:[128, 8192, 64]-[128, 128, 64]=float:[128, 8192, 128]
            ("matmul_052", "batchmatmul_run",
             ((128,), 8192, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float-float:[131072, 1024]-[1024, 1024]=float:[131072, 1024]
            ("matmul_053", "batchmatmul_run",
             ((), 131072, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[131072, 1024]-[1024, 4096]=float:[131072, 4096]
            ("matmul_054", "batchmatmul_run",
             ((), 131072, 4096, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[131072, 4096]-[4096, 1024]=float:[131072, 1024]
            ("matmul_055", "batchmatmul_run",
             ((), 131072, 1024, 4096, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[160, 1024]-[1024, 1024]=float:[160, 1024]
            (
                "matmul_056", "batchmatmul_run",
                ((), 160, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[160, 1024]-[21128, 1024]=float:[160, 21128]
            ("matmul_057", "batchmatmul_run",
             ((), 160, 21128, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[160, 21128]-[160, 1024]=float:[21128, 1024]
            ("matmul_058", "batchmatmul_run",
             ((), 21128, 1024, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[160, 21128]-[21128, 1024]=float:[160, 1024]
            ("matmul_059", "batchmatmul_run",
             ((), 160, 1024, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[16, 1024]-[1024, 1024]=float:[16, 1024]
            ("matmul_060", "batchmatmul_run", ((), 16, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[16, 128, 1]-[16, 1, 128]=float:[16, 128, 128]
            ("matmul_061", "batchmatmul_run", ((16,), 128, 128, 1, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[16, 16, 128, 128]-[16, 16, 128, 64]=float:[16, 16, 128, 64]
            ("matmul_062", "batchmatmul_run",
             ((16, 16), 128, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[16, 16, 128, 64]-[16, 16, 128, 64]=float:[16, 16, 128, 128]
            ("matmul_063", "batchmatmul_run",
             ((16, 16), 128, 128, 128, (), "float32", False, True, "batch_matmul_output")),
            # float-float:[16, 2]-[2, 1024]=float:[16, 1024]
            ("matmul_064", "batchmatmul_run", ((), 16, 1024, 2, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[16384, 1024]-[1024, 1024]=float:[16384, 1024]
            ("matmul_065", "batchmatmul_run",
             ((), 16384, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[16384, 1024]-[1024, 4096]=float:[16384, 4096]
            ("matmul_066", "batchmatmul_run",
             ((), 16384, 4096, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[16384, 4096]-[4096, 1024]=float:[16384, 1024]
            ("matmul_067", "batchmatmul_run",
             ((), 16384, 1024, 4096, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[20, 1024]-[1024, 1024]=float:[20, 1024]
            ("matmul_068", "batchmatmul_run", ((), 20, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[20, 1024]-[21128, 1024]=float:[20, 21128]
            (
                "matmul_069", "batchmatmul_run",
                ((), 20, 21128, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[20, 21128]-[20, 1024]=float:[21128, 1024]
            ("matmul_070", "batchmatmul_run",
             ((), 21128, 1024, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[20, 21128]-[21128, 1024]=float:[20, 1024]
            (
                "matmul_071", "batchmatmul_run",
                ((), 20, 1024, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[20480, 1024]-[1024, 1024]=float:[20480, 1024]
            ("matmul_072", "batchmatmul_run",
             ((), 20480, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[20480, 1024]-[21128, 1024]=float:[20480, 21128]
            ("matmul_073", "batchmatmul_run",
             ((), 20480, 21128, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[20480, 21128]-[20480, 1024]=float:[21128, 1024]
            ("matmul_074", "batchmatmul_run",
             ((), 21128, 1024, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[20480, 21128]-[21128, 1024]=float:[20480, 1024]
            ("matmul_075", "batchmatmul_run",
             ((), 20480, 1024, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[2048, 1024]-[1024, 1024]=float:[2048, 1024]
            ("matmul_076", "batchmatmul_run",
             ((), 2048, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[2048, 1024]-[1024, 4096]=float:[2048, 4096]
            ("matmul_077", "batchmatmul_run",
             ((), 2048, 4096, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[2048, 4096]-[4096, 1024]=float:[2048, 1024]
            ("matmul_078", "batchmatmul_run",
             ((), 2048, 1024, 4096, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[2, 1024]-[1024, 1024]=float:[2, 1024]
            ("matmul_079", "batchmatmul_run", ((), 2, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[2, 128, 1]-[2, 1, 128]=float:[2, 128, 128]
            ("matmul_080", "batchmatmul_run", ((2,), 128, 128, 1, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[2, 16, 128, 128]-[2, 16, 128, 64]=float:[2, 16, 128, 64]
            ("matmul_081", "batchmatmul_run",
             ((2, 16), 128, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[2, 16, 128, 64]-[2, 16, 128, 64]=float:[2, 16, 128, 128]
            ("matmul_082", "batchmatmul_run",
             ((2, 16), 128, 128, 128, (), "float32", False, True, "batch_matmul_output")),
            # float-float:[2, 2]-[2, 1024]=float:[2, 1024]
            ("matmul_083", "batchmatmul_run", ((), 2, 1024, 2, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[2560, 1024]-[1024, 1024]=float:[2560, 1024]
            ("matmul_084", "batchmatmul_run",
             ((), 2560, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[2560, 1024]-[21128, 1024]=float:[2560, 21128]
            ("matmul_085", "batchmatmul_run",
             ((), 2560, 21128, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[2560, 21128]-[21128, 1024]=float:[2560, 1024]
            ("matmul_086", "batchmatmul_run",
             ((), 2560, 1024, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[2560, 21128]-[2560, 1024]=float:[21128, 1024]
            ("matmul_087", "batchmatmul_run",
             ((), 21128, 1024, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[256, 1024]-[1024, 1024]=float:[256, 1024]
            (
                "matmul_088", "batchmatmul_run",
                ((), 256, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[256, 1024]-[1024, 4096]=float:[256, 4096]
            (
                "matmul_089", "batchmatmul_run",
                ((), 256, 4096, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[256, 128, 1]-[256, 1, 128]=float:[256, 128, 128]
            (
                "matmul_090", "batchmatmul_run",
                ((256,), 128, 128, 1, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[256, 16, 128, 128]-[256, 16, 128, 64]=float:[256, 16, 128, 64]
            ("matmul_091", "batchmatmul_run",
             ((256, 16), 128, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[256, 16, 128, 64]-[256, 16, 128, 64]=float:[256, 16, 128, 128]
            ("matmul_092", "batchmatmul_run",
             ((256, 16), 128, 128, 128, (), "float32", False, True, "batch_matmul_output")),
            # float-float:[256, 2]-[2, 1024]=float:[256, 1024]
            ("matmul_093", "batchmatmul_run", ((), 256, 1024, 2, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[256, 4096]-[4096, 1024]=float:[256, 1024]
            (
                "matmul_094", "batchmatmul_run",
                ((), 256, 1024, 4096, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[320, 1024]-[1024, 1024]=float:[320, 1024]
            (
                "matmul_095", "batchmatmul_run",
                ((), 320, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[320, 1024]-[21128, 1024]=float:[320, 21128]
            ("matmul_096", "batchmatmul_run",
             ((), 320, 21128, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[320, 21128]-[21128, 1024]=float:[320, 1024]
            ("matmul_097", "batchmatmul_run",
             ((), 320, 1024, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[320, 21128]-[320, 1024]=float:[21128, 1024]
            ("matmul_098", "batchmatmul_run",
             ((), 21128, 1024, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[32, 1024]-[1024, 1024]=float:[32, 1024]
            ("matmul_099", "batchmatmul_run", ((), 32, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[32, 128, 1]-[32, 1, 128]=float:[32, 128, 128]
            (
                "matmul_0100", "batchmatmul_run",
                ((32,), 128, 128, 1, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[32, 16, 128, 128]-[32, 16, 128, 64]=float:[32, 16, 128, 64]
            ("matmul_0101", "batchmatmul_run",
             ((32, 16), 128, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[32, 16, 128, 64]-[32, 16, 128, 64]=float:[32, 16, 128, 128]
            ("matmul_0102", "batchmatmul_run",
             ((32, 16), 128, 128, 128, (), "float32", False, True, "batch_matmul_output")),
            # float-float:[32, 2]-[2, 1024]=float:[32, 1024]
            ("matmul_0103", "batchmatmul_run", ((), 32, 1024, 2, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[32768, 1024]-[1024, 1024]=float:[32768, 1024]
            ("matmul_0104", "batchmatmul_run",
             ((), 32768, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[32768, 1024]-[1024, 4096]=float:[32768, 4096]
            ("matmul_0105", "batchmatmul_run",
             ((), 32768, 4096, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[32768, 4096]-[4096, 1024]=float:[32768, 1024]
            ("matmul_0106", "batchmatmul_run",
             ((), 32768, 1024, 4096, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[40, 1024]-[1024, 1024]=float:[40, 1024]
            (
                "matmul_0107", "batchmatmul_run",
                ((), 40, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[40, 1024]-[21128, 1024]=float:[40, 21128]
            ("matmul_0108", "batchmatmul_run",
             ((), 40, 21128, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[40, 21128]-[21128, 1024]=float:[40, 1024]
            ("matmul_0109", "batchmatmul_run",
             ((), 40, 1024, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[40, 21128]-[40, 1024]=float:[21128, 1024]
            ("matmul_0110", "batchmatmul_run",
             ((), 21128, 1024, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[4096, 1024]-[1024, 1024]=float:[4096, 1024]
            ("matmul_0111", "batchmatmul_run",
             ((), 4096, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[4096, 1024]-[1024, 4096]=float:[4096, 4096]
            ("matmul_0112", "batchmatmul_run",
             ((), 4096, 4096, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[4096, 4096]-[4096, 1024]=float:[4096, 1024]
            ("matmul_0113", "batchmatmul_run",
             ((), 4096, 1024, 4096, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[4, 1024]-[1024, 1024]=float:[4, 1024]
            ("matmul_0114", "batchmatmul_run", ((), 4, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[4, 128, 1]-[4, 1, 128]=float:[4, 128, 128]
            ("matmul_0115", "batchmatmul_run", ((4,), 128, 128, 1, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[4, 16, 128, 128]-[4, 16, 128, 64]=float:[4, 16, 128, 64]
            ("matmul_0116", "batchmatmul_run",
             ((4, 16), 128, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[4, 16, 128, 64]-[4, 16, 128, 64]=float:[4, 16, 128, 128]
            ("matmul_0117", "batchmatmul_run",
             ((4, 16), 128, 128, 128, (), "float32", False, True, "batch_matmul_output")),
            # float-float:[4, 2]-[2, 1024]=float:[4, 1024]
            ("matmul_0118", "batchmatmul_run", ((), 4, 1024, 2, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[5120, 1024]-[1024, 1024]=float:[5120, 1024]
            ("matmul_0119", "batchmatmul_run",
             ((), 5120, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[5120, 1024]-[21128, 1024]=float:[5120, 21128]
            ("matmul_0120", "batchmatmul_run",
             ((), 5120, 21128, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[5120, 21128]-[21128, 1024]=float:[5120, 1024]
            ("matmul_0121", "batchmatmul_run",
             ((), 5120, 1024, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[5120, 21128]-[5120, 1024]=float:[21128, 1024]
            ("matmul_0122", "batchmatmul_run",
             ((), 21128, 1024, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[512, 1024]-[1024, 1024]=float:[512, 1024]
            ("matmul_0123", "batchmatmul_run",
             ((), 512, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[512, 1024]-[1024, 4096]=float:[512, 4096]
            ("matmul_0124", "batchmatmul_run",
             ((), 512, 4096, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[512, 128, 1]-[512, 1, 128]=float:[512, 128, 128]
            ("matmul_0125", "batchmatmul_run",
             ((512,), 128, 128, 1, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[512, 16, 128, 128]-[512, 16, 128, 64]=float:[512, 16, 128, 64]
            ("matmul_0126", "batchmatmul_run",
             ((512, 16), 128, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[512, 16, 128, 64]-[512, 16, 128, 64]=float:[512, 16, 128, 128]
            ("matmul_0127", "batchmatmul_run",
             ((512, 16), 128, 128, 128, (), "float32", False, True, "batch_matmul_output")),
            # float-float:[512, 2]-[2, 1024]=float:[512, 1024]
            ("matmul_0128", "batchmatmul_run", ((), 512, 1024, 2, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[512, 4096]-[4096, 1024]=float:[512, 1024]
            ("matmul_0129", "batchmatmul_run",
             ((), 512, 1024, 4096, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[640, 1024]-[1024, 1024]=float:[640, 1024]
            ("matmul_0130", "batchmatmul_run",
             ((), 640, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[640, 1024]-[21128, 1024]=float:[640, 21128]
            ("matmul_0131", "batchmatmul_run",
             ((), 640, 21128, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[640, 21128]-[21128, 1024]=float:[640, 1024]
            ("matmul_0132", "batchmatmul_run",
             ((), 640, 1024, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[640, 21128]-[640, 1024]=float:[21128, 1024]
            ("matmul_0133", "batchmatmul_run",
             ((), 21128, 1024, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[65536, 1024]-[1024, 1024]=float:[65536, 1024]
            ("matmul_0134", "batchmatmul_run",
             ((), 65536, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[65536, 1024]-[1024, 4096]=float:[65536, 4096]
            ("matmul_0135", "batchmatmul_run",
             ((), 65536, 4096, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[65536, 4096]-[4096, 1024]=float:[65536, 1024]
            ("matmul_0136", "batchmatmul_run",
             ((), 65536, 1024, 4096, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[80, 1024]-[1024, 1024]=float:[80, 1024]
            (
                "matmul_0137", "batchmatmul_run",
                ((), 80, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[80, 1024]-[21128, 1024]=float:[80, 21128]
            ("matmul_0138", "batchmatmul_run",
             ((), 80, 21128, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[80, 21128]-[21128, 1024]=float:[80, 1024]
            ("matmul_0139", "batchmatmul_run",
             ((), 80, 1024, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[80, 21128]-[80, 1024]=float:[21128, 1024]
            ("matmul_0140", "batchmatmul_run",
             ((), 21128, 1024, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[8, 1024]-[1024, 1024]=float:[8, 1024]
            ("matmul_0141", "batchmatmul_run", ((), 8, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[8, 128, 1]-[8, 1, 128]=float:[8, 128, 128]
            ("matmul_0142", "batchmatmul_run", ((8,), 128, 128, 1, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[8, 16, 128, 128]-[8, 16, 128, 64]=float:[8, 16, 128, 64]
            ("matmul_0143", "batchmatmul_run",
             ((8, 16), 128, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[8, 16, 128, 64]-[8, 16, 128, 64]=float:[8, 16, 128, 128]
            ("matmul_0144", "batchmatmul_run",
             ((8, 16), 128, 128, 128, (), "float32", False, True, "batch_matmul_output")),
            # float-float:[8, 2]-[2, 1024]=float:[8, 1024]
            ("matmul_0145", "batchmatmul_run", ((), 8, 1024, 2, (), "float32", False, False, "batch_matmul_output")),

            # Neg OP
            # float:[10240]=float:[10240]
            ("neg_001", "neg_run", ((10240,), "float32")),
            # float:[1024]=float:[1024]
            ("neg_002", "neg_run", ((1024,), "float32")),
            # float:[128]=float:[128]
            ("neg_003", "neg_run", ((128,), "float32")),
            # float:[160]=float:[160]
            ("neg_004", "neg_run", ((160,), "float32")),
            # float:[16]=float:[16]
            ("neg_005", "neg_run", ((16,), "float32")),
            # float:[1]=float:[1]
            ("neg_006", "neg_run", ((1,), "float32")),
            # float:[20480]=float:[20480]
            ("neg_007", "neg_run", ((20480,), "float32")),
            # float:[20]=float:[20]
            ("neg_008", "neg_run", ((20,), "float32")),
            # float:[2560]=float:[2560]
            ("neg_009", "neg_run", ((2560,), "float32")),
            # float:[256]=float:[256]
            ("neg_0010", "neg_run", ((256,), "float32")),
            # float:[2]=float:[2]
            ("neg_0011", "neg_run", ((2,), "float32")),
            # float:[320]=float:[320]
            ("neg_0012", "neg_run", ((320,), "float32")),
            # float:[32]=float:[32]
            ("neg_0013", "neg_run", ((32,), "float32")),
            # float:[40]=float:[40]
            ("neg_0014", "neg_run", ((40,), "float32")),
            # float:[4]=float:[4]
            ("neg_0015", "neg_run", ((4,), "float32")),
            # float:[5120]=float:[5120]
            ("neg_0016", "neg_run", ((5120,), "float32")),
            # float:[512]=float:[512]
            ("neg_0017", "neg_run", ((512,), "float32")),
            # float:[640]=float:[640]
            ("neg_0018", "neg_run", ((640,), "float32")),
            # float:[80]=float:[80]
            ("neg_0019", "neg_run", ((80,), "float32")),
            # float:[8]=float:[8]
            ("neg_0020", "neg_run", ((8,), "float32")),

            # onehot OP
            # int32-int32-float-float:[10240]-[]-[]-[]=float:[10240, 21128]
            ("one_hot_001", "one_hot_run", ((10240,), 21128, "int32", 1, 0, -1)),
            # int32-int32-float-float:[1024]-[]-[]-[]=float:[1024, 2]
            ("one_hot_002", "one_hot_run", ((1024,), 2, "int32", 1, 0, -1)),
            # int32-int32-float-float:[128]-[]-[]-[]=float:[128, 2]
            ("one_hot_003", "one_hot_run", ((128,), 2, "int32", 1, 0, -1)),
            # int32-int32-float-float:[160]-[]-[]-[]=float:[160, 21128]
            ("one_hot_004", "one_hot_run", ((160,), 21128, "int32", 1, 0, -1)),
            # int32-int32-float-float:[16]-[]-[]-[]=float:[16, 2]
            ("one_hot_005", "one_hot_run", ((16,), 2, "int32", 1, 0, -1)),
            # int32-int32-float-float:[1]-[]-[]-[]=float:[2]
            ("one_hot_006", "one_hot_run", ((1,), 2, "int32", 1, 0, -1)),
            # int32-int32-float-float:[20480]-[]-[]-[]=float:[20480, 21128]
            ("one_hot_007", "one_hot_run", ((20480,), 21128, "int32", 1, 0, -1)),
            # int32-int32-float-float:[20]-[]-[]-[]=float:[20, 21128]
            ("one_hot_008", "one_hot_run", ((20,), 21128, "int32", 1, 0, -1)),
            # int32-int32-float-float:[2560]-[]-[]-[]=float:[2560, 21128]
            ("one_hot_009", "one_hot_run", ((2560,), 21128, "int32", 1, 0, -1)),
            # int32-int32-float-float:[256]-[]-[]-[]=float:[256, 2]
            ("one_hot_0010", "one_hot_run", ((256,), 2, "int32", 1, 0, -1)),
            # int32-int32-float-float:[2]-[]-[]-[]=float:[2, 2]
            ("one_hot_0011", "one_hot_run", ((2,), 2, "int32", 1, 0, -1)),
            # int32-int32-float-float:[320]-[]-[]-[]=float:[320, 21128]
            ("one_hot_0012", "one_hot_run", ((320,), 21128, "int32", 1, 0, -1)),
            # int32-int32-float-float:[32]-[]-[]-[]=float:[32, 2]
            ("one_hot_0013", "one_hot_run", ((32,), 2, "int32", 1, 0, -1)),
            # int32-int32-float-float:[40]-[]-[]-[]=float:[40, 21128]
            ("one_hot_0014", "one_hot_run", ((40,), 21128, "int32", 1, 0, -1)),
            # int32-int32-float-float:[4]-[]-[]-[]=float:[4, 2]
            ("one_hot_0015", "one_hot_run", ((4,), 2, "int32", 1, 0, -1)),
            # int32-int32-float-float:[5120]-[]-[]-[]=float:[5120, 21128]
            ("one_hot_0016", "one_hot_run", ((5120,), 21128, "int32", 1, 0, -1)),
            # int32-int32-float-float:[512]-[]-[]-[]=float:[512, 2]
            ("one_hot_0017", "one_hot_run", ((512,), 2, "int32", 1, 0, -1)),
            # int32-int32-float-float:[640]-[]-[]-[]=float:[640, 21128]
            ("one_hot_0018", "one_hot_run", ((640,), 21128, "int32", 1, 0, -1)),
            # int32-int32-float-float:[80]-[]-[]-[]=float:[80, 21128]
            ("one_hot_0019", "one_hot_run", ((80,), 21128, "int32", 1, 0, -1)),
            # int32-int32-float-float:[8]-[]-[]-[]=float:[8, 2]
            ("one_hot_0020", "one_hot_run", ((8,), 2, "int32", 1, 0, -1)),

            # sub OP
            # float-float:[1024, 1024]-[1024, 1024]=float:[1024, 1024]
            ("sub_001", "sub_run", [(1024, 1024), (1024, 1024), "float32"]),
            # float-float:[1024]-[1024]=float:[1024]
            ("sub_002", "sub_run", [(1024,), (1024,), "float32"]),
            # float-float:[1024, 16, 128, 128]-[1024, 16, 128, 1]=float:[1024, 16, 128, 128]
            ("sub_003", "sub_run", [(1024, 16, 128, 128,), (1024, 16, 128, 1), "float32"]),
            # float-float:[1024, 4096]-[1024, 4096]=float:[1024, 4096]
            ("sub_004", "sub_run", [(1024, 4096,), (1024, 4096), "float32"]),
            # float-float:[1]-[1024, 1, 128, 128]=float:[1024, 1, 128, 128]
            ("sub_005", "sub_run", [(1,), (1024, 1, 128, 128), "float32"]),
            # float-float:[1]-[1, 1, 128, 128]=float:[1, 1, 128, 128]
            ("sub_006", "sub_run", [(1,), (1, 1, 128, 128), "float32"]),
            # float-float:[1]-[128, 1, 128, 128]=float:[128, 1, 128, 128]
            ("sub_007", "sub_run", [(1,), (128, 1, 128, 128), "float32"]),
            # float-float:[1]-[16, 1, 128, 128]=float:[16, 1, 128, 128]
            ("sub_008", "sub_run", [(1,), (16, 1, 128, 128), "float32"]),
            # float-float:[1, 16, 128, 128]-[1, 16, 128, 1]=float:[1, 16, 128, 128]
            ("sub_009", "sub_run", [(1, 16, 128, 128,), (1, 16, 128, 1), "float32"]),
            # float-float:[1]-[1]=float:[1]
            ("sub_0010", "sub_run", [(1,), (1,), "float32"]),
            # float-float:[1]-[2, 1, 128, 128]=float:[2, 1, 128, 128]
            ("sub_0011", "sub_run", [(1,), (2, 1, 128, 128), "float32"]),
            # float-float:[1]-[256, 1, 128, 128]=float:[256, 1, 128, 128]
            ("sub_0012", "sub_run", [(1,), (256, 1, 128, 128), "float32"]),
            # float-float:[128, 16, 128, 128]-[128, 16, 128, 1]=float:[128, 16, 128, 128]
            ("sub_0013", "sub_run", [(128, 16, 128, 128,), (128, 16, 128, 1), "float32"]),
            # float-float:[1]-[32, 1, 128, 128]=float:[32, 1, 128, 128]
            ("sub_0014", "sub_run", [(1,), (32, 1, 128, 128), "float32"]),
            # float-float:[1]-[4, 1, 128, 128]=float:[4, 1, 128, 128]
            ("sub_0015", "sub_run", [(1,), (4, 1, 128, 128), "float32"]),
            # float-float:[1]-[512, 1, 128, 128]=float:[512, 1, 128, 128]
            ("sub_0016", "sub_run", [(1,), (512, 1, 128, 128), "float32"]),
            # float-float:[16, 16, 128, 128]-[16, 16, 128, 1]=float:[16, 16, 128, 128]
            ("sub_0017", "sub_run", [(16, 16, 128, 128,), (16, 16, 128, 1), "float32"]),
            # float-float:[1]-[8, 1, 128, 128]=float:[8, 1, 128, 128]
            ("sub_0018", "sub_run", [(1,), (8, 1, 128, 128), "float32"]),
            # float-float:[2, 1024]-[2, 1024]=float:[2, 1024]
            ("sub_0019", "sub_run", [(2, 1024,), (2, 1024), "float32"]),
            # float-float:[21128, 1024]-[21128, 1024]=float:[21128, 1024
            ("sub_0020", "sub_run", [(21128, 1024,), (21128, 1024), "float32"]),
            # float-float:[21128]-[21128]=float:[21128]
            ("sub_0021", "sub_run", [(21128,), (21128,), "float32"]),
            # float-float:[2, 16, 128, 128]-[2, 16, 128, 1]=float:[2, 16, 128, 128]
            ("sub_0022", "sub_run", [(2, 16, 128, 128,), (2, 16, 128, 1), "float32"]),
            # float-float:[2]-[2]=float:[2]
            ("sub_0023", "sub_run", [(2,), (2,), "float32"]),
            # float-float:[256, 16, 128, 128]-[256, 16, 128, 1]=float:[256, 16, 128, 128]
            ("sub_0024", "sub_run", [(256, 16, 128, 128,), (256, 16, 128, 1), "float32"]),
            # float-float:[32, 16, 128, 128]-[32, 16, 128, 1]=float:[32, 16, 128, 128]
            ("sub_0025", "sub_run", [(32, 16, 128, 128,), (32, 16, 128, 1), "float32"]),
            # float-float:[33, 64]-[33, 64]=float:[33, 64]
            ("sub_0026", "sub_run", [(33, 64,), (33, 64), "float32"]),
            # float-float:[4096, 1024]-[4096, 1024]=float:[4096, 1024]
            ("sub_0027", "sub_run", [(4096, 1024,), (4096, 1024), "float32"]),
            # float-float:[4096]-[4096]=float:[4096]
            ("sub_0028", "sub_run", [(4096,), (4096,), "float32"]),
            # float-float:[4, 16, 128, 128]-[4, 16, 128, 1]=float:[4, 16, 128, 128]
            ("sub_0029", "sub_run", [(4, 16, 128, 128,), (4, 16, 128, 1), "float32"]),
            # float-float:[512, 16, 128, 128]-[512, 16, 128, 1]=float:[512, 16, 128, 128]
            ("sub_0030", "sub_run", [(512, 16, 128, 128,), (512, 16, 128, 1), "float32"]),
            # float-float:[8, 16, 128, 128]-[8, 16, 128, 1]=float:[8, 16, 128, 128]
            ("sub_0031", "sub_run", [(8, 16, 128, 128,), (8, 16, 128, 1), "float32"]),
            # int32-int32:[128, 128]-[128, 128]=int32:[128, 128]
            ("sub_0032", "sub_run", [(128, 128,), (128, 128), "int32"]),

            # sum OP
            # float-int32:[10240]-[-1]=float:[1]
            ("sum_001", "sum_run", ((10240,), (-1,), False, "float32")),
            # float-int32:[10240, 21128]-[-1]=float:[21128]
            ("sum_002", "sum_run", ((10240, 21128), (-1,), False, "float32")),
            # float-int32:[1024, 1024]-[-1]=float:[1024]
            ("sum_003", "sum_run", ((1024, 1024), (-1,), False, "float32")),
            # float-int32:[1024, 16, 128, 128]-[-1]=float:[1024, 16, 128, 1]
            ("sum_004", "sum_run", ((1024, 16, 128, 128), (-1,), False, "float32")),
            # float-int32:[1024]-[-1]=float:[1]
            ("sum_005", "sum_run", ((1024,), (-1,), False, "float32")),
            # float-int32:[1024, 4096]-[-1]=float:[1024]
            ("sum_006", "sum_run", ((1024, 4096), (-1,), False, "float32")),
            # float-int32:[1, 16, 128, 128]-[-1]=float:[1, 16, 128, 1]
            ("sum_007", "sum_run", ((1, 16, 128, 128), (-1,), False, "float32")),
            # float-int32:[128, 16, 128, 128]-[-1]=float:[128, 16, 128, 1]
            ("sum_008", "sum_run", ((128, 16, 128, 128), (-1,), False, "float32")),
            # float-int32:[160]-[-1]=float:[1]
            ("sum_009", "sum_run", ((160,), (-1,), False, "float32")),
            # float-int32:[160, 21128]-[-1]=float:[21128]
            ("sum_0010", "sum_run", ((160, 21128), (-1,), False, "float32")),
            # float-int32:[16, 16, 128, 128]-[-1]=float:[16, 16, 128, 1]
            ("sum_0011", "sum_run", ((16, 16, 128, 128), (-1,), False, "float32")),
            # float-int32:[20]-[-1]=float:[1]
            ("sum_0012", "sum_run", ((20,), (-1,), False, "float32")),
            # float-int32:[20, 21128]-[-1]=float:[21128]
            ("sum_0013", "sum_run", ((20, 21128), (-1,), False, "float32")),
            # float-int32:[20480]-[-1]=float:[1]
            ("sum_0014", "sum_run", ((20480,), (-1,), False, "float32")),
            # float-int32:[20480, 21128]-[-1]=float:[21128]
            ("sum_0015", "sum_run", ((20480, 21128), (-1,), False, "float32")),
            # float-int32:[2, 1024]-[-1]=float:[2]
            ("sum_0016", "sum_run", ((2, 1024), (-1,), False, "float32")),
            # float-int32:[21128, 1024]-[-1]=float:[21128]
            ("sum_0017", "sum_run", ((21128, 1024), (-1,), False, "float32")),
            # float-int32:[21128]-[-1]=float:[1]
            ("sum_0018", "sum_run", ((21128,), (-1,), False, "float32")),
            # float-int32:[2, 16, 128, 128]-[-1]=float:[2, 16, 128, 1]
            ("sum_0019", "sum_run", ((2, 16, 128, 128), (-1,), False, "float32")),
            # float-int32:[2]-[-1]=float:[1]
            ("sum_0020", "sum_run", ((2,), (-1,), False, "float32")),
            # float-int32:[2560]-[-1]=float:[1]
            ("sum_0021", "sum_run", ((2560,), (-1,), False, "float32")),
            # float-int32:[2560, 21128]-[-1]=float:[21128]
            ("sum_0022", "sum_run", ((2560, 21128), (-1,), False, "float32")),
            # float-int32:[256, 16, 128, 128]-[-1]=float:[256, 16, 128, 1]
            ("sum_0023", "sum_run", ((256, 16, 128, 128), (-1,), False, "float32")),
            # float-int32:[320]-[-1]=float:[1]
            ("sum_0024", "sum_run", ((320,), (-1,), False, "float32")),
            # float-int32:[320, 21128]-[-1]=float:[21128]
            ("sum_0025", "sum_run", ((320, 21128), (-1,), False, "float32")),
            # float-int32:[32, 16, 128, 128]-[-1]=float:[32, 16, 128, 1]
            ("sum_0026", "sum_run", ((32, 16, 128, 128), (-1,), False, "float32")),
            # float-int32:[33, 64]-[-1]=float:[33]
            ("sum_0027", "sum_run", ((33, 64), (-1,), False, "float32")),
            # float-int32:[40]-[-1]=float:[1]
            ("sum_0028", "sum_run", ((40,), (-1,), False, "float32")),
            # float-int32:[40, 21128]-[-1]=float:[21128]
            ("sum_0029", "sum_run", ((40, 21128), (-1,), False, "float32")),
            # float-int32:[4096, 1024]-[-1]=float:[4096]
            ("sum_0030", "sum_run", ((4096, 1024), (-1,), False, "float32")),
            # float-int32:[4096]-[-1]=float:[1
            ("sum_0031", "sum_run", ((4096,), (-1,), False, "float32")),
            # float-int32:[4, 16, 128, 128]-[-1]=float:[4, 16, 128, 1]
            ("sum_0032", "sum_run", ((4, 16, 128, 128), (-1,), False, "float32")),
            # float-int32:[5120]-[-1]=float:[1]
            ("sum_0033", "sum_run", ((5120,), (-1,), False, "float32")),
            # float-int32:[5120, 21128]-[-1]=float:[21128]
            ("sum_0034", "sum_run", ((5120, 21128), (-1,), False, "float32")),
            # float-int32:[512, 16, 128, 128]-[-1]=float:[512, 16, 128, 1]
            ("sum_0035", "sum_run", ((512, 16, 128, 128), (-1,), False, "float32")),
            # float-int32:[640]-[-1]=float:[1]
            ("sum_0036", "sum_run", ((640,), (-1,), False, "float32")),
            # float-int32:[640, 21128]-[-1]=float:[21128]
            ("sum_0037", "sum_run", ((640, 21128), (-1,), False, "float32")),
            # float-int32:[80]-[-1]=float:[1]
            ("sum_0038", "sum_run", ((80,), (-1,), False, "float32")),
            # float-int32:[80, 21128]-[-1]=float:[21128]
            ("sum_0039", "sum_run", ((80, 21128), (-1,), False, "float32")),
            # float-int32:[8, 16, 128, 128]-[-1]=float:[8, 16, 128, 1]
            ("sum_0040", "sum_run", ((8, 16, 128, 128), (-1,), False, "float32")),

            # StridedSlice OP
            # float-int32-int32-int32:[1024, 128, 1024]-[3]-[3]-[3]=float:[1024, 1, 1024]
            ("strided_slice_001", "strided_slice_run",
             ((1024, 128, 1024), [0, 0, 0], [1024, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),
            # float-int32-int32-int32:[1, 128, 1024]-[3]-[3]-[3]=float:[1, 1, 1024]
            ("strided_slice_002", "strided_slice_run",
             ((1, 128, 1024), [0, 0, 0], [1, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),
            # float-int32-int32-int32:[128, 128, 1024]-[3]-[3]-[3]=float:[128, 1, 1024]
            ("strided_slice_003", "strided_slice_run",
             ((128, 128, 1024), [0, 0, 0], [128, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),
            # float-int32-int32-int32:[16, 128, 1024]-[3]-[3]-[3]=float:[16, 1, 1024]
            ("strided_slice_004", "strided_slice_run",
             ((16, 128, 1024), [0, 0, 0], [16, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),
            # float-int32-int32-int32:[2, 128, 1024]-[3]-[3]-[3]=float:[2, 1, 1024]
            ("strided_slice_005", "strided_slice_run",
             ((2, 128, 1024), [0, 0, 0], [2, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),
            # float-int32-int32-int32:[256, 128, 1024]-[3]-[3]-[3]=float:[256, 1, 1024]
            ("strided_slice_006", "strided_slice_run",
             ((256, 128, 1024), [0, 0, 0], [256, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),
            # float-int32-int32-int32:[32, 128, 1024]-[3]-[3]-[3]=float:[32, 1, 1024]
            ("strided_slice_007", "strided_slice_run",
             ((32, 128, 1024), [0, 0, 0], [32, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),
            # float-int32-int32-int32:[4, 128, 1024]-[3]-[3]-[3]=float:[4, 1, 1024]
            ("strided_slice_008", "strided_slice_run",
             ((4, 128, 1024), [0, 0, 0], [4, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),
            # float-int32-int32-int32:[512, 128, 1024]-[3]-[3]-[3]=float:[512, 1, 1024]
            ("strided_slice_009", "strided_slice_run",
             ((512, 128, 1024), [0, 0, 0], [512, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),
            # float-int32-int32-int32:[8, 128, 1024]-[3]-[3]-[3]=float:[8, 1, 1024]
            ("strided_slice_0010", "strided_slice_run",
             ((8, 128, 1024), [0, 0, 0], [8, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),

            # StridedSliceGrad OP
            # float-int32-int32-int32-int32:[1024, 1, 1024]-[3]-[3]-[3]-[3]=float:[1024, 128, 1024]
            ("strided_slice_grad_001", "strided_slice_grad_run",
             [(1024, 128, 1024), [0, 0, 0], [1024, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, (1024, 1, 1024), "int32"]),
            # float-int32-int32-int32-int32:[1, 1, 1024]-[3]-[3]-[3]-[3]=float:[1, 128, 1024]
            ("strided_slice_grad_002", "strided_slice_grad_run",
             [(1, 128, 1024), [0, 0, 0], [1, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, (1, 1, 1024), "int32"]),
            # float-int32-int32-int32-int32:[128, 1, 1024]-[3]-[3]-[3]-[3]=float:[128, 128, 1024]
            ("strided_slice_grad_003", "strided_slice_grad_run",
             [(128, 128, 1024), [0, 0, 0], [128, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, (128, 1, 1024), "int32"]),
            # float-int32-int32-int32-int32:[16, 1, 1024]-[3]-[3]-[3]-[3]=float:[16, 128, 1024]
            ("strided_slice_grad_004", "strided_slice_grad_run",
             [(16, 128, 1024), [0, 0, 0], [16, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, (16, 1, 1024), "int32"]),
            # float-int32-int32-int32-int32:[2, 1, 1024]-[3]-[3]-[3]-[3]=float:[2, 128, 1024]
            ("strided_slice_grad_005", "strided_slice_grad_run",
             [(2, 128, 1024), [0, 0, 0], [2, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, (2, 1, 1024), "int32"]),
            # float-int32-int32-int32-int32:[256, 1, 1024]-[3]-[3]-[3]-[3]=float:[256, 128, 1024]
            ("strided_slice_grad_006", "strided_slice_grad_run",
             [(256, 128, 1024), [0, 0, 0], [256, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, (256, 1, 1024), "int32"]),
            # float-int32-int32-int32-int32:[32, 1, 1024]-[3]-[3]-[3]-[3]=float:[32, 128, 1024]
            ("strided_slice_grad_007", "strided_slice_grad_run",
             [(32, 128, 1024), [0, 0, 0], [32, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, (32, 1, 1024), "int32"]),
            # float-int32-int32-int32-int32:[4, 1, 1024]-[3]-[3]-[3]-[3]=float:[4, 128, 1024]
            ("strided_slice_grad_008", "strided_slice_grad_run",
             [(4, 128, 1024), [0, 0, 0], [4, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, (4, 1, 1024), "int32"]),
            # float-int32-int32-int32-int32:[512, 1, 1024]-[3]-[3]-[3]-[3]=float:[512, 128, 1024]
            ("strided_slice_grad_009", "strided_slice_grad_run",
             [(512, 128, 1024), [0, 0, 0], [512, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, (512, 1, 1024), "int32"]),
            # float-int32-int32-int32-int32:[8, 1, 1024]-[3]-[3]-[3]-[3]=float:[8, 128, 1024]
            ("strided_slice_grad_0010", "strided_slice_grad_run",
             [(8, 128, 1024), [0, 0, 0], [8, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, (8, 1, 1024), "int32"]),

            # tanh OP
            # float:[1024, 1024]=float:[1024, 1024]
            ("tanh_001", "tanh_run", ((1024, 1024), "float32")),
            # float:[1, 1024]=float:[1, 1024]
            ("tanh_002", "tanh_run", ((1, 1024), "float32")),
            # float:[128, 1024]=float:[128, 1024]
            ("tanh_003", "tanh_run", ((128, 1024), "float32")),
            # float:[16, 1024]=float:[16, 1024]
            ("tanh_004", "tanh_run", ((16, 1024), "float32")),
            # float:[2, 1024]=float:[2, 1024]
            ("tanh_005", "tanh_run", ((2, 1024), "float32")),
            # float:[256, 1024]=float:[256, 1024]
            ("tanh_006", "tanh_run", ((256, 1024), "float32")),
            # float:[32, 1024]=float:[32, 1024]
            ("tanh_007", "tanh_run", ((32, 1024), "float32")),
            # float:[4, 1024]=float:[4, 1024]
            ("tanh_008", "tanh_run", ((4, 1024), "float32")),
            # float:[512, 1024]=float:[512, 1024]
            ("tanh_009", "tanh_run", ((512, 1024), "float32")),
            # float:[8, 1024]=float:[8, 1024]
            ("tanh_010", "tanh_run", ((8, 1024), "float32")),

            # tanh OP
            # float:[1024, 1024]=float:[1024, 1024]
            ("tanh_grad_001", "tanh_grad_run", ((1024, 1024), "float32")),
            # float:[1, 1024]=float:[1, 1024]
            ("tanh_grad_002", "tanh_grad_run", ((1, 1024), "float32")),
            # float:[128, 1024]=float:[128, 1024]
            ("tanh_grad_003", "tanh_grad_run", ((128, 1024), "float32")),
            # float:[16, 1024]=float:[16, 1024]
            ("tanh_grad_004", "tanh_grad_run", ((16, 1024), "float32")),
            # float:[2, 1024]=float:[2, 1024]
            ("tanh_grad_005", "tanh_grad_run", ((2, 1024), "float32")),
            # float:[256, 1024]=float:[256, 1024]
            ("tanh_grad_006", "tanh_grad_run", ((256, 1024), "float32")),
            # float:[32, 1024]=float:[32, 1024]
            ("tanh_grad_007", "tanh_grad_run", ((32, 1024), "float32")),
            # float:[4, 1024]=float:[4, 1024]
            ("tanh_grad_008", "tanh_grad_run", ((4, 1024), "float32")),
            # float:[512, 1024]=float:[512, 1024]
            ("tanh_grad_009", "tanh_grad_run", ((512, 1024), "float32")),
            # float:[8, 1024]=float:[8, 1024]
            ("tanh_grad_010", "tanh_grad_run", ((8, 1024), "float32")),

            # reshape OP
            # float-int32:[10240, 1]-[2]=float:[10240]
            ("reshape_0001", "reshape_run", [(10240, 1), (10240,), "float32"]),
            ("reshape_0002", "reshape_run", [(10240,), (10240, 1), "float32"]),
            # float-int32:[1024, 1024]-[2]=float:[1024, 1024]
            ("reshape_0003", "reshape_run", [(1024, 1024), (1024, 1024), "float32"]),
            # float-int32:[1024, 1024]-[2]=float:[8, 128, 1024]
            ("reshape_0004", "reshape_run", [(1024, 1024), (8, 128, 1024), "float32"]),
            ("reshape_0005", "reshape_run", [(8, 128, 1024), (1024, 1024), "float32"]),
            # float-int32:[1024, 1024]-[2]=float:[8, 128, 16, 64]
            ("reshape_0006", "reshape_run", [(1024, 1024), (8, 128, 16, 64), "float32"]),
            ("reshape_0007", "reshape_run", [(8, 128, 16, 64), (1024, 1024), "float32"]),
            # float-int32:[1024, 1024]-[3]=float:[1024, 1, 1024]
            ("reshape_0008", "reshape_run", [(1024, 1024), (1024, 1, 1024), "float32"]),
            ("reshape_0009", "reshape_run", [(1024, 1, 1024), (1024, 1024), "float32"]),
            # float-int32:[1024, 128, 1024]-[2]=float:[131072, 1024]
            ("reshape_0010", "reshape_run", [(1024, 128, 1024), (131072, 1024), "float32"]),
            ("reshape_0011", "reshape_run", [(131072, 1024), (1024, 128, 1024), "float32"]),
            # float-int32:[1024, 128, 16, 64]-[2]=float:[131072, 1024]
            ("reshape_0012", "reshape_run", [(1024, 128, 16, 64), (131072, 1024), "float32"]),
            ("reshape_0013", "reshape_run", [(131072, 1024), (1024, 128, 16, 64), "float32"]),
            # float-int32:[1024, 1]-[2]=float:[1024]
            ("reshape_0014", "reshape_run", [(1024, 1), (1024,), "float32"]),
            ("reshape_0015", "reshape_run", [(1024,), (1024, 1), "float32"]),
            # float-int32:[1024, 4096]-[2]=float:[1024, 4096]
            ("reshape_0016", "reshape_run", [(1024, 4096), (1024, 4096), "float32"]),
            # float-int32:[1, 1024]-[3]=float:[1, 1, 1024]
            ("reshape_0017", "reshape_run", [(1, 1024), (1, 1, 1024), "float32"]),
            ("reshape_0018", "reshape_run", [(1, 1, 1024), (1, 1024), "float32"]),
            # float-int32:[1, 128, 1024]-[2]=float:[128, 1024]
            ("reshape_0019", "reshape_run", [(1, 128, 1024), (128, 1024), "float32"]),
            ("reshape_0020", "reshape_run", [(128, 1024), (1, 128, 1024), "float32"]),
            # float-int32:[1, 128, 16, 64]-[2]=float:[128, 1024]
            ("reshape_0021", "reshape_run", [(1, 128, 16, 64), (128, 1024), "float32"]),
            ("reshape_0022", "reshape_run", [(128, 1024), (1, 128, 16, 64), "float32"]),
            # float-int32:[1, 1]-[2]=float:[1]
            ("reshape_0023", "reshape_run", [(1, 1), (1,), "float32"]),
            ("reshape_0024", "reshape_run", [(1,), (1, 1), "float32"]),
            # float-int32:[128, 1024, 16, 128]-[3]=float:[128, 16384, 128]
            ("reshape_0025", "reshape_run", [(128, 1024, 16, 128), (128, 16384, 128), "float32"]),
            ("reshape_0026", "reshape_run", [(128, 16384, 128), (128, 1024, 16, 128), "float32"]),
            # float-int32:[128, 1024, 16, 64]-[3]=float:[128, 16384, 64]
            ("reshape_0027", "reshape_run", [(128, 1024, 16, 64), (128, 16384, 64), "float32"]),
            ("reshape_0028", "reshape_run", [(128, 16384, 64), (128, 1024, 16, 64), "float32"]),
            # float-int32:[128, 1024]-[2]=float:[128, 1024]
            ("reshape_0029", "reshape_run", [(128, 1024), (128, 1024), "float32"]),
            # float-int32:[128, 1024]-[3]=float:[128, 1, 1024]
            ("reshape_0030", "reshape_run", [(128, 1024), (128, 1, 1024), "float32"]),
            ("reshape_0031", "reshape_run", [(128, 1, 1024), (128, 1024), "float32"]),
            # float-int32:[128, 1, 16, 128]-[3]=float:[128, 16, 128]
            ("reshape_0032", "reshape_run", [(128, 1, 16, 128), (128, 16, 128), "float32"]),
            ("reshape_0033", "reshape_run", [(128, 16, 128), (128, 1, 16, 128), "float32"]),
            # float-int32:[128, 1, 16, 64]-[3]=float:[128, 16, 64]
            ("reshape_0034", "reshape_run", [(128, 1, 16, 64), (128, 16, 64), "float32"]),
            ("reshape_0035", "reshape_run", [(128, 16, 64), (128, 1, 16, 64), "float32"]),
            # float-int32:[128, 128, 1024]-[2]=float:[16384, 1024]
            ("reshape_0036", "reshape_run", [(128, 128, 1024), (16384, 1024), "float32"]),
            ("reshape_0037", "reshape_run", [(16384, 1024), (128, 128, 1024), "float32"]),
            # float-int32:[128, 128, 128]-[3]=float:[128, 8, 16, 128]
            ("reshape_0038", "reshape_run", [(128, 128, 128), (128, 8, 16, 128), "float32"]),
            ("reshape_0039", "reshape_run", [(128, 8, 16, 128), (128, 128, 128), "float32"]),
            # float-int32:[128, 128, 16, 128]-[3]=float:[128, 2048, 128]
            ("reshape_0040", "reshape_run", [(128, 128, 16, 128), (128, 2048, 128), "float32"]),
            ("reshape_0041", "reshape_run", [(128, 2048, 128), (128, 128, 16, 128), "float32"]),
            # float-int32:[128, 128, 16, 64]-[2]=float:[16384, 1024]
            ("reshape_0042", "reshape_run", [(128, 128, 16, 64), (16384, 1024), "float32"]),
            ("reshape_0043", "reshape_run", [(16384, 1024), (128, 128, 16, 64), "float32"]),
            # float-int32:[128, 128, 16, 64]-[3]=float:[128, 2048, 64]
            ("reshape_0044", "reshape_run", [(128, 128, 16, 64), (128, 2048, 64), "float32"]),
            ("reshape_0045", "reshape_run", [(128, 2048, 64), (128, 128, 16, 64), "float32"]),
            # float-int32:[128, 128, 64]-[3]=float:[128, 8, 16, 64]
            ("reshape_0046", "reshape_run", [(128, 128, 64), (128, 8, 16, 64), "float32"]),
            ("reshape_0047", "reshape_run", [(128, 8, 16, 64), (128, 128, 64), "float32"]),
            # float-int32:[128, 1]-[2]=float:[128]
            ("reshape_0048", "reshape_run", [(128, 1), (128,), "float32"]),
            ("reshape_0049", "reshape_run", [(128,), (128, 1), "float32"]),
            # float-int32:[128, 16, 16, 128]-[3]=float:[128, 256, 128]
            ("reshape_0050", "reshape_run", [(128, 16, 16, 128), (128, 256, 128), "float32"]),
            ("reshape_0051", "reshape_run", [(128, 256, 128), (128, 16, 16, 128), "float32"]),
            # float-int32:[128, 16, 16, 64]-[3]=float:[128, 256, 64]
            ("reshape_0052", "reshape_run", [(128, 16, 16, 64), (128, 256, 64), "float32"]),
            ("reshape_0053", "reshape_run", [(128, 256, 64), (128, 16, 16, 64), "float32"]),
            # float-int32:[128, 2, 16, 128]-[3]=float:[128, 32, 128]
            ("reshape_0054", "reshape_run", [(128, 2, 16, 128), (128, 32, 128), "float32"]),
            ("reshape_0055", "reshape_run", [(128, 32, 128), (128, 2, 16, 128), "float32"]),
            # float-int32:[128, 2, 16, 64]-[3]=float:[128, 32, 64]
            ("reshape_0056", "reshape_run", [(128, 2, 16, 64), (128, 32, 64), "float32"]),
            ("reshape_0057", "reshape_run", [(128, 32, 64), (128, 2, 16, 64), "float32"]),
            # float-int32:[128, 256, 16, 128]-[3]=float:[128, 4096, 128]
            ("reshape_0058", "reshape_run", [(128, 256, 16, 128), (128, 4096, 128), "float32"]),
            ("reshape_0059", "reshape_run", [(128, 4096, 128), (128, 256, 16, 128), "float32"]),
            # float-int32:[128, 256, 16, 64]-[3]=float:[128, 4096, 64]
            ("reshape_0060", "reshape_run", [(128, 256, 16, 64), (128, 4096, 64), "float32"]),
            ("reshape_0061", "reshape_run", [(128, 4096, 64), (128, 256, 16, 64), "float32"]),
            # float-int32:[128, 32, 16, 128]-[3]=float:[128, 512, 128]
            ("reshape_0062", "reshape_run", [(128, 32, 16, 128), (128, 512, 128), "float32"]),
            ("reshape_0063", "reshape_run", [(128, 512, 128), (128, 32, 16, 128), "float32"]),
            # float-int32:[128, 32, 16, 64]-[3]=float:[128, 512, 64]
            ("reshape_0064", "reshape_run", [(128, 32, 16, 64), (128, 512, 64), "float32"]),
            ("reshape_0065", "reshape_run", [(128, 512, 64), (128, 32, 16, 64), "float32"]),
            # float-int32:[128, 4, 16, 128]-[3]=float:[128, 64, 128]
            ("reshape_0066", "reshape_run", [(128, 4, 16, 128), (128, 64, 128), "float32"]),
            ("reshape_0067", "reshape_run", [(128, 64, 128), (128, 4, 16, 128), "float32"]),
            # float-int32:[128, 4, 16, 64]-[3]=float:[128, 64, 64]
            ("reshape_0068", "reshape_run", [(128, 4, 16, 64), (128, 64, 64), "float32"]),
            ("reshape_0069", "reshape_run", [(128, 64, 64), (128, 4, 16, 64), "float32"]),
            # float-int32:[128, 512, 16, 128]-[3]=float:[128, 8192, 128]
            ("reshape_0070", "reshape_run", [(128, 512, 16, 128), (128, 8192, 128), "float32"]),
            ("reshape_0071", "reshape_run", [(128, 8192, 128), (128, 512, 16, 128), "float32"]),
            # float-int32:[128, 512, 16, 64]-[3]=float:[128, 8192, 64]
            ("reshape_0072", "reshape_run", [(128, 512, 16, 64), (128, 8192, 64), "float32"]),
            ("reshape_0073", "reshape_run", [(128, 8192, 64), (128, 512, 16, 64), "float32"]),
            # float-int32:[131072, 1024]-[2]=float:[131072, 1024]
            ("reshape_0074", "reshape_run", [(131072, 1024), (131072, 1024), "float32"]),
            # float-int32:[160, 1]-[2]=float:[160]
            ("reshape_0075", "reshape_run", [(160, 1), (160,), "float32"]),
            ("reshape_0076", "reshape_run", [(160,), (160, 1), "float32"]),
            # float-int32:[16, 1024]-[3]=float:[16, 1, 1024]
            ("reshape_0077", "reshape_run", [(16, 1024), (16, 1, 1024), "float32"]),
            ("reshape_0078", "reshape_run", [(16, 1, 1024), (16, 1024), "float32"]),
            # float-int32:[16, 128, 1024]-[2]=float:[2048, 1024]
            ("reshape_0079", "reshape_run", [(16, 128, 1024), (2048, 1024), "float32"]),
            ("reshape_0080", "reshape_run", [(2048, 1024), (16, 128, 1024), "float32"]),
            # float-int32:[16, 128, 16, 64]-[2]=float:[2048, 1024]
            ("reshape_0081", "reshape_run", [(16, 128, 16, 64), (2048, 1024), "float32"]),
            ("reshape_0082", "reshape_run", [(2048, 1024), (16, 128, 16, 64), "float32"]),
            # float-int32:[16, 1]-[2]=float:[16]
            ("reshape_0083", "reshape_run", [(16, 1), (16,), "float32"]),
            ("reshape_0084", "reshape_run", [(16,), (16, 1), "float32"]),
            # float-int32:[16384, 1024]-[2]=float:[16384, 1024]
            ("reshape_0085", "reshape_run", [(16384, 1024), (16384, 1024), "float32"]),
            # float-int32:[20, 1]-[2]=float:[20]
            ("reshape_0086", "reshape_run", [(20, 1), (20,), "float32"]),
            ("reshape_0087", "reshape_run", [(20,), (20, 1), "float32"]),
            # float-int32:[20480, 1]-[2]=float:[20480]
            ("reshape_0088", "reshape_run", [(20480, 1), (20480,), "float32"]),
            ("reshape_0089", "reshape_run", [(20480,), (20480, 1), "float32"]),
            # float-int32:[2048, 1024]-[2]=float:[2048, 1024]
            ("reshape_0090", "reshape_run", [(2048, 1024), (2048, 1024), "float32"]),
            # float-int32:[2, 1024]-[2]=float:[2, 1024]
            ("reshape_0091", "reshape_run", [(2, 1024), (2, 1024), "float32"]),
            # float-int32:[2, 1024]-[3]=float:[2, 1, 1024]
            ("reshape_0092", "reshape_run", [(2, 1024), (2, 1, 1024), "float32"]),
            ("reshape_0093", "reshape_run", [(2, 1, 1024), (2, 1024), "float32"]),
            # float-int32:[21128, 1024]-[2]=float:[21128, 1024]
            ("reshape_0094", "reshape_run", [(21128, 1024), (21128, 1024), "float32"]),
            # float-int32:[2, 128, 1024]-[2]=float:[256, 1024]
            ("reshape_0095", "reshape_run", [(2, 128, 1024), (256, 1024), "float32"]),
            ("reshape_0096", "reshape_run", [(256, 1024), (2, 128, 1024), "float32"]),
            # float-int32:[2, 128, 16, 64]-[2]=float:[256, 1024]
            ("reshape_0097", "reshape_run", [(2, 128, 16, 64), (256, 1024), "float32"]),
            ("reshape_0098", "reshape_run", [(256, 1024), (2, 128, 16, 64), "float32"]),
            # float-int32:[2, 1]-[2]=float:[2]
            ("reshape_0099", "reshape_run", [(2, 1), (2,), "float32"]),
            ("reshape_0100", "reshape_run", [(2,), (2, 1), "float32"]),
            # float-int32:[2560, 1]-[2]=float:[2560]
            ("reshape_0101", "reshape_run", [(2560, 1), (2560,), "float32"]),
            ("reshape_0102", "reshape_run", [(2560,), (2560, 1), "float32"]),
            # float-int32:[256, 1024]-[2]=float:[256, 1024]
            ("reshape_0103", "reshape_run", [(256, 1024), (256, 1024), "float32"]),
            # float-int32:[256, 1024]-[3]=float:[256, 1, 1024]
            ("reshape_0104", "reshape_run", [(256, 1024), (256, 1, 1024), "float32"]),
            ("reshape_0105", "reshape_run", [(256, 1, 1024), (256, 1024), "float32"]),
            # float-int32:[256, 128, 1024]-[2]=float:[32768, 1024]
            ("reshape_0106", "reshape_run", [(256, 128, 1024), (32768, 1024), "float32"]),
            ("reshape_0107", "reshape_run", [(32768, 1024), (256, 128, 1024), "float32"]),
            # float-int32:[256, 128, 16, 64]-[2]=float:[32768, 1024]
            ("reshape_0108", "reshape_run", [(256, 128, 16, 64), (32768, 1024), "float32"]),
            ("reshape_0109", "reshape_run", [(32768, 1024), (256, 128, 16, 64), "float32"]),
            # float-int32:[256, 1]-[2]=float:[256]
            ("reshape_0110", "reshape_run", [(256, 1), (256,), "float32"]),
            ("reshape_0111", "reshape_run", [(256,), (256, 1), "float32"]),
            # float-int32:[320, 1]-[2]=float:[320]
            ("reshape_0112", "reshape_run", [(320, 1), (320,), "float32"]),
            ("reshape_0113", "reshape_run", [(320,), (320, 1), "float32"]),
            # float-int32:[32, 1024]-[3]=float:[32, 1, 1024]
            ("reshape_0114", "reshape_run", [(32, 1024), (32, 1, 1024), "float32"]),
            ("reshape_0115", "reshape_run", [(32, 1, 1024), (32, 1024), "float32"]),
            # float-int32:[32, 128, 1024]-[2]=float:[4096, 1024]
            ("reshape_0116", "reshape_run", [(32, 128, 1024), (4096, 1024), "float32"]),
            ("reshape_0117", "reshape_run", [(4096, 1024), (32, 128, 1024), "float32"]),
            # float-int32:[32, 128, 16, 64]-[2]=float:[4096, 1024]
            ("reshape_0118", "reshape_run", [(32, 128, 16, 64), (4096, 1024), "float32"]),
            ("reshape_0119", "reshape_run", [(4096, 1024), (32, 128, 16, 64), "float32"]),
            # float-int32:[32, 1]-[2]=float:[32]
            ("reshape_0120", "reshape_run", [(32, 1), (32,), "float32"]),
            ("reshape_0121", "reshape_run", [(32,), (32, 1), "float32"]),
            # float-int32:[32768, 1024]-[2]=float:[32768, 1024]
            ("reshape_0122", "reshape_run", [(32768, 1024), (32768, 1024), "float32"]),
            # float-int32:[33, 64]-[2]=float:[33, 64]
            ("reshape_0123", "reshape_run", [(33, 64), (33, 64), "float32"]),
            # float-int32:[40, 1]-[2]=float:[40]
            ("reshape_0124", "reshape_run", [(40, 1), (40,), "float32"]),
            ("reshape_0125", "reshape_run", [(40,), (40, 1), "float32"]),
            # float-int32:[4096, 1024]-[2]=float:[4096, 1024]
            ("reshape_0126", "reshape_run", [(4096, 1024), (4096, 1024), "float32"]),
            # float-int32:[4, 1024]-[3]=float:[4, 1, 1024]
            ("reshape_0127", "reshape_run", [(4, 1024), (4, 1, 1024), "float32"]),
            ("reshape_0128", "reshape_run", [(4, 1, 1024), (4, 1024), "float32"]),
            # float-int32:[4, 128, 1024]-[2]=float:[512, 1024]
            ("reshape_0129", "reshape_run", [(4, 128, 1024), (512, 1024), "float32"]),
            ("reshape_0130", "reshape_run", [(512, 1024), (4, 128, 1024), "float32"]),
            # float-int32:[4, 128, 16, 64]-[2]=float:[512, 1024]
            ("reshape_0131", "reshape_run", [(4, 128, 16, 64), (512, 1024), "float32"]),
            ("reshape_0132", "reshape_run", [(512, 1024), (4, 128, 16, 64), "float32"]),
            # float-int32:[4, 1]-[2]=float:[4]
            ("reshape_0133", "reshape_run", [(4, 1), (4,), "float32"]),
            ("reshape_0134", "reshape_run", [(4,), (4, 1), "float32"]),
            # float-int32:[5120, 1]-[2]=float:[5120]
            ("reshape_0135", "reshape_run", [(5120, 1), (5120,), "float32"]),
            ("reshape_0136", "reshape_run", [(5120,), (5120, 1), "float32"]),
            # float-int32:[512, 1024]-[2]=float:[512, 1024]
            ("reshape_0137", "reshape_run", [(512, 1024), (512, 1024), "float32"]),
            # float-int32:[512, 1024]-[3]=float:[512, 1, 1024]
            ("reshape_0138", "reshape_run", [(512, 1024), (512, 1, 1024), "float32"]),
            ("reshape_0139", "reshape_run", [(512, 1, 1024), (512, 1024), "float32"]),
            # float-int32:[512, 128, 1024]-[2]=float:[65536, 1024]
            ("reshape_0140", "reshape_run", [(512, 128, 1024), (65536, 1024), "float32"]),
            ("reshape_0141", "reshape_run", [(65536, 1024), (512, 128, 1024), "float32"]),
            # float-int32:[512, 128, 16, 64]-[2]=float:[65536, 1024]
            ("reshape_0142", "reshape_run", [(512, 128, 16, 64), (65536, 1024), "float32"]),
            ("reshape_0143", "reshape_run", [(65536, 1024), (512, 128, 16, 64), "float32"]),
            # float-int32:[512, 1]-[2]=float:[512]
            ("reshape_0144", "reshape_run", [(512, 1), (512,), "float32"]),
            ("reshape_0145", "reshape_run", [(512,), (512, 1), "float32"]),
            # float-int32:[640, 1]-[2]=float:[640]
            ("reshape_0146", "reshape_run", [(640, 1), (640,), "float32"]),
            ("reshape_0147", "reshape_run", [(640,), (640, 1), "float32"]),
            # float-int32:[65536, 1024]-[2]=float:[65536, 1024]
            ("reshape_0148", "reshape_run", [(65536, 1024), (65536, 1024), "float32"]),
            # float-int32:[80, 1]-[2]=float:[80]
            ("reshape_0149", "reshape_run", [(80, 1), (80,), "float32"]),
            ("reshape_0150", "reshape_run", [(80,), (80, 1), "float32"]),
            # float-int32:[8, 1024]-[3]=float:[8, 1, 1024]
            ("reshape_0151", "reshape_run", [(8, 1024), (8, 1, 1024), "float32"]),
            ("reshape_0152", "reshape_run", [(8, 1, 1024), (8, 1024), "float32"]),
            # float-int32:[8, 1]-[2]=float:[8]
            ("reshape_0153", "reshape_run", [(8, 1), (8,), "float32"]),
            ("reshape_0154", "reshape_run", [(8,), (8, 1), "float32"]),
            # int32-int32:[1024, 1, 128]-[3]=int32:[1024, 128]
            ("reshape_0155", "reshape_run", [(1024, 1, 128), (1024, 128), "int32"]),
            ("reshape_0156", "reshape_run", [(1024, 128), (1024, 1, 128), "int32"]),
            # int32-int32:[1024, 1]-[2]=int32:[1024]
            ("reshape_0157", "reshape_run", [(1024, 1), (1024,), "int32"]),
            ("reshape_0158", "reshape_run", [(1024,), (1024, 1), "int32"]),
            # int32-int32:[1, 1, 128]-[3]=int32:[1, 128]
            ("reshape_0159", "reshape_run", [(1, 1, 128), (1, 128), "int32"]),
            ("reshape_0160", "reshape_run", [(1, 128), (1, 1, 128), "int32"]),
            # int32-int32:[1, 1]-[2]=int32:[1]
            ("reshape_0161", "reshape_run", [(1, 1), (1,), "int32"]),
            ("reshape_0162", "reshape_run", [(1,), (1, 1), "int32"]),
            # int32-int32:[128, 1, 128]-[3]=int32:[128, 128]
            ("reshape_0163", "reshape_run", [(128, 1, 128), (128, 128), "int32"]),
            ("reshape_0164", "reshape_run", [(128, 128), (128, 1, 128), "int32"]),
            # int32-int32:[128, 128]-[2]=int32:[16384]
            ("reshape_0165", "reshape_run", [(128, 128), (16384,), "int32"]),
            ("reshape_0166", "reshape_run", [(16384,), (128, 128), "int32"]),
            # int32-int32:[128, 1]-[2]=int32:[128]
            ("reshape_0167", "reshape_run", [(128, 1), (128,), "int32"]),
            ("reshape_0168", "reshape_run", [(128,), (128, 1), "int32"]),
            # int32-int32:[16, 1, 128]-[3]=int32:[16, 128]
            ("reshape_0169", "reshape_run", [(16, 1, 128), (16, 128), "int32"]),
            ("reshape_0170", "reshape_run", [(16, 128), (16, 1, 128), "int32"]),
            # int32-int32:[16, 1]-[2]=int32:[16]
            ("reshape_0171", "reshape_run", [(16, 1), (16,), "int32"]),
            ("reshape_0172", "reshape_run", [(16,), (16, 1), "int32"]),
            # int32-int32:[2, 1, 128]-[3]=int32:[2, 128]
            ("reshape_0173", "reshape_run", [(2, 1, 128), (2, 128), "int32"]),
            ("reshape_0174", "reshape_run", [(2, 128), (2, 1, 128), "int32"]),
            # int32-int32:[2, 1]-[2]=int32:[2]
            ("reshape_0175", "reshape_run", [(2, 1), (2,), "int32"]),
            ("reshape_0176", "reshape_run", [(2,), (2, 1), "int32"]),
            # int32-int32:[256, 1, 128]-[3]=int32:[256, 128]
            ("reshape_0177", "reshape_run", [(256, 1, 128), (256, 128), "int32"]),
            ("reshape_0178", "reshape_run", [(256, 128), (256, 1, 128), "int32"]),
            # int32-int32:[256, 1]-[2]=int32:[256]
            ("reshape_0179", "reshape_run", [(256, 1), (256,), "int32"]),
            ("reshape_0180", "reshape_run", [(256,), (256, 1), "int32"]),
            # int32-int32:[32, 1, 128]-[3]=int32:[32, 128]
            ("reshape_0181", "reshape_run", [(32, 1, 128), (32, 128), "int32"]),
            ("reshape_0182", "reshape_run", [(32, 128), (32, 1, 128), "int32"]),
            # int32-int32:[32, 1]-[2]=int32:[32]
            ("reshape_0183", "reshape_run", [(32, 1), (32,), "int32"]),
            ("reshape_0184", "reshape_run", [(32,), (32, 1), "int32"]),
            # int32-int32:[4, 1, 128]-[3]=int32:[4, 128]
            ("reshape_0185", "reshape_run", [(4, 1, 128), (4, 128), "int32"]),
            ("reshape_0186", "reshape_run", [(4, 128), (4, 1, 128), "int32"]),
            # int32-int32:[4, 1]-[2]=int32:[4]
            ("reshape_0187", "reshape_run", [(4, 1), (4,), "int32"]),
            ("reshape_0188", "reshape_run", [(4,), (4, 1), "int32"]),
            # int32-int32:[512, 1, 128]-[3]=int32:[512, 128]
            ("reshape_0189", "reshape_run", [(512, 1, 128), (512, 128), "int32"]),
            ("reshape_0190", "reshape_run", [(512, 128), (512, 1, 128), "int32"]),
            # int32-int32:[512, 1]-[2]=int32:[512]
            ("reshape_0191", "reshape_run", [(512, 1), (512,), "int32"]),
            ("reshape_0192", "reshape_run", [(512,), (512, 1), "int32"]),
            # int32-int32:[8, 1, 128]-[3]=int32:[8, 128]
            ("reshape_0193", "reshape_run", [(8, 1, 128), (8, 128), "int32"]),
            ("reshape_0194", "reshape_run", [(8, 128), (8, 1, 128), "int32"]),
            # int32-int32:[8, 1]-[2]=int32:[8]
            ("reshape_0195", "reshape_run", [(8, 1), (8,), "int32"]),
            ("reshape_0196", "reshape_run", [(8,), (8, 1), "int32"]),
            # float-int32:[10240]-[]=float:[512, 20]
            ("reshape_0197", "reshape_run", [(10240,), (512, 20), "float32"]),
            ("reshape_0198", "reshape_run", [(512, 20), (10240,), "float32"]),
            # float-int32:[1024, 20]-[]=float:[20480]
            ("reshape_0199", "reshape_run", [(1024, 20), (20480,), "float32"]),
            ("reshape_0200", "reshape_run", [(20480,), (1024, 20), "float32"]),
            # float-int32:[1024]-[]=float:[1024]
            ("reshape_0201", "reshape_run", [(1024,), (1024,), "float32"]),
            # float-int32:[1024]-[]=float:[1, 1024]
            ("reshape_0202", "reshape_run", [(1024,), (1, 1024), "float32"]),
            ("reshape_0203", "reshape_run", [(1, 1024), (1024,), "float32"]),
            # float-int32:[1, 20]-[]=float:[20]
            ("reshape_0204", "reshape_run", [(1, 20), (20,), "float32"]),
            ("reshape_0205", "reshape_run", [(20,), (1, 20), "float32"]),
            # float-int32:[1, 21128]-[]=float:[21128]
            ("reshape_0206", "reshape_run", [(1, 21128), (21128,), "float32"]),
            ("reshape_0207", "reshape_run", [(21128,), (1, 21128), "float32"]),
            # float-int32:[128, 20]-[]=float:[2560]
            ("reshape_0208", "reshape_run", [(128, 20), (2560,), "float32"]),
            ("reshape_0209", "reshape_run", [(2560,), (128, 20), "float32"]),
            # float-int32:[1, 2]-[]=float:[2]
            ("reshape_0210", "reshape_run", [(1, 2), (2,), "float32"]),
            ("reshape_0211", "reshape_run", [(2,), (1, 2), "float32"]),
            # float-int32:[1, 4096]-[]=float:[4096]
            ("reshape_0212", "reshape_run", [(1, 4096), (4096,), "float32"]),
            ("reshape_0213", "reshape_run", [(4096,), (1, 4096), "float32"]),
            # float-int32:[160]-[]=float:[8, 20]
            ("reshape_0214", "reshape_run", [(160,), (8, 20), "float32"]),
            ("reshape_0215", "reshape_run", [(8, 20), (160,), "float32"]),
            # float-int32:[16, 20]-[]=float:[320]
            ("reshape_0216", "reshape_run", [(16, 20), (320,), "float32"]),
            ("reshape_0217", "reshape_run", [(320,), (16, 20), "float32"]),
            # float-int32:[1]-[]=float:[1]
            ("reshape_0218", "reshape_run", [(1,), (1,), "float32"]),
            # float-int32:[21128]-[]=float:[21128]
            ("reshape_0219", "reshape_run", [(21128,), (21128,), "float32"]),
            # float-int32:[2, 20]-[]=float:[40]
            ("reshape_0220", "reshape_run", [(2, 20), (40,), "float32"]),
            ("reshape_0221", "reshape_run", [(40,), (2, 20), "float32"]),
            # float-int32:[256, 20]-[]=float:[5120]
            ("reshape_0222", "reshape_run", [(256, 20), (5120,), "float32"]),
            ("reshape_0223", "reshape_run", [(5120,), (256, 20), "float32"]),
            # float-int32:[2]-[]=float:[2]
            ("reshape_0224", "reshape_run", [(2,), (2,), "float32"]),
            # float-int32:[32, 20]-[]=float:[640]
            ("reshape_0225", "reshape_run", [(32, 20), (640,), "float32"]),
            ("reshape_0226", "reshape_run", [(640,), (32, 20), "float32"]),
            # float-int32:[4096]-[]=float:[4096]
            ("reshape_0227", "reshape_run", [(4096,), (4096,), "float32"]),
            # float-int32:[4, 20]-[]=float:[80]
            ("reshape_0228", "reshape_run", [(4, 20), (80,), "float32"]),
            ("reshape_0229", "reshape_run", [(80,), (4, 20), "float32"]),
            # int32-int32:[10240]-[]=int32:[512, 20]
            ("reshape_0230", "reshape_run", [(10240,), (512, 20), "int32"]),
            ("reshape_0231", "reshape_run", [(512, 20), (10240,), "int32"]),
            # int32-int32:[1024, 128, 1]-[]=int32:[131072]
            ("reshape_0232", "reshape_run", [(1024, 128, 1), (131072,), "int32"]),
            ("reshape_0233", "reshape_run", [(131072,), (1024, 128, 1), "int32"]),
            # int32-int32:[1024, 128]-[]=int32:[131072]
            ("reshape_0234", "reshape_run", [(1024, 128), (131072,), "int32"]),
            ("reshape_0235", "reshape_run", [(131072,), (1024, 128), "int32"]),
            # int32-int32:[1024, 20]-[]=int32:[20480]
            ("reshape_0236", "reshape_run", [(1024, 20), (20480,), "int32"]),
            ("reshape_0237", "reshape_run", [(20480,), (1024, 20), "int32"]),
            # int32-int32:[1024]-[]=int32:[8, 128]
            ("reshape_0238", "reshape_run", [(1024,), (8, 128), "int32"]),
            ("reshape_0239", "reshape_run", [(8, 128), (1024,), "int32"]),
            # int32-int32:[1024]-[]=int32:[8, 128, 1]
            ("reshape_0240", "reshape_run", [(1024,), (8, 128, 1), "int32"]),
            ("reshape_0241", "reshape_run", [(8, 128, 1), (1024,), "int32"]),
            # int32-int32:[1, 128, 1]-[]=int32:[128]
            ("reshape_0242", "reshape_run", [(1, 128, 1), (128,), "int32"]),
            ("reshape_0243", "reshape_run", [(128,), (1, 128, 1), "int32"]),
            # int32-int32:[1, 128]-[]=int32:[128]
            ("reshape_0244", "reshape_run", [(1, 128), (128,), "int32"]),
            ("reshape_0245", "reshape_run", [(128,), (1, 128), "int32"]),
            # int32-int32:[1, 20]-[]=int32:[20]
            ("reshape_0246", "reshape_run", [(1, 20), (20,), "int32"]),
            ("reshape_0247", "reshape_run", [(20,), (1, 20), "int32"]),
            # int32-int32:[128, 128, 1]-[]=int32:[16384]
            ("reshape_0248", "reshape_run", [(128, 128, 1), (16384,), "int32"]),
            ("reshape_0249", "reshape_run", [(16384,), (128, 128, 1), "int32"]),
            # int32-int32:[128, 20]-[]=int32:[2560]
            ("reshape_0250", "reshape_run", [(128, 20), (2560,), "int32"]),
            ("reshape_0251", "reshape_run", [(2560,), (128, 20), "int32"]),
            # int32-int32:[160]-[]=int32:[8, 20]
            ("reshape_0252", "reshape_run", [(160,), (8, 20), "int32"]),
            ("reshape_0253", "reshape_run", [(8, 20), (160,), "int32"]),
            # int32-int32:[16, 128, 1]-[]=int32:[2048]
            ("reshape_0254", "reshape_run", [(16, 128, 1), (2048,), "int32"]),
            ("reshape_0255", "reshape_run", [(2048,), (16, 128, 1), "int32"]),
            # int32-int32:[16, 128]-[]=int32:[2048]
            ("reshape_0256", "reshape_run", [(16, 128), (2048,), "int32"]),
            ("reshape_0257", "reshape_run", [(2048,), (16, 128), "int32"]),
            # int32-int32:[16, 20]-[]=int32:[320]
            ("reshape_0258", "reshape_run", [(16, 20), (320,), "int32"]),
            ("reshape_0259", "reshape_run", [(320,), (16, 20), "int32"]),
            # int32-int32:[2, 128, 1]-[]=int32:[256]
            ("reshape_0260", "reshape_run", [(2, 128, 1), (256,), "int32"]),
            ("reshape_0261", "reshape_run", [(256,), (2, 128, 1), "int32"]),
            # int32-int32:[2, 128]-[]=int32:[256]
            ("reshape_0262", "reshape_run", [(2, 128), (256,), "int32"]),
            ("reshape_0263", "reshape_run", [(256,), (2, 128), "int32"]),
            # int32-int32:[2, 20]-[]=int32:[40]
            ("reshape_0264", "reshape_run", [(2, 20), (40,), "int32"]),
            ("reshape_0265", "reshape_run", [(40,), (2, 20), "int32"]),
            # int32-int32:[256, 128, 1]-[]=int32:[32768]
            ("reshape_0266", "reshape_run", [(256, 128, 1), (32768,), "int32"]),
            ("reshape_0267", "reshape_run", [(32768,), (256, 128, 1), "int32"]),
            # int32-int32:[256, 128]-[]=int32:[32768]
            ("reshape_0268", "reshape_run", [(256, 128), (32768,), "int32"]),
            ("reshape_0269", "reshape_run", [(32768,), (256, 128), "int32"]),
            # int32-int32:[256, 20]-[]=int32:[5120]
            ("reshape_0270", "reshape_run", [(256, 20), (5120,), "int32"]),
            ("reshape_0271", "reshape_run", [(5120,), (256, 20), "int32"]),
            # int32-int32:[32, 128, 1]-[]=int32:[4096]
            ("reshape_0272", "reshape_run", [(32, 128, 1), (4096,), "int32"]),
            ("reshape_0273", "reshape_run", [(4096,), (32, 128, 1), "int32"]),
            # int32-int32:[32, 128]-[]=int32:[4096]
            ("reshape_0274", "reshape_run", [(32, 128), (4096,), "int32"]),
            ("reshape_0275", "reshape_run", [(4096,), (32, 128), "int32"]),
            # int32-int32:[32, 20]-[]=int32:[640]
            ("reshape_0276", "reshape_run", [(32, 20), (640,), "int32"]),
            ("reshape_0277", "reshape_run", [(640,), (32, 20), "int32"]),
            # int32-int32:[4, 128, 1]-[]=int32:[512]
            ("reshape_0278", "reshape_run", [(4, 128, 1), (512,), "int32"]),
            ("reshape_0279", "reshape_run", [(512,), (4, 128, 1), "int32"]),
            # int32-int32:[4, 128]-[]=int32:[512]
            ("reshape_0280", "reshape_run", [(4, 128), (512,), "int32"]),
            ("reshape_0281", "reshape_run", [(512,), (4, 128), "int32"]),
            # int32-int32:[4, 20]-[]=int32:[80]
            ("reshape_0282", "reshape_run", [(4, 20), (80,), "int32"]),
            ("reshape_0283", "reshape_run", [(80,), (4, 20), "int32"]),
            # int32-int32:[512, 128, 1]-[]=int32:[65536]
            ("reshape_0284", "reshape_run", [(512, 128, 1), (65536,), "int32"]),
            ("reshape_0285", "reshape_run", [(65536,), (512, 128, 1), "int32"]),
            # int32-int32:[512, 128]-[]=int32:[65536]
            ("reshape_0286", "reshape_run", [(512, 128), (65536,), "int32"]),
            ("reshape_0287", "reshape_run", [(65536,), (512, 128), "int32"]),

            # softmax OP
            # float:[1024, 16, 128, 128]=float:[1024, 16, 128, 128]
            ("softmax_001", "softmax_run", ((1024, 16, 128, 128), "float32", -1, "cce_softmax_fp32")),
            # float:[1, 16, 128, 128]=float:[1, 16, 128, 128]
            ("softmax_002", "softmax_run", ((1, 16, 128, 128), "float32", -1, "cce_softmax_fp32")),
            # float:[128, 16, 128, 128]=float:[128, 16, 128, 128]
            ("softmax_003", "softmax_run", ((128, 16, 128, 128), "float32", -1, "cce_softmax_fp32")),
            # float:[16, 16, 128, 128]=float:[16, 16, 128, 128]
            ("softmax_004", "softmax_run", ((16, 16, 128, 128), "float32", -1, "cce_softmax_fp32")),
            # float:[2, 16, 128, 128]=float:[2, 16, 128, 128]
            ("softmax_005", "softmax_run", ((2, 16, 128, 128), "float32", -1, "cce_softmax_fp32")),
            # float:[256, 16, 128, 128]=float:[256, 16, 128, 128]
            ("softmax_006", "softmax_run", ((256, 16, 128, 128), "float32", -1, "cce_softmax_fp32")),
            # float:[32, 16, 128, 128]=float:[32, 16, 128, 128]
            ("softmax_007", "softmax_run", ((32, 16, 128, 128), "float32", -1, "cce_softmax_fp32")),
            # float:[4, 16, 128, 128]=float:[4, 16, 128, 128]
            ("softmax_008", "softmax_run", ((4, 16, 128, 128), "float32", -1, "cce_softmax_fp32")),
            # float:[512, 16, 128, 128]=float:[512, 16, 128, 128]
            ("softmax_009", "softmax_run", ((512, 16, 128, 128), "float32", -1, "cce_softmax_fp32")),
            # float:[8, 16, 128, 128]=float:[8, 16, 128, 128]
            ("softmax_010", "softmax_run", ((8, 16, 128, 128), "float32", -1, "cce_softmax_fp32")),

            # pow OP
            # float - float:[1280, 768] - [] = float:[1280, 768]
            ("pow_001", "pow_run", ((1280, 768), (1,), 'float32')),
            # float - float:[] - [] = float:[]
            ("pow_002", "pow_run", ((1,), (1,), 'float32')),
            # half - half:[8192, 3072] - [] = half:[8192, 3072]
            ("pow_003", "pow_run", ((8192, 3072), (8192, 3072), 'float16')),

            # reciprocal OP
            # float - float:[160, 1024] = float:[160, 1024]
            ("reciprocal_001", "reciprocal_run", ((160, 1024), 'float32'),),
            # float - float:[] = float:[]
            ("reciprocal_002", "reciprocal_run", ((1,), 'float32'),),

            # addn OP
            # float-float:[64, 128, 1024]-[64, 128, 1024]=float:[64, 128, 1024]
            ("test_bert_addn_003_053", "addn_run", ((64, 128, 1024), "float32", 2)),
            # float-float:[64, 16, 128, 128]-[64, 16, 128, 128]=float:[64, 16, 128, 128]
            ("test_bert_addn_003_054", "addn_run", ((64, 16, 128, 128), "float32", 2)),
            # float-float:[64, 16, 128, 64]-[64, 16, 128, 64]=float:[64, 16, 128, 64]
            ("test_bert_addn_003_055", "addn_run", ((64, 16, 128, 64), "float32", 2)),
            # float-float:[8192, 1024]-[8192, 1024]=float:[8192, 1024]
            ("test_bert_addn_003_056", "addn_run", ((8192, 1024), "float32", 2)),
            # float-float-float:[8192, 1024]-[8192, 1024]-[8192, 1024]=float:[8192, 1024]
            ("test_bert_addn_003_057", "addn_run", ((8192, 1024), "float32", 3)),

            # dropout OP
            # float:[64, 128, 1024]=float:[64, 128, 1024]
            ("test_bert_dropout_003_031", "dropout_run", ((64, 128, 1024), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[64, 16, 128, 128]=float:[64, 16, 128, 128]
            ("test_bert_dropout_003_032", "dropout_run", ((64, 16, 128, 128), 1.0, "float32", "cce_dropout_do_mask")),
            # float:[8192, 1024]=float:[8192, 1024]
            ("test_bert_dropout_003_033", "dropout_run", ((8192, 1024), 1.0, "float32", "cce_dropout_do_mask")),

            # gelu_grad OP
            # float:[1280, 1024]=float:[1280, 1024]
            ("test_bert_gelu_grad_003_021", "gelu_grad_run", ((1280, 1024), "float32")),
            # float:[8192, 4096]=float:[8192, 4096]
            ("test_bert_gelu_grad_003_022", "gelu_grad_run", ((8192, 4096), "float32")),

            # gelu OP
            # float:[1280, 1024]=float:[1280, 1024]
            ("test_bert_gelu_003_021", "gelu_run", ((1280, 1024), "float32")),
            # float:[8192, 4096]=float:[8192, 4096]
            ("test_bert_gelu_003_022", "gelu_run", ((8192, 4096), "float32")),

            # UnsortedSegmentSum OP
            # float-int32-int32:[1280, 1024]-[1280]-[]=float:[8192, 1024]
            ("test_bert_unsortedsegmentsum_003_032", "unsortedsegmentsum_run", ([1280, 1024], [1280], 8192, "float32")),
            # float-int32-int32:[8192, 1024]-[8192]-[]=float:[2, 1024]
            ("test_bert_unsortedsegmentsum_003_033", "unsortedsegmentsum_run", ([8192, 1024], [8192], 2, "float32")),
            # float-int32-int32:[8192, 1024]-[8192]-[]=float:[21128, 1024]
            ("test_bert_unsortedsegmentsum_003_0034", "unsortedsegmentsum_run",
             ([8192, 1024], [8192], 21128, "float32")),

            # MatMul OP
            # float-float:[1280, 1024]-[1024, 1024]=float:[1280, 1024]
            ("test_bert_batch_matmul_003_146", "batchmatmul_run",
             ((), 1280, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[1280, 1024]-[21128, 1024]=float:[1280, 21128]
            ("test_bert_batch_matmul_003_147", "batchmatmul_run",
             ((), 1280, 21128, 1024, (), "float32", False, True, "batch_matmul_output")),
            # float-float:[1280, 21128]-[1280, 1024]=float:[21128, 1024]
            ("test_bert_batch_matmul_003_148", "batchmatmul_run",
             ((), 21128, 1024, 1280, (), "float32", True, False, "batch_matmul_output")),
            # float-float:[1280, 21128]-[21128, 1024]=float:[1280, 1024]
            ("test_bert_batch_matmul_003_149", "batchmatmul_run",
             ((), 1280, 1024, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[128, 1024, 128]-[128, 1024, 64]=float:[128, 128, 64]
            ("test_bert_batch_matmul_003_150", "batchmatmul_run",
             ((128,), 128, 64, 1024, (), "float32", True, False, "batch_matmul_output")),
            # float-float:[128, 1024, 128]-[128, 128, 64]=float:[128, 1024, 64]
            ("test_bert_batch_matmul_003_151", "batchmatmul_run",
             ((128,), 1024, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[128, 1024, 64]-[128, 128, 64]=float:[128, 1024, 128]
            ("test_bert_batch_matmul_003_152", "batchmatmul_run",
             ((128,), 1024, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float-float:[64, 1024]-[1024, 1024]=float:[64, 1024]
            ("test_bert_batch_matmul_003_153", "batchmatmul_run",
             ((), 64, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[64, 16, 128, 128]-[64, 16, 128, 64]=float:[64, 16, 128, 64]
            ("test_bert_batch_matmul_003_154", "batchmatmul_run",
             ((64, 16), 128, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[64, 16, 128, 64]-[64, 16, 128, 64]=float:[64, 16, 128, 128]
            ("test_bert_batch_matmul_003_155", "batchmatmul_run",
             ((64, 16), 128, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float-float:[64, 2]-[2, 1024]=float:[64, 1024]
            ("test_bert_batch_matmul_003_156", "batchmatmul_run",
             ((), 64, 1024, 2, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[8192, 1024]-[1024, 1024]=float:[8192, 1024]
            ("test_bert_batch_matmul_003_157", "batchmatmul_run",
             ((), 8192, 1024, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[8192, 1024]-[1024, 4096]=float:[8192, 4096]
            ("test_bert_batch_matmul_003_158", "batchmatmul_run",
             ((), 8192, 4096, 1024, (), "float32", False, False, "batch_matmul_output")),
            # float-float:[8192, 4096]-[4096, 1024]=float:[8192, 1024]
            ("test_bert_batch_matmul_003_159", "batchmatmul_run",
             ((), 8192, 1024, 4096, (), "float32", False, False, "batch_matmul_output")),

            # LayerNormGrad OP
            # float:[1280, 1024]=float:[1280, 1024]
            ("test_bert_layer_norm_grad_003_031", "fused_layer_norm_grad_run", ((1280, 1024), -1, -1, "float32")),
            # float:[64, 128, 1024]=float:[64, 128, 1024]
            ("test_bert_layer_norm_grad_003_032", "fused_layer_norm_grad_run", ((64, 128, 1024), -1, -1, "float32")),
            # float:[8192, 1024]=float:[8192, 1024]
            ("test_bert_layer_norm_grad_003_033", "fused_layer_norm_grad_run", ((8192, 1024), -1, -1, "float32")),

            # LayerNorm OP
            # float:[1280, 1024]=float:[1280, 1024]
            ("test_bert_layernorm_003_031", "fused_layernorm_run", ((1280, 1024), -1, -1, "float32")),
            # float:[64, 128, 1024]=float:[64, 128, 1024]
            ("test_bert_layernorm_003_032", "fused_layernorm_run", ((64, 128, 1024), -1, -1, "float32")),
            # float:[8192, 1024]=float:[8192, 1024]
            ("test_bert_layernorm_003_033", "fused_layernorm_run", ((8192, 1024), -1, -1, "float32")),

            # Reshape OP
            # float-int32:[1024, 128, 12, 64]-[4]=float:[131072, 768]
            ("test_bert_reshape_003_0288", "reshape_run", [(1024, 128, 12, 64), (131072, 768), "float32"]),
            # float-int32:[1024, 128, 768]-[3]=float:[131072, 768]
            ("test_bert_reshape_003_0289", "reshape_run", [(1024, 128, 768), (131072, 768), "float32"]),
            # float-int32:[1024, 1, 768]-[3]=float:[1024, 768]
            ("test_bert_reshape_003_0290", "reshape_run", [(1024, 1, 768), (1024, 768), "float32"]),
            # float-int32:[1024, 768]-[2]=float:[8, 128, 12, 64]
            ("test_bert_reshape_003_0291", "reshape_run", [(1024, 768), (8, 128, 12, 64), "float32"]),
            # float-int32:[1024, 768]-[2]=float:[8, 128, 768]
            ("test_bert_reshape_003_0292", "reshape_run", [(1024, 768), (8, 128, 768), "float32"]),
            # float-int32:[1, 128, 12, 64]-[4]=float:[128, 768]
            ("test_bert_reshape_003_0293", "reshape_run", [(1, 128, 12, 64), (128, 768), "float32"]),
            # float-int32:[1, 128, 768]-[3]=float:[128, 768]
            ("test_bert_reshape_003_0294", "reshape_run", [(1, 128, 768), (128, 768), "float32"]),
            # float-int32:[1, 1, 768]-[3]=float:[1, 768]
            ("test_bert_reshape_003_0295", "reshape_run", [(1, 1, 768), (1, 768), "float32"]),
            # float-int32:[1280, 1]-[2]=float:[1280]
            ("test_bert_reshape_003_0296", "reshape_run", [(1280, 1), (1280,), "float32"]),
            # float-int32:[128, 1024, 12, 128]-[4]=float:[128, 12288, 128]
            ("test_bert_reshape_003_0297", "reshape_run", [(128, 1024, 12, 128), (128, 12288, 128), "float32"]),
            # float-int32:[128, 1024, 12, 64]-[4]=float:[128, 12288, 64]
            ("test_bert_reshape_003_0298", "reshape_run", [(128, 1024, 12, 64), (128, 12288, 64), "float32"]),
            # float-int32:[128, 1024, 128]-[3]=float:[128, 64, 16, 128]
            ("test_bert_reshape_003_0299", "reshape_run", [(128, 1024, 128), (128, 64, 16, 128), "float32"]),
            # float-int32:[128, 1024, 128]-[4]=float:[128, 64, 16, 128]
            ("test_bert_reshape_003_0300", "reshape_run", [(128, 1024, 128), (128, 64, 16, 128), "float32"]),
            # float-int32:[128, 1024, 64]-[3]=float:[128, 64, 16, 64]
            ("test_bert_reshape_003_0301", "reshape_run", [(128, 1024, 64), (128, 64, 16, 64), "float32"]),
            # float-int32:[128, 1024, 64]-[4]=float:[128, 64, 16, 64]
            ("test_bert_reshape_003_0302", "reshape_run", [(128, 1024, 64), (128, 64, 16, 64), "float32"]),
            # float-int32:[128, 1, 12, 128]-[4]=float:[128, 12, 128]
            ("test_bert_reshape_003_0303", "reshape_run", [(128, 1, 12, 128), (128, 12, 128), "float32"]),
            # float-int32:[128, 1, 12, 64]-[4]=float:[128, 12, 64]
            ("test_bert_reshape_003_0304", "reshape_run", [(128, 1, 12, 64), (128, 12, 64), "float32"]),
            # float-int32:[128, 12, 128]-[3]=float:[128, 1, 12, 128]
            ("test_bert_reshape_003_0305", "reshape_run", [(128, 12, 128), (128, 1, 12, 128), "float32"]),
            # float-int32:[128, 12288, 128]-[3]=float:[128, 1024, 12, 128]
            ("test_bert_reshape_003_0306", "reshape_run", [(128, 12288, 128), (128, 1024, 12, 128), "float32"]),
            # float-int32:[128, 12288, 64]-[3]=float:[128, 1024, 12, 64]
            ("test_bert_reshape_003_0307", "reshape_run", [(128, 12288, 64), (128, 1024, 12, 64), "float32"]),
            # float-int32:[128, 12, 64]-[3]=float:[128, 1, 12, 64]
            ("test_bert_reshape_003_0308", "reshape_run", [(128, 12, 64), (128, 1, 12, 64), "float32"]),
            # float-int32:[128, 128, 12, 128]-[4]=float:[128, 1536, 128]
            ("test_bert_reshape_003_0309", "reshape_run", [(128, 128, 12, 128), (128, 1536, 128), "float32"]),
            # float-int32:[128, 128, 12, 64]-[4]=float:[128, 1536, 64]
            ("test_bert_reshape_003_0310", "reshape_run", [(128, 128, 12, 64), (128, 1536, 64), "float32"]),
            # float-int32:[128, 128, 12, 64]-[4]=float:[16384, 768]
            ("test_bert_reshape_003_0311", "reshape_run", [(128, 128, 12, 64), (16384, 768), "float32"]),
            # float-int32:[128, 128, 768]-[3]=float:[16384, 768]
            ("test_bert_reshape_003_0312", "reshape_run", [(128, 128, 768), (16384, 768), "float32"]),
            # float-int32:[128, 1536, 128]-[3]=float:[128, 128, 12, 128]
            ("test_bert_reshape_003_0313", "reshape_run", [(128, 1536, 128), (128, 128, 12, 128), "float32"]),
            # float-int32:[128, 1536, 64]-[3]=float:[128, 128, 12, 64]
            ("test_bert_reshape_003_0314", "reshape_run", [(128, 1536, 64), (128, 128, 12, 64), "float32"]),
            # float-int32:[128, 16, 12, 128]-[4]=float:[128, 192, 128]
            ("test_bert_reshape_003_0315", "reshape_run", [(128, 16, 12, 128), (128, 192, 128), "float32"]),
            # float-int32:[128, 16, 12, 64]-[4]=float:[128, 192, 64]
            ("test_bert_reshape_003_0316", "reshape_run", [(128, 16, 12, 64), (128, 192, 64), "float32"]),
            # float-int32:[128, 1, 768]-[3]=float:[128, 768]
            ("test_bert_reshape_003_0317", "reshape_run", [(128, 1, 768), (128, 768), "float32"]),
            # float-int32:[128, 192, 128]-[3]=float:[128, 16, 12, 128]
            ("test_bert_reshape_003_0318", "reshape_run", [(128, 192, 128), (128, 16, 12, 128), "float32"]),
            # float-int32:[128, 192, 64]-[3]=float:[128, 16, 12, 64]
            ("test_bert_reshape_003_0319", "reshape_run", [(128, 192, 64), (128, 16, 12, 64), "float32"]),
            # float-int32:[128, 2, 12, 128]-[4]=float:[128, 24, 128]
            ("test_bert_reshape_003_0320", "reshape_run", [(128, 2, 12, 128), (128, 24, 128), "float32"]),
            # float-int32:[128, 2, 12, 64]-[4]=float:[128, 24, 64]
            ("test_bert_reshape_003_0321", "reshape_run", [(128, 2, 12, 64), (128, 24, 64), "float32"]),
            # float-int32:[128, 24, 128]-[3]=float:[128, 2, 12, 128]
            ("test_bert_reshape_003_0322", "reshape_run", [(128, 24, 128), (128, 2, 12, 128), "float32"]),
            # float-int32:[128, 24, 64]-[3]=float:[128, 2, 12, 64]
            ("test_bert_reshape_003_0323", "reshape_run", [(128, 24, 64), (128, 2, 12, 64), "float32"]),
            # float-int32:[128, 256, 12, 128]-[4]=float:[128, 3072, 128]
            ("test_bert_reshape_003_0324", "reshape_run", [(128, 256, 12, 128), (128, 3072, 128), "float32"]),
            # float-int32:[128, 256, 12, 64]-[4]=float:[128, 3072, 64]
            ("test_bert_reshape_003_0325", "reshape_run", [(128, 256, 12, 64), (128, 3072, 64), "float32"]),
            # float-int32:[128, 3072, 128]-[3]=float:[128, 256, 12, 128]
            ("test_bert_reshape_003_0326", "reshape_run", [(128, 3072, 128), (128, 256, 12, 128), "float32"]),
            # float-int32:[128, 3072, 64]-[3]=float:[128, 256, 12, 64]
            ("test_bert_reshape_003_0327", "reshape_run", [(128, 3072, 64), (128, 256, 12, 64), "float32"]),
            # float-int32:[128, 32, 12, 128]-[4]=float:[128, 384, 128]
            ("test_bert_reshape_003_0328", "reshape_run", [(128, 32, 12, 128), (128, 384, 128), "float32"]),
            # float-int32:[128, 32, 12, 64]-[4]=float:[128, 384, 64]
            ("test_bert_reshape_003_0329", "reshape_run", [(128, 32, 12, 64), (128, 384, 64), "float32"]),
            # float-int32:[128, 384, 128]-[3]=float:[128, 32, 12, 128]
            ("test_bert_reshape_003_0330", "reshape_run", [(128, 384, 128), (128, 32, 12, 128), "float32"]),
            # float-int32:[128, 384, 64]-[3]=float:[128, 32, 12, 64]
            ("test_bert_reshape_003_0331", "reshape_run", [(128, 384, 64), (128, 32, 12, 64), "float32"]),
            # float-int32:[128, 4, 12, 128]-[4]=float:[128, 48, 128]
            ("test_bert_reshape_003_0332", "reshape_run", [(128, 4, 12, 128), (128, 48, 128), "float32"]),
            # float-int32:[128, 4, 12, 64]-[4]=float:[128, 48, 64]
            ("test_bert_reshape_003_0333", "reshape_run", [(128, 4, 12, 64), (128, 48, 64), "float32"]),
            # float-int32:[128, 48, 128]-[3]=float:[128, 4, 12, 128]
            ("test_bert_reshape_003_0334", "reshape_run", [(128, 48, 128), (128, 4, 12, 128), "float32"]),
            # float-int32:[128, 48, 64]-[3]=float:[128, 4, 12, 64]
            ("test_bert_reshape_003_0335", "reshape_run", [(128, 48, 64), (128, 4, 12, 64), "float32"]),
            # float-int32:[128, 512, 12, 128]-[4]=float:[128, 6144, 128]
            ("test_bert_reshape_003_0336", "reshape_run", [(128, 512, 12, 128), (128, 6144, 128), "float32"]),
            # float-int32:[128, 512, 12, 64]-[4]=float:[128, 6144, 64]
            ("test_bert_reshape_003_0337", "reshape_run", [(128, 512, 12, 64), (128, 6144, 64), "float32"]),
            # float-int32:[128, 6144, 128]-[3]=float:[128, 512, 12, 128]
            ("test_bert_reshape_003_0338", "reshape_run", [(128, 6144, 128), (128, 512, 12, 128), "float32"]),
            # float-int32:[128, 6144, 64]-[3]=float:[128, 512, 12, 64]
            ("test_bert_reshape_003_0339", "reshape_run", [(128, 6144, 64), (128, 512, 12, 64), "float32"]),
            # float-int32:[128, 64, 12, 128]-[4]=float:[128, 768, 128]
            ("test_bert_reshape_003_0340", "reshape_run", [(128, 64, 12, 128), (128, 768, 128), "float32"]),
            # float-int32:[128, 64, 12, 64]-[4]=float:[128, 768, 64]
            ("test_bert_reshape_003_0341", "reshape_run", [(128, 64, 12, 64), (128, 768, 64), "float32"]),
            # float-int32:[128, 64, 16, 128]-[3]=float:[128, 1024, 128]
            ("test_bert_reshape_003_0342", "reshape_run", [(128, 64, 16, 128), (128, 1024, 128), "float32"]),
            # float-int32:[128, 64, 16, 128]-[4]=float:[128, 1024, 128]
            ("test_bert_reshape_003_0343", "reshape_run", [(128, 64, 16, 128), (128, 1024, 128), "float32"]),
            # float-int32:[128, 64, 16, 64]-[3]=float:[128, 1024, 64]
            ("test_bert_reshape_003_0344", "reshape_run", [(128, 64, 16, 64), (128, 1024, 64), "float32"]),
            # float-int32:[128, 64, 16, 64]-[4]=float:[128, 1024, 64]
            ("test_bert_reshape_003_0345", "reshape_run", [(128, 64, 16, 64), (128, 1024, 64), "float32"]),
            # float-int32:[128, 768, 128]-[3]=float:[128, 64, 12, 128]
            ("test_bert_reshape_003_0346", "reshape_run", [(128, 768, 128), (128, 64, 12, 128), "float32"]),
            # float-int32:[128, 768]-[2]=float:[1, 128, 12, 64]
            ("test_bert_reshape_003_0347", "reshape_run", [(128, 768), (1, 128, 12, 64), "float32"]),
            # float-int32:[128, 768]-[2]=float:[1, 128, 768]
            ("test_bert_reshape_003_0348", "reshape_run", [(128, 768), (1, 128, 768), "float32"]),
            # float-int32:[128, 768, 64]-[3]=float:[128, 64, 12, 64]
            ("test_bert_reshape_003_0349", "reshape_run", [(128, 768, 64), (128, 64, 12, 64), "float32"]),
            # float-int32:[128, 8, 12, 128]-[4]=float:[128, 96, 128]
            ("test_bert_reshape_003_0350", "reshape_run", [(128, 8, 12, 128), (128, 96, 128), "float32"]),
            # float-int32:[128, 8, 12, 64]-[4]=float:[128, 96, 64]
            ("test_bert_reshape_003_0351", "reshape_run", [(128, 8, 12, 64), (128, 96, 64), "float32"]),
            # float-int32:[128, 96, 128]-[3]=float:[128, 8, 12, 128]
            ("test_bert_reshape_003_0352", "reshape_run", [(128, 96, 128), (128, 8, 12, 128), "float32"]),
            # float-int32:[128, 96, 64]-[3]=float:[128, 8, 12, 64]
            ("test_bert_reshape_003_0353", "reshape_run", [(128, 96, 64), (128, 8, 12, 64), "float32"]),
            # float-int32:[131072, 768]-[2]=float:[1024, 128, 12, 64]
            ("test_bert_reshape_003_0354", "reshape_run", [(131072, 768), (1024, 128, 12, 64), "float32"]),
            # float-int32:[131072, 768]-[2]=float:[1024, 128, 768]
            ("test_bert_reshape_003_0355", "reshape_run", [(131072, 768), (1024, 128, 768), "float32"]),
            # float-int32:[16, 128, 12, 64]-[4]=float:[2048, 768]
            ("test_bert_reshape_003_0356", "reshape_run", [(16, 128, 12, 64), (2048, 768), "float32"]),
            # float-int32:[16, 128, 768]-[3]=float:[2048, 768]
            ("test_bert_reshape_003_0357", "reshape_run", [(16, 128, 768), (2048, 768), "float32"]),
            # float-int32:[16, 1, 768]-[3]=float:[16, 768]
            ("test_bert_reshape_003_0358", "reshape_run", [(16, 1, 768), (16, 768), "float32"]),
            # float-int32:[16384, 768]-[2]=float:[128, 128, 12, 64]
            ("test_bert_reshape_003_0359", "reshape_run", [(16384, 768), (128, 128, 12, 64), "float32"]),
            # float-int32:[16384, 768]-[2]=float:[128, 128, 768]
            ("test_bert_reshape_003_0360", "reshape_run", [(16384, 768), (128, 128, 768), "float32"]),
            # float-int32:[2048, 768]-[2]=float:[16, 128, 12, 64]
            ("test_bert_reshape_003_0361", "reshape_run", [(2048, 768), (16, 128, 12, 64), "float32"]),
            # float-int32:[2048, 768]-[2]=float:[16, 128, 768]
            ("test_bert_reshape_003_0362", "reshape_run", [(2048, 768), (16, 128, 768), "float32"]),
            # float-int32:[2, 128, 12, 64]-[4]=float:[256, 768]
            ("test_bert_reshape_003_0363", "reshape_run", [(2, 128, 12, 64), (256, 768), "float32"]),
            # float-int32:[2, 128, 768]-[3]=float:[256, 768]
            ("test_bert_reshape_003_0364", "reshape_run", [(2, 128, 768), (256, 768), "float32"]),
            # float-int32:[2, 1, 768]-[3]=float:[2, 768]
            ("test_bert_reshape_003_0365", "reshape_run", [(2, 1, 768), (2, 768), "float32"]),
            # float-int32:[256, 128, 12, 64]-[4]=float:[32768, 768]
            ("test_bert_reshape_003_0366", "reshape_run", [(256, 128, 12, 64), (32768, 768), "float32"]),
            # float-int32:[256, 128, 768]-[3]=float:[32768, 768]
            ("test_bert_reshape_003_0367", "reshape_run", [(256, 128, 768), (32768, 768), "float32"]),
            # float-int32:[256, 1, 768]-[3]=float:[256, 768]
            ("test_bert_reshape_003_0368", "reshape_run", [(256, 1, 768), (256, 768), "float32"]),
            # float-int32:[256, 768]-[2]=float:[2, 128, 12, 64]
            ("test_bert_reshape_003_0369", "reshape_run", [(256, 768), (2, 128, 12, 64), "float32"]),
            # float-int32:[256, 768]-[2]=float:[2, 128, 768]
            ("test_bert_reshape_003_0370", "reshape_run", [(256, 768), (2, 128, 768), "float32"]),
            # float-int32:[32, 128, 12, 64]-[4]=float:[4096, 768]
            ("test_bert_reshape_003_0371", "reshape_run", [(32, 128, 12, 64), (4096, 768), "float32"]),
            # float-int32:[32, 128, 768]-[3]=float:[4096, 768]
            ("test_bert_reshape_003_0372", "reshape_run", [(32, 128, 768), (4096, 768), "float32"]),
            # float-int32:[32, 1, 768]-[3]=float:[32, 768]
            ("test_bert_reshape_003_0373", "reshape_run", [(32, 1, 768), (32, 768), "float32"]),
            # float-int32:[32768, 768]-[2]=float:[256, 128, 12, 64]
            ("test_bert_reshape_003_0374", "reshape_run", [(32768, 768), (256, 128, 12, 64), "float32"]),
            # float-int32:[32768, 768]-[2]=float:[256, 128, 768]
            ("test_bert_reshape_003_0375", "reshape_run", [(32768, 768), (256, 128, 768), "float32"]),
            # float-int32:[4096, 768]-[2]=float:[32, 128, 12, 64]
            ("test_bert_reshape_003_0376", "reshape_run", [(4096, 768), (32, 128, 12, 64), "float32"]),
            # float-int32:[4096, 768]-[2]=float:[32, 128, 768]
            ("test_bert_reshape_003_0377", "reshape_run", [(4096, 768), (32, 128, 768), "float32"]),
            # float-int32:[4, 128, 12, 64]-[4]=float:[512, 768]
            ("test_bert_reshape_003_0378", "reshape_run", [(4, 128, 12, 64), (512, 768), "float32"]),
            # float-int32:[4, 128, 768]-[3]=float:[512, 768]
            ("test_bert_reshape_003_0379", "reshape_run", [(4, 128, 768), (512, 768), "float32"]),
            # float-int32:[4, 1, 768]-[3]=float:[4, 768]
            ("test_bert_reshape_003_0380", "reshape_run", [(4, 1, 768), (4, 768), "float32"]),
            # float-int32:[512, 128, 12, 64]-[4]=float:[65536, 768]
            ("test_bert_reshape_003_0381", "reshape_run", [(512, 128, 12, 64), (65536, 768), "float32"]),
            # float-int32:[512, 128, 768]-[3]=float:[65536, 768]
            ("test_bert_reshape_003_0382", "reshape_run", [(512, 128, 768), (65536, 768), "float32"]),
            # float-int32:[512, 1, 768]-[3]=float:[512, 768]
            ("test_bert_reshape_003_0383", "reshape_run", [(512, 1, 768), (512, 768), "float32"]),
            # float-int32:[512, 768]-[2]=float:[4, 128, 12, 64]
            ("test_bert_reshape_003_0384", "reshape_run", [(512, 768), (4, 128, 12, 64), "float32"]),
            # float-int32:[512, 768]-[2]=float:[4, 128, 768]
            ("test_bert_reshape_003_0385", "reshape_run", [(512, 768), (4, 128, 768), "float32"]),
            # float-int32:[64, 1024]-[3]=float:[64, 1, 1024]
            ("test_bert_reshape_003_0386", "reshape_run", [(64, 1024), (64, 1, 1024), "float32"]),
            # float-int32:[64, 1, 1024]-[3]=float:[64, 1024]
            ("test_bert_reshape_003_0387", "reshape_run", [(64, 1, 1024), (64, 1024), "float32"]),
            # float-int32:[64, 128, 1024]-[2]=float:[8192, 1024]
            ("test_bert_reshape_003_0388", "reshape_run", [(64, 128, 1024), (8192, 1024), "float32"]),
            # float-int32:[64, 128, 1024]-[3]=float:[8192, 1024]
            ("test_bert_reshape_003_0389", "reshape_run", [(64, 128, 1024), (8192, 1024), "float32"]),
            # float-int32:[64, 128, 12, 64]-[4]=float:[8192, 768]
            ("test_bert_reshape_003_0390", "reshape_run", [(64, 128, 12, 64), (8192, 768), "float32"]),
            # float-int32:[64, 128, 16, 64]-[2]=float:[8192, 1024]
            ("test_bert_reshape_003_0391", "reshape_run", [(64, 128, 16, 64), (8192, 1024), "float32"]),
            # float-int32:[64, 128, 16, 64]-[4]=float:[8192, 1024]
            ("test_bert_reshape_003_0392", "reshape_run", [(64, 128, 16, 64), (8192, 1024), "float32"]),
            # float-int32:[64, 128, 768]-[3]=float:[8192, 768]
            ("test_bert_reshape_003_0393", "reshape_run", [(64, 128, 768), (8192, 768), "float32"]),
            # float-int32:[64, 1, 768]-[3]=float:[64, 768]
            ("test_bert_reshape_003_0394", "reshape_run", [(64, 1, 768), (64, 768), "float32"]),
            # float-int32:[65536, 768]-[2]=float:[512, 128, 12, 64]
            ("test_bert_reshape_003_0395", "reshape_run", [(65536, 768), (512, 128, 12, 64), "float32"]),
            # float-int32:[65536, 768]-[2]=float:[512, 128, 768]
            ("test_bert_reshape_003_0396", "reshape_run", [(65536, 768), (512, 128, 768), "float32"]),
            # float-int32:[8, 128, 12, 64]-[4]=float:[1024, 768]
            ("test_bert_reshape_003_0397", "reshape_run", [(8, 128, 12, 64), (1024, 768), "float32"]),
            # float-int32:[8, 128, 768]-[3]=float:[1024, 768]
            ("test_bert_reshape_003_0398", "reshape_run", [(8, 128, 768), (1024, 768), "float32"]),
            # float-int32:[8, 1, 768]-[3]=float:[8, 768]
            ("test_bert_reshape_003_0399", "reshape_run", [(8, 1, 768), (8, 768), "float32"]),
            # float-int32:[8192, 1024]-[2]=float:[64, 128, 1024]
            ("test_bert_reshape_003_0400", "reshape_run", [(8192, 1024), (64, 128, 1024), "float32"]),
            # float-int32:[8192, 1024]-[2]=float:[64, 128, 16, 64]
            ("test_bert_reshape_003_0401", "reshape_run", [(8192, 1024), (64, 128, 16, 64), "float32"]),
            # float-int32:[8192, 1024]-[2]=float:[8192, 1024]
            ("test_bert_reshape_003_0402", "reshape_run", [(8192, 1024), (8192, 1024), "float32"]),
            # float-int32:[8192, 1024]-[3]=float:[64, 128, 1024]
            ("test_bert_reshape_003_0403", "reshape_run", [(8192, 1024), (64, 128, 1024), "float32"]),
            # float-int32:[8192, 1024]-[4]=float:[64, 128, 16, 64]
            ("test_bert_reshape_003_0404", "reshape_run", [(8192, 1024), (64, 128, 16, 64), "float32"]),
            # float-int32:[8192, 768]-[2]=float:[64, 128, 12, 64]
            ("test_bert_reshape_003_0405", "reshape_run", [(8192, 768), (64, 128, 12, 64), "float32"]),
            # float-int32:[8192, 768]-[2]=float:[64, 128, 768]
            ("test_bert_reshape_003_0406", "reshape_run", [(8192, 768), (64, 128, 768), "float32"]),

            # Softmax OP
            # float:[64, 16, 128, 128]=float:[64, 16, 128, 128]
            ("test_bert_softmax_003_011", "softmax_run", ((64, 16, 128, 128), "float32", -1, "cce_softmax_fp32")),

            # StridedSlice	OP
            # float-int32-int32-int32:[64, 128, 1024]-[3]-[3]-[3]=float:[64, 1, 1024]
            ("test_bert_strided_slice_003_012", "strided_slice_run",
             ((64, 128, 1024), [0, 0, 0], [64, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),

            # StridedSliceGrad	OP
            # float-int32-int32-int32-int32:[64, 1, 1024]-[3]-[3]-[3]-[3]=float:[64, 128, 1024]
            ("test_bert_strided_slice_grad_003_011", "strided_slice_grad_run",
             [(64, 128, 1024), [0, 0, 0], [64, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, (64, 1, 1024), "int32"]),

            # Sub OP
            # float-float:[64, 16, 128, 128]-[64, 16, 128, 1]=float:[64, 16, 128, 128]
            ("test_bert_sub_003_033", "sub_run", [(64, 16, 128, 128,), (64, 16, 128, 1), "float32"]),

            # Sum OP
            # float-int32:[10240, 21128]-[-1]=float:[10240]
            ("test_bert_sum_003_0041", "sum_run", ((10240, 21128), (-1,), False, "float32")),
            # float-int32:[1024, 2]-[-1]=float:[1024]
            ("test_bert_sum_003_0042", "sum_run", ((1024, 2), (-1,), False, "float32")),
            # float-int32:[1, 2]-[-1]=float:[1]
            ("test_bert_sum_003_0043", "sum_run", ((1, 2), (-1,), False, "float32")),
            # float-int32:[1280, 21128]-[-1]=float:[1280]
            ("test_bert_sum_003_0044", "sum_run", ((1280, 21128), (-1,), False, "float32")),
            # float-int32:[128, 2]-[-1]=float:[128]
            ("test_bert_sum_003_0045", "sum_run", ((128, 2), (-1,), False, "float32")),
            # float-int32:[160, 21128]-[-1]=float:[160]
            ("test_bert_sum_003_0046", "sum_run", ((160, 21128), (-1,), False, "float32")),
            # float-int32:[16, 2]-[-1]=float:[16]
            ("test_bert_sum_003_0047", "sum_run", ((16, 2), (-1,), False, "float32")),
            # float-int32:[20, 21128]-[-1]=float:[20]
            ("test_bert_sum_003_0048", "sum_run", ((20, 21128), (-1,), False, "float32")),
            # float-int32:[20480, 21128]-[-1]=float:[20480]
            ("test_bert_sum_003_0049", "sum_run", ((20480, 21128), (-1,), False, "float32")),
            # float-int32:[2, 2]-[-1]=float:[2]
            ("test_bert_sum_003_0050", "sum_run", ((2, 2), (-1,), False, "float32")),
            # float-int32:[2560, 21128]-[-1]=float:[2560]
            ("test_bert_sum_003_0051", "sum_run", ((2560, 21128), (-1,), False, "float32")),
            # float-int32:[256, 2]-[-1]=float:[256]
            ("test_bert_sum_003_0052", "sum_run", ((256, 2), (-1,), False, "float32")),
            # float-int32:[320, 21128]-[-1]=float:[320]
            ("test_bert_sum_003_0053", "sum_run", ((320, 21128), (-1,), False, "float32")),
            # float-int32:[32, 2]-[-1]=float:[32]
            ("test_bert_sum_003_0054", "sum_run", ((32, 2), (-1,), False, "float32")),
            # float-int32:[40, 21128]-[-1]=float:[40]
            ("test_bert_sum_003_0055", "sum_run", ((40, 21128), (-1,), False, "float32")),
            # float-int32:[4, 2]-[-1]=float:[4]
            ("test_bert_sum_003_0056", "sum_run", ((4, 2), (-1,), False, "float32")),
            # float-int32:[5120, 21128]-[-1]=float:[5120]
            ("test_bert_sum_003_0057", "sum_run", ((5120, 21128), (-1,), False, "float32")),
            # float-int32:[512, 2]-[-1]=float:[512]
            ("test_bert_sum_003_0058", "sum_run", ((512, 2), (-1,), False, "float32")),
            # float-int32:[640, 21128]-[-1]=float:[640]
            ("test_bert_sum_003_0059", "sum_run", ((640, 21128), (-1,), False, "float32")),
            # float-int32:[64, 16, 128, 128]-[-1]=float:[64, 16, 128, 1]
            ("test_bert_sum_003_0060", "sum_run", ((64, 16, 128, 128), (-1,), False, "float32")),
            # float-int32:[64, 2]-[-1]=float:[64]
            ("test_bert_sum_003_0061", "sum_run", ((64, 2), (-1,), False, "float32")),
            # float-int32:[80, 21128]-[-1]=float:[80]
            ("test_bert_sum_003_0062", "sum_run", ((80, 21128), (-1,), False, "float32")),
            # float-int32:[8, 2]-[-1]=float:[8]
            ("test_bert_sum_003_0063", "sum_run", ((8, 2), (-1,), False, "float32")),

            # TanhGrad	OP
            # f#loat:[64, 1024]=float:[64, 1024]
            ("test_bert_tanh_grad_003_001", "tanh_grad_run", ((64, 1024), "float32")),

            # Tanh	OP
            # float:[64, 1024]=float:[64, 1024]
            ("test_bert_tanh_003_001", "tanh_run", ((64, 1024), "float32")),

            # Transpose	OP
            # float-int32:[1280, 1024]-[2]=float:[1280, 1024]				(0,1)
            ("test_bert_transpose_003_0086", "transpose_run", ((1280, 1024), (0, 1), "float32")),
            # float-int32:[128, 64, 16, 128]-[4]=float:[64, 16, 128, 128]	(1,2,0,3)
            ("test_bert_transpose_003_0087", "transpose_run", ((128, 64, 16, 128), (1, 2, 0, 3), "float32")),
            # float-int32:[128, 64, 16, 64]-[4]=float:[64, 16, 128, 64]	(1,2,0,3)
            ("test_bert_transpose_003_0088", "transpose_run", ((128, 64, 16, 64), (1, 2, 0, 3), "float32")),
            # float-int32:[64, 128, 16, 64]-[4]=float:[64, 16, 128, 64]	(0,2,1,3)
            ("test_bert_transpose_003_0089", "transpose_run", ((64, 128, 16, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[64, 16, 128, 128]-[4]=float:[128, 64, 16, 128]	(2,0,1,3)
            ("test_bert_transpose_003_0090", "transpose_run", ((64, 16, 128, 128), (2, 0, 1, 3), "float32")),
            # float-int32:[64, 16, 128, 64]-[4]=float:[128, 64, 16, 64]	(2,0,1,3)
            ("test_bert_transpose_003_0091", "transpose_run", ((64, 16, 128, 64), (2, 0, 1, 3), "float32")),
            # float-int32:[64, 16, 128, 64]-[4]=float:[64, 128, 16, 64]	(0,2,1,3)
            ("test_bert_transpose_003_0092", "transpose_run", ((64, 16, 128, 64), (0, 2, 1, 3), "float32")),
            # float-int32:[8192, 1024]-[2]=float:[8192, 1024]				(0,1)
            ("test_bert_transpose_003_0093", "transpose_run", ((8192, 1024), (0, 1), "float32")),

            # Square OP
            # float32:[1024, 1024] = float32:[1024, 1024]
            ("test_bert_square_003_001", "square_run", ((1024, 1024), "float32", "cce_mod_fp32")),
            # float32:[1024, 4096] = float32:[1024, 4096]
            ("test_bert_square_003_002", "square_run", ((1024, 4096), "float32", "cce_mod_fp32")),
            # float32:[1024] = float32:[1024]
            ("test_bert_square_003_003", "square_run", ((1024,), "float32", "cce_mod_fp32")),
            # float32:[2, 1024] = float32:[2, 1024]
            ("test_bert_square_003_004", "square_run", ((2, 1024), "float32", "cce_mod_fp32")),
            # float32:[21128, 1024] = float32:[21128, 1024]
            ("test_bert_square_003_005", "square_run", ((21128, 1024), "float32", "cce_mod_fp32")),
            # float32:[21128, 768] = float32:[21128, 768]
            ("test_bert_square_003_006", "square_run", ((21128, 768), "float32", "cce_mod_fp32")),
            # float32:[21128] = float32:[21128]
            ("test_bert_square_003_007", "square_run", ((21128,), "float32", "cce_mod_fp32")),
            # float32:[2, 768] = float32:[2, 768]
            ("test_bert_square_003_008", "square_run", ((2, 768), "float32", "cce_mod_fp32")),
            # float32:[2] = float32:[2]
            ("test_bert_square_003_009", "square_run", ((2,), "float32", "cce_mod_fp32")),
            # float32:[3072, 768] = float32:[3072, 768]
            ("test_bert_square_003_010", "square_run", ((3072, 768), "float32", "cce_mod_fp32")),
            # float32:[3072] = float32:[3072]
            ("test_bert_square_003_011", "square_run", ((3072,), "float32", "cce_mod_fp32")),
            # float32:[33, 64] = float32:[33, 64]
            ("test_bert_square_003_012", "square_run", ((33, 64), "float32", "cce_mod_fp32")),
            # float32:[4096, 1024] = float32:[4096, 1024]
            ("test_bert_square_003_013", "square_run", ((4096, 1024), "float32", "cce_mod_fp32")),
            # float32:[4096] = float32:[4096]
            ("test_bert_square_003_014", "square_run", ((4096,), "float32", "cce_mod_fp32")),
            # float32:[768, 3072] = float32:[768, 3072]
            ("test_bert_square_003_015", "square_run", ((768, 3072), "float32", "cce_mod_fp32")),
            # float32:[768, 768] = float32:[768, 768]
            ("test_bert_square_003_016", "square_run", ((768, 768), "float32", "cce_mod_fp32")),
            # float32:[768] = float32:[768]
            ("test_bert_square_003_017", "square_run", ((768,), "float32", "cce_mod_fp32")),

            # FusedMinimumOrMaximumGrad OP
            # float-float:[1024, 128, 1024]-[1024, 128, 1024]
            ("test_bert_fused_min_or_max_grad_003_001", "fused_minimum_or_maximum_grad_run",
             ((1024, 128, 1024), (1024, 128, 1024), (1,), True, True, "GE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[1024, 128, 768]-[1024, 128, 768]
            ("test_bert_fused_min_or_max_grad_003_002", "fused_minimum_or_maximum_grad_run",
             ((1024, 128, 768), (1024, 128, 768), (1,), True, True, "GE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[1, 128, 1024]-[1, 128, 1024]
            ("test_bert_fused_min_or_max_grad_003_003", "fused_minimum_or_maximum_grad_run",
             ((1, 128, 1024), (1, 128, 1024), (1,), True, True, "GE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[1, 128, 768]-[1, 128, 768]
            ("test_bert_fused_min_or_max_grad_003_004", "fused_minimum_or_maximum_grad_run",
             ((1, 128, 768), (1, 128, 768), (1,), True, True, "GE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[128, 128, 1024]-[128, 128, 1024]
            ("test_bert_fused_min_or_max_grad_003_005", "fused_minimum_or_maximum_grad_run",
             ((128, 128, 1024), (128, 128, 1024), (1,), True, True, "GE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[128, 128, 64]-[128, 128, 64]
            ("test_bert_fused_min_or_max_grad_003_006", "fused_minimum_or_maximum_grad_run",
             ((128, 128, 64), (128, 128, 64), (1,), True, True, "GE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[128, 128, 768]-[128, 128, 768]
            ("test_bert_fused_min_or_max_grad_003_007", "fused_minimum_or_maximum_grad_run",
             ((128, 128, 768), (128, 128, 768), (1,), True, True, "GE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[16, 128, 1024]-[16, 128, 1024]
            ("test_bert_fused_min_or_max_grad_003_008", "fused_minimum_or_maximum_grad_run",
             ((16, 128, 1024), (16, 128, 1024), (1,), True, True, "GE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[16, 128, 768]-[16, 128, 768]
            ("test_bert_fused_min_or_max_grad_003_009", "fused_minimum_or_maximum_grad_run",
             ((16, 128, 768), (16, 128, 768), (1,), True, True, "GE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[2, 128, 1024]-[2, 128, 1024]
            ("test_bert_fused_min_or_max_grad_003_010", "fused_minimum_or_maximum_grad_run",
             ((2, 128, 1024), (2, 128, 1024), (1,), True, True, "GE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[2, 128, 768]-[2, 128, 768]
            ("test_bert_fused_min_or_max_grad_003_011", "fused_minimum_or_maximum_grad_run",
             ((2, 128, 768), (2, 128, 768), (1,), True, True, "GE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[256, 128, 1024]-[256, 128, 1024]
            ("test_bert_fused_min_or_max_grad_003_012", "fused_minimum_or_maximum_grad_run",
             ((256, 128, 1024), (256, 128, 1024), (1,), True, True, "GE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[256, 128, 768]-[256, 128, 768]
            ("test_bert_fused_min_or_max_grad_003_013", "fused_minimum_or_maximum_grad_run",
             ((256, 128, 768), (256, 128, 768), (1,), True, True, "GE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[32, 128, 1024]-[32, 128, 1024]
            ("test_bert_fused_min_or_max_grad_003_014", "fused_minimum_or_maximum_grad_run",
             ((32, 128, 1024), (32, 128, 1024), (1,), True, True, "GE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[32, 128, 768]-[32, 128, 768]
            ("test_bert_fused_min_or_max_grad_003_015", "fused_minimum_or_maximum_grad_run",
             ((32, 128, 768), (32, 128, 768), (1,), True, True, "GE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[4, 128, 1024]-[4, 128, 1024]
            ("test_bert_fused_min_or_max_grad_003_016", "fused_minimum_or_maximum_grad_run",
             ((4, 128, 1024), (4, 128, 1024), (1,), True, True, "GE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[4, 128, 768]-[4, 128, 768]
            ("test_bert_fused_min_or_max_grad_003_017", "fused_minimum_or_maximum_grad_run",
             ((4, 128, 768), (4, 128, 768), (1,), True, True, "GE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[512, 128, 1024]-[512, 128, 1024]
            ("test_bert_fused_min_or_max_grad_003_018", "fused_minimum_or_maximum_grad_run",
             ((512, 128, 1024), (512, 128, 1024), (1,), True, True, "GE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[512, 128, 768]-[512, 128, 768]
            ("test_bert_fused_min_or_max_grad_003_019", "fused_minimum_or_maximum_grad_run",
             ((512, 128, 768), (512, 128, 768), (1,), True, True, "GE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[64, 128, 1024]-[64, 128, 1024]
            ("test_bert_fused_min_or_max_grad_003_020", "fused_minimum_or_maximum_grad_run",
             ((64, 128, 1024), (64, 128, 1024), (1,), True, True, "GE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[64, 128, 768]-[64, 128, 768]
            ("test_bert_fused_min_or_max_grad_003_021", "fused_minimum_or_maximum_grad_run",
             ((64, 128, 768), (64, 128, 768), (1,), True, True, "GE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[8, 128, 1024]-[8, 128, 1024]
            ("test_bert_fused_min_or_max_grad_003_022", "fused_minimum_or_maximum_grad_run",
             ((8, 128, 1024), (8, 128, 1024), (1,), True, True, "GE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[8, 128, 768]-[8, 128, 768]
            ("test_bert_fused_min_or_max_grad_003_023", "fused_minimum_or_maximum_grad_run",
             ((8, 128, 768), (8, 128, 768), (1,), True, True, "GE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[1024, 128, 1024]-[1024, 128, 1024]
            ("test_bert_fused_min_or_max_grad_003_024", "fused_minimum_or_maximum_grad_run",
             ((1024, 128, 1024), (1024, 128, 1024), (1,), True, True, "LE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[1024, 128, 768]-[1024, 128, 768]
            ("test_bert_fused_min_or_max_grad_003_025", "fused_minimum_or_maximum_grad_run",
             ((1024, 128, 768), (1024, 128, 768), (1,), True, True, "LE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[1, 128, 1024]-[1, 128, 1024]
            ("test_bert_fused_min_or_max_grad_003_026", "fused_minimum_or_maximum_grad_run",
             ((1, 128, 1024), (1, 128, 1024), (1,), True, True, "LE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[1, 128, 768]-[1, 128, 768]
            ("test_bert_fused_min_or_max_grad_003_027", "fused_minimum_or_maximum_grad_run",
             ((1, 128, 768), (1, 128, 768), (1,), True, True, "LE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[128, 128, 1024]-[128, 128, 1024]
            ("test_bert_fused_min_or_max_grad_003_028", "fused_minimum_or_maximum_grad_run",
             ((128, 128, 1024), (128, 128, 1024), (1,), True, True, "LE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[128, 128, 64]-[128, 128, 64]
            ("test_bert_fused_min_or_max_grad_003_029", "fused_minimum_or_maximum_grad_run",
             ((128, 128, 64), (128, 128, 64), (1,), True, True, "LE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[128, 128, 768]-[128, 128, 768]
            ("test_bert_fused_min_or_max_grad_003_030", "fused_minimum_or_maximum_grad_run",
             ((128, 128, 768), (128, 128, 768), (1,), True, True, "LE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[16, 128, 1024]-[16, 128, 1024]
            ("test_bert_fused_min_or_max_grad_003_031", "fused_minimum_or_maximum_grad_run",
             ((16, 128, 1024), (16, 128, 1024), (1,), True, True, "LE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[16, 128, 768]-[16, 128, 768]
            ("test_bert_fused_min_or_max_grad_003_032", "fused_minimum_or_maximum_grad_run",
             ((16, 128, 768), (16, 128, 768), (1,), True, True, "LE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[2, 128, 1024]-[2, 128, 1024]
            ("test_bert_fused_min_or_max_grad_003_033", "fused_minimum_or_maximum_grad_run",
             ((2, 128, 1024), (2, 128, 1024), (1,), True, True, "LE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[2, 128, 768]-[2, 128, 768]
            ("test_bert_fused_min_or_max_grad_003_034", "fused_minimum_or_maximum_grad_run",
             ((2, 128, 768), (2, 128, 768), (1,), True, True, "LE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[256, 128, 1024]-[256, 128, 1024]
            ("test_bert_fused_min_or_max_grad_003_035", "fused_minimum_or_maximum_grad_run",
             ((256, 128, 1024), (256, 128, 1024), (1,), True, True, "LE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[256, 128, 768]-[256, 128, 768]
            ("test_bert_fused_min_or_max_grad_003_036", "fused_minimum_or_maximum_grad_run",
             ((256, 128, 768), (256, 128, 768), (1,), True, True, "LE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[32, 128, 1024]-[32, 128, 1024]
            ("test_bert_fused_min_or_max_grad_003_037", "fused_minimum_or_maximum_grad_run",
             ((32, 128, 1024), (32, 128, 1024), (1,), True, True, "LE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[32, 128, 768]-[32, 128, 768]
            ("test_bert_fused_min_or_max_grad_003_038", "fused_minimum_or_maximum_grad_run",
             ((32, 128, 768), (32, 128, 768), (1,), True, True, "LE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[4, 128, 1024]-[4, 128, 1024]
            ("test_bert_fused_min_or_max_grad_003_039", "fused_minimum_or_maximum_grad_run",
             ((4, 128, 1024), (4, 128, 1024), (1,), True, True, "LE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[4, 128, 768]-[4, 128, 768]
            ("test_bert_fused_min_or_max_grad_003_040", "fused_minimum_or_maximum_grad_run",
             ((4, 128, 768), (4, 128, 768), (1,), True, True, "LE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[512, 128, 1024]-[512, 128, 1024]
            ("test_bert_fused_min_or_max_grad_003_041", "fused_minimum_or_maximum_grad_run",
             ((512, 128, 1024), (512, 128, 1024), (1,), True, True, "LE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[512, 128, 768]-[512, 128, 768]
            ("test_bert_fused_min_or_max_grad_003_042", "fused_minimum_or_maximum_grad_run",
             ((512, 128, 768), (512, 128, 768), (1,), True, True, "LE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[64, 128, 1024]-[64, 128, 1024]
            ("test_bert_fused_min_or_max_grad_003_043", "fused_minimum_or_maximum_grad_run",
             ((64, 128, 1024), (64, 128, 1024), (1,), True, True, "LE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[64, 128, 768]-[64, 128, 768]
            ("test_bert_fused_min_or_max_grad_003_044", "fused_minimum_or_maximum_grad_run",
             ((64, 128, 768), (64, 128, 768), (1,), True, True, "LE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[8, 128, 1024]-[8, 128, 1024]
            ("test_bert_fused_min_or_max_grad_003_045", "fused_minimum_or_maximum_grad_run",
             ((8, 128, 1024), (8, 128, 1024), (1,), True, True, "LE", "float32", "cce_min_max_grad_fp16")),
            # float-float:[8, 128, 768]-[8, 128, 768]
            ("test_bert_fused_min_or_max_grad_003_046", "fused_minimum_or_maximum_grad_run",
             ((8, 128, 768), (8, 128, 768), (1,), True, True, "LE", "float32", "cce_min_max_grad_fp16")),

            # bias_add_grad OP
            # float:[10240, 1024] = float:[1024]
            ("test_bert_bias_add_grad_001", "bias_add_grad_run", ([10240, 1024, 1, 1], "NCHW", "float32")),
            # float:[1024, 1024] = float:[1024]
            ("test_bert_bias_add_grad_002", "bias_add_grad_run", ([1024, 1024, 1, 1], "NCHW", "float32")),
            # float:[1, 1024] = float:[1024]
            ("test_bert_bias_add_grad_003", "bias_add_grad_run", ([1, 1024, 1, 1], "NCHW", "float32")),
            # float:[1280, 1024] = float:[1024]
            ("test_bert_bias_add_grad_004", "bias_add_grad_run", ([1280, 1024, 1, 1], "NCHW", "float32")),
            # float:[128, 1024] = float:[1024]
            ("test_bert_bias_add_grad_005", "bias_add_grad_run", ([128, 1024, 1, 1], "NCHW", "float32")),
            # float:[131072, 1024] = float:[1024]
            ("test_bert_bias_add_grad_006", "bias_add_grad_run", ([131072, 1024, 1, 1], "NCHW", "float32")),
            # float:[160, 1024] = float:[1024]
            ("test_bert_bias_add_grad_007", "bias_add_grad_run", ([160, 1024, 1, 1], "NCHW", "float32")),
            # float:[16, 1024] = float:[1024]
            ("test_bert_bias_add_grad_008", "bias_add_grad_run", ([16, 1024, 1, 1], "NCHW", "float32")),
            # float:[16384, 1024] = float:[1024]
            ("test_bert_bias_add_grad_009", "bias_add_grad_run", ([16384, 1024, 1, 1], "NCHW", "float32")),
            # float:[20, 1024] = float:[1024]
            ("test_bert_bias_add_grad_010", "bias_add_grad_run", ([20, 1024, 1, 1], "NCHW", "float32")),
            # float:[20480, 1024] = float:[1024]
            ("test_bert_bias_add_grad_011", "bias_add_grad_run", ([20480, 1024, 1, 1], "NCHW", "float32")),
            # float:[2048, 1024] = float:[1024]
            ("test_bert_bias_add_grad_012", "bias_add_grad_run", ([2048, 1024, 1, 1], "NCHW", "float32")),
            # float:[2, 1024] = float:[1024]
            ("test_bert_bias_add_grad_013", "bias_add_grad_run", ([2, 1024, 1, 1], "NCHW", "float32")),
            # float:[2560, 1024] = float:[1024]
            ("test_bert_bias_add_grad_014", "bias_add_grad_run", ([2560, 1024, 1, 1], "NCHW", "float32")),
            # float:[256, 1024] = float:[1024]
            ("test_bert_bias_add_grad_015", "bias_add_grad_run", ([256, 1024, 1, 1], "NCHW", "float32")),
            # float:[320, 1024] = float:[1024]
            ("test_bert_bias_add_grad_016", "bias_add_grad_run", ([320, 1024, 1, 1], "NCHW", "float32")),
            # float:[32, 1024] = float:[1024]
            ("test_bert_bias_add_grad_017", "bias_add_grad_run", ([32, 1024, 1, 1], "NCHW", "float32")),
            # float:[32768, 1024] = float:[1024]
            ("test_bert_bias_add_grad_018", "bias_add_grad_run", ([32768, 1024, 1, 1], "NCHW", "float32")),
            # float:[40, 1024] = float:[1024]
            ("test_bert_bias_add_grad_019", "bias_add_grad_run", ([40, 1024, 1, 1], "NCHW", "float32")),
            # float:[4096, 1024] = float:[1024]
            ("test_bert_bias_add_grad_020", "bias_add_grad_run", ([4096, 1024, 1, 1], "NCHW", "float32")),
            # float:[4, 1024] = float:[1024]
            ("test_bert_bias_add_grad_021", "bias_add_grad_run", ([4, 1024, 1, 1], "NCHW", "float32")),
            # float:[5120, 1024] = float:[1024]
            ("test_bert_bias_add_grad_022", "bias_add_grad_run", ([5120, 1024, 1, 1], "NCHW", "float32")),
            # float:[512, 1024] = float:[1024]
            ("test_bert_bias_add_grad_023", "bias_add_grad_run", ([512, 1024, 1, 1], "NCHW", "float32")),
            # float:[640, 1024] = float:[1024]
            ("test_bert_bias_add_grad_024", "bias_add_grad_run", ([640, 1024, 1, 1], "NCHW", "float32")),
            # float:[64, 1024] = float:[1024]
            ("test_bert_bias_add_grad_025", "bias_add_grad_run", ([64, 1024, 1, 1], "NCHW", "float32")),
            # float:[65536, 1024] = float:[1024]
            ("test_bert_bias_add_grad_026", "bias_add_grad_run", ([65536, 1024, 1, 1], "NCHW", "float32")),
            # float:[80, 1024] = float:[1024]
            ("test_bert_bias_add_grad_027", "bias_add_grad_run", ([80, 1024, 1, 1], "NCHW", "float32")),
            # float:[8, 1024] = float:[1024]
            ("test_bert_bias_add_grad_028", "bias_add_grad_run", ([8, 1024, 1, 1], "NCHW", "float32")),
            # float:[8192, 1024] = float:[1024]
            ("test_bert_bias_add_grad_029", "bias_add_grad_run", ([8192, 1024, 1, 1], "NCHW", "float32")),
            # float:[1024, 2] = float:[2]
            ("test_bert_bias_add_grad_030", "bias_add_grad_run", ([1024, 2, 1, 1], "NCHW", "float32")),
            # float:[1, 2] = float:[2]
            ("test_bert_bias_add_grad_031", "bias_add_grad_run", ([1, 2, 1, 1], "NCHW", "float32")),
            # float:[128, 2] = float:[2]
            ("test_bert_bias_add_grad_032", "bias_add_grad_run", ([128, 2, 1, 1], "NCHW", "float32")),
            # float:[16, 2] = float:[2]
            ("test_bert_bias_add_grad_033", "bias_add_grad_run", ([16, 2, 1, 1], "NCHW", "float32")),
            # float:[2, 2] = float:[2]
            ("test_bert_bias_add_grad_034", "bias_add_grad_run", ([2, 2, 1, 1], "NCHW", "float32")),
            # float:[256, 2] = float:[2]
            ("test_bert_bias_add_grad_035", "bias_add_grad_run", ([256, 2, 1, 1], "NCHW", "float32")),
            # float:[32, 2] = float:[2]
            ("test_bert_bias_add_grad_036", "bias_add_grad_run", ([32, 2, 1, 1], "NCHW", "float32")),
            # float:[4, 2] = float:[2]
            ("test_bert_bias_add_grad_037", "bias_add_grad_run", ([4, 2, 1, 1], "NCHW", "float32")),
            # float:[512, 2] = float:[2]
            ("test_bert_bias_add_grad_038", "bias_add_grad_run", ([512, 2, 1, 1], "NCHW", "float32")),
            # float:[64, 2] = float:[2]
            ("test_bert_bias_add_grad_039", "bias_add_grad_run", ([64, 2, 1, 1], "NCHW", "float32")),
            # float:[8, 2] = float:[2]
            ("test_bert_bias_add_grad_040", "bias_add_grad_run", ([8, 2, 1, 1], "NCHW", "float32")),
            # float:[1024, 3072] = float:[3072]
            ("test_bert_bias_add_grad_041", "bias_add_grad_run", ([1024, 3072, 1, 1], "NCHW", "float32")),
            # float:[128, 3072] = float:[3072]
            ("test_bert_bias_add_grad_042", "bias_add_grad_run", ([128, 3072, 1, 1], "NCHW", "float32")),
            # float:[131072, 3072] = float:[3072]
            ("test_bert_bias_add_grad_043", "bias_add_grad_run", ([131072, 3072, 1, 1], "NCHW", "float32")),
            # float:[16384, 3072] = float:[3072]
            ("test_bert_bias_add_grad_044", "bias_add_grad_run", ([16384, 3072, 1, 1], "NCHW", "float32")),
            # float:[2048, 3072] = float:[3072]
            ("test_bert_bias_add_grad_045", "bias_add_grad_run", ([2048, 3072, 1, 1], "NCHW", "float32")),
            # float:[256, 3072] = float:[3072]
            ("test_bert_bias_add_grad_046", "bias_add_grad_run", ([256, 3072, 1, 1], "NCHW", "float32")),
            # float:[32768, 3072] = float:[3072]
            ("test_bert_bias_add_grad_047", "bias_add_grad_run", ([32768, 3072, 1, 1], "NCHW", "float32")),
            # float:[4096, 3072] = float:[3072]
            ("test_bert_bias_add_grad_048", "bias_add_grad_run", ([4096, 3072, 1, 1], "NCHW", "float32")),
            # float:[512, 3072] = float:[3072]
            ("test_bert_bias_add_grad_049", "bias_add_grad_run", ([512, 3072, 1, 1], "NCHW", "float32")),

            # float:[65536, 3072] = float:[3072]
            ("test_bert_bias_add_grad_050", "bias_add_grad_run", ([65536, 3072, 1, 1], "NCHW", "float32")),
            # float:[8192, 3072] = float:[3072]
            ("test_bert_bias_add_grad_051", "bias_add_grad_run", ([8192, 3072, 1, 1], "NCHW", "float32")),
            # float:[1024, 4096] = float:[4096]
            ("test_bert_bias_add_grad_052", "bias_add_grad_run", ([1024, 4096, 1, 1], "NCHW", "float32")),
            # float:[128, 4096] = float:[4096]
            ("test_bert_bias_add_grad_053", "bias_add_grad_run", ([128, 4096, 1, 1], "NCHW", "float32")),
            # float:[131072, 4096] = float:[4096]
            ("test_bert_bias_add_grad_054", "bias_add_grad_run", ([131072, 4096, 1, 1], "NCHW", "float32")),
            # float:[16384, 4096] = float:[4096]
            ("test_bert_bias_add_grad_055", "bias_add_grad_run", ([16384, 4096, 1, 1], "NCHW", "float32")),
            # float:[2048, 4096] = float:[4096]
            ("test_bert_bias_add_grad_056", "bias_add_grad_run", ([2048, 4096, 1, 1], "NCHW", "float32")),
            # float:[256, 4096] = float:[4096]
            ("test_bert_bias_add_grad_057", "bias_add_grad_run", ([256, 4096, 1, 1], "NCHW", "float32")),
            # float:[32768, 4096] = float:[4096]
            ("test_bert_bias_add_grad_058", "bias_add_grad_run", ([32768, 4096, 1, 1], "NCHW", "float32")),
            # float:[4096, 4096] = float:[4096]
            ("test_bert_bias_add_grad_059", "bias_add_grad_run", ([4096, 4096, 1, 1], "NCHW", "float32")),
            # float:[512, 4096] = float:[4096]
            ("test_bert_bias_add_grad_060", "bias_add_grad_run", ([512, 4096, 1, 1], "NCHW", "float32")),
            # float:[65536, 4096] = float:[4096]
            ("test_bert_bias_add_grad_061", "bias_add_grad_run", ([65536, 4096, 1, 1], "NCHW", "float32")),
            # float:[8192, 4096] = float:[4096]
            ("test_bert_bias_add_grad_062", "bias_add_grad_run", ([8192, 4096, 1, 1], "NCHW", "float32")),
            # float:[10240, 768] = float:[768]
            ("test_bert_bias_add_grad_063", "bias_add_grad_run", ([10240, 768, 1, 1], "NCHW", "float32")),
            # float:[1024, 768] = float:[768]
            ("test_bert_bias_add_grad_064", "bias_add_grad_run", ([1024, 768, 1, 1], "NCHW", "float32")),
            # float:[1280, 768] = float:[768]
            ("test_bert_bias_add_grad_065", "bias_add_grad_run", ([1280, 768, 1, 1], "NCHW", "float32")),
            # float:[128, 768] = float:[768]
            ("test_bert_bias_add_grad_066", "bias_add_grad_run", ([128, 768, 1, 1], "NCHW", "float32")),
            # float:[131072, 768] = float:[768]
            ("test_bert_bias_add_grad_067", "bias_add_grad_run", ([131072, 768, 1, 1], "NCHW", "float32")),
            # float:[160, 768] = float:[768]
            ("test_bert_bias_add_grad_068", "bias_add_grad_run", ([160, 768, 1, 1], "NCHW", "float32")),
            # float:[16384, 768] = float:[768]
            ("test_bert_bias_add_grad_069", "bias_add_grad_run", ([16384, 768, 1, 1], "NCHW", "float32")),
            # float:[16, 768] = float:[768]
            ("test_bert_bias_add_grad_070", "bias_add_grad_run", ([16, 768, 1, 1], "NCHW", "float32")),
            # float:[1, 768] = float:[768]
            ("test_bert_bias_add_grad_071", "bias_add_grad_run", ([1, 768, 1, 1], "NCHW", "float32")),
            # float:[20480, 768] = float:[768]
            ("test_bert_bias_add_grad_072", "bias_add_grad_run", ([20480, 768, 1, 1], "NCHW", "float32")),
            # float:[2048, 768] = float:[768]
            ("test_bert_bias_add_grad_073", "bias_add_grad_run", ([2048, 768, 1, 1], "NCHW", "float32")),
            # float:[20, 768] = float:[768]
            ("test_bert_bias_add_grad_074", "bias_add_grad_run", ([20, 768, 1, 1], "NCHW", "float32")),
            # float:[2560, 768] = float:[768]
            ("test_bert_bias_add_grad_075", "bias_add_grad_run", ([2560, 768, 1, 1], "NCHW", "float32")),
            # float:[256, 768] = float:[768]
            ("test_bert_bias_add_grad_076", "bias_add_grad_run", ([256, 768, 1, 1], "NCHW", "float32")),
            # float:[2, 768] = float:[768]
            ("test_bert_bias_add_grad_077", "bias_add_grad_run", ([2, 768, 1, 1], "NCHW", "float32")),
            # float:[320, 768] = float:[768]
            ("test_bert_bias_add_grad_078", "bias_add_grad_run", ([320, 768, 1, 1], "NCHW", "float32")),
            # float:[32, 768] = float:[768]
            ("test_bert_bias_add_grad_079", "bias_add_grad_run", ([32, 768, 1, 1], "NCHW", "float32")),
            # float:[32768, 768] = float:[768]
            ("test_bert_bias_add_grad_080", "bias_add_grad_run", ([32768, 768, 1, 1], "NCHW", "float32")),
            # float:[40, 768] = float:[768]
            ("test_bert_bias_add_grad_081", "bias_add_grad_run", ([40, 768, 1, 1], "NCHW", "float32")),
            # float:[4096, 768] = float:[768]
            ("test_bert_bias_add_grad_082", "bias_add_grad_run", ([4096, 768, 1, 1], "NCHW", "float32")),
            # float:[4, 768] = float:[768]
            ("test_bert_bias_add_grad_083", "bias_add_grad_run", ([4, 768, 1, 1], "NCHW", "float32")),
            # float:[5120, 768] = float:[768]
            ("test_bert_bias_add_grad_084", "bias_add_grad_run", ([5120, 768, 1, 1], "NCHW", "float32")),
            # float:[512, 768] = float:[768]
            ("test_bert_bias_add_grad_085", "bias_add_grad_run", ([512, 768, 1, 1], "NCHW", "float32")),
            # float:[640, 768] = float:[768]
            ("test_bert_bias_add_grad_086", "bias_add_grad_run", ([640, 768, 1, 1], "NCHW", "float32")),
            # float:[64, 768] = float:[768]
            ("test_bert_bias_add_grad_087", "bias_add_grad_run", ([64, 768, 1, 1], "NCHW", "float32")),
            # float:[65536, 768] = float:[768]
            ("test_bert_bias_add_grad_088", "bias_add_grad_run", ([65536, 768, 1, 1], "NCHW", "float32")),
            # float:[80, 768] = float:[768]
            ("test_bert_bias_add_grad_089", "bias_add_grad_run", ([80, 768, 1, 1], "NCHW", "float32")),
            # float:[8192, 768] = float:[768]
            ("test_bert_bias_add_grad_090", "bias_add_grad_run", ([8192, 768, 1, 1], "NCHW", "float32")),
            # float:[8, 768] = float:[768]
            ("test_bert_bias_add_grad_091", "bias_add_grad_run", ([8, 768, 1, 1], "NCHW", "float32")),

            ("test_bert_div_003_001", "div_run", ((21128, 1024), (1,), "float32")),
            ("test_bert_div_003_002", "div_run", ((2, 1024), (1,), "float32")),
            ("test_bert_div_003_003", "div_run", ((1024,), (1,), "float32")),
            ("test_bert_div_003_004", "div_run", ((1024, 1024), (1,), "float32")),
            ("test_bert_div_003_005", "div_run", ((33, 64), (1,), "float32")),
            ("test_bert_div_003_006", "div_run", ((1024, 4096), (1,), "float32")),
            ("test_bert_div_003_007", "div_run", ((4096,), (1,), "float32")),
            ("test_bert_div_003_008", "div_run", ((4096, 1024), (1,), "float32")),
            ("test_bert_div_003_009", "div_run", ((21128,), (1,), "float32")),
            ("test_bert_div_003_010", "div_run", ((2,), (1,), "float32")),

            # float-float:[10240, 1024]-[1024, 1024]=float:[10240, 1024]
            ("test_bert_batch_matmul_003_160", "batchmatmul_run",
             ((), 10240, 1024, 1024, (1024,), "float32", False, False, "batch_matmul_output")),
            # float-float:[10240, 768]-[768, 768]=float:[10240, 768]
            ("test_bert_batch_matmul_003_161", "batchmatmul_run",
             ((), 10240, 768, 768, (768,), "float32", False, False, "batch_matmul_output")),
            # float-float:[1024, 1024]-[1024, 1024]=float:[1024, 1024]
            ("test_bert_batch_matmul_003_162", "batchmatmul_run",
             ((), 1024, 1024, 1024, (1024,), "float32", False, False, "batch_matmul_output")),
            # float-float:[1024, 4096]-[1024, 4096]=float:[1024, 1024]
            ("test_bert_batch_matmul_003_163", "batchmatmul_run",
             ((), 1024, 1024, 4096, (1024,), "float32", False, True, "batch_matmul_output")),
            # float-float:[1024, 1024]-[2, 1024]=float:[1024, 2]
            ("test_bert_batch_matmul_003_164", "batchmatmul_run",
             ((), 1024, 2, 1024, (2,), "float32", False, True, "batch_matmul_output")),
            # float-float:[1024, 768]-[2, 768]=float:[1024, 2]
            ("test_bert_batch_matmul_003_165", "batchmatmul_run",
             ((), 1024, 2, 768, (2,), "float32", False, True, "batch_matmul_output")),
            # float-float:[1024, 768]-[3072, 768]=float:[1024, 3072]
            ("test_bert_batch_matmul_003_166", "batchmatmul_run",
             ((), 1024, 3072, 768, (3072,), "float32", False, True, "batch_matmul_output")),
            # float-float:[1024, 1024]-[4096, 1024]=float:[1024, 4096]
            ("test_bert_batch_matmul_003_167", "batchmatmul_run",
             ((), 1024, 4096, 1024, (4096,), "float32", False, True, "batch_matmul_output")),
            # float-float:[1024, 3072]-[768, 3072]=float:[1024, 768]
            ("test_bert_batch_matmul_003_168", "batchmatmul_run",
             ((), 1024, 768, 3072, (768,), "float32", False, True, "batch_matmul_output")),
            # float-float:[1024, 768]-[768, 768]=float:[1024, 768]
            ("test_bert_batch_matmul_003_169", "batchmatmul_run",
             ((), 1024, 768, 768, (768,), "float32", False, False, "batch_matmul_output")),
            # float-float:[1, 1024]-[1024, 1024]=float:[1, 1024]
            ("test_bert_batch_matmul_003_170", "batchmatmul_run",
             ((), 1, 1024, 1024, (1024,), "float32", False, False, "batch_matmul_output")),
            # float-float:[1280, 1024]-[1024, 1024]=float:[1280, 1024]
            ("test_bert_batch_matmul_003_171", "batchmatmul_run",
             ((), 1280, 1024, 1024, (1024,), "float32", False, False, "batch_matmul_output")),
            # float-float:[1280, 768]-[768, 768]=float:[1280, 768]
            ("test_bert_batch_matmul_003_172", "batchmatmul_run",
             ((), 1280, 768, 768, (768,), "float32", False, False, "batch_matmul_output")),
            # float-float:[128, 1024]-[1024, 1024]=float:[128, 1024]
            ("test_bert_batch_matmul_003_173", "batchmatmul_run",
             ((), 128, 1024, 1024, (1024,), "float32", False, False, "batch_matmul_output")),
            # float-float:[128, 4096]-[1024, 4096]=float:[128, 1024]
            ("test_bert_batch_matmul_003_174", "batchmatmul_run",
             ((), 128, 1024, 4096, (1024,), "float32", False, True, "batch_matmul_output")),
            # float-float:[128, 1024]-[2, 1024]=float:[128, 2]
            ("test_bert_batch_matmul_003_175", "batchmatmul_run",
             ((), 128, 2, 1024, (2,), "float32", False, True, "batch_matmul_output")),
            # float-float:[128, 768]-[2, 768]=float:[128, 2]
            ("test_bert_batch_matmul_003_176", "batchmatmul_run",
             ((), 128, 2, 768, (2,), "float32", False, True, "batch_matmul_output")),
            # float-float:[128, 768]-[3072, 768]=float:[128, 3072]
            ("test_bert_batch_matmul_003_177", "batchmatmul_run",
             ((), 128, 3072, 768, (3072,), "float32", False, True, "batch_matmul_output")),
            # float-float:[128, 1024]-[4096, 1024]=float:[128, 4096]
            ("test_bert_batch_matmul_003_178", "batchmatmul_run",
             ((), 128, 4096, 1024, (4096,), "float32", False, True, "batch_matmul_output")),
            # float-float:[128, 3072]-[768, 3072]=float:[128, 768]
            ("test_bert_batch_matmul_003_179", "batchmatmul_run",
             ((), 128, 768, 3072, (768,), "float32", False, True, "batch_matmul_output")),
            # float-float:[128, 768]-[768, 768]=float:[128, 768]
            ("test_bert_batch_matmul_003_180", "batchmatmul_run",
             ((), 128, 768, 768, (768,), "float32", False, False, "batch_matmul_output")),
            # float-float:[1, 1024]-[2, 1024]=float:[1, 2]
            ("test_bert_batch_matmul_003_181", "batchmatmul_run",
             ((), 1, 2, 1024, (2,), "float32", False, True, "batch_matmul_output")),
            # float-float:[1, 768]-[2, 768]=float:[1, 2]
            ("test_bert_batch_matmul_003_182", "batchmatmul_run",
             ((), 1, 2, 768, (2,), "float32", False, True, "batch_matmul_output")),
            # float-float:[131072, 1024]-[1024, 1024]=float:[131072, 1024]
            ("test_bert_batch_matmul_003_183", "batchmatmul_run",
             ((), 131072, 1024, 1024, (1024,), "float32", False, False, "batch_matmul_output")),
            # float-float:[131072, 4096]-[1024, 4096]=float:[131072, 1024]
            ("test_bert_batch_matmul_003_184", "batchmatmul_run",
             ((), 131072, 1024, 4096, (1024,), "float32", False, True, "batch_matmul_output")),
            # float-float:[131072, 768]-[3072, 768]=float:[131072, 3072]
            ("test_bert_batch_matmul_003_185", "batchmatmul_run",
             ((), 131072, 3072, 768, (3072,), "float32", False, True, "batch_matmul_output")),
            # float-float:[131072, 1024]-[4096, 1024]=float:[131072, 4096]
            ("test_bert_batch_matmul_003_186", "batchmatmul_run",
             ((), 131072, 4096, 1024, (4096,), "float32", False, True, "batch_matmul_output")),
            # float-float:[131072, 3072]-[768, 3072]=float:[131072, 768]
            ("test_bert_batch_matmul_003_187", "batchmatmul_run",
             ((), 131072, 768, 3072, (768,), "float32", False, True, "batch_matmul_output")),
            # float-float:[131072, 768]-[768, 768]=float:[131072, 768]
            ("test_bert_batch_matmul_003_188", "batchmatmul_run",
             ((), 131072, 768, 768, (768,), "float32", False, False, "batch_matmul_output")),
            # float-float:[160, 1024]-[1024, 1024]=float:[160, 1024]
            ("test_bert_batch_matmul_003_189", "batchmatmul_run",
             ((), 160, 1024, 1024, (1024,), "float32", False, False, "batch_matmul_output")),
            # float-float:[160, 768]-[768, 768]=float:[160, 768]
            ("test_bert_batch_matmul_003_190", "batchmatmul_run",
             ((), 160, 768, 768, (768,), "float32", False, False, "batch_matmul_output")),
            # float-float:[16, 1024]-[1024, 1024]=float:[16, 1024]
            ("test_bert_batch_matmul_003_191", "batchmatmul_run",
             ((), 16, 1024, 1024, (1024,), "float32", False, False, "batch_matmul_output")),
            # float-float:[16, 1024]-[2, 1024]=float:[16, 2]
            ("test_bert_batch_matmul_003_192", "batchmatmul_run",
             ((), 16, 2, 1024, (2,), "float32", False, True, "batch_matmul_output")),
            # float-float:[16, 768]-[2, 768]=float:[16, 2]
            ("test_bert_batch_matmul_003_193", "batchmatmul_run",
             ((), 16, 2, 768, (2,), "float32", False, True, "batch_matmul_output")),
            # float-float:[16384, 1024]-[1024, 1024]=float:[16384, 1024]
            ("test_bert_batch_matmul_003_194", "batchmatmul_run",
             ((), 16384, 1024, 1024, (1024,), "float32", False, False, "batch_matmul_output")),
            # float-float:[16384, 4096]-[1024, 4096]=float:[16384, 1024]
            ("test_bert_batch_matmul_003_195", "batchmatmul_run",
             ((), 16384, 1024, 4096, (1024,), "float32", False, True, "batch_matmul_output")),
            # float-float:[16384, 768]-[3072, 768]=float:[16384, 3072]
            ("test_bert_batch_matmul_003_196", "batchmatmul_run",
             ((), 16384, 3072, 768, (3072,), "float32", False, True, "batch_matmul_output")),
            # float-float:[16384, 1024]-[4096, 1024]=float:[16384, 4096]
            ("test_bert_batch_matmul_003_197", "batchmatmul_run",
             ((), 16384, 4096, 1024, (4096,), "float32", False, True, "batch_matmul_output")),
            # float-float:[16384, 3072]-[768, 3072]=float:[16384, 768]
            ("test_bert_batch_matmul_003_198", "batchmatmul_run",
             ((), 16384, 768, 3072, (768,), "float32", False, True, "batch_matmul_output")),
            # float-float:[16384, 768]-[768, 768]=float:[16384, 768]
            ("test_bert_batch_matmul_003_199", "batchmatmul_run",
             ((), 16384, 768, 768, (768,), "float32", False, False, "batch_matmul_output")),
            # float-float:[16, 768]-[768, 768]=float:[16, 768]
            ("test_bert_batch_matmul_003_200", "batchmatmul_run",
             ((), 16, 768, 768, (768,), "float32", False, False, "batch_matmul_output")),
            # float-float:[1, 768]-[768, 768]=float:[1, 768]
            ("test_bert_batch_matmul_003_201", "batchmatmul_run",
             ((), 1, 768, 768, (768,), "float32", False, False, "batch_matmul_output")),
            # float-float:[20, 1024]-[1024, 1024]=float:[20, 1024]
            ("test_bert_batch_matmul_003_202", "batchmatmul_run",
             ((), 20, 1024, 1024, (1024,), "float32", False, False, "batch_matmul_output")),
            # float-float:[20480, 1024]-[1024, 1024]=float:[20480, 1024]
            ("test_bert_batch_matmul_003_203", "batchmatmul_run",
             ((), 20480, 1024, 1024, (1024,), "float32", False, False, "batch_matmul_output")),
            # float-float:[20480, 768]-[768, 768]=float:[20480, 768]
            ("test_bert_batch_matmul_003_204", "batchmatmul_run",
             ((), 20480, 768, 768, (768,), "float32", False, False, "batch_matmul_output")),
            # float-float:[2048, 1024]-[1024, 1024]=float:[2048, 1024]
            ("test_bert_batch_matmul_003_205", "batchmatmul_run",
             ((), 2048, 1024, 1024, (1024,), "float32", False, False, "batch_matmul_output")),
            # float-float:[2048, 4096]-[1024, 4096]=float:[2048, 1024]
            ("test_bert_batch_matmul_003_206", "batchmatmul_run",
             ((), 2048, 1024, 4096, (1024,), "float32", False, True, "batch_matmul_output")),
            # float-float:[2048, 768]-[3072, 768]=float:[2048, 3072]
            ("test_bert_batch_matmul_003_207", "batchmatmul_run",
             ((), 2048, 3072, 768, (3072,), "float32", False, True, "batch_matmul_output")),
            # float-float:[2048, 1024]-[4096, 1024]=float:[2048, 4096]
            ("test_bert_batch_matmul_003_208", "batchmatmul_run",
             ((), 2048, 4096, 1024, (4096,), "float32", False, True, "batch_matmul_output")),
            # float-float:[2048, 3072]-[768, 3072]=float:[2048, 768]
            ("test_bert_batch_matmul_003_209", "batchmatmul_run",
             ((), 2048, 768, 3072, (768,), "float32", False, True, "batch_matmul_output")),
            # float-float:[2048, 768]-[768, 768]=float:[2048, 768]
            ("test_bert_batch_matmul_003_210", "batchmatmul_run",
             ((), 2048, 768, 768, (768,), "float32", False, False, "batch_matmul_output")),
            # float-float:[20, 768]-[768, 768]=float:[20, 768]
            ("test_bert_batch_matmul_003_211", "batchmatmul_run",
             ((), 20, 768, 768, (768,), "float32", False, False, "batch_matmul_output")),
            # float-float:[2, 1024]-[1024, 1024]=float:[2, 1024]
            ("test_bert_batch_matmul_003_212", "batchmatmul_run",
             ((), 2, 1024, 1024, (1024,), "float32", False, False, "batch_matmul_output")),
            # float-float:[2, 1024]-[2, 1024]=float:[2, 2]
            ("test_bert_batch_matmul_003_213", "batchmatmul_run",
             ((), 2, 2, 1024, (2,), "float32", False, True, "batch_matmul_output")),
            # float-float:[2, 768]-[2, 768]=float:[2, 2]
            ("test_bert_batch_matmul_003_214", "batchmatmul_run",
             ((), 2, 2, 768, (2,), "float32", False, True, "batch_matmul_output")),
            # float-float:[2560, 1024]-[1024, 1024]=float:[2560, 1024]
            ("test_bert_batch_matmul_003_215", "batchmatmul_run",
             ((), 2560, 1024, 1024, (1024,), "float32", False, False, "batch_matmul_output")),
            # float-float:[2560, 768]-[768, 768]=float:[2560, 768]
            ("test_bert_batch_matmul_003_216", "batchmatmul_run",
             ((), 2560, 768, 768, (768,), "float32", False, False, "batch_matmul_output")),
            # float-float:[256, 1024]-[1024, 1024]=float:[256, 1024]
            ("test_bert_batch_matmul_003_217", "batchmatmul_run",
             ((), 256, 1024, 1024, (1024,), "float32", False, False, "batch_matmul_output")),
            # float-float:[256, 4096]-[1024, 4096]=float:[256, 1024]
            ("test_bert_batch_matmul_003_218", "batchmatmul_run",
             ((), 256, 1024, 4096, (1024,), "float32", False, True, "batch_matmul_output")),
            # float-float:[256, 1024]-[2, 1024]=float:[256, 2]
            ("test_bert_batch_matmul_003_219", "batchmatmul_run",
             ((), 256, 2, 1024, (2,), "float32", False, True, "batch_matmul_output")),
            # float-float:[256, 768]-[2, 768]=float:[256, 2]
            ("test_bert_batch_matmul_003_220", "batchmatmul_run",
             ((), 256, 2, 768, (2,), "float32", False, True, "batch_matmul_output")),
            # float-float:[256, 768]-[3072, 768]=float:[256, 3072]
            ("test_bert_batch_matmul_003_221", "batchmatmul_run",
             ((), 256, 3072, 768, (3072,), "float32", False, True, "batch_matmul_output")),
            # float-float:[256, 1024]-[4096, 1024]=float:[256, 4096]
            ("test_bert_batch_matmul_003_222", "batchmatmul_run",
             ((), 256, 4096, 1024, (4096,), "float32", False, True, "batch_matmul_output")),
            # float-float:[256, 3072]-[768, 3072]=float:[256, 768]
            ("test_bert_batch_matmul_003_223", "batchmatmul_run",
             ((), 256, 768, 3072, (768,), "float32", False, True, "batch_matmul_output")),
            # float-float:[256, 768]-[768, 768]=float:[256, 768]
            ("test_bert_batch_matmul_003_224", "batchmatmul_run",
             ((), 256, 768, 768, (768,), "float32", False, False, "batch_matmul_output")),
            # float-float:[2, 768]-[768, 768]=float:[2, 768]
            ("test_bert_batch_matmul_003_225", "batchmatmul_run",
             ((), 2, 768, 768, (768,), "float32", False, False, "batch_matmul_output")),
            # float-float:[320, 1024]-[1024, 1024]=float:[320, 1024]
            ("test_bert_batch_matmul_003_226", "batchmatmul_run",
             ((), 320, 1024, 1024, (1024,), "float32", False, False, "batch_matmul_output")),
            # float-float:[320, 768]-[768, 768]=float:[320, 768]
            ("test_bert_batch_matmul_003_227", "batchmatmul_run",
             ((), 320, 768, 768, (768,), "float32", False, False, "batch_matmul_output")),
            # float-float:[32, 1024]-[1024, 1024]=float:[32, 1024]
            ("test_bert_batch_matmul_003_228", "batchmatmul_run",
             ((), 32, 1024, 1024, (1024,), "float32", False, False, "batch_matmul_output")),
            # float-float:[32, 1024]-[2, 1024]=float:[32, 2]
            ("test_bert_batch_matmul_003_229", "batchmatmul_run",
             ((), 32, 2, 1024, (2,), "float32", False, True, "batch_matmul_output")),
            # float-float:[32, 768]-[2, 768]=float:[32, 2]
            ("test_bert_batch_matmul_003_230", "batchmatmul_run",
             ((), 32, 2, 768, (2,), "float32", False, True, "batch_matmul_output")),
            # float-float:[32768, 1024]-[1024, 1024]=float:[32768, 1024]
            ("test_bert_batch_matmul_003_231", "batchmatmul_run",
             ((), 32768, 1024, 1024, (1024,), "float32", False, False, "batch_matmul_output")),
            # float-float:[32768, 4096]-[1024, 4096]=float:[32768, 1024]
            ("test_bert_batch_matmul_003_232", "batchmatmul_run",
             ((), 32768, 1024, 4096, (1024,), "float32", False, True, "batch_matmul_output")),
            # float-float:[32768, 768]-[3072, 768]=float:[32768, 3072]
            ("test_bert_batch_matmul_003_233", "batchmatmul_run",
             ((), 32768, 3072, 768, (3072,), "float32", False, True, "batch_matmul_output")),
            # float-float:[32768, 1024]-[4096, 1024]=float:[32768, 4096]
            ("test_bert_batch_matmul_003_234", "batchmatmul_run",
             ((), 32768, 4096, 1024, (4096,), "float32", False, True, "batch_matmul_output")),
            # float-float:[32768, 3072]-[768, 3072]=float:[32768, 768]
            ("test_bert_batch_matmul_003_235", "batchmatmul_run",
             ((), 32768, 768, 3072, (768,), "float32", False, True, "batch_matmul_output")),
            # float-float:[32768, 768]-[768, 768]=float:[32768, 768]
            ("test_bert_batch_matmul_003_236", "batchmatmul_run",
             ((), 32768, 768, 768, (768,), "float32", False, False, "batch_matmul_output")),
            # float-float:[32, 768]-[768, 768]=float:[32, 768]
            ("test_bert_batch_matmul_003_237", "batchmatmul_run",
             ((), 32, 768, 768, (768,), "float32", False, False, "batch_matmul_output")),
            # float-float:[40, 1024]-[1024, 1024]=float:[40, 1024]
            ("test_bert_batch_matmul_003_238", "batchmatmul_run",
             ((), 40, 1024, 1024, (1024,), "float32", False, False, "batch_matmul_output")),
            # float-float:[40, 768]-[768, 768]=float:[40, 768]
            ("test_bert_batch_matmul_003_239", "batchmatmul_run",
             ((), 40, 768, 768, (768,), "float32", False, False, "batch_matmul_output")),
            # float-float:[4096, 1024]-[1024, 1024]=float:[4096, 1024]
            ("test_bert_batch_matmul_003_240", "batchmatmul_run",
             ((), 4096, 1024, 1024, (1024,), "float32", False, False, "batch_matmul_output")),
            # float-float:[4096, 4096]-[1024, 4096]=float:[4096, 1024]
            ("test_bert_batch_matmul_003_241", "batchmatmul_run",
             ((), 4096, 1024, 4096, (1024,), "float32", False, True, "batch_matmul_output")),
            # float-float:[4096, 768]-[3072, 768]=float:[4096, 3072]
            ("test_bert_batch_matmul_003_242", "batchmatmul_run",
             ((), 4096, 3072, 768, (3072,), "float32", False, True, "batch_matmul_output")),
            # float-float:[4096, 1024]-[4096, 1024]=float:[4096, 4096]
            ("test_bert_batch_matmul_003_243", "batchmatmul_run",
             ((), 4096, 4096, 1024, (4096,), "float32", False, True, "batch_matmul_output")),
            # float-float:[4096, 3072]-[768, 3072]=float:[4096, 768]
            ("test_bert_batch_matmul_003_244", "batchmatmul_run",
             ((), 4096, 768, 3072, (768,), "float32", False, True, "batch_matmul_output")),
            # float-float:[4096, 768]-[768, 768]=float:[4096, 768]
            ("test_bert_batch_matmul_003_245", "batchmatmul_run",
             ((), 4096, 768, 768, (768,), "float32", False, False, "batch_matmul_output")),
            # float-float:[4, 1024]-[1024, 1024]=float:[4, 1024]
            ("test_bert_batch_matmul_003_246", "batchmatmul_run",
             ((), 4, 1024, 1024, (1024,), "float32", False, False, "batch_matmul_output")),
            # float-float:[4, 1024]-[2, 1024]=float:[4, 2]
            ("test_bert_batch_matmul_003_247", "batchmatmul_run",
             ((), 4, 2, 1024, (2,), "float32", False, True, "batch_matmul_output")),
            # float-float:[4, 768]-[2, 768]=float:[4, 2]
            ("test_bert_batch_matmul_003_248", "batchmatmul_run",
             ((), 4, 2, 768, (2,), "float32", False, True, "batch_matmul_output")),
            # float-float:[4, 768]-[768, 768]=float:[4, 768]
            ("test_bert_batch_matmul_003_249", "batchmatmul_run",
             ((), 4, 768, 768, (768,), "float32", False, False, "batch_matmul_output")),
            # float-float:[5120, 1024]-[1024, 1024]=float:[5120, 1024]
            ("test_bert_batch_matmul_003_250", "batchmatmul_run",
             ((), 5120, 1024, 1024, (1024,), "float32", False, False, "batch_matmul_output")),
            # float-float:[5120, 768]-[768, 768]=float:[5120, 768]
            ("test_bert_batch_matmul_003_251", "batchmatmul_run",
             ((), 5120, 768, 768, (768,), "float32", False, False, "batch_matmul_output")),
            # float-float:[512, 1024]-[1024, 1024]=float:[512, 1024]
            ("test_bert_batch_matmul_003_252", "batchmatmul_run",
             ((), 512, 1024, 1024, (1024,), "float32", False, False, "batch_matmul_output")),
            # float-float:[512, 4096]-[1024, 4096]=float:[512, 1024]
            ("test_bert_batch_matmul_003_253", "batchmatmul_run",
             ((), 512, 1024, 4096, (1024,), "float32", False, True, "batch_matmul_output")),
            # float-float:[512, 1024]-[2, 1024]=float:[512, 2]
            ("test_bert_batch_matmul_003_254", "batchmatmul_run",
             ((), 512, 2, 1024, (2,), "float32", False, True, "batch_matmul_output")),
            # float-float:[512, 768]-[2, 768]=float:[512, 2]
            ("test_bert_batch_matmul_003_255", "batchmatmul_run",
             ((), 512, 2, 768, (2,), "float32", False, True, "batch_matmul_output")),
            # float-float:[512, 768]-[3072, 768]=float:[512, 3072]
            ("test_bert_batch_matmul_003_256", "batchmatmul_run",
             ((), 512, 3072, 768, (3072,), "float32", False, True, "batch_matmul_output")),
            # float-float:[512, 1024]-[4096, 1024]=float:[512, 4096]
            ("test_bert_batch_matmul_003_257", "batchmatmul_run",
             ((), 512, 4096, 1024, (4096,), "float32", False, True, "batch_matmul_output")),
            # float-float:[512, 3072]-[768, 3072]=float:[512, 768]
            ("test_bert_batch_matmul_003_258", "batchmatmul_run",
             ((), 512, 768, 3072, (768,), "float32", False, True, "batch_matmul_output")),
            # float-float:[512, 768]-[768, 768]=float:[512, 768]
            ("test_bert_batch_matmul_003_259", "batchmatmul_run",
             ((), 512, 768, 768, (768,), "float32", False, False, "batch_matmul_output")),
            # float-float:[640, 1024]-[1024, 1024]=float:[640, 1024]
            ("test_bert_batch_matmul_003_260", "batchmatmul_run",
             ((), 640, 1024, 1024, (1024,), "float32", False, False, "batch_matmul_output")),
            # float-float:[640, 768]-[768, 768]=float:[640, 768]
            ("test_bert_batch_matmul_003_261", "batchmatmul_run",
             ((), 640, 768, 768, (768,), "float32", False, False, "batch_matmul_output")),
            # float-float:[64, 1024]-[1024, 1024]=float:[64, 1024]
            ("test_bert_batch_matmul_003_262", "batchmatmul_run",
             ((), 64, 1024, 1024, (1024,), "float32", False, False, "batch_matmul_output")),
            # float-float:[64, 1024]-[2, 1024]=float:[64, 2]
            ("test_bert_batch_matmul_003_263", "batchmatmul_run",
             ((), 64, 2, 1024, (2,), "float32", False, True, "batch_matmul_output")),
            # float-float:[64, 768]-[2, 768]=float:[64, 2]
            ("test_bert_batch_matmul_003_264", "batchmatmul_run",
             ((), 64, 2, 768, (2,), "float32", False, True, "batch_matmul_output")),
            # float-float:[64, 768]-[768, 768]=float:[64, 768]
            ("test_bert_batch_matmul_003_265", "batchmatmul_run",
             ((), 64, 768, 768, (768,), "float32", False, False, "batch_matmul_output")),
            # float-float:[65536, 1024]-[1024, 1024]=float:[65536, 1024]
            ("test_bert_batch_matmul_003_266", "batchmatmul_run",
             ((), 65536, 1024, 1024, (1024,), "float32", False, False, "batch_matmul_output")),
            # float-float:[65536, 4096]-[1024, 4096]=float:[65536, 1024]
            ("test_bert_batch_matmul_003_267", "batchmatmul_run",
             ((), 65536, 1024, 4096, (1024,), "float32", False, True, "batch_matmul_output")),
            # float-float:[65536, 768]-[3072, 768]=float:[65536, 3072]
            ("test_bert_batch_matmul_003_268", "batchmatmul_run",
             ((), 65536, 3072, 768, (3072,), "float32", False, True, "batch_matmul_output")),
            # float-float:[65536, 1024]-[4096, 1024]=float:[65536, 4096]
            ("test_bert_batch_matmul_003_269", "batchmatmul_run",
             ((), 65536, 4096, 1024, (4096,), "float32", False, True, "batch_matmul_output")),
            # float-float:[65536, 3072]-[768, 3072]=float:[65536, 768]
            ("test_bert_batch_matmul_003_270", "batchmatmul_run",
             ((), 65536, 768, 3072, (768,), "float32", False, True, "batch_matmul_output")),
            # float-float:[65536, 768]-[768, 768]=float:[65536, 768]
            ("test_bert_batch_matmul_003_271", "batchmatmul_run",
             ((), 65536, 768, 768, (768,), "float32", False, False, "batch_matmul_output")),
            # float-float:[80, 1024]-[1024, 1024]=float:[80, 1024]
            ("test_bert_batch_matmul_003_272", "batchmatmul_run",
             ((), 80, 1024, 1024, (1024,), "float32", False, False, "batch_matmul_output")),
            # float-float:[80, 768]-[768, 768]=float:[80, 768]
            ("test_bert_batch_matmul_003_273", "batchmatmul_run",
             ((), 80, 768, 768, (768,), "float32", False, False, "batch_matmul_output")),
            # float-float:[8, 1024]-[1024, 1024]=float:[8, 1024]
            ("test_bert_batch_matmul_003_274", "batchmatmul_run",
             ((), 8, 1024, 1024, (1024,), "float32", False, False, "batch_matmul_output")),
            # float-float:[8192, 1024]-[1024, 1024]=float:[8192, 1024]
            ("test_bert_batch_matmul_003_275", "batchmatmul_run",
             ((), 8192, 1024, 1024, (1024,), "float32", False, False, "batch_matmul_output")),
            # float-float:[8192, 4096]-[1024, 4096]=float:[8192, 1024]
            ("test_bert_batch_matmul_003_276", "batchmatmul_run",
             ((), 8192, 1024, 4096, (1024,), "float32", False, True, "batch_matmul_output")),
            # float-float:[8192, 768]-[3072, 768]=float:[8192, 3072]
            ("test_bert_batch_matmul_003_277", "batchmatmul_run",
             ((), 8192, 3072, 768, (3072,), "float32", False, True, "batch_matmul_output")),
            # float-float:[8192, 1024]-[4096, 1024]=float:[8192, 4096]
            ("test_bert_batch_matmul_003_278", "batchmatmul_run",
             ((), 8192, 4096, 1024, (4096,), "float32", False, True, "batch_matmul_output")),
            # float-float:[8192, 3072]-[768, 3072]=float:[8192, 768]
            ("test_bert_batch_matmul_003_279", "batchmatmul_run",
             ((), 8192, 768, 3072, (768,), "float32", False, True, "batch_matmul_output")),
            # float-float:[8192, 768]-[768, 768]=float:[8192, 768]
            ("test_bert_batch_matmul_003_280", "batchmatmul_run",
             ((), 8192, 768, 768, (768,), "float32", False, False, "batch_matmul_output")),
            # float-float:[8, 1024]-[2, 1024]=float:[8, 2]
            ("test_bert_batch_matmul_003_281", "batchmatmul_run",
             ((), 8, 2, 1024, (2,), "float32", False, True, "batch_matmul_output")),
            # float-float:[8, 768]-[2, 768]=float:[8, 2]
            ("test_bert_batch_matmul_003_282", "batchmatmul_run",
             ((), 8, 2, 768, (2,), "float32", False, True, "batch_matmul_output")),
            # float-float:[8, 768]-[768, 768]=float:[8, 768]
            ("test_bert_batch_matmul_003_283", "batchmatmul_run",
             ((), 8, 768, 768, (768,), "float32", False, False, "batch_matmul_output")),
        ]


def print_args():
    cls = TestBert001()
    cls.setup()
    cls.print_args()


if __name__ == "__main__":
    print_args()
