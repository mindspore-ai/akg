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


class TestSsd001(BaseCaseRun):
    def setup(self):
        """
        testcase preparcondition
        :return:
        """
        case_name = "test_ssd_all_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        if not super(TestSsd001, self).setup():
            return False

        self.test_args = [
            ("test_ssd_concat_8_4_4_001", "concat_run",
             ([[8, 4, 4], [8, 36, 4], [8, 150, 4], [8, 600, 4], [8, 2116, 4], [8, 5776, 4]], "float32", 1),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_concat_8_4_6_001", "concat_run",
             ([[8, 4, 6], [8, 36, 6], [8, 150, 6], [8, 600, 6], [8, 2116, 6], [8, 5776, 6]], "float32", 1),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_focal_loss_8_8732_6_x_8_8732_6_001", "focal_loss_run",
             ((8, 8732, 6), "float16", "int32", 2.0, "focalloss"), ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_focal_loss_grad_8_8732_6_002", "focalloss_grad_run",
             ((8, 8732, 6), "float32", "float32", 2),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_smooth_l1_loss_8_8732_4_x_8_8732_4_x_8_8732_002", "smooth_l1_loss_run",
             ((8, 8732, 4), "float32", (8, 8732, 4), "float32",
              (8, 8732), "int32", 0, 1.0, "smooth_l1_loss_output"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_smooth_l1_loss_grad_8_8732_4_002", "smooth_l1_loss_grad_run",
             ((8, 8732, 4), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_slice_8_8732_4_x_0_0_0_x_8_4_4_002", "slice_run",
             ((8, 8732, 4), (0, 0, 0), (8, 4, 4), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_slice_8_8732_4_x_0_190_0_x_8_600_4_002", "slice_run",
             ((8, 8732, 4), (0, 190, 0), (8, 600, 4), "float32"), ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_slice_8_8732_4_x_0_2956_0_x_8_5776_4_002", "slice_run",
             ((8, 8732, 4), (0, 2956, 0), (8, 5776, 4), "float32"), ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_slice_8_8732_4_x_0_40_0_x_8_150_4_002", "slice_run",
             ((8, 8732, 4), (0, 40, 0), (8, 150, 4), "float32"), ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_slice_8_8732_4_x_0_4_0_x_8_36_4_002", "slice_run",
             ((8, 8732, 4), (0, 4, 0), (8, 36, 4), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_slice_8_8732_4_x_0_790_0_x_8_2166_4_002", "slice_run",
             ((8, 8732, 4), (0, 790, 0), (8, 2166, 4), "float32"), ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_slice_8_8732_6_x_0_0_0_x_8_4_6_002", "slice_run",
             ((8, 8732, 6), (0, 0, 0), (8, 4, 6), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_slice_8_8732_6_x_0_190_0_x_8_600_6_002", "slice_run",
             ((8, 8732, 6), (0, 190, 0), (8, 600, 6), "float32"), ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_slice_8_8732_6_x_0_2956_0_x_8_5776_6_002", "slice_run",
             ((8, 8732, 6), (0, 2956, 0), (8, 5776, 6), "float32"), ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_slice_8_8732_6_x_0_40_0_x_8_150_6_002", "slice_run",
             ((8, 8732, 6), (0, 40, 0), (8, 150, 6), "float32"), ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_slice_8_8732_6_x_0_4_0_x_8_36_6_002", "slice_run",
             ((8, 8732, 6), (0, 4, 0), (8, 36, 6), "float32"), [
                 "level0", "rpc", "rpc_cloud"]),
            ("test_ssd_slice_8_8732_6_x_0_790_0_x_8_2166_6_002", "slice_run",
             ((8, 8732, 6), (0, 790, 0), (8, 2166, 6), "float32"), ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_one_hot_8_8732_001", "one_hot_run", ((8, 8732), 16, "int32", 1, 0, -1),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_reshape_1_002", "reshape_run", ((1,), (1,), "float32"), ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_reshape_8_002", "reshape_run", ((8,), (8, 1), "float32"), ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_reshape_8_1_1_16_002", "reshape_run", ((8, 1, 1, 16), (8, 4, 4), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_reshape_8_1_1_24_002", "reshape_run", [(8, 1, 1, 24), (8, 4, 6), "float32"],
             ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_reshape_8_10_10_24_002", "reshape_run", ((8, 10, 10, 24), (8, 600, 4), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_reshape_8_10_10_36_002", "reshape_run", ((8, 10, 10, 36), (8, 600, 6), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_reshape_8_150_4_002", "reshape_run", ((8, 150, 4), (8, 5, 5, 24), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_reshape_8_150_6_002", "reshape_run", ((8, 150, 6), (8, 5, 5, 36), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_reshape_8_19_19_24_002", "reshape_run", ((8, 19, 19, 24), (8, 2166, 4), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_reshape_8_19_19_36_002", "reshape_run", ((8, 19, 19, 36), (8, 2166, 6), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_reshape_8_2166_4_002", "reshape_run", ((8, 2166, 4), (8, 19, 19, 24), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_reshape_8_2166_6_002", "reshape_run", ((8, 2166, 6), (8, 19, 19, 36), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_reshape_8_3_3_16_002", "reshape_run", ((8, 3, 3, 16), (8, 36, 4), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_reshape_8_3_3_24_002", "reshape_run", ((8, 3, 3, 24), (8, 36, 6), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_reshape_8_36_4_002", "reshape_run", ((8, 36, 4), (8, 3, 3, 16), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_reshape_8_36_6_002", "reshape_run", ((8, 36, 6), (8, 3, 3, 24), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_reshape_8_38_38_16_002", "reshape_run", ((8, 38, 38, 16), (8, 5776, 4), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_reshape_8_38_38_24_002", "reshape_run", ((8, 38, 38, 24), (8, 5776, 6), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_reshape_8_4_4_002", "reshape_run", ((8, 4, 4), (8, 1, 1, 16), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_reshape_8_4_6_002", "reshape_run", ((8, 4, 6), (8, 1, 1, 24), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_reshape_8_5_5_24_002", "reshape_run", ((8, 5, 5, 24), (8, 150, 4), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_reshape_8_5_5_36_002", "reshape_run", ((8, 5, 5, 36), (8, 150, 6), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_reshape_8_5776_4_002", "reshape_run", ((8, 5776, 4), (8, 38, 38, 16), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_reshape_8_5776_6_002", "reshape_run", ((8, 5776, 6), (8, 38, 38, 24), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_reshape_8_600_4_002", "reshape_run", ((8, 600, 4), (8, 10, 10, 24), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_reshape_8_600_6_002", "reshape_run", ((8, 600, 6), (8, 10, 10, 36), "float32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_sum_8_8732_002", "sum_run", ((8, 8732), (1,), False, "float32"), ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_reduce_mean_8_002", "mean_run", ((8,), "float32", (0,), True, "cce_mean_1_8_fp32"),
             ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_tile_1_002", "tile_run", ((1,), "float32", (64,)), ["level0", "rpc", "rpc_cloud"]),
            ("test_ssd_tile_8_1_002", "tile_run", ((8, 1), "float32", (1, 64)),
             ["level0", "rpc", "rpc_cloud"]),
        ]


def print_args():
    cls = TestSsd001()
    cls.setup()
    cls.print_args()


if __name__ == "__main__":
    print_args()
