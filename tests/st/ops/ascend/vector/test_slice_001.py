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
from tests.common.test_run.slice_run import slice_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestSlice(TestBase):
    def setup(self):
        case_name = "test_akg_slice_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            # random compile fail in daily build
            ("slice_daily", slice_run, ((4, 131, 131, 256), (0, 0, 0, 0), (4, 129, 129, 256), "float16")),

            ("slice_02", slice_run, ((512, 1024), (2, 0), (128, -1), "float16")),
            ("slice_4", slice_run, ((830, 1000, 1), (4, 40, 0), (513, 513, 1), "int32")),
            # SSD testcases
            ("slice1", slice_run, ((8, 8732, 6), (0, 0, 0), (8, 4, -1), "float16")),
            ("slice2", slice_run, ((8, 8732, 6), (0, 0, 0), (8, 4, -1), "float32")),
            ("slice3", slice_run, ((8, 8732, 4), (0, 0, 0), (-1, 4, 4), "float16")),
            ("slice4", slice_run, ((8, 8732, 4), (0, 0, 0), (-1, 4, 4), "float32")),
            ("slice5", slice_run, ((8, 8732, 6), (0, 4, 0), (8, 36, 6), "float16")),
            ("slice6", slice_run, ((8, 8732, 6), (0, 4, 0), (8, 36, 6), "float32")),
            ("slice7", slice_run, ((8, 8732, 4), (0, 4, 0), (8, 36, 4), "float16")),
            ("slice8", slice_run, ((8, 8732, 4), (0, 4, 0), (8, 36, 4), "float32")),
            ("slice9", slice_run, ((8, 8732, 6), (0, 40, 0), (8, 150, 6), "float16")),
            ("slice10", slice_run, ((8, 8732, 6), (0, 40, 0), (8, 150, 6), "float32")),
            ("slice11", slice_run, ((8, 8732, 4), (0, 40, 0), (8, 150, 4), "float16")),
            ("slice12", slice_run, ((8, 8732, 4), (0, 40, 0), (8, 150, 4), "float32")),
            ("slice13", slice_run, ((8, 8732, 6), (0, 190, 0), (8, 600, 6), "float16")),
            ("slice14", slice_run, ((8, 8732, 6), (0, 190, 0), (8, 600, 6), "float32")),
            ("slice15", slice_run, ((8, 8732, 4), (0, 190, 0), (8, 600, 4), "float16")),
            ("slice16", slice_run, ((8, 8732, 4), (0, 190, 0), (8, 600, 4), "float32")),
            ("slice17", slice_run, ((8, 8732, 6), (0, 790, 0), (8, 2166, 6), "float16")),
            ("slice18", slice_run, ((8, 8732, 6), (0, 790, 0), (8, 2166, 6), "float32")),
            ("slice19", slice_run, ((8, 8732, 4), (0, 790, 0), (8, 2166, 4), "float16")),
            ("slice20", slice_run, ((8, 8732, 4), (0, 790, 0), (8, 2166, 4), "float32")),
            ("slice21", slice_run, ((8, 8732, 6), (0, 2956, 0), (8, -1, 6), "float16")),
            ("slice22", slice_run, ((8, 8732, 6), (0, 2956, 0), (8, -1, 6), "float32")),
            ("slice23", slice_run, ((8, 8732, 4), (0, 2956, 0), (8, 5776, 4), "float16")),
            ("slice24", slice_run, ((8, 8732, 4), (0, 2956, 0), (8, 5776, 4), "float32")),
        ]
        self.testarg_cloud = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("slice_02", slice_run, ((512, 1024), (2, 0), (128, -1), "float32")),
        ]
        self.testarg_level1 = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("slice_03", slice_run, ((128, 128, 1024), (0, 64, 0), (-1, 1, -1), "float16")),
            # Deeplab v3
            ("slice_4", slice_run, ((830, 1000, 1), (4, 40, 0), (513, 513, 1), "int32")),
            ("slice_5", slice_run, ((4, 129, 129, 304), (0, 0, 0, 50), (4, 129, 129, 48), "float16")),
            ("slice_6", slice_run, ((625, 513, 3), (60, 0, 0), (513, 513, 3), "float16")),
            ("slice_7", slice_run, ((830, 1000, 3), (70, 7, 0), (513, 513, 3), "float16")),
            ("slice_8", slice_run, ((666, 1000, 3), (80, 8, 0), (513, 513, 3), "float16")),
            ("slice_9", slice_run, ((550, 625, 1), (9, 90, 0), (513, 513, 1), "int32")),
            ("slice_10", slice_run, ((513, 625, 1), (0, 10, 0), (513, 513, 1), "int32")),
            ("slice_11", slice_run, ((591, 875, 1), (11, 110, 0), (513, 513, 1), "int32")),
            ("slice_12", slice_run, ((1000, 626, 3), (120, 12, 0), (513, 513, 3), "float16")),
            ("slice_13", slice_run, ((562, 750, 1), (13, 130, 0), (513, 513, 1), "int32")),
            ("slice_14", slice_run, ((4, 131, 131, 256), (0, 0, 0, 0), (4, 129, 129, 256), "float16")),
            ("slice_15", slice_run, ((2,), (0,), (1,), "int32")),
            ("slice_16", slice_run, ((656, 875, 3), (16, 160, 0), (513, 513, 3), "float16")),
            ("slice_17", slice_run, ((1000, 750, 1), (170, 17, 0), (513, 513, 1), "int32")),
            ("slice_18", slice_run, ((624, 875, 1), (18, 180, 0), (513, 513, 1), "int32")),
            ("slice_19", slice_run, ((750, 513, 1), (190, 0, 0), (513, 513, 1), "int32")),
            ("slice_20", slice_run, ((1000, 750, 3), (200, 20, 0), (513, 513, 3), "float16")),
            ("slice_21", slice_run, ((4, 33, 33, 1280), (0, 0, 0, 210), (4, 33, 33, 256), "float16")),
            ("slice_22", slice_run, ((750, 1000, 3), (22, 220, 0), (513, 513, 3), "float16")),
            ("slice_23", slice_run, ((4, 129, 129, 304), (0, 0, 0, 23), (4, 129, 129, 256), "float16")),
            ("slice_24", slice_run, ((576, 750, 1), (24, 24, 0), (513, 513, 1), "int32")),
            ("slice_25", slice_run, ((513, 625, 3), (0, 25, 0), (513, 513, 3), "float16")),
            ("slice_26", slice_run, ((4, 67, 67, 728), (0, 0, 0, 0), (4, 65, 65, 728), "float16")),
            ("slice_27", slice_run, ((513, 875, 1), (0, 27, 0), (513, 513, 1), "int32")),
            ("slice_28", slice_run, ((624, 875, 3), (28, 28, 0), (513, 513, 3), "float16")),
            ("slice_29", slice_run, ((576, 750, 3), (29, 29, 0), (513, 513, 3), "float16")),
            ("slice_30", slice_run, ((666, 1000, 1), (30, 300, 0), (513, 513, 1), "int32")),
            ("slice_31", slice_run, ((582, 875, 3), (31, 310, 0), (513, 513, 3), "float16")),
            ("slice_32", slice_run, ((627, 750, 1), (32, 32, 0), (513, 513, 1), "int32")),
            ("slice_33", slice_run, ((591, 875, 3), (33, 330, 0), (513, 513, 3), "float16")),
            ("slice_34", slice_run, ((550, 625, 3), (34, 34, 0), (513, 513, 3), "float16")),
            ("slice_35", slice_run, ((513, 750, 1), (0, 35, 0), (513, 513, 1), "int32")),
            ("slice_36", slice_run, ((513, 750, 3), (0, 36, 0), (513, 513, 3), "float16")),
            ("slice_37", slice_run, ((875, 656, 3), (37, 37, 0), (513, 513, 3), "float16")),
            ("slice_38", slice_run, ((513, 513, 1), (0, 0, 0), (513, 513, 1), "int32")),
            ("slice_39", slice_run, ((513, 875, 3), (0, 39, 0), (513, 513, 3), "float16")),
            ("slice_40", slice_run, ((656, 875, 1), (40, 4, 0), (513, 513, 1), "int32")),
            ("slice_41", slice_run, ((582, 875, 1), (41, 41, 0), (513, 513, 1), "int32")),
            ("slice_42", slice_run, ((513, 513, 3), (0, 0, 0), (513, 513, 3), "float16")),
            ("slice_43", slice_run, ((750, 513, 3), (43, 0, 0), (513, 513, 3), "float16")),
            ("slice_44", slice_run, ((627, 750, 3), (44, 44, 0), (513, 513, 3), "float16")),
            ("slice_45", slice_run, ((562, 750, 3), (45, 45, 0), (513, 513, 3), "float16")),
            ("slice_46", slice_run, ((750, 1000, 1), (46, 46, 0), (513, 513, 1), "int32")),
            ("slice_47", slice_run, ((625, 513, 1), (47, 0, 0), (513, 513, 1), "int32")),
            ("slice_48", slice_run, ((4, 2), (0, 1), (4, 1), "int32")),
            ("slice_49", slice_run, ((1000, 626, 1), (49, 49, 0), (513, 513, 1), "int32")),
            ("slice_50", slice_run, ((875, 656, 1), (50, 5, 0), (513, 513, 1), "int32")),
            ("slice_51", slice_run, ((4, 259, 259, 128), (0, 1, 1, 0), (4, 257, 257, 128), "float16")),
        ]
        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

    def test_run_cloud(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_cloud)

    def test_run_level1(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_level1)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
