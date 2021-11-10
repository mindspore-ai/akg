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
batch_to_space_nd
"""
import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.ascend.batch_to_space_nd_run import batch_to_space_nd_run


class TestCase(TestBase):

    def setup(self):
        case_name = "test_batch_to_space_nd_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.test_level0 = [
            # 3. float-int32-int32:[36,6,6,2048]-[2]-[2,2]=float:[1,33,33,2048]
            ("batch_to_space_nd_run3", batch_to_space_nd_run, ((36, 6, 6, 2048), "float16", (6, 6), ((3, 0), (3, 0)), "batch_to_space_nd_output")),
            # 4. float-int32-int32:[4,17,17,1536]-[2]-[2,2]=float:[1,33,33,1536]
            ("batch_to_space_nd_run4", batch_to_space_nd_run, ((4, 17, 17, 1536), "float16", (2, 2), ((1, 0), (1, 0)), "batch_to_space_nd_output")),
            # 6. float-int32-int32:[16,17,17,1024]-[2]-[2,2]=float:[4,33,33,1024]
            ("batch_to_space_nd_run6", batch_to_space_nd_run, ((16, 17, 17, 1024), "float16", (2, 2), ((1, 0), (1, 0)), "batch_to_space_nd_output")),
            # 7. float-int32-int32:[144,3,3,2048]-[2]-[2,2]=float:[1,33,33,2048]
            ("batch_to_space_nd_run7", batch_to_space_nd_run, ((144, 3, 3, 2048), "float16", (12, 12), ((3, 0), (3, 0)), "batch_to_space_nd_output")),
            # 9. float-int32-int32:[4,2,2,960]-[2]-[2,2]=float:[1,3,3,960]
            ("batch_to_space_nd_run9", batch_to_space_nd_run, ((4, 2, 2, 960), "float16", (2, 2), ((1, 0), (1, 0)), "batch_to_space_nd_output")),
            # 12 .float-int32-int32:[16,19,19,1024]-[2]-[2,2]=float:[4,33,33,1024]
            ("batch_to_space_nd_run12", batch_to_space_nd_run, ((16, 19, 19, 1024), "float16", (2, 2), ((5, 0), (0, 5)), "batch_to_space_nd_output")),
            # 15. float-int32-int32:[324,2,2,2048]-[2]-[2,2]=float:[1,33,33,2048]
            ("batch_to_space_nd_run15", batch_to_space_nd_run, ((324, 2, 2, 2048), "float16", (18, 18), ((3, 0), (0, 3)), "batch_to_space_nd_output")),
            # 16. float-int32-int32:[4,17,17,1024]-[2]-[2,2]=float:[1,33,33,1024]
            ("batch_to_space_nd_run16", batch_to_space_nd_run, ((4, 17, 17, 1024), "float16", (2, 2), ((1, 0), (1, 0)), "batch_to_space_nd_output")),
        ]
        self.test_level1 = [
            # 1. float-int32-int32:[1296,4,4,2048]-[2]-[2,2]=float:[4,33,33,2048]
            ("batch_to_space_nd_run1", batch_to_space_nd_run, ((1296, 4, 4, 2048), "float16", (18, 18), ((39, 0), (39, 0)), "batch_to_space_nd_output")),
            # 2. float-int32-int32:[16,17,17,1536]-[2]-[2,2]=float:[4,33,33,1536]
            ("batch_to_space_nd_run2", batch_to_space_nd_run, ((16, 17, 17, 1536), "float16", (2, 2), ((1, 0), (1, 0)), "batch_to_space_nd_output")),
            # 5 .float-int32-int32:[144,6,6,2048]-[2]-[2,2]=float:[4,33,33,2048]
            ("batch_to_space_nd_run5", batch_to_space_nd_run, ((144, 6, 6, 2048), "float16", (6, 6), ((3, 0), (3, 0)), "batch_to_space_nd_output")),
            # 8. float-int32-int32:[144,8,8,2048]-[2]-[2,2]=float:[4,33,33,2048]
            ("batch_to_space_nd_run8", batch_to_space_nd_run, ((144, 8, 8, 2048), "float16", (6, 6), ((15, 0), (15, 0)), "batch_to_space_nd_output")),
            # 10. float-int32-int32:[16,19,19,1536]-[2]-[2,2]=float:[4,33,33,1536]
            ("batch_to_space_nd_run10", batch_to_space_nd_run, ((16, 19, 19, 1536), "float16", (2, 2), ((5, 0), (5, 0)), "batch_to_space_nd_output")),
            # 11. float-int32-int32:[1296,2,2,2048]-[2]-[2,2]=float:[4,33,33,2048]
            ("batch_to_space_nd_run11", batch_to_space_nd_run, ((1296, 2, 2, 2048), "float16", (18, 18), ((3, 0), (3, 0)), "batch_to_space_nd_output")),
            # 13. float-int32-int32:[576,3,3,2048]-[2]-[2,2]=float:[4,33,33,2048]
            ("batch_to_space_nd_run13", batch_to_space_nd_run, ((576, 3, 3, 2048), "float16", (12, 12), ((3, 0), (0, 3)), "batch_to_space_nd_output")),
            # 14. float-int32-int32:[576,5,5,2048]-[2]-[2,2]=float:[4,33,33,2048]
            ("batch_to_space_nd_run14", batch_to_space_nd_run, ((576, 5, 5, 2048), "float16", (12, 12), ((27, 0), (0, 27)), "batch_to_space_nd_output")),
        ]

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run_ci(self):
        self.common_run(self.test_level0)

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run_daily_ci(self):
        self.common_run(self.test_level1)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
