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
space_to_batch_nd
"""
import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.space_to_batch_nd_run import space_to_batch_nd_run


class TestCase(TestBase):
    def setup(self):
        case_name = "test_auto_space_to_batch_nd_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # 5. float-int32-int32:[1,33,33,1024]-[2]-[2,2]=float:[4,19,19,1024]
            ("space_to_batch_nd_run5", space_to_batch_nd_run, ((1, 33, 33, 1024), "float16", (2, 2), ((5, 0), (5, 0)), "space_to_batch_nd_output")),
            # 10. float-int32-int32:[1,33,33,1536]-[2]-[2,2]=float:[4,19,19,1536]
            ("space_to_batch_nd_run10", space_to_batch_nd_run, ((1, 33, 33, 1536), "float16", (2, 2), ((5, 0), (5, 0)), "space_to_batch_nd_output")),
            # 13. float-int32-int32:[1,3,3,960]-[2]-[2,2]=float:[4,4,4,960]
            ("space_to_batch_nd_run13", space_to_batch_nd_run, ((1, 3, 3, 960), "float16", (2, 2), ((5, 0), (5, 0)), "space_to_batch_nd_output")),
            # 17. float32 test
            ("space_to_batch_nd_run17", space_to_batch_nd_run, ((2, 16, 22, 3), "float32", (4, 3), ((2, 2), (1, 1)), "space_to_batch_nd_output")),
        ]
        self.testarg_debug = [
            # 1. float-int32-int32:[1,33,33,2048]-[2]-[2,2]=float:[324,4,4,2048]
            ("space_to_batch_nd_run1", space_to_batch_nd_run, ((1, 33, 33, 2048), "float16", (18, 18), ((39, 0), (39, 0)), "space_to_batch_nd_output")),
            # 2. float-int32-int32:[4,33,33,2048]-[2]-[2,2]=float:[576,3,3,2048]
            ("space_to_batch_nd_run2", space_to_batch_nd_run, ((4, 33, 33, 2048), "float16", (12, 12), ((3, 0), (3, 0)), "space_to_batch_nd_output")),
            # 3. float-int32-int32:[1,33,33,2048]-[2]-[2,2]=float:[144,5,5,2048]
            ("space_to_batch_nd_run3", space_to_batch_nd_run, ((1, 33, 33, 2048), "float16", (12, 12), ((27, 0), (27, 0)), "space_to_batch_nd_output")),
            # 4. float-int32-int32:[4,33,33,2048]-[2]-[2,2]=float:[144,6,6,2048]
            ("space_to_batch_nd_run4", space_to_batch_nd_run, ((4, 33, 33, 2048), "float16", (6, 6), ((3, 0), (3, 0)), "space_to_batch_nd_output")),
            # 6. float-int32-int32:[4,33,33,2048]-[2]-[2,2]=float:[1296,4,4,2048]
            ("space_to_batch_nd_run6", space_to_batch_nd_run, ((4, 33, 33, 2048), "float16", (18, 18), ((39, 0), (39, 0)), "space_to_batch_nd_output")),
            # 7. float-int32-int32:[4,33,33,1024]-[2]-[2,2]=float:[16,17,17,1024]
            ("space_to_batch_nd_run7", space_to_batch_nd_run, ((4, 33, 33, 1024), "float16", (2, 2), ((1, 0), (1, 0)), "space_to_batch_nd_output")),
            # 8. float-int32-int32:[4,33,33,2048]-[2]-[2,2]=float:[576,5,5,2048]
            ("space_to_batch_nd_run8", space_to_batch_nd_run, ((4, 33, 33, 2048), "float16", (12, 12), ((27, 0), (27, 0)), "space_to_batch_nd_output")),
            # 9. float-int32-int32:[4,33,33,2048]-[2]-[2,2]=float:[144,8,8,2048]
            ("space_to_batch_nd_run9", space_to_batch_nd_run, ((4, 33, 33, 2048), "float16", (6, 6), ((15, 0), (15, 0)), "space_to_batch_nd_output")),
            # 11. float-int32-int32:[4,33,33,1536]-[2]-[2,2]=float:[16,17,17,1536]
            ("space_to_batch_nd_run11", space_to_batch_nd_run, ((4, 33, 33, 1536), "float16", (2, 2), ((1, 0), (1, 0)), "space_to_batch_nd_output")),
            # 12. float-int32-int32:[1,33,33,2048]-[2]-[2,2]=float:[36,8,8,2048]
            ("space_to_batch_nd_run12", space_to_batch_nd_run, ((1, 33, 33, 2048), "float16", (6, 6), ((15, 0), (15, 0)), "space_to_batch_nd_output")),
            # 14. float-int32-int32:[4,33,33,1536]-[2]-[2,2]=float:[16,19,19,1536]
            ("space_to_batch_nd_run14", space_to_batch_nd_run, ((4, 33, 33, 1536), "float16", (2, 2), ((5, 0), (5, 0)), "space_to_batch_nd_output")),
            # 15. float-int32-int32:[4,33,33,2048]-[2]-[2,2]=float:[1296,2,2,2048]
            ("space_to_batch_nd_run15", space_to_batch_nd_run, ((4, 33, 33, 2048), "float16", (18, 18), ((3, 0), (3, 0)), "space_to_batch_nd_output")),
            # 16. float-int32-int32:[4,33,33,1024]-[2]-[2,2]=float:[16,19,19,1024]
            ("space_to_batch_nd_run16", space_to_batch_nd_run, ((4, 33, 33, 1024), "float16", (2, 2), ((5, 0), (5, 0)), "space_to_batch_nd_output")),
        ]
        return

    @pytest.mark.level2
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

    def test_run_level2(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_debug)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
