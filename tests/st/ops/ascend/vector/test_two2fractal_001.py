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
tf_transpose
"""
import os
import pytest
from tests.common.base import TestBase


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_tf_four2five"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # caseflag,opfuncname,testRunArgs, dimArgs
            # To zn
            ("two2fractal_run_001", "two2fractal_run", ([32, 32], 'zN', "float16")),
            ("two2fractal_run_002", "two2fractal_run", ([125, 256], 'zN', "float16")),
            ("two2fractal_run_003", "two2fractal_run", ([32, 32], 'zZ', "float16")),
            ("two2fractal_run_004", "two2fractal_run", ([125, 256], 'zZ', "float16")),
            ("two2fractal_run_005", "two2fractal_run", ([32, 32], 'nZ', "float16")),
            ("two2fractal_run_006", "two2fractal_run_sort", ([125, 256], 'nZ', "float16")),

            ("two2fractal_run_007_f32", "two2fractal_run_isl", ([32, 32], 'zN', "float32")),
            ("two2fractal_run_008_f32", "two2fractal_run_islshift", ([125, 256], 'zN', "float32")),
            ("two2fractal_run_009_f32", "two2fractal_run_isl", ([32, 32], 'zZ', "float32")),
            ("two2fractal_run_010_f32", "two2fractal_run_islshift", ([125, 256], 'zZ', "float32")),
            ("two2fractal_run_011_f32", "two2fractal_run_isl", ([32, 32], 'nZ', "float32")),

            # Matmul shape
            #("two2fractal_run_012_3d", two2fractal_run, ([128, 128, 1536], 'zN', "float16")),
            ("two2fractal_run_013_3d", "two2fractal_run_isl", ([128, 768, 128], 'zN', "float32")),
            #("two2fractal_run_014_3d", two2fractal_run, ([128, 128, 6144], 'zN', "float16")),
            ("two2fractal_run_015_3d", "two2fractal_run", ([128, 6144, 64], 'zN', "float16")),
            ("two2fractal_run_016_3d", "two2fractal_run", ([128, 128, 1536], 'zZ', "float16")),
            #("two2fractal_run_017_3d", two2fractal_run, ([128, 768, 128], 'zZ', "float16")),
            ("two2fractal_run_018_3d", "two2fractal_run", ([128, 128, 6144], 'zZ', "float16")),
            #("two2fractal_run_019_3d", two2fractal_run, ([128, 6144, 64], 'zZ', "float16")),

            ("two2fractal_run_020_4d", "two2fractal_run", ([64, 12, 128, 128], 'zN', "float16")),
            ("two2fractal_run_021_4d", "two2fractal_run", ([64, 12, 128, 128], 'zZ', "float16")),
            ("two2fractal_run_022_4d", "two2fractal_run", ([64, 12, 128, 128], 'nZ', "float16")),

            # Failed case
            # ("two2fractal_run_003", two2fractal_run, ([128, 230], 'zN', "float16")),
            # ("two2fractal_run_003", two2fractal_run, ([128, 230], 'nZ', "float16")),
            # ("two2fractal_run_004_f32", two2fractal_run, ([125, 256], 'nZ', "float32")),

        ]

        self.testarg_rpc_cloud = [
        ]

        self.testarg_level1 = [
            ("two2fractal_run_001", "two2fractal_run", ([2048, 256], 'zN', "float16")),
            ("two2fractal_run_002", "two2fractal_run", ([753, 1024], 'zN', "float16")),
            ("two2fractal_run_003", "two2fractal_run", ([4796, 256], 'zN', "float16")),
            ("two2fractal_run_004", "two2fractal_run", ([2048, 256], 'nZ', "float16")),
            ("two2fractal_run_005", "two2fractal_run", ([753, 1024], 'nZ', "float16")),
            ("two2fractal_run_006", "two2fractal_run", ([4796, 256], 'nZ', "float16")),
        ]
        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run_level1(self):
        self.common_run(self.testarg_level1)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
