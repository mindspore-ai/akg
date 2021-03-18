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
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # caseflag,opfuncname,testRunArgs, dimArgs
            # zn to 2d
            ("fractal2two_run_001", "fractal2two_run", ([2, 2, 16, 16], [32, 32], 'zN', "float16", "float16")),
            ("fractal2two_run_002", "fractal2two_run", ([8, 16, 16, 16], [256, 128], 'zN', "float16", "float16")),

            # zz to 2d
            ("fractal2two_run_003", "fractal2two_run", ([2, 2, 16, 16], [32, 32], 'zZ', "float16", "float16")),
            ("fractal2two_run_004", "fractal2two_run", ([8, 16, 16, 16], [128, 256], 'zZ', "float16", "float16")),

            # # Matmul shape
            # zn to 2d
            ("fractal2two_run_005_3d", "fractal2two_run", ([128, 4, 8, 16, 16], [128, 128, 64], 'zN', "float16", "float16")),
            ("fractal2two_run_006_3d", "fractal2two_run", ([128, 8, 48, 16, 16], [128, 768, 128], 'zN', "float16", "float16")),
            ("fractal2two_run_007_4d", "fractal2two_run", ([64, 12, 4, 8, 16, 16], [64, 12, 128, 64], 'zN', "float16", "float16")),
            ("fractal2two_run_008_4d", "fractal2two_run", ([64, 12, 4, 8, 16, 16], [64, 12, 128, 64], 'zN', "float32", "float32")),

            # zz to 2d
            ("fractal2two_run_009_3d", "fractal2two_run", ([128, 4, 8, 16, 16], [128, 64, 128], 'zZ', "float16", "float16")),
            ("fractal2two_run_010_3d", "fractal2two_run", ([128, 8, 48, 16, 16], [128, 128, 768], 'zZ', "float16", "float16")),
            ("fractal2two_run_011_4d", "fractal2two_run", ([64, 12, 4, 8, 16, 16], [64, 12, 64, 128], 'zZ', "float16", "float16")),
            ("fractal2two_run_012_4d", "fractal2two_run", ([64, 12, 4, 8, 16, 16], [64, 12, 64, 128], 'zZ', "float32", "float32")),
        ]

        self.testarg_rpc_cloud = [
        ]
        self.testarg_level1 = [
            ("fractal2two_run_002", "fractal2two_run", ([288, 288, 16, 16], [4608, 4608], 'zN', "float16", "float16")),
            ("fractal2two_run_002", "fractal2two_run", ([288, 288, 16, 16], [4608, 4608], 'zZ', "float16", "float16")),
            ("fractal2two_run_002", "fractal2two_run", ([288, 288, 16, 16], [4608, 4608], 'zN', "float32", "float32")),
            ("fractal2two_run_002", "fractal2two_run", ([288, 288, 16, 16], [4608, 4608], 'zZ', "float32", "float32")),
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

    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

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


if __name__ == "__main__":
    t = TestCase()
    t.setup()
    t.test_run()
    t.teardown()
