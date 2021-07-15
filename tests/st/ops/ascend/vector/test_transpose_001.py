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
transpose
"""
import os
import pytest
from tests.common.base import TestBase


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_transpose_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # caseflag,opfuncname,testRunArgs, dimArgs
            ###
            ("transpose_run31", "transpose_run", ((8, 24, 1, 1), (0, 2, 3, 1), "float32")),
            ("transpose_run31", "transpose_run", ((8, 24, 3, 3), (0, 2, 3, 1), "float32")),
            ("transpose_run31", "transpose_run", ((8, 36, 5, 5), (0, 2, 3, 1), "float32")),
            ("transpose_run31", "transpose_run", ((8, 36, 10, 10), (0, 2, 3, 1), "float32")),
            ("transpose_run31", "transpose_run", ((8, 36, 19, 19), (0, 2, 3, 1), "float32")),
            ("transpose_run31", "transpose_run", ((8, 24, 38, 38), (0, 2, 3, 1), "float32")),
            # ("transpose_run31", "transpose_run", ((8,1,1,24),(0,3,1,2), "float32")),
            ("transpose_run31", "transpose_run", ((8, 16, 1, 1), (0, 2, 3, 1), "float32")),
            ("transpose_run31", "transpose_run", ((8, 16, 3, 3), (0, 2, 3, 1), "float32")),
            ("transpose_run31", "transpose_run", ((8, 24, 5, 5), (0, 2, 3, 1), "float32")),
            ("transpose_run31", "transpose_run", ((8, 24, 10, 10), (0, 2, 3, 1), "float32")),
            ("transpose_run31", "transpose_run", ((8, 24, 19, 19), (0, 2, 3, 1), "float32")),
            ("transpose_run31", "transpose_run", ((8, 16, 38, 38), (0, 2, 3, 1), "float32")),
            ("transpose_run31", "transpose_run", ((8, 1, 1, 16), (0, 3, 1, 2), "float32")),

            # ("transpose_run31", "transpose_run", ((8,3,3,24),(0,3,1,2), "float32")),
            ("transpose_run31", "transpose_run", ((8, 3, 3, 16), (0, 3, 1, 2), "float32")),
            # ("transpose_run31", "transpose_run", ((8,3,3,24),(0,3,1,2),"float32")),
            # ("transpose_run31", "transpose_run", ((8,5,5,36),(0,3,1,2), "float32")),
            # ("transpose_run31", "transpose_run", ((8,10,10,36),(0,3,1,2),"float32")),
            # ("transpose_run31", "transpose_run", ((8,19,19,36),(0,3,1,2),"float32")),
            # ("transpose_run31", "transpose_run", ((8,38,38,24),(0,3,1,2),"float32")),
            # ("transpose_run31", "transpose_run", ((8,10,10,24),(0,3,1,2),"float32")),
            # ("transpose_run31", "transpose_run", ((8,19,19,24),(0,3,1,2),"float32")),
            ("transpose_run31", "transpose_run", ((8, 38, 38, 16), (0, 3, 1, 2), "float32")),
            ##("transpose_run1", "transpose_run", ((64, 16, 128, 64), (0, 2, 1, 3), "float16"),((4,4),(4,4),(128,128),(64,64))),
            ##("transpose_run2", "transpose_run", ((8, 16, 128, 64), (0, 2, 1, 3), "float16"),((2,2),(2,2),(128,128),(64,64))),
            ("transpose_run3", "transpose_run", ((8, 128, 16, 64), (0, 2, 1, 3), "float16")),
            ##("transpose_run4", "transpose_run", ((64, 128, 16, 64), (0, 2, 1, 3), "float16"),((4,4),(8,8),(16,16),(64,64)))
            ##("transpose_run5", "transpose_run", ((64, 512, 16, 64), (0, 2, 1, 3), "float16"),((4,4),(8,8),(16,16),(64,64))),
            ##("transpose_run6", "transpose_run", ((64, 16, 512, 64), (0, 2, 1, 3), "float16"),((4,4),(8,8),(16,16),(64,64))),
            ("transpose_run7", "transpose_run", ((16, 16), (1, 0,), "float16"), ((16, 16), (16, 16))),
            ("transpose_run8", "transpose_run", ((5, 12), (1, 0,), "float16"), ((16, 1), (16, 1))),
            ("transpose_run9", "transpose_run", ((128, 16), (1, 0,), "float16"), ((128, 128), (128, 128))),
            ("transpose_run10", "transpose_run", ((128, 128), (1, 0,), "float16"), ((128, 128), (128, 128))),
            ("transpose_run11", "transpose_run", ((16, 128), (1, 0,), "float16"), ((128, 128), (128, 128))),
            ("transpose_run11", "transpose_run", ((16, 9), (1, 0,), "float16"), ((16, 1), (16, 1))),
            ("transpose_run12", "transpose_run", ((4, 8, 16, 16), (0, 1, 3, 2), "float16"),
             ((32, 32), (32, 32), (32, 32), (32, 32))),
            ("transpose_run13", "transpose_run", ((4, 8, 128, 128), (0, 1, 3, 2), "float16"),
             ((1, 1), (1, 1), (128, 128), (128, 128))),
            # ("transpose_run14", "transpose_run", ((4, 8, 64, 16), (0, 1, 3, 2), "float16"),((64, 64), (64, 64), (64,64),(64,64))),

        ]

        self.testarg_rpc_cloud = [
            # caseflag,opfuncname,testRunArgs, dimArgs
            ("transpose_run1_fp32", "transpose_run", ((64, 16, 128, 64), (0, 2, 1, 3), "float32")),
            ("transpose_run2_fp32", "transpose_run", ((8, 16, 128, 64), (0, 2, 1, 3), "float32")),
            ("transpose_run3_fp32", "transpose_run", ((8, 128, 16, 64), (0, 2, 1, 3), "float32")),
            ("transpose_run4_fp32", "transpose_run", ((64, 128, 16, 64), (0, 2, 1, 3), "float32")),

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
            ("transpose_006", "transpose_run", ((64, 12, 128, 128), (2, 0, 1, 3), "float32")),
            # half - int32:[128, 64, 12, 64] - [4] = half:[64, 12, 128, 64]
            ("transpose_007", "transpose_run", ((128, 64, 12, 64), (1, 2, 0, 3), "float32")),
            # half - int32:[128, 64, 12, 128] - [4] = half:[64, 12, 128, 128]
            ("transpose_008", "transpose_run", ((128, 64, 12, 128), (1, 2, 0, 3), "float32")),
            # half - int32:[64, 128, 12, 64] - [4] = half:[64, 12, 128, 64]
            ("transpose_009", "transpose_run", ((64, 128, 12, 64), (0, 2, 1, 3), "float32")),
            # half - int32:[64, 12, 128, 64] - [4] = half:[128, 64, 12, 64]
            ("transpose_010", "transpose_run", ((64, 12, 128, 64), (2, 0, 1, 3), "float32")),
        ]
        self.testarg_level1 = [
            # caseflag,opfuncname,testRunArgs, dimArgs

            # run fail ("transpose_run1", transpose_run, ((64, 16, 128, 64), (0, 2, 1, 3), "float16"),((4,4),(4,4),(128,128),(64,64))),
            # run fail ("transpose_run2", transpose_run, ((8, 16, 128, 64), (0, 2, 1, 3), "float16"),((2,2),(2,2),(128,128),(64,64))),
            # run fail ("transpose_run5", transpose_run, ((64, 512, 16, 64), (0, 2, 1, 3), "float16"),((4,4),(8,8),(16,16),(64,64))),
            # run fail ("transpose_run6", transpose_run, ((64, 16, 512, 64), (0, 2, 1, 3), "float16"),((4,4),(8,8),(16,16),(64,64))),
            # level 0 ("transpose_run3", transpose_run, ((8, 128, 16, 64), (0, 2, 1, 3), "float16"),((4,4),(8,8),(16,16),(64,64))),
            ("transpose_run4", "transpose_run", ((64, 128, 16, 64), (0, 2, 1, 3), "float16"),
             ((4, 4), (8, 8), (16, 16), (64, 64)))

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

    '''
    def test_run_rpc_cloud(self):
        self.common_run([self.testarg_rpc_cloud[0]])

    def test_run_level1(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_level1)
	'''

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
