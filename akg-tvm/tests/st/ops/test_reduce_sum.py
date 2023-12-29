# Copyright 2021 Huawei Technologies Co., Ltd
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

import os
import pytest
import akg.utils as utils
from tests.common.base import TestBase
from tests.common.test_run import reduce_sum_run

############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_sum_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("001_sum", reduce_sum_run, ((32, 64, 32), (0, 1), False, "float16")),
            ("002_sum", reduce_sum_run, ((17, 5, 7), (0, 1, 2), True, "float16")),
            ("003_sum", reduce_sum_run, ((17, 5, 127), (0,), True, "float16")),
            ("004_sum", reduce_sum_run, ((17, 5, 17), (0,), True, "float16")),
            ("005_sum", reduce_sum_run, ((17, 17), (0,), True, "float16")),
            ("006_sum", reduce_sum_run, ((64, 128, 1024), (0, 1), True, "float32")),
            ("007_sum", reduce_sum_run, ((64, 128, 1024), (), True, "float16")),
            ("test_sum_001", reduce_sum_run, ((32, 15, 7, 16), (0,), False, "float16")),

            # fail ("test_sum_002",reduce_sum_run,((32,15,7,16), 1, False, "float16"), ((32, 0), (7, 0),(16,16),(15,16))),

            # fail ("test_sum_003",reduce_sum_run,((32,13), 1, False, "float16"), ((32, 0), (13, 32))),

            # fail ("test_sum_004", reduce_sum_run, ((32, 16, 13), 2, False, "float16"), ((32, 1), (16, 1), (13, 13))),
            ("test_sum_005", reduce_sum_run, ((64, 128, 1024), (0, 1), False, "float16"), ),
            ("test_sum_006", reduce_sum_run, ((1, 8), (1,), False, "float16")),
            ("test_sum_007", reduce_sum_run, ((64, 128, 1024), (2,), False, "float16"), ),

            # fail ("test_sum_008", reduce_sum_run, ((1280, 30522), (1,), False, "float16"), ((8, 8), (1024, 1024))),

            ("test_sum_009", reduce_sum_run, ((64, 2), (1,), False, "float16")),
            ("test_sum_010", reduce_sum_run, ((1, 1280), (1,), False, "float16")),
            ("test_sum_011", reduce_sum_run, ((64, 128, 1), (2,), False, "float16")),
            ("test_sum_012", reduce_sum_run, ((8192, 1024), (1,), False, "float16")),

            # fail ("test_sum_013", reduce_sum_run, ((1280, 30522), (1,), False, "float16"), ((1, 1), (30522, 30522))),

            ("test_sum_014", reduce_sum_run, ((1, 398), (1,), False, "float16")),
            ("test_sum_015", reduce_sum_run, ((1280, 1), (1,), False, "float16")),

            # fail ("test_sum_016", reduce_sum_run, ((160,30522), (1,), False, "float16"), ((1, 1),(30522,30522))),

            ("test_sum_017", reduce_sum_run, ((1280, 1024), (1,), False, "float16")),
            ("test_sum_018", reduce_sum_run, ((64, 2), (1,), False, "float16")),

            # fail ("test_sum_019", reduce_sum_run, ((64,128,1024), (0,), False, "float16"), ((2, 2),(8,8),(1024,1024))),

            ("test_sum_020", reduce_sum_run, ((64, 16, 128, 128), (3,), False, "float16")),
            ("test_sum_021", reduce_sum_run, ((1, 160), (1,), False, "float16")),
            ("test_sum_022", reduce_sum_run, ((8, 2), (1,), False, "float16")),
            ("test_sum_023", reduce_sum_run, ((8092, 1024), (0,), False, "float16"),),
            ("test_sum_024", reduce_sum_run, ((8192, 1), (1,), False, "float16")),
            ("test_sum_025", reduce_sum_run, ((1280, 1024), (0,), False, "float16")),
            ("test_sum_026", reduce_sum_run, ((4, 3, 8, 8), (0, 2, 3), False, "float16")),
            ("test_sum_027", reduce_sum_run, ((4, 3, 8, 8), (1, 3), False, "float16")),
            ("test_sum_028", reduce_sum_run, ((8, 256), (1,), False, "float16")),

            # case29 is added for multi_last_axis_reduction
            ("test_sum_029", reduce_sum_run, ((17, 5, 7), (0, 1, 2), False, "float16")),
        ]

        self.testarg_rpc_cloud = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            # float - int32:[64, 128, 1024] - [2] = float:[1024]:ok
            ("001_sum", reduce_sum_run, ((64, 128, 1024), (0, 1), False, "float32")),
            # float - int32:[8] - [1] = float:[]:
            ("002_sum", reduce_sum_run, ((8,), (0,), False, "float32")),
            # float - int32:[64, 128, 1024] - [1] = float:[64, 128]
            ("003_sum", reduce_sum_run, ((64, 128, 1024), (2,), False, "float32")),
            # float - int32:[1280, 30522] - [1] = float:[1280]
            ("004_sum", reduce_sum_run, ((1280, 30522), (1,), False, "float32")),
            # float - int32:[64, 2] - [] = float:[64, 1]
            ("005_sum", reduce_sum_run, ((64, 2), (1,), True, "float32")),
            # float - int32:[1280] - [1] = float:[]
            ("006_sum", reduce_sum_run, ((1280,), (0,), False, "float32")),
            # float - int32:[64, 128, 1] - [1] = float:[64, 128]
            ("007_sum", reduce_sum_run, ((64, 128, 1), (2,), False, "float32")),
            # float - int32:[8192, 1024] - [1] = float:[8192]:ok
            ("008_sum", reduce_sum_run, ((8192, 1024), (1,), False, "float32")),
            # float - int32:[1280, 30522] - [] = float:[1280, 1]:ok
            ("009_sum", reduce_sum_run, ((1280, 30522), (1,), True, "float32")),
            # float - int32:[398] - [1] = float:[]
            ("010_sum", reduce_sum_run, ((398,), (0,), True, "float32")),
            # float - int32:[1280, 1] - [1] = float:[1280]:ok
            ("011_sum", reduce_sum_run, ((1280, 1), (1,), False, "float32")),
            # float - int32:[160, 30522] - [1] = float:[160]:ok
            ("012_sum", reduce_sum_run, ((160, 30522), (1,), False, "float32")),
            # float - int32:[1280, 1024] - [1] = float:[1280]:ok
            ("013_sum", reduce_sum_run, ((1280, 1024), (1,), False, "float32")),
            # float - int32:[64, 2] - [] = float:[64]:ok
            ("014_sum", reduce_sum_run, ((64, 2), (1,), False, "float32")),
            # float - int32:[64, 128, 1024] - [1] = float:[128, 1024]:
            ("015_sum", reduce_sum_run, ((64, 128, 1024), (0,), False, "float32")),
            # float - int32:[64, 16, 128, 128] - [] = float:[64, 16, 128, 1]:ok
            ("016_sum", reduce_sum_run, ((64, 16, 128, 128), (3,), True, "float32")),
            # float - int32:[160] - [1] = float:[]
            ("017_sum", reduce_sum_run, ((160,), (0,), True, "float32")),
            # float - int32:[8, 2] - [] = float:[8]
            ("018_sum", reduce_sum_run, ((8, 2), (1,), True, "float32")),
            # float - int32:[8192, 1024] - [1] = float:[1024]:ok
            ("019_sum", reduce_sum_run, ((8192, 1024), (0,), False, "float32")),
            # float - int32:[8192, 1] - [1] = float:[8192]:ok
            ("020_sum", reduce_sum_run, ((8192, 1), (1,), False, "float32")),
            # float - int32:[1280, 1024] - [1] = float:[1024]:ok
            ("021_sum", reduce_sum_run, ((1280, 1024), (0,), False, "float32")),
            # float - int32:[3072] - [1] = float:[1]
            ("022_sum", reduce_sum_run, ((3072,), (0,), True, "float32")),
            # float - int32:[64, 128, 768] - [2] = float:[768]
            ("023_sum", reduce_sum_run, ((64, 128, 768), (0, 1), False, "float32")),
            # half - int32:[8192, 1] - [1] = half:[8192]
            ("024_sum", reduce_sum_run, ((8092, 1), (1,), False, "float16")),
            # float - int32:[1280, 768] - [1] = float:[1280]
            ("025_sum", reduce_sum_run, ((1280, 768), (1,), False, "float32")),
            # float - int32:[3072, 768] - [2] = float:[1, 1]
            ("026_sum", reduce_sum_run, ((3072, 768), (0, 1), True, "float32")),
            # float - int32:[768, 3072] - [2] = float:[1, 1]
            ("027_sum", reduce_sum_run, ((768, 3072), (0, 1), True, "float32")),
            # float - int32:[1280, 768] - [1] = float:[768]
            ("028_sum", reduce_sum_run, ((1280, 768), (0,), False, "float32")),
            # float - int32:[1280] - [1] = float:[]
            ("029_sum", reduce_sum_run, ((1280,), (0,), True, "float32")),
            # float - int32:[21128] - [1] = float:[1]
            ("030_sum", reduce_sum_run, ((21128,), (0,), True, "float32")),
            # float - int32:[21128, 768] - [2] = float:[1, 1]
            ("031_sum", reduce_sum_run, ((21128, 768), (0, 1), True, "float32")),
            # float - int32:[1280, 21128] - [1] = float:[1280]
            ("032_sum", reduce_sum_run, ((21128, 768), (1,), True, "float32")),
            # half - int32:[64, 12, 128, 128] - [] = half:[64, 12, 128, 1]
            ("033_sum", reduce_sum_run, ((64, 12, 128, 128), (3,), True, "float16")),
            # half - int32:[8192, 768] - [1] = half:[768]
            ("034_sum", reduce_sum_run, ((8192, 768), (0,), False, "float16")),
            # float - int32:[64, 2] - [] = float:[64]
            ("035_sum", reduce_sum_run, ((64, 2), (1,), False, "float32")),
            # float - int32:[768, 768] - [2] = float:[1, 1]
            ("036_sum", reduce_sum_run, ((768, 768), (0, 1), False, "float32")),
            # half - int32:[8192, 768] - [1] = half:[8192]
            ("037_sum", reduce_sum_run, ((8192, 768), (1,), False, "float16")),
            # float - int32:[2] - [1] = float:[1]
            ("038_sum", reduce_sum_run, ((2, 1), (0,), False, "float32")),
            # float - int32:[1280, 1] - [1] = float:[1280]
            ("039_sum", reduce_sum_run, ((1280, 1), (1,), False, "float32")),
            # float - int32:[33, 64] - [2] = float:[1, 1]
            ("040_sum", reduce_sum_run, ((33, 64), (0, 1), True, "float32")),
            # float - int32:[2, 768] - [2] = float:[1, 1]
            ("041_sum", reduce_sum_run, ((2, 768), (0, 1), True, "float32")),
            # float - int32:[768] - [1] = float:[1]
            ("042_sum", reduce_sum_run, ((768, 1), (0,), False, "float32")),
            # float - int32:[64, 128, 1] - [1] = float:[64, 128]
            ("043_sum", reduce_sum_run, ((64, 128, 1), (2,), False, "float32")),
            # float - int32:[64, 128, 768] - [1] = float:[64, 128]
            ("044_sum", reduce_sum_run, ((64, 128, 768), (2,), False, "float32")),
            # float - int32:[64, 2] - [] = float:[64, 1]
            ("045_sum", reduce_sum_run, ((64, 2), (1,), True, "float32")),
            # float - int32:[1280, 21128] - [] = float:[1280, 1]
            ("046_sum", reduce_sum_run, ((1280, 21128), (1,), False, "float32")),

            ("005_sum", reduce_sum_run, ((64, 128, 1024), (2,), False, "float32")),
            # fail for now:
            # ("004_sum", reduce_sum_run, ((8,), (0,), False, "float32")),
            # ("008_sum", reduce_sum_run, ((1280,), (0,), False, "float32")),
            # ("012_sum", reduce_sum_run, ((398,), (0,), False, "float32")),
            # ("019_sum", reduce_sum_run, ((160,), (0,), False, "float32")),
            #
            # ("006_sum", reduce_sum_run, ((1280, 30522), (1,), False, "float32")),
            # ("007_sum", reduce_sum_run, ((64, 2), 1, True, "float32")),
            # ("009_sum", reduce_sum_run, ((64, 128, 1), (2,), False, "float32")),
            # ("010_sum", reduce_sum_run, ((8192, 1024), (1,), False, "float32")),
            # ("011_sum", reduce_sum_run, ((1280, 30522), (1,), True, "float32")),
            # ("013_sum", reduce_sum_run, ((1280, 1), (1,), False, "float32")),
            # ("014_sum", reduce_sum_run, ((160, 30522), (1,), False, "float32")),
            # ("015_sum", reduce_sum_run, ((1280, 1024), (1,), False, "float32")),
            # ("016_sum", reduce_sum_run, ((64, 2), 1, False, "float32")),
            # ("017_sum", reduce_sum_run, ((64, 128, 1024), (0,), False, "float32")),
            # ("018_sum", reduce_sum_run, ((64, 16, 128, 128), (3,), True, "float32")),
            # ("020_sum", reduce_sum_run, ((8, 2), 1, False, "float32")),
            # ("021_sum", reduce_sum_run, ((8192, 1024), (0,), False, "float32")),
            # ("022_sum", reduce_sum_run, ((8192, 1), (1,), False, "float32")),
            # ("023_sum", reduce_sum_run, ((1280, 1024), (0,), False, "float32")),
            ("001_ssd_sum", reduce_sum_run, ((8, 8732, 4), (2,), False, "float32")),

        ]

        self.testarg_cloud_level0 = [
            ("test_sum_006", reduce_sum_run, ((1, 8), (1,), False, "float32"), ((8, 8),)),
            ("test_sum_007", reduce_sum_run, ((1, 8), (), False, "float32"))
        ]
        self.testarg_level2 = [
            # caseflag,opfuncname,testRunArgs, dimArgs

            ("test_sum_002", reduce_sum_run, ((32, 15, 7, 16), 1, False, "float16")),

            ("test_sum_003", reduce_sum_run, ((32, 13), 1, False, "float16")),

            ("test_sum_004", reduce_sum_run, ((32, 16, 13), 2, False, "float16")),

            ("test_sum_007", reduce_sum_run, ((64, 128, 1024), (2,), False, "float16")),

            ("test_sum_008", reduce_sum_run, ((1280, 30522), (1,), False, "float16")),

            ("test_sum_013", reduce_sum_run, ((1280, 30522), (1,), False, "float16")),

            ("test_sum_016", reduce_sum_run, ((160, 30522), (1,), False, "float16")),

            ("test_sum_019", reduce_sum_run, ((64, 128, 1024), (0,), False, "float16")),

            # ("test_sum_020", reduce_sum_run, ((2,255,13), (0,1), False, "float16"), ((1, 1),(1,1),(1,1))),
            # ("test_sum_021", reduce_sum_run, ((16, 16, 16, 16), (1, 2), False, "float32")),

        ]

        self.test_args = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("000_case", reduce_sum_run, ((9, 1024, 1024), 1, True, 'float32'), ["level0"]),
            ("001_case", reduce_sum_run, ((9, 1024, 1024), 2, True, 'float32'), ["level0"]),
            ("002_case", reduce_sum_run, ((9, 1024, 1024), None, True, 'float32'), ["level0"]),
            ("003_case", reduce_sum_run, ((10240,), None, True, 'float32'), ["level0"]),
        ]
        return True

    def teardown(self):
        self._log.info("{0} Teardown".format(self.casename))
        super(TestCase, self).teardown()
        return

    @pytest.mark.level0
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_gpu_level0(self):
        return self.run_cases(self.test_args, utils.CUDA, "level0")
    
    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_cpu_level0(self):
        return self.run_cases(self.test_args, utils.LLVM, "level0")

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_run_rpc_cloud(self):
        self.common_run([self.testarg_rpc_cloud[15], self.testarg_rpc_cloud[-1]])

    def test_run_cloud_level0(self):
        self.common_run(self.testarg_cloud_level0)

    def test_run_level2(self):
        self.common_run(self.testarg_level2)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return


if __name__ == "__main__":
    t = TestCase()
    t.setup()
    t.test_run()
    t.teardown()
