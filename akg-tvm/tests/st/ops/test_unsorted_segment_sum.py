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
from tests.common.test_run import unsorted_segment_sum_run

############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):
    def setup(self):
        case_name = "unsorted_segment_sum"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.test_args = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("000_case", unsorted_segment_sum_run, ((108365, 8, 1), (108365,), 19717, "float32"), ["level0"]),
        ]

        self.args_ascend = [
            # testflag, opfuncname, testRunArgs, dimArgs
            ("001_uss_1280_1024_8192_fp16", unsorted_segment_sum_run, ([1280, 1024], [1280], 8192, "float16"), ["level0"]),
            ("002_uss_1280_1024_8192_fp32", unsorted_segment_sum_run, ([1280, 1024], [1280], 8192, "float32"), ["level0"]),
            # ("003_uss_128_128_64_32_fp16", unsorted_segment_sum_run, ([128, 128, 64], [128, 128], 34, "float16"), ["level0"]),
            #("004_uss_128_128_64_32_fp32",  unsorted_segment_sum_run, ([128, 128, 64], [128, 128], 34, "float32"), ["level0"]),
            # ("001_uss_1280_1024_1280",  unsorted_segment_sum_run, ([38714,1024], [38714], 30522, "float32"), ["level1"]),
            #("001_uss_1280_1024_1280",  unsorted_segment_sum_run, ([128, 128, 64], [128,128], 33, "float32"), ["level1"]),
        ]
        self.args_rpc_cloud = [
            # testflag, opfuncname, testRunArgs, dimArgs
            ("bert_unsortedsegmentsum_001", unsorted_segment_sum_run, ([1280, 768], [1280], 8192, "float32"), ["level0"]),
        ]
        return True

    def teardown(self):
        self._log.info("{0} Teardown".format(self.casename))
        super(TestCase, self).teardown()
        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        return self.run_cases(self.args_ascend, utils.CCE, "level0")

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run_level1(self):
        return self.run_cases(self.args_ascend, utils.CCE, "level1")

    def test_run_rpc_cloud(self):
        self.common_run(self.args_rpc_cloud)

    @pytest.mark.level0
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_gpu_level0(self):
        return self.run_cases(self.test_args, utils.CUDA, "level0")
    
    # @pytest.mark.level0
    # @pytest.mark.platform_x86_cpu
    # @pytest.mark.env_onecard
    # def test_cpu_level0(self):
    #     return self.run_cases(self.test_args, utils.LLVM, "level0")