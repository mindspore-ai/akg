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
from tests.common.test_run import batch_matmul_run

############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):
    def __init__(self):
        self.case_name = "batch_matmul"
        self.case_path = os.getcwd()
        self.args_default = [
            ("000_case", batch_matmul_run, ((128, 64), (128, 64), 'float32', 'float32', "NHDT", "NHDT", "NHDT",
                (1, ), False, False), ["level0"]),
        ]
        self.args_gpu = [
            ("001_case", batch_matmul_run, ((32, 12, 128, 128), (32, 12, 128, 64), 'float16', 'float16', "NHDT",
                "NHTD", "NHDT", (1, ), False, True), ["level0"]),
            ("002_case", batch_matmul_run, ((256, 128), (64, 128), 'float16', 'float16', "NHDT", "NHDT", "NHDT",
                (1, ), False, True), ["level0"]),
            ("003_case", batch_matmul_run, ((128, 32), (128, 512), 'float16', 'float16', "NHTD", "NHTD", "NHDT",
                (1, ), False, True), ["level0"]),
            ("004_case", batch_matmul_run, ((128, 64), (64, 32), 'float16', 'float16', "NHDT", "NHTD", "NHDT",
                (1, ), False, True), ["level0"]),
            ("005_case", batch_matmul_run, ((32, 1, 128, 64), (32, 1, 64, 32), 'float16', 'float16', "NHDT", "NHTD", "NHDT",
                (1, ), False, True), ["level0"]),
        ]

    def setup(self):
        self.params_init(self.case_name, self.case_path)
        return True

    def run_gpu_level0(self):
        return self.run_cases(self.args_default + self.args_gpu, utils.CUDA, "level0")
    
    def run_cpu_level0(self):
        return self.run_cases(self.args_default, utils.LLVM, "level0")

    def teardown(self):
        self._log.info("{0} Teardown".format(self.casename))
        super(TestCase, self).teardown()
        return

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_level0():
    test_case = TestCase()
    test_case.setup()
    test_case.run_gpu_level0()
    test_case.teardown()

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cpu_level0():
    test_case = TestCase()
    test_case.setup()
    test_case.run_cpu_level0()
    test_case.teardown()