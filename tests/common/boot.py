# Copyright 2019-2023 Huawei Technologies Co., Ltd
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
This utils is used for running akg test case more easy and relaxed. even by
python common line. the run command is equal to testarg of testcase

example 1: use string test_run function to avoid import it explicitly
-------------------
boot.run("bias_add_64_1024", "bias_add_run", ([64, 1024], "float16"), [(8, 8), (1024, 1024)])

example 2: use test_run function directly
-------------------
from test_run.bias_add_run import bias_add_run
boot.run("bias_add_64_1024", bias_add_run, ([64, 1024], "float16"), [(8, 8), (1024, 1024)])
"""
import os
import sys
from tests.common.base import TestBase


class TestCase(TestBase):
    def setup(self, case, build_only=False):
        self._build_only = False
        func = case[1]
        case_name = "test_akg_boot"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        if isinstance(func, str):
            try:
                lib = "tests.common.test_run.ascend." + func
                exec("import " + lib)
            except Exception as e:
                print("import fail: {} try another path...".format(e))
                lib = "tests.common.test_run." + func
                exec("import " + lib)
                
            mod = sys.modules[lib]
            self._build_only = build_only and hasattr(mod, func.split('_run')[0] + "_compile")
            # backward compatible with xxx_run function entry
            if hasattr(mod, func):
                case = list(case)
                case[1] = getattr(mod, func)
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [case]

    def test_run(self, is_conv=False):
        if self._build_only:
            self.common_run(self.testarg, mode='compile', is_conv=is_conv)
        else:
            self.common_run(self.testarg, is_conv=is_conv)

    def test_lower(self, is_conv=False):
        return self.common_run(self.testarg, mode='lower', is_conv=is_conv)

    def teardown(self):
        self._log.info("============= {0} Teardown============".format(self.casename))


def run(*case):
    a = TestCase()
    a.setup(case)
    a.test_run()
    a.teardown()


def run_conv(*case):
    a = TestCase()
    a.setup(case)
    a.test_run(True)
    a.teardown()


def build(*case):
    a = TestCase()
    a.setup(case, True)
    a.test_run()
    a.teardown()

def lower(*case):
    a = TestCase()
    a.setup(case, True)
    res = a.test_lower()
    a.teardown()
    return res