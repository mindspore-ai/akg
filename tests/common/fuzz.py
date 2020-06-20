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
This utils is used for running fuzz test case more easy and relaxed. even by
python common line. the run command is equal to testarg of testcase

use string test_run function to avoid import it explicitly
-------------------
import fuzz
fuzz.run("add", [(677, 1474), (677, 1474)], ['float32', 'float32'], [], 'add_float32')
"""

import os
import time
from base import TestBase
from akg.utils import kernel_exec as utils


class TestCase(TestBase):
    def setup(self, case):
        self.test_arg = case
        case_name = "test_akg_fuzz"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

    def test_run(self):
        self.run_op_build_compile(self.translate_func_name(self.test_arg))

    def translate_func_name(self, args):
        """
        args such as ('reshape', 'reshape.reshape', ([(435, 888)], ['float32'], [[435, 888]], 'reshape_float32'))
        """
        args_list = [args[0], self.import_get_func(args[1])]
        for arg in args[2]:
            args_list.append(arg)
        return tuple(args_list)

    def import_get_func(self, func_name):
        """
        import get dsl function
        """
        op_fromlist = ["akg.ops.array.", "akg.ops.nn.", "akg.ops.math.", "akg.ops.optimizers.", "akg.ops.state."]
        func_name = func_name.split(".")
        for op_from_path in op_fromlist:
            try:
                op_fromlist = op_from_path + func_name[0]
                op_func_py = __import__(op_fromlist, fromlist=func_name[0])
                op_func = getattr(op_func_py, func_name[1])
            except ImportError:
                continue
            if op_func is not None:
                break
        else:
            op_fromlist = "test_op." + func_name[0]
            op_func_py = __import__(op_fromlist, fromlist=func_name[0])
            op_func = getattr(op_func_py, func_name[1])
        return op_func

    def run_op_build_compile(self, arg):
        arg_info = (arg[0], arg[1].__name__, arg[2:])
        mod = None
        t0 = time.time()
        try:
            mod = utils.op_build_test(*arg[1:])
        except Exception as e:
            self._log.error(e)
            TestBase.pandora_logger_.traceback()
        finally:
            if not mod:
                self._log.error("run_op_build_compile :: circle {0} fail !".format(arg))
                self._log.error("run_op_build_compile :: compile failed !")
                result = "fail"
            else:
                result = "succ"
                t1 = time.time()
                self._log.info("run_fuzz_result_count func time test: args:%s, result:%s, running:%s seconds",
                               arg_info, result, str(t1 - t0))
        if not result:
            assert result
        return True

    def teardown(self):
        self._log.info("============= {0} Teardown============".format(self.casename))


def run(*case):
    a = TestCase()
    a.setup(case)
    a.test_run()
    a.teardown()
