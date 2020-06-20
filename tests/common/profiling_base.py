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

import os
from multiprocessing import Process

from base import TestBase, PERFORMANCE_TEST
from akg.utils.kernel_exec import PERFORMANCE_TEST_FILE

TIMEOUT = 600


class ProfilingTestBase(TestBase):

    def __init__(self, casename, testcases):
        """
        testcase preparcondition
        :return:
        """
        casepath = os.getcwd()
        super(ProfilingTestBase, self).__init__(casename, casepath)
        self.testcases = testcases

    def setup(self):
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.result_file = os.path.join(self.caselog_path, self.casename + ".csv")
        os.environ[PERFORMANCE_TEST] = "True"
        os.environ[PERFORMANCE_TEST_FILE] = self.result_file
        return

    def _get_test_case_perf(self, test_case):
        _, func, args, _ = self.ana_args(test_case)
        func_name = func if isinstance(func, str) else func.__name__
        operator_name = func_name.split("_run")[0]
        p_file = open(self.result_file, 'a+')
        p_file.write("%s; %s; " % (operator_name, args))
        p_file.close()
        is_conv = True if "conv" in operator_name else False
        self.common_run([test_case], is_conv=is_conv)

    def test_run_perf(self):
        """
        run case.
        :return:
        """
        for test_case in self.testcases:
            # For the profiling tool, each test case must run with a new process
            p = Process(target=self._get_test_case_perf, args=(test_case,))
            p.start()
            p.join(timeout=TIMEOUT)
            if p.is_alive():
                p.terminate()
                raise RuntimeError("process for {0} timeout!".format(test_case))

    def teardown(self):
        """
        clean environment
        :return:
        """
        os.environ.pop(PERFORMANCE_TEST_FILE)
        os.environ.pop(PERFORMANCE_TEST)
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
