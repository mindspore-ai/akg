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
################################################

Testcase_PrepareCondition:

Testcase_TestSteps:

Testcase_ExpectedResult:

"""
import os
from tests.common.base import TestBase


class BaseCaseRun(TestBase):
    def setup(self):
        if not super(BaseCaseRun, self).setup():
            return False
        self._log.info("{0} Setup case".format(self.casename))
        return True

    def print_args(self):
        case_arg_attr = self.get_env_var("CASE_ARG_ATTR")
        all_case_flag = self.get_env_var("ALL_CASE_RUN")
        for index, arg in enumerate(self.test_args):
            arg = list(arg)
            if not all_case_flag and len(arg) >= 4 and "Unavailable" in arg[3]:
                continue
            if case_arg_attr and len(arg) >= 4 and case_arg_attr not in arg[3]:
                continue
            print("{0}&{1}&{2}".format(self.casename, index, arg[0]))

    def base_case_run_func(self, attr_flag=None):
        env_dic = os.environ
        if not env_dic.get('TEST_INDEX'):
            for arg in self.test_args:
                case_result = self.run_test_arg_func([arg], attr=attr_flag)
                if not case_result:
                    self._log.info("{0} run failed".format(arg))
                    assert False
        else:
            test_index = int(env_dic.get('TEST_INDEX'))
            self._log.info("test_index:%s", test_index)
            if 0 <= test_index < len(self.test_args):
                arg = self.test_args[test_index]
                self._log.info(arg)
                case_result = self.run_test_arg_func([arg], attr=attr_flag)
                if not case_result:
                    self._log.info("{0} base_case_run_func failed".format(arg))
                    assert False
            else:
                self.print_args()
                self._log.info("test_index is error, test_index:{0}".format(test_index))
                assert False
        return True

    def test_run_level0(self):
        return self.base_case_run_func(attr_flag="level0")

    def test_run_level1(self):
        return self.base_case_run_func(attr_flag="level1")

    def test_run_rpc(self):
        return self.base_case_run_func(attr_flag="rpc")

    def test_run_rpc_cloud(self):
        return self.base_case_run_func(attr_flag="rpc_cloud")

    def teardown(self):
        self._log.info("{0} Teardown".format(self.casename))
        super(BaseCaseRun, self).teardown()
