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

import sys

sys.path.append(os.getcwd())
from base_all_run import BaseCaseRun


class TestAutodiff001(BaseCaseRun):
    def setup(self):
        """
        testcase preparcondition
        :return:
        """
        case_name = "test_autodiff_all_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        if not super(TestAutodiff001, self).setup():
            return False

        self.test_args = []


def print_args():
    cls = TestAutodiff001()
    cls.setup()
    cls.print_args()


if __name__ == "__main__":
    print_args()
