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

from nose.plugins.attrib import attr

from test_resnet50_all_001 import TestResnet50_001 as Resnet50Case
from test_alexnet_all_001 import TestAlexnet as AlexnetCase
from test_lenet_all_001 import TestLenet as LenetCase
from test_autodiff_all_001 import TestAutodiff001 as AutodiffCase


class TestAllnet():
    def setup(self):
        """
        testcase preparcondition
        :return:
        """
        casename = "test_allnet_all_001"
        casepath = os.getcwd()
        self.case_net_list = [Resnet50Case(), AlexnetCase(), LenetCase(), AutodiffCase()]


def print_args():
    cls = TestAllnet()
    cls.setup()
    for case_net in cls.case_net_list:
        case_net.print_args()


if __name__ == "__main__":
    print_args()
