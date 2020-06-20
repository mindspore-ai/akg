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

import numpy as np
from test_op import div_mod_issue


def div_mod_issue_run(data_shape, weight_shape, casenumber, unused):
    mod = div_mod_issue.div_mod_issue(data_shape, weight_shape, casenumber)

    # DUMMY values
    out_data = np.full(data_shape, 0, 'float16')
    expect = np.full(data_shape, 0, 'float16')

    return input, out_data, expect, True
