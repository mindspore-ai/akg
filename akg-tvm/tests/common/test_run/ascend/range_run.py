# Copyright 2019-2021 Huawei Technologies Co., Ltd
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
sqrt run define
"""

import numpy as np
import akg
import akg.lang.ascend
import akg.tvm
from akg.utils import kernel_exec as utils
from tests.common.test_op.ascend import range as tvm_range
from tests.common.tensorio import compare_tensor


def range_run(start, limit, delta, dtype, attrs):
    t_range = tvm_range.range_value(start, limit, delta, dtype)
    # Create module
    sch = akg.tvm.create_schedule(t_range.op)
    kernel_name = "range"
    with akg.build_config(add_lower_pass=utils.debug_mode(0), dump_pass_ir=True):
        mod = akg.build(sch, [t_range], "cce", name=kernel_name, attrs=attrs, polyhedral=True)
        print(mod.imported_modules[0].get_source())
    # Generate data for testing the op
    expect = np.asarray(list(range(start, limit, delta)))

    output = np.full((max(0, (limit - start) / delta),), np.nan, dtype)
    output = utils.mod_launch(mod, (output, ), expect=expect)

    return tuple(), output, expect, compare_tensor(output, expect, rtol=5e-03, equal_nan=True)
