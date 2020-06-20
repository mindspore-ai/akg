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
from tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from akg.ops.state import clear_zero
from base import get_rtol_atol


def clear_zero_run(shape, dtype, attrs, kernel_name="clear_zero"):
    expect = np.full(shape, 0, dtype)
    data = np.full(shape, np.nan, dtype)
    inout = data

    mod = utils.op_build_test(clear_zero.clear_zero, [shape], [dtype], kernel_name=kernel_name)
    inout = utils.mod_launch(mod, (inout,), outputs=(-1,), expect=expect)
    rtol, atol = get_rtol_atol("clear_zero", dtype)
    return (data), inout, expect, compare_tensor(inout, expect, rtol=rtol, atol=atol, equal_nan=True)
