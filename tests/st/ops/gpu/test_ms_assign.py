# Copyright 2020 Huawei Technologies Co., Ltd
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
# limitations under the License
import numpy as np
from akg.ms.gpu import assign
from akg.utils import kernel_exec as utils
from tests.common.gen_random import random_gaussian


def gen_data(dtype, ref_shape, val_shape):
    if dtype == "float32":
        ref = random_gaussian(ref_shape, miu=1, sigma=0.1).astype(np.float32)
        val = random_gaussian(val_shape, miu=1, sigma=0.1).astype(np.float32)
    expect = val
    return ref, val, expect


def test_ms_assign(dtype, ref_shape, val_shape):
    ref, val, expect = gen_data(dtype, ref_shape, val_shape)
    mod = utils.op_build_test(assign.Assign, (ref_shape, val_shape), (dtype, dtype), kernel_name="assign")
    fake_output = np.full(val_shape, np.nan, dtype)
    result = utils.mod_launch(mod, (ref, val, fake_output), expect=expect)
    assert np.allclose(result, expect, rtol=5e-03, atol=1.e-8)


if __name__ == '__main__':
    test_ms_assign('float32', (16, 16), (16, 16))
