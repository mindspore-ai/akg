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
from akg.ms.gpu import Mul
from gen_random import random_gaussian
from akg.utils import kernel_exec as utils

def gen_data(shape, dtype):
    support_list = {"float16": np.float16, "float32": np.float32}
    lhd = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
    rhd = random_gaussian(shape, miu=1, sigma=0.1).astype(support_list[dtype])
    expect = np.multiply(lhd, rhd)
    output = np.full(shape, np.nan, dtype)
    return lhd, rhd, output, expect

def test_ms_mul(shape, dtype):
    mod = utils.op_build(Mul, (shape, shape), (dtype, dtype))    
    lhd, rhd, output, expect = gen_data(shape, dtype)
    output = utils.mod_launch(mod, (lhd, rhd, output), expect = expect)
    np.allclose(output, expect, rtol=5e-03, atol=1.e-8)

if __name__ == '__main__':
    test_ms_mul((1024, 4096), 'float32')
