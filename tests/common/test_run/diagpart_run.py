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
from test_op.diagpart import diagpart
from gen_random import random_gaussian

def diagpart_run(shape, dtype, kernel_name, attrs, cce_path="./"):
    def gen_data(shape, dtype):
        data = random_gaussian(shape, miu=1, sigma=0.1).astype(dtype)
        out_shape = []
        rank = len(shape)
        for i in range(rank // 2):
            out_shape.append(shape[i])
        expect = np.full(out_shape, 0.0, dtype)
        if rank == 2:
            for i in range(shape[0]):
                expect[i] = data[i, i]
        elif rank == 4:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    expect[i, j] = data[i, j, i, j]
        elif rank == 6:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for m in range(shape[2]):
                        expect[i, j, m] = data[i, j, m, i, j, m]
        elif rank == 8:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for m in range(shape[2]):
                        for n in range(shape[3]):
                            expect[i, j, m, n] = data[i, j, m, n, i, j, m, n]
        else:
            raise RuntimeError("diagpart only support even rank (2/4/6/8) while the rank is {}".format(rank))
        return data, expect

    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        testdata, expect = gen_data(shape, dtype)
        mod = utils.op_build_test(diagpart, [shape], [dtype], kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            out_shape = expect.shape
            out = np.full(out_shape, np.nan, dtype)
            return mod, expect, (testdata, out)
        else:
            return mod
    else:
        testdata, expect = gen_data(shape, dtype)
        out_shape = expect.shape

        out = np.full(out_shape, np.nan, dtype)
        mod = utils.op_build_test(diagpart, [shape], [dtype], kernel_name=kernel_name, attrs=attrs)
        res = utils.mod_launch(mod, (testdata, out), expect=expect)
        cpr_res = compare_tensor(res, expect, rtol=5e-03, equal_nan=True)
        return testdata, res, expect, cpr_res
